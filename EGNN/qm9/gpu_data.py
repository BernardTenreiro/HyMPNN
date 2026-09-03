"""GPU-resident data pipeline.

The whole processed QM9 dataset is 86 MB against 143 GB of VRAM, so there is no
reason to stream it. Uploading once and batching on-device removes, rather than
merely overlaps, everything the host was doing per batch:

  * the DataLoader and its worker processes, pinning and per-batch H2D copies
  * collate: pad_sequence, drop_zeros, the (bs, n, n) edge_mask, and the
    nonzero() that compressed it
  * preprocess_input, which recomputed the same per-molecule node features
    every epoch even though they never change

Node features and the per-molecule atom counts are precomputed once for the
whole split. A batch is then an index_select plus one edge-index build.

Semantics match the DataLoader path: same padding-trim rule (n = max atoms in
the batch), same compressed edge set (ordered pairs of real atoms, no
self-loops, rows ascending), same node/edge masks.
"""
import torch


class GPUBatches:
    """Batch iterator over a split held entirely in device memory."""

    def __init__(self, data, prop, charge_power, charge_scale, device,
                 dtype=torch.float32, perm=None):
        d = {k: v for k, v in data.items()}
        idx = perm if perm is not None else torch.arange(len(d['charges']))
        self.device, self.dtype = device, dtype

        charges = d['charges'][idx].to(device)                     # (M, n)
        self.positions = d['positions'][idx].to(device, dtype)     # (M, n, 3)
        self.label = d[prop][idx].to(device, dtype)                # (M,)
        self.atom_mask = (charges > 0)                             # (M, n)
        self.num_atoms = self.atom_mask.sum(1)                     # (M,)

        # Node features are a pure function of (one_hot, charges) -- constant
        # across epochs, so pay for them once instead of once per batch.
        one_hot = d['one_hot'][idx].to(device, dtype)
        ch = charges.to(device, dtype)
        powers = torch.arange(charge_power + 1., device=device, dtype=dtype)
        ct = (ch.unsqueeze(-1) / charge_scale).pow(powers)          # (M, n, P+1)
        self.nodes = (one_hot.unsqueeze(-1) * ct.unsqueeze(-2)).view(
            *charges.shape, -1)                                     # (M, n, 15)
        self.M = charges.size(0)
        self.max_n = charges.size(1)

    def __len__(self):
        return self.M

    @staticmethod
    def _edges(atom_mask, n_nodes):
        """Compressed dense edge index: ordered pairs of real atoms, no self-loops.

        Identical set and order to collate_fn's compressed path, so `rows` is
        ascending -- which the fused dense kernel's segmented aggregation needs.
        """
        em = atom_mask.unsqueeze(2) & atom_mask.unsqueeze(1)         # (bs, n, n)
        em = em & ~torch.eye(n_nodes, dtype=torch.bool,
                             device=atom_mask.device).unsqueeze(0)
        keep = em.reshape(-1).nonzero(as_tuple=True)[0]
        b = torch.div(keep, n_nodes * n_nodes, rounding_mode='floor')
        rem = keep - b * (n_nodes * n_nodes)
        i = torch.div(rem, n_nodes, rounding_mode='floor')
        return b * n_nodes + i, b * n_nodes + (rem - i * n_nodes)

    def epoch(self, batch_size, shuffle=True, generator=None, fixed_n=None):
        order = (torch.randperm(self.M, device=self.device, generator=generator)
                 if shuffle else torch.arange(self.M, device=self.device))
        for s in range(0, self.M, batch_size):
            idx = order[s:s + batch_size]
            bs = idx.numel()
            # Trim padding to the batch's widest molecule, as collate_fn does.
            # fixed_n keeps shapes static instead (costs a little node work,
            # but makes CUDA-graph capture possible).
            n = self.max_n if fixed_n else int(self.num_atoms[idx].max())
            am = self.atom_mask[idx, :n]
            rows, cols = self._edges(am, n)
            yield {
                'nodes': self.nodes[idx, :n].reshape(bs * n, -1),
                'x': self.positions[idx, :n].reshape(bs * n, -1),
                'atom_mask': am.reshape(bs * n, 1).to(self.dtype),
                'edges': [rows, cols],
                'label': self.label[idx],
                'n_nodes': n,
                'batch_size': bs,
            }


def build(dataloaders, prop, charge_power, charge_scale, device):
    """One GPUBatches per split, honouring each split's shuffle permutation."""
    out = {}
    for split, dl in dataloaders.items():
        ds = dl.dataset
        out[split] = GPUBatches(ds.data, prop, charge_power, charge_scale,
                                device, perm=ds.perm)
    return out
