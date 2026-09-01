from models.gcl import E_GCL, unsorted_segment_sum
import torch
from torch import nn
from typing import Dict, List, Optional, Sequence, Tuple


class E_GCL_mask(E_GCL):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_attr_dim=0, act_fn=nn.ReLU(), recurrent=True, coords_weight=1.0, attention=False):
        E_GCL.__init__(self, input_nf, output_nf, hidden_nf, edges_in_d=edges_in_d, nodes_att_dim=nodes_attr_dim, act_fn=act_fn, recurrent=recurrent, coords_weight=coords_weight, attention=attention)

        del self.coord_mlp
        self.act_fn = act_fn

    # NOTE: coord_model is never called (E_GCL_mask.forward does not update
    # coords), and self.coord_mlp was deleted above.

    def forward(self, h, edge_index, coord, node_mask, edge_mask, edge_attr=None, node_attr=None, n_nodes=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)

        edge_feat = edge_feat * edge_mask

        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr


class EGNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, coords_weight=1.0, attention=False, node_attr=1):
        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers

        ### Encoder
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        self.node_attr = node_attr
        if node_attr:
            n_node_attr = in_node_nf
        else:
            n_node_attr = 0
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL_mask(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, nodes_attr_dim=n_node_attr, act_fn=act_fn, recurrent=True, coords_weight=coords_weight, attention=attention))

        self.node_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                      act_fn,
                                      nn.Linear(self.hidden_nf, self.hidden_nf))

        self.graph_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                       act_fn,
                                       nn.Linear(self.hidden_nf, 1))
        self.to(self.device)

    def forward(self, h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes,
                charges=None, sparse_edges_per_layer=None):
        h = self.embedding(h0)
        for i in range(0, self.n_layers):
            if self.node_attr:
                h, _, _ = self._modules["gcl_%d" % i](h, edges, x, node_mask, edge_mask, edge_attr=edge_attr, node_attr=h0, n_nodes=n_nodes)
            else:
                h, _, _ = self._modules["gcl_%d" % i](h, edges, x, node_mask, edge_mask, edge_attr=edge_attr,
                                                      node_attr=None, n_nodes=n_nodes)

        h = self.node_dec(h)
        h = h * node_mask
        h = h.view(-1, n_nodes, self.hidden_nf)
        h = torch.sum(h, dim=1)
        pred = self.graph_dec(h)
        return pred.squeeze(1)


###############################################################################
# Frame ordering / edge coloring utilities
###############################################################################

_ATOMIC_MASS: Dict[int, float] = {
    1: 1.008, 2: 4.003, 3: 6.941, 4: 9.012, 5: 10.811,
    6: 12.011, 7: 14.007, 8: 15.999, 9: 18.998, 10: 20.180,
    11: 22.990, 12: 24.305, 13: 26.982, 14: 28.086, 15: 30.974,
    16: 32.065, 17: 35.453, 18: 39.948, 19: 39.098, 20: 40.078,
    35: 79.904, 53: 126.904,
}
_MAX_Z = 100


def _build_mass_table(max_z: int = _MAX_Z) -> torch.Tensor:
    table = torch.zeros(max_z + 1, dtype=torch.float32)
    for z, m in _ATOMIC_MASS.items():
        if z <= max_z:
            table[z] = m
    for z in range(max_z + 1):
        if table[z] == 0:
            table[z] = float(z)
    return table


def _node_scores(charges: torch.Tensor, scoring: str, mass_table: torch.Tensor) -> torch.Tensor:
    z = charges.long().clamp(0, mass_table.size(0) - 1)
    if scoring == "atomic_number":
        return z.float()
    mass = mass_table.to(z.device)[z]
    if scoring == "mass":
        return mass
    if scoring == "mass_noh":
        mass = mass.clone()
        mass[z == 1] = 0.0
        return mass
    if scoring == "penalized_h":
        mass = mass.clone()
        mass[z == 1] = -1.0
        return mass
    raise ValueError(f"Unknown frame_scoring='{scoring}'.")


def _unique_undirected_pairs(edge_index):
    rows, cols = edge_index
    e_count = rows.size(0)
    pair_map: Dict[Tuple[int, int], int] = {}
    pairs: List[Tuple[int, int]] = []
    edge_to_pair = torch.empty(e_count, dtype=torch.long, device=rows.device)
    for e in range(e_count):
        a, b = int(rows[e]), int(cols[e])
        u, v = (a, b) if a < b else (b, a)
        key = (u, v)
        idx = pair_map.get(key)
        if idx is None:
            idx = len(pairs)
            pair_map[key] = idx
            pairs.append(key)
        edge_to_pair[e] = idx
    return pairs, edge_to_pair


def _greedy_pair_coloring(pairs):
    colors = [-1] * len(pairs)
    node_to_pairs: Dict[int, List[int]] = {}
    for p_idx, (u, v) in enumerate(pairs):
        node_to_pairs.setdefault(u, []).append(p_idx)
        node_to_pairs.setdefault(v, []).append(p_idx)
    for p_idx, (u, v) in enumerate(pairs):
        used = {colors[n] for n in node_to_pairs[u] + node_to_pairs[v] if colors[n] != -1}
        c = 0
        while c in used:
            c += 1
        colors[p_idx] = c
    return colors


def _color_masks_single(edge_index):
    pairs, edge_to_pair = _unique_undirected_pairs(edge_index)
    if len(pairs) == 0:
        return []
    pair_colors = _greedy_pair_coloring(pairs)
    num_colors = max(pair_colors) + 1
    pair_colors_t = torch.tensor(pair_colors, dtype=torch.long, device=edge_to_pair.device)
    edge_colors = pair_colors_t[edge_to_pair]
    return [(edge_colors == c) for c in range(num_colors)]


def _score_color_classes(color_masks, edge_index, node_scores):
    rows, cols = edge_index
    scores = []
    for mask in color_masks:
        mask = mask.bool()
        if mask.sum().item() == 0:
            scores.append(float("inf"))
            continue
        scores.append((node_scores[rows[mask]] + node_scores[cols[mask]]).mean().item())
    return scores


def _sandwich_order(sorted_colors):
    out, lo, hi, take_low = [], 0, len(sorted_colors) - 1, True
    while lo <= hi:
        if take_low:
            out.append(sorted_colors[lo]); lo += 1
        else:
            out.append(sorted_colors[hi]); hi -= 1
        take_low = not take_low
    return out


def build_frame_schedule_single(edge_index, charges, n_layers, frame_ordering, frame_scoring, mass_table):
    scoring_override = {
        "sandwich_atomic": "atomic_number",
        "sandwich_mass": "mass",
        "sandwich_mass_noh": "mass_noh",
        "sandwich_penalized_h": "penalized_h",
    }
    scoring = scoring_override.get(frame_ordering, frame_scoring)
    ns = _node_scores(charges, scoring, mass_table)
    color_masks = _color_masks_single(edge_index)

    if not color_masks:
        raise ValueError("No edges found — cannot build frame schedule.")

    scores = _score_color_classes(color_masks, edge_index, ns)
    sorted_colors = sorted(range(len(scores)), key=lambda c: (scores[c], c))

    if frame_ordering == "sort_repeat":
        base = sorted_colors
    elif frame_ordering in scoring_override:
        base = _sandwich_order(sorted_colors)
    else:
        raise ValueError(f"Unsupported frame_ordering='{frame_ordering}'.")

    schedule = [base[t % len(base)] for t in range(n_layers)]
    return color_masks, schedule


###############################################################################
# Preprocessing: precompute per-molecule active edge (row, col) pairs per layer
###############################################################################

#Version 1: store (row, col) pairs of active edges per layer, as LongTensors on CPU. During batch assembly, these are concatenated and moved to GPU.
# def precompute_molecule_colorings(
#     dataloaders: Dict,
#     n_layers: int,
#     frame_ordering: str,
#     frame_scoring: str,
# ) -> Dict[tuple, List[Tuple[torch.Tensor, torch.Tensor]]]:
#     """
#     Iterate over ALL splits once and precompute per-molecule, per-layer
#     active edge (row, col) pairs in LOCAL node indices (0..n_real-1).

#     Returns:
#         cache: dict mapping charge_tuple -> list of n_layers tuples of
#                (rows_local, cols_local) LongTensors, each containing only
#                the active directed edges for that layer.
#     """
#     mass_table = _build_mass_table()
#     cache: Dict[tuple, List[Tuple[torch.Tensor, torch.Tensor]]] = {}

#     for split_name, loader in dataloaders.items():
#         for data in loader:
#             batch_size, n_nodes, _ = data['positions'].size()
#             charges_batch = data['charges']
#             atom_mask_batch = data['atom_mask']

#             for g in range(batch_size):
#                 n_real = int(atom_mask_batch[g].sum().item())
#                 if n_real < 2:
#                     continue

#                 charges_g = charges_batch[g, :n_real].long()
#                 cache_key = tuple(charges_g.tolist())

#                 if cache_key in cache:
#                     continue

#                 # Build local fully-connected edge index
#                 idx_i = torch.arange(n_real).unsqueeze(1).expand(n_real, n_real).reshape(-1)
#                 idx_j = torch.arange(n_real).unsqueeze(0).expand(n_real, n_real).reshape(-1)
#                 no_self = idx_i != idx_j
#                 local_rows_t = idx_i[no_self]
#                 local_cols_t = idx_j[no_self]

#                 color_masks, schedule = build_frame_schedule_single(
#                     edge_index=(local_rows_t, local_cols_t),
#                     charges=charges_g,
#                     n_layers=n_layers,
#                     frame_ordering=frame_ordering,
#                     frame_scoring=frame_scoring,
#                     mass_table=mass_table,
#                 )

#                 # For each layer, store the (row, col) pairs of active edges
#                 layer_edges = []
#                 for layer_idx in range(n_layers):
#                     color_idx = schedule[layer_idx]
#                     mask = color_masks[color_idx]
#                     active_rows = local_rows_t[mask]
#                     active_cols = local_cols_t[mask]
#                     layer_edges.append((active_rows, active_cols))
#                 cache[cache_key] = layer_edges

#     print(f"Precomputed colorings for {len(cache)} unique molecule compositions")
#     return cache


# def assemble_batch_sparse_edges(
#     coloring_cache: Dict[tuple, List[Tuple[torch.Tensor, torch.Tensor]]],
#     charges_batch: torch.Tensor,
#     atom_mask_batch: torch.Tensor,
#     n_nodes: int,
#     n_layers: int,
#     device: torch.device,
# ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
#     """
#     Assemble per-layer compact edge lists from precomputed colorings.

#     Returns:
#         sparse_edges: list of n_layers tuples of (rows, cols, edge_mask),
#                       where rows/cols are global node indices and edge_mask
#                       is all-ones of shape (num_active_edges, 1).
#                       These contain ONLY the active edges — no padding.
#     """
#     batch_size = charges_batch.size(0)

#     # Collect per-layer lists of (rows, cols) to concatenate
#     per_layer_rows: List[List[torch.Tensor]] = [[] for _ in range(n_layers)]
#     per_layer_cols: List[List[torch.Tensor]] = [[] for _ in range(n_layers)]

#     for g in range(batch_size):
#         n_real = int(atom_mask_batch[g].sum().item())
#         if n_real < 2:
#             continue

#         charges_g = charges_batch[g, :n_real].long()
#         cache_key = tuple(charges_g.tolist())
#         cached = coloring_cache.get(cache_key)
#         if cached is None:
#             continue

#         node_offset = g * n_nodes

#         for layer_idx in range(n_layers):
#             local_rows, local_cols = cached[layer_idx]
#             per_layer_rows[layer_idx].append(local_rows + node_offset)
#             per_layer_cols[layer_idx].append(local_cols + node_offset)

#     # Concatenate into single tensors per layer
#     sparse_edges = []
#     for layer_idx in range(n_layers):
#         if per_layer_rows[layer_idx]:
#             rows = torch.cat(per_layer_rows[layer_idx]).to(device)
#             cols = torch.cat(per_layer_cols[layer_idx]).to(device)
#         else:
#             rows = torch.zeros(0, dtype=torch.long, device=device)
#             cols = torch.zeros(0, dtype=torch.long, device=device)
#         edge_mask = torch.ones(rows.size(0), 1, dtype=torch.float32, device=device)
#         sparse_edges.append((rows, cols, edge_mask))

#     return sparse_edges

#Version 2: store (row, col) pairs of active edges per layer, as LongTensors on CPU. During batch assembly, these are concatenated and moved to GPU. Also store edge_mask tensors (all ones) for direct use by GCL layers.
def precompute_molecule_colorings(
    dataloaders: Dict,
    n_layers: int,
    frame_ordering: str,
    frame_scoring: str,
) -> Dict[tuple, Dict[str, List[torch.Tensor]]]:
    mass_table = _build_mass_table()
    cache: Dict[tuple, Dict[str, List[torch.Tensor]]] = {}

    for loader in dataloaders.values():
        for data in loader:
            batch_size, n_nodes, _ = data['positions'].size()
            charges_batch = data['charges']
            atom_mask_batch = data['atom_mask']

            for g in range(batch_size):
                n_real = int(atom_mask_batch[g].sum().item())
                if n_real < 2:
                    continue

                charges_g = charges_batch[g, :n_real].long().view(-1)
                cache_key = tuple(charges_g.tolist())
                if cache_key in cache:
                    continue

                idx_i = torch.arange(n_real).unsqueeze(1).expand(n_real, n_real).reshape(-1)
                idx_j = torch.arange(n_real).unsqueeze(0).expand(n_real, n_real).reshape(-1)
                no_self = idx_i != idx_j
                local_rows_t = idx_i[no_self]
                local_cols_t = idx_j[no_self]

                color_masks, schedule = build_frame_schedule_single(
                    edge_index=(local_rows_t, local_cols_t),
                    charges=charges_g,
                    n_layers=n_layers,
                    frame_ordering=frame_ordering,
                    frame_scoring=frame_scoring,
                    mass_table=mass_table,
                )

                layer_rows, layer_cols, layer_counts = [], [], []
                for layer_idx in range(n_layers):
                    color_idx = schedule[layer_idx]
                    mask = color_masks[color_idx]
                    active_rows = local_rows_t[mask].contiguous()
                    active_cols = local_cols_t[mask].contiguous()
                    layer_rows.append(active_rows)
                    layer_cols.append(active_cols)
                    layer_counts.append(int(active_rows.numel()))

                cache[cache_key] = {
                    "rows": layer_rows,
                    "cols": layer_cols,
                    "counts": layer_counts,
                }

    print(f"Precomputed colorings for {len(cache)} unique molecule compositions")
    return cache


def assemble_batch_sparse_edges(
    coloring_cache: Dict[tuple, Dict[str, List[torch.Tensor]]],
    charges_batch: torch.Tensor,
    atom_mask_batch: torch.Tensor,
    n_nodes: int,
    n_layers: int,
    device: torch.device,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if charges_batch.dim() == 3 and charges_batch.size(-1) == 1:
        charges_batch = charges_batch.squeeze(-1)
    batch_size = charges_batch.size(0)

    groups: Dict[tuple, List[int]] = {}
    for g in range(batch_size):
        n_real = int(atom_mask_batch[g].sum().item())
        if n_real < 2:
            continue
        charges_g = charges_batch[g, :n_real].long().view(-1)
        cache_key = tuple(charges_g.tolist())
        if cache_key not in coloring_cache:
            raise RuntimeError(f"Cache miss for charges={cache_key}")
        groups.setdefault(cache_key, []).append(g)

    per_layer_rows_cpu: List[List[torch.Tensor]] = [[] for _ in range(n_layers)]
    per_layer_cols_cpu: List[List[torch.Tensor]] = [[] for _ in range(n_layers)]

    for cache_key, graph_ids in groups.items():
        cached = coloring_cache[cache_key]
        offsets = torch.tensor(graph_ids, dtype=torch.long) * n_nodes
        for layer_idx in range(n_layers):
            base_rows = cached["rows"][layer_idx]
            base_cols = cached["cols"][layer_idx]
            if base_rows.numel() == 0:
                continue
            rows = (base_rows.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)
            cols = (base_cols.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)
            per_layer_rows_cpu[layer_idx].append(rows)
            per_layer_cols_cpu[layer_idx].append(cols)

    sparse_edges = []
    for layer_idx in range(n_layers):
        if per_layer_rows_cpu[layer_idx]:
            rows = torch.cat(per_layer_rows_cpu[layer_idx], dim=0).to(device, non_blocking=True)
            cols = torch.cat(per_layer_cols_cpu[layer_idx], dim=0).to(device, non_blocking=True)
        else:
            rows = torch.zeros(0, dtype=torch.long, device=device)
            cols = torch.zeros(0, dtype=torch.long, device=device)
        edge_mask = torch.ones(rows.size(0), 1, dtype=torch.float32, device=device)
        sparse_edges.append((rows, cols, edge_mask))
    return sparse_edges

###############################################################################
# SparseEGNN
###############################################################################

class SparseEGNN(nn.Module):
    """
    Frame-ordered sparse EGNN using the original QM9 data pipeline.

    Identical to EGNN in every way except that at layer i, only the active
    edges for that color class are passed to the GCL layer — NOT the full
    graph with a mask. This means tensor operations scale with the number
    of active edges (a matching ≈ N/2 per molecule) rather than N^2.

    Usage:
        1. Create model
        2. Call precompute_molecule_colorings() once on all dataloaders
        3. Each batch, call assemble_batch_sparse_edges() to get sparse_edges
        4. Pass sparse_edges_per_layer= to forward()
    """

    def __init__(
        self,
        in_node_nf,
        in_edge_nf,
        hidden_nf,
        device='cpu',
        act_fn=nn.SiLU(),
        n_layers=4,
        coords_weight=1.0,
        attention=False,
        node_attr=1,
        frame_ordering='sort_repeat',
        frame_scoring='atomic_number',
    ):
        super(SparseEGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.node_attr = node_attr
        self.frame_ordering = frame_ordering
        self.frame_scoring = frame_scoring

        self.register_buffer("mass_table", _build_mass_table())

        # Exactly the same layers as EGNN
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        n_node_attr = in_node_nf if node_attr else 0
        for i in range(n_layers):
            self.add_module("gcl_%d" % i, E_GCL_mask(
                hidden_nf, hidden_nf, hidden_nf,
                edges_in_d=in_edge_nf,
                nodes_attr_dim=n_node_attr,
                act_fn=act_fn,
                recurrent=True,
                coords_weight=coords_weight,
                attention=attention,
            ))

        self.node_dec = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
        )
        self.graph_dec = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1),
        )
        self.to(device)

    def forward(self, h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes,
                charges=None, sparse_edges_per_layer=None):
        """
        Args:
            h0, x, node_mask, n_nodes: same as EGNN
            edges, edge_attr, edge_mask: from the standard pipeline (unused by
                sparse layers, kept for signature compat)
            sparse_edges_per_layer: list of n_layers tuples (rows, cols, edge_mask)
                from assemble_batch_sparse_edges(). REQUIRED.
        """
        if sparse_edges_per_layer is None:
            raise ValueError(
                "SparseEGNN requires sparse_edges_per_layer. "
                "Call precompute_molecule_colorings() before training and "
                "assemble_batch_sparse_edges() each batch."
            )

        h = self.embedding(h0)
        for i in range(self.n_layers):
            sparse_rows, sparse_cols, sparse_emask = sparse_edges_per_layer[i]
            sparse_edge_index = [sparse_rows, sparse_cols]

            node_attr_i = h0 if self.node_attr else None
            h, _, _ = self._modules["gcl_%d" % i](
                h, sparse_edge_index, x, node_mask, sparse_emask,
                edge_attr=None, node_attr=node_attr_i, n_nodes=n_nodes,
            )

        h = self.node_dec(h)
        h = h * node_mask
        h = h.view(-1, n_nodes, self.hidden_nf)
        h = torch.sum(h, dim=1)
        pred = self.graph_dec(h)
        return pred.squeeze(1)






