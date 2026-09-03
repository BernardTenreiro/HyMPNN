import os
import torch

def compute_mean_mad(dataloaders, label_property):
    values = dataloaders['train'].dataset.data[label_property]
    meann = torch.mean(values)
    ma = torch.abs(values - meann)
    mad = torch.mean(ma)
    return meann, mad

edges_dic = {}
def get_adj_matrix(n_nodes, batch_size, device):
    """Fully-connected edge index for `batch_size` graphs of `n_nodes` nodes each.

    The result depends only on (n_nodes, batch_size, device), so it is built once
    and reused.  The previous version never wrote into `edges_dic`, so every call
    rebuilt batch_size * n_nodes**2 indices with nested Python loops.

    The vectorised construction produces exactly the same index values in exactly
    the same order as those loops (rows vary slowest, cols fastest, offset by
    batch_idx * n_nodes), so results are bitwise unchanged.
    """
    key = (n_nodes, batch_size, str(device))
    cached = None if os.environ.get('EGNN_BASELINE') else edges_dic.get(key)
    if cached is not None:
        return cached

    base = torch.arange(n_nodes)
    rows = base.view(-1, 1).expand(n_nodes, n_nodes).reshape(-1)
    cols = base.view(1, -1).expand(n_nodes, n_nodes).reshape(-1)
    offsets = (torch.arange(batch_size) * n_nodes).view(-1, 1)

    edges = [(rows.view(1, -1) + offsets).reshape(-1).to(device),
             (cols.view(1, -1) + offsets).reshape(-1).to(device)]
    edges_dic[key] = edges
    return edges

def preprocess_input(one_hot, charges, charge_power, charge_scale, device):
    charge_tensor = (charges.unsqueeze(-1) / charge_scale).pow(
        torch.arange(charge_power + 1., device=device, dtype=torch.float32))
    charge_tensor = charge_tensor.view(charges.shape + (1, charge_power + 1))
    atom_scalars = (one_hot.unsqueeze(-1) * charge_tensor).view(charges.shape[:2] + (-1,))
    return atom_scalars