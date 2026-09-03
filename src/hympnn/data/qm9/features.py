"""Feature construction and target normalization for QM9."""

import os

import torch
from torch.utils.data import Subset


def _property_values(dataset, label_property):
    """Return property values in the same order exposed by a dataset."""
    if isinstance(dataset, Subset):
        values = _property_values(dataset.dataset, label_property)
        indices = torch.as_tensor(dataset.indices, dtype=torch.long)
        return values[indices]

    if hasattr(dataset, "data"):
        values = dataset.data[label_property]
        permutation = getattr(dataset, "perm", None)
        if permutation is not None:
            values = values[permutation]
        return values[: len(dataset)]

    return torch.stack([dataset[index][label_property] for index in range(len(dataset))])


def compute_mean_mad(dataloaders, label_property):
    values = _property_values(dataloaders["train"].dataset, label_property)
    mean = torch.mean(values)
    mean_absolute_deviation = torch.mean(torch.abs(values - mean))
    if mean_absolute_deviation == 0:
        raise ValueError("target mean absolute deviation is zero")
    return mean, mean_absolute_deviation


_EDGE_CACHE = {}


def fully_connected_edges(n_nodes, batch_size, device):
    """Fully-connected edge index for `batch_size` graphs of `n_nodes` nodes each.

    The result depends only on ``(n_nodes, batch_size, device)``, so it is built
    once and reused.

    The vectorised construction produces exactly the same index values in exactly
    the same order as those loops (rows vary slowest, cols fastest, offset by
    batch_idx * n_nodes), so results are bitwise unchanged.
    """
    key = (n_nodes, batch_size, str(device))
    disable_cache = "HYMPNN_DISABLE_EDGE_CACHE" in os.environ
    disable_cache |= "EGNN_BASELINE" in os.environ
    cached = None if disable_cache else _EDGE_CACHE.get(key)
    if cached is not None:
        return cached

    base = torch.arange(n_nodes)
    rows = base.view(-1, 1).expand(n_nodes, n_nodes).reshape(-1)
    cols = base.view(1, -1).expand(n_nodes, n_nodes).reshape(-1)
    offsets = (torch.arange(batch_size) * n_nodes).view(-1, 1)

    edges = [
        (rows.view(1, -1) + offsets).reshape(-1).to(device),
        (cols.view(1, -1) + offsets).reshape(-1).to(device),
    ]
    _EDGE_CACHE[key] = edges
    return edges


def build_node_features(one_hot, charges, charge_power, charge_scale, device):
    charge_tensor = (charges.unsqueeze(-1) / charge_scale).pow(
        torch.arange(charge_power + 1.0, device=device, dtype=torch.float32)
    )
    charge_tensor = charge_tensor.view(charges.shape + (1, charge_power + 1))
    atom_scalars = (one_hot.unsqueeze(-1) * charge_tensor).view(charges.shape[:2] + (-1,))
    return atom_scalars
