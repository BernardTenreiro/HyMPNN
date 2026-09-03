"""Batch collation for variable-size QM9 molecules."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


def _stack(values: list[Tensor]) -> Tensor:
    if not torch.is_tensor(values[0]):
        return torch.tensor(values)
    if values[0].ndim == 0:
        return torch.stack(values)
    return pad_sequence(values, batch_first=True, padding_value=0)


def _drop_empty_columns(values: Tensor, columns_to_keep: Tensor) -> Tensor:
    if not torch.is_tensor(values) or values.ndim < 2:
        return values
    return values[:, columns_to_keep, ...]


def collate_batch(molecules: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Pad molecules, construct masks, and compress dense edges."""
    batch = {
        property_name: _stack([molecule[property_name] for molecule in molecules])
        for property_name in molecules[0]
    }
    columns_to_keep = batch["charges"].sum(dim=0) > 0
    batch = {
        property_name: _drop_empty_columns(values, columns_to_keep)
        for property_name, values in batch.items()
    }

    atom_mask = batch["charges"] > 0
    batch["atom_mask"] = atom_mask
    batch_size, node_count = atom_mask.shape
    edge_mask = atom_mask.unsqueeze(1) & atom_mask.unsqueeze(2)
    edge_mask &= ~torch.eye(node_count, dtype=torch.bool).unsqueeze(0)
    batch["edge_mask"] = edge_mask.reshape(batch_size * node_count * node_count, 1)

    kept_edges = edge_mask.reshape(-1).nonzero(as_tuple=True)[0]
    graph_indices = torch.div(kept_edges, node_count * node_count, rounding_mode="floor")
    graph_offsets = kept_edges - graph_indices * node_count * node_count
    local_rows = torch.div(graph_offsets, node_count, rounding_mode="floor")
    local_columns = graph_offsets - local_rows * node_count
    batch["dense_rows"] = graph_indices * node_count + local_rows
    batch["dense_cols"] = graph_indices * node_count + local_columns
    return batch


# Kept as a short alias for callers that accept the standard PyTorch naming.
collate_fn = collate_batch
