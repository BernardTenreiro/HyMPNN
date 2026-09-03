"""Feature construction and target normalization for QM9."""

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


def build_node_features(one_hot, charges, charge_power, charge_scale, device):
    charge_tensor = (charges.unsqueeze(-1) / charge_scale).pow(
        torch.arange(charge_power + 1.0, device=device, dtype=torch.float32)
    )
    charge_tensor = charge_tensor.view(charges.shape + (1, charge_power + 1))
    atom_scalars = (one_hot.unsqueeze(-1) * charge_tensor).view(charges.shape[:2] + (-1,))
    return atom_scalars
