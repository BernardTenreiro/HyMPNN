"""PyTorch dataset for processed QM9 tensors."""

from __future__ import annotations

import logging
from collections.abc import Mapping

from torch import Tensor
from torch.utils.data import Dataset


class QM9Dataset(Dataset):
    """A processed QM9 split with derived atom-type features."""

    def __init__(
        self,
        data: Mapping[str, Tensor],
        included_species: Tensor,
        sample_count: int = -1,
        subtract_thermochemical_energy: bool = True,
    ) -> None:
        self.data = dict(data)
        available_samples = len(self.data["charges"])
        if sample_count < 0:
            self.sample_count = available_samples
        else:
            self.sample_count = min(sample_count, available_samples)
            if sample_count > available_samples:
                logging.warning(
                    "Requested %d samples, but the split contains only %d.",
                    sample_count,
                    available_samples,
                )

        if subtract_thermochemical_energy:
            targets = [key.removesuffix("_thermo") for key in data if key.endswith("_thermo")]
            for target in targets:
                self.data[target] = self.data[target] - self.data[f"{target}_thermo"].to(
                    self.data[target].dtype
                )

        self.included_species = included_species
        self.data["one_hot"] = (
            self.data["charges"].unsqueeze(-1).eq(included_species.view(1, 1, -1))
        )
        self.num_species = included_species.numel()
        self.max_charge = included_species.max()
        self.statistics: dict[str, tuple[Tensor, Tensor]] = {}
        self._calculate_statistics()

    def _calculate_statistics(self) -> None:
        self.statistics = {
            key: (value.mean(), value.std())
            for key, value in self.data.items()
            if isinstance(value, Tensor) and value.ndim == 1 and value.is_floating_point()
        }

    def convert_units(self, conversion_factors: Mapping[str, float]) -> None:
        """Scale target properties in place using ``conversion_factors``."""
        for property_name, factor in conversion_factors.items():
            if property_name in self.data:
                self.data[property_name].mul_(factor)
        self._calculate_statistics()

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {key: value[index] for key, value in self.data.items()}
