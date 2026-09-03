"""QM9 dataset and DataLoader construction."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .collate import collate_batch
from .dataset import QM9Dataset
from .preparation import prepare_qm9_dataset

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA_ROOT = Path(os.environ.get("HYMPNN_DATA_DIR", REPOSITORY_ROOT / "data"))
QM9_UNIT_CONVERSIONS = {
    "U0": 27.2114,
    "U": 27.2114,
    "G": 27.2114,
    "H": 27.2114,
    "zpve": 27211.4,
    "gap": 27.2114,
    "homo": 27.2114,
    "lumo": 27.2114,
}


def _environment_flag(name: str, legacy_name: str | None = None) -> bool:
    return name in os.environ or (legacy_name is not None and legacy_name in os.environ)


def _worker_options(worker_count: int) -> dict[str, Any]:
    """Return multiprocessing options shared by all data loaders."""
    if worker_count <= 0:
        return {}
    return {
        "multiprocessing_context": "fork",
        "persistent_workers": True,
        "pin_memory": True,
        "prefetch_factor": 4,
    }


def create_dataloader(
    dataset: Dataset,
    split: str,
    batch_size: int,
    worker_count: int,
    collate: Callable = collate_batch,
) -> DataLoader:
    """Construct a consistently configured DataLoader for one split."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        num_workers=worker_count,
        collate_fn=collate,
        **_worker_options(worker_count),
    )


def rebuild_dataloaders(
    dataloaders: Mapping[str, DataLoader],
    batch_size: int,
    worker_count: int,
    collate: Callable,
) -> dict[str, DataLoader]:
    """Recreate loaders with a different collator, preserving datasets."""
    datasets = {split: loader.dataset for split, loader in dataloaders.items()}
    shutdown_dataloaders(dataloaders)
    return {
        split: create_dataloader(dataset, split, batch_size, worker_count, collate)
        for split, dataset in datasets.items()
    }


def shutdown_dataloaders(dataloaders: Mapping[str, DataLoader]) -> None:
    """Stop persistent workers instead of waiting for interpreter shutdown."""
    for loader in dataloaders.values():
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()
            loader._iterator = None


def _load_splits(
    data_root: Path, force_prepare: bool = False
) -> dict[str, dict[str, torch.Tensor]]:
    processed_directory = data_root / "processed" / "qm9"
    raw_directory = data_root / "raw" / "qm9"
    split_paths = {
        split: processed_directory / f"{split}.npz" for split in ("train", "valid", "test")
    }
    existence = [path.exists() for path in split_paths.values()]
    if force_prepare or not any(existence):
        prepare_qm9_dataset(raw_directory, processed_directory)
    elif not all(existence):
        missing = ", ".join(str(path) for path in split_paths.values() if not path.exists())
        raise FileNotFoundError(f"QM9 is only partially processed; missing: {missing}")

    splits: dict[str, dict[str, torch.Tensor]] = {}
    for split, path in split_paths.items():
        with np.load(path) as archive:
            splits[split] = {key: torch.from_numpy(value) for key, value in archive.items()}
    return splits


def _included_species(splits: Mapping[str, Mapping[str, torch.Tensor]]) -> torch.Tensor:
    species = torch.cat([data["charges"].unique() for data in splits.values()]).unique(sorted=True)
    species = species[species != 0]
    for split, data in splits.items():
        split_species = data["charges"].unique(sorted=True)
        split_species = split_species[split_species != 0]
        if not torch.equal(split_species, species):
            raise ValueError(f"QM9 split {split!r} does not contain every atomic species")
    return species


def retrieve_dataloaders(
    batch_size: int,
    worker_count: int = 1,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> tuple[dict[str, DataLoader], torch.Tensor]:
    """Load QM9 and return training, validation, and test data loaders."""
    tensor_splits = _load_splits(
        data_root,
        force_prepare=_environment_flag("HYMPNN_FORCE_PREPARE", "QM9_FORCE_DOWNLOAD"),
    )
    species = _included_species(tensor_splits)
    datasets = {
        split: QM9Dataset(data, species, subtract_thermochemical_energy=True)
        for split, data in tensor_splits.items()
    }
    for qm9_dataset in datasets.values():
        qm9_dataset.convert_units(QM9_UNIT_CONVERSIONS)

    dataloaders = {
        split: create_dataloader(dataset, split, batch_size, worker_count)
        for split, dataset in datasets.items()
    }
    return dataloaders, datasets["train"].max_charge
