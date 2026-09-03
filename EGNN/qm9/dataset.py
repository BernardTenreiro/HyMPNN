"""QM9 dataset and DataLoader construction helpers."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Any

from torch.utils.data import DataLoader, Dataset

from qm9.args import init_argparse
from qm9.data.collate import collate_fn
from qm9.data.utils import initialize_datasets

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


def _worker_options(num_workers: int) -> dict[str, Any]:
    """Return options used by every multiprocessing DataLoader.

    Python 3.14 defaults to ``forkserver``, which cannot spawn workers on the
    target cluster. Explicitly requesting ``fork`` avoids that failure.
    ``EGNN_NO_PIN=1`` disables pinning for systems where the pinning thread
    competes with CUDA kernel dispatch.
    """
    if num_workers <= 0:
        return {}
    return {
        "multiprocessing_context": "fork",
        "persistent_workers": True,
        "pin_memory": "EGNN_NO_PIN" not in os.environ,
        "prefetch_factor": 4,
    }


def create_dataloader(
    dataset: Dataset,
    split: str,
    batch_size: int,
    num_workers: int,
    collate: Callable = collate_fn,
) -> DataLoader:
    """Construct a consistently configured DataLoader for one dataset split."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        num_workers=num_workers,
        collate_fn=collate,
        **_worker_options(num_workers),
    )


def rebuild_dataloaders(
    dataloaders: Mapping[str, DataLoader],
    batch_size: int,
    num_workers: int,
    collate: Callable,
) -> dict[str, DataLoader]:
    """Recreate loaders with a new collator while preserving their datasets."""
    datasets = {split: loader.dataset for split, loader in dataloaders.items()}
    shutdown_dataloaders(dataloaders)
    return {
        split: create_dataloader(dataset, split, batch_size, num_workers, collate)
        for split, dataset in datasets.items()
    }


def shutdown_dataloaders(dataloaders: Mapping[str, DataLoader]) -> None:
    """Stop persistent workers instead of waiting for interpreter shutdown.

    Coloring precomputation consumes all original loaders. Rebuilding those
    loaders without shutting them down used to leave obsolete workers alive
    for the rest of training and could hang Python during process teardown.
    """
    for loader in dataloaders.values():
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()
            loader._iterator = None


def retrieve_dataloaders(
    batch_size: int, num_workers: int = 1
) -> tuple[dict[str, DataLoader], object]:
    """Initialize QM9 and return its train, validation, and test loaders."""
    args = init_argparse("qm9")
    args, datasets, _, charge_scale = initialize_datasets(
        args,
        args.datadir,
        "qm9",
        subtract_thermo=args.subtract_thermo,
        force_download=args.force_download,
    )

    for qm9_dataset in datasets.values():
        qm9_dataset.convert_units(QM9_UNIT_CONVERSIONS)

    dataloaders = {
        split: create_dataloader(dataset, split, batch_size, num_workers)
        for split, dataset in datasets.items()
    }
    return dataloaders, charge_scale
