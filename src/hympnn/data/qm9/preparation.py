"""Prepare reproducible train, validation, and test splits from raw QM9."""

from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path

import numpy as np

from .parsing import parse_qm9_xyz, parse_xyz_collection

ARCHIVE_NAME = "dsgdb9nsd.xyz.tar.bz2"
EXCLUSIONS_NAME = "uncharacterized.txt"
ATOM_REFERENCES_NAME = "atomref.txt"
EXCLUSIONS_URL = "https://springernature.figshare.com/ndownloader/files/3195404"
ATOM_REFERENCES_URL = "https://springernature.figshare.com/ndownloader/files/3195395"
MOLECULE_COUNT = 133_885
EXCLUDED_MOLECULE_COUNT = 3_054
TRAINING_MOLECULE_COUNT = 100_000
THERMOCHEMICAL_TARGETS = ("zpve", "U0", "U", "H", "G", "Cv")
ATOMIC_NUMBERS = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}


def prepare_qm9_dataset(raw_directory: Path, output_directory: Path) -> None:
    """Process raw QM9 files into deterministic NumPy split archives."""
    raw_directory = Path(raw_directory)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    archive_path = raw_directory / ARCHIVE_NAME
    if not archive_path.exists():
        raise FileNotFoundError(
            f"QM9 archive not found at {archive_path}. See data/README.md for setup."
        )

    splits = _generate_split_indices(raw_directory, output_directory)
    subset_size = int(os.environ.get("QM9_SUBSET", "0"))
    if subset_size > 0:
        logging.warning("QM9_SUBSET=%d: truncating each split for a smoke test", subset_size)
        splits = {name: indices[:subset_size] for name, indices in splits.items()}

    prepared = {
        split: parse_xyz_collection(archive_path, parse_qm9_xyz, indices)
        for split, indices in splits.items()
    }
    references = _load_atom_references(raw_directory, output_directory)
    if references:
        prepared = {
            split: _add_thermochemical_targets(data, references) for split, data in prepared.items()
        }
    else:
        logging.warning(
            "No atom references available; energy targets will not be converted to atomization energies"
        )

    for split, data in prepared.items():
        np.savez_compressed(output_directory / f"{split}.npz", **data)


def _generate_split_indices(
    raw_directory: Path,
    output_directory: Path,
) -> dict[str, np.ndarray]:
    exclusions_path = raw_directory / EXCLUSIONS_NAME
    temporary_download = False
    if not exclusions_path.exists():
        exclusions_path = output_directory / EXCLUSIONS_NAME
        logging.info("Downloading %s", EXCLUSIONS_NAME)
        urllib.request.urlretrieve(EXCLUSIONS_URL, exclusions_path)
        temporary_download = True

    try:
        excluded_indices = []
        with exclusions_path.open(encoding="utf-8") as exclusions_file:
            for line in exclusions_file:
                fields = line.split()
                if fields and fields[0].isdigit():
                    excluded_indices.append(int(fields[0]) - 1)
        if len(excluded_indices) != EXCLUDED_MOLECULE_COUNT:
            raise ValueError(
                f"Expected {EXCLUDED_MOLECULE_COUNT} excluded molecules, "
                f"found {len(excluded_indices)}"
            )
    finally:
        if temporary_download:
            exclusions_path.unlink(missing_ok=True)

    included_indices = np.setdiff1d(
        np.arange(MOLECULE_COUNT), np.asarray(excluded_indices), assume_unique=True
    )
    random_generator = np.random.RandomState(0)
    permutation = random_generator.permutation(len(included_indices))
    test_count = int(0.1 * len(included_indices))
    validation_count = len(included_indices) - TRAINING_MOLECULE_COUNT - test_count
    training, validation, test = np.split(
        permutation,
        [TRAINING_MOLECULE_COUNT, TRAINING_MOLECULE_COUNT + validation_count],
    )
    return {
        "train": included_indices[training],
        "valid": included_indices[validation],
        "test": included_indices[test],
    }


def _load_atom_references(
    raw_directory: Path,
    output_directory: Path,
) -> dict[str, dict[int, float]]:
    reference_path = raw_directory / ATOM_REFERENCES_NAME
    temporary_download = False
    if not reference_path.exists():
        reference_path = output_directory / ATOM_REFERENCES_NAME
        logging.info("Downloading %s", ATOM_REFERENCES_NAME)
        try:
            urllib.request.urlretrieve(ATOM_REFERENCES_URL, reference_path)
        except OSError as error:
            logging.warning("Could not download atom references: %s", error)
            return {}
        temporary_download = True

    references = {target: {} for target in THERMOCHEMICAL_TARGETS}
    try:
        with reference_path.open(encoding="utf-8") as reference_file:
            for line in reference_file:
                fields = line.split()
                if not fields or fields[0] not in ATOMIC_NUMBERS:
                    continue
                atomic_number = ATOMIC_NUMBERS[fields[0]]
                for target, value in zip(THERMOCHEMICAL_TARGETS, fields[1:]):
                    references[target][atomic_number] = float(value)
    finally:
        if temporary_download:
            reference_path.unlink(missing_ok=True)
    return references if any(references.values()) else {}


def _add_thermochemical_targets(data, references):
    charges = data["charges"].numpy()
    charge_counts = {
        int(atomic_number): (charges == atomic_number).sum(axis=1)
        for atomic_number in np.unique(charges)
    }
    for target, atomic_references in references.items():
        thermochemical_energy = np.zeros(len(data[target]))
        for atomic_number, molecule_counts in charge_counts.items():
            if atomic_number in atomic_references:
                thermochemical_energy += atomic_references[atomic_number] * molecule_counts
        data[f"{target}_thermo"] = thermochemical_energy
    return data
