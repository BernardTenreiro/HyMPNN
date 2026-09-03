"""Parsing utilities for the raw QM9 XYZ archive."""

from __future__ import annotations

import tarfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import BinaryIO

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

ATOMIC_NUMBERS = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}


def parse_xyz_collection(
    source: Path,
    parser: Callable[[BinaryIO], dict[str, Tensor]],
    indices: Iterable[int] | None = None,
) -> dict[str, Tensor]:
    """Parse selected entries from an XYZ directory or tar archive."""
    selected_indices = None if indices is None else set(indices)
    if tarfile.is_tarfile(source):
        archive = tarfile.open(source, "r")
        members = archive.getmembers()
        selected = (
            members
            if selected_indices is None
            else [member for index, member in enumerate(members) if index in selected_indices]
        )

        def open_entry(member):
            file_object = archive.extractfile(member)
            if file_object is None:
                raise ValueError(f"Could not read {member.name!r} from {source}")
            return file_object

    elif source.is_dir():
        entries = sorted(source.iterdir())
        selected = (
            entries
            if selected_indices is None
            else [entry for index, entry in enumerate(entries) if index in selected_indices]
        )

        def open_entry(entry):
            return entry.open("rb")

        archive = None
    else:
        raise ValueError(f"QM9 source must be a directory or tar archive: {source}")

    try:
        molecules = []
        for entry in selected:
            with open_entry(entry) as file_object:
                molecules.append(parser(file_object))
    finally:
        if archive is not None:
            archive.close()

    if not molecules:
        raise ValueError(f"No molecules selected from {source}")
    properties = molecules[0].keys()
    if not all(properties == molecule.keys() for molecule in molecules):
        raise ValueError("QM9 molecules do not expose a consistent property set")

    return {
        key: pad_sequence(values, batch_first=True) if values[0].ndim else torch.stack(values)
        for key in properties
        if (values := [molecule[key] for molecule in molecules])
    }


def parse_qm9_xyz(data_file: BinaryIO) -> dict[str, Tensor]:
    """Parse one GDB9 XYZ entry into molecular tensors."""
    lines = [line.decode("utf-8") for line in data_file.readlines()]
    atom_count = int(lines[0])
    raw_properties = lines[1].split()
    xyz_lines = lines[2 : atom_count + 2]
    frequencies = lines[atom_count + 2]

    charges: list[int] = []
    positions: list[list[float]] = []
    for line in xyz_lines:
        atom, x, y, z, _ = line.replace("*^", "e").split()
        charges.append(ATOMIC_NUMBERS[atom])
        positions.append([float(x), float(y), float(z)])

    property_names = [
        "index",
        "A",
        "B",
        "C",
        "mu",
        "alpha",
        "homo",
        "lumo",
        "gap",
        "r2",
        "zpve",
        "U0",
        "U",
        "H",
        "G",
        "Cv",
    ]
    values = [int(raw_properties[1]), *[float(value) for value in raw_properties[2:]]]
    molecule: dict[str, object] = dict(zip(property_names, values))
    molecule.update(
        {
            "num_atoms": atom_count,
            "charges": charges,
            "positions": positions,
            "omega1": max(float(frequency) for frequency in frequencies.split()),
        }
    )
    return {key: torch.tensor(value) for key, value in molecule.items()}
