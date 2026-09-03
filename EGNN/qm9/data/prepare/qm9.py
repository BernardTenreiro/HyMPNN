import logging
import os
import urllib
import urllib.request
from os.path import exists
from os.path import join as join

import numpy as np

from qm9.data.prepare.process import process_xyz_files, process_xyz_gdb9
from qm9.data.prepare.utils import cleanup_file, is_int

# Raw QM9 files (dsgdb9nsd.xyz.tar.bz2, uncharacterized.txt, optionally
# atomref.txt) ship with the repo under EGNN/data.  This file lives at
# EGNN/qm9/data/prepare/qm9.py, so EGNN/ is three levels up.  Resolving from
# __file__ keeps everything working no matter what directory the job is
# launched from; QM9_RAW_DIR overrides it.
EGNN_ROOT = os.path.abspath(join(os.path.dirname(__file__), "..", "..", ".."))
RAW_DATA_DIR = os.environ.get("QM9_RAW_DIR", join(EGNN_ROOT, "data"))

# Set QM9_SUBSET=<n> to truncate every split to n molecules (fast smoke tests).
QM9_SUBSET = int(os.environ.get("QM9_SUBSET", "0"))


def download_dataset_qm9(
    datadir, dataname, splits=None, calculate_thermo=True, exclude=True, cleanup=True
):
    """
    Download and prepare the QM9 (GDB9) dataset.
    """
    # Define directory for which data will be output.
    gdb9dir = join(*[datadir, dataname])

    # Important to avoid a race condition
    os.makedirs(gdb9dir, exist_ok=True)

    logging.info(
        "Downloading and processing GDB9 dataset. Output will be in directory: {}.".format(gdb9dir)
    )

    # logging.info('Beginning download of GDB9 dataset!')
    # gdb9_url_data = 'https://springernature.figshare.com/ndownloader/files/3195389'
    # gdb9_tar_data = join(gdb9dir, 'dsgdb9nsd.xyz.tar.bz2')
    # gdb9_tar_file = join(gdb9dir, 'dsgdb9nsd.xyz.tar.bz2')
    # gdb9_tar_data =
    # tardata = tarfile.open(gdb9_tar_file, 'r')
    # files = tardata.getmembers()

    gdb9_tar_data = join(RAW_DATA_DIR, "dsgdb9nsd.xyz.tar.bz2")
    if not exists(gdb9_tar_data):
        raise FileNotFoundError(
            "QM9 archive not found at {}. Place dsgdb9nsd.xyz.tar.bz2 there or "
            "set QM9_RAW_DIR.".format(gdb9_tar_data)
        )
    logging.info("Using existing local QM9 archive: {}".format(gdb9_tar_data))

    # urllib.request.urlretrieve(gdb9_url_data, filename=gdb9_tar_data)
    # logging.info('GDB9 dataset downloaded successfully!')

    # If splits are not specified, automatically generate them.
    if splits is None:
        splits = gen_splits_gdb9(gdb9dir, cleanup)

    # Process GDB9 dataset, and return dictionary of splits
    gdb9_data = {}
    for split, split_idx in splits.items():
        gdb9_data[split] = process_xyz_files(
            gdb9_tar_data, process_xyz_gdb9, file_idx_list=split_idx, stack=True
        )

    # Subtract thermochemical energy if desired.
    if calculate_thermo:
        # Download thermochemical energy from GDB9 dataset, and then process it into a dictionary
        therm_energy = get_thermo_dict(gdb9dir, cleanup)

        if therm_energy:
            # For each of train/validation/test split, add the thermochemical energy
            for split_idx, split_data in gdb9_data.items():
                gdb9_data[split_idx] = add_thermo_targets(split_data, therm_energy)
        else:
            logging.warning(
                "No atomref.txt available -- skipping thermochemical targets. "
                "Properties homo/lumo/gap/alpha/mu/Cv-free targets are unaffected, "
                "but zpve/U0/U/H/G/Cv will NOT have atomisation energies subtracted."
            )

    # Save processed GDB9 data into train/validation/test splits
    logging.info("Saving processed data:")
    for split, data in gdb9_data.items():
        savedir = join(gdb9dir, split + ".npz")
        np.savez_compressed(savedir, **data)

    logging.info("Processing/saving complete!")


def gen_splits_gdb9(gdb9dir, cleanup=True):
    """
    Generate GDB9 training/validation/test splits used.

    First, use the file 'uncharacterized.txt' in the GDB9 figshare to find a
    list of excluded molecules.

    Second, create a list of molecule ids, and remove the excluded molecule
    indices.

    Third, assign 100k molecules to the training set, 10% to the test set,
    and the remaining to the validation set.

    Finally, generate torch.tensors which give the molecule ids for each
    set.
    """
    logging.info("Splits were not specified! Automatically generating.")

    # Prefer the copy shipped in RAW_DATA_DIR; only hit figshare if it is absent.
    gdb9_txt_excluded = join(RAW_DATA_DIR, "uncharacterized.txt")
    downloaded_excluded = None
    if not exists(gdb9_txt_excluded):
        gdb9_url_excluded = "https://springernature.figshare.com/ndownloader/files/3195404"
        downloaded_excluded = join(gdb9dir, "uncharacterized.txt")
        logging.info("uncharacterized.txt not found locally, downloading.")
        urllib.request.urlretrieve(gdb9_url_excluded, filename=downloaded_excluded)
        gdb9_txt_excluded = downloaded_excluded

    # First get list of excluded indices
    excluded_strings = []
    with open(gdb9_txt_excluded) as f:
        lines = f.readlines()
        excluded_strings = [line.split()[0] for line in lines if len(line.split()) > 0]

    excluded_idxs = [int(idx) - 1 for idx in excluded_strings if is_int(idx)]

    assert len(excluded_idxs) == 3054, (
        "There should be exactly 3054 excluded atoms. Found {}".format(len(excluded_idxs))
    )

    # Now, create a list of indices
    Ngdb9 = 133885
    Nexcluded = 3054

    included_idxs = np.array(sorted(list(set(range(Ngdb9)) - set(excluded_idxs))))

    # Now generate random permutations to assign molecules to training/validation/test sets.
    Nmols = Ngdb9 - Nexcluded

    Ntrain = 100000
    Ntest = int(0.1 * Nmols)
    Nvalid = Nmols - (Ntrain + Ntest)

    # Generate random permutation
    np.random.seed(0)
    data_perm = np.random.permutation(Nmols)

    # Now use the permutations to generate the indices of the dataset splits.
    # train, valid, test, extra = np.split(included_idxs[data_perm], [Ntrain, Ntrain+Nvalid, Ntrain+Nvalid+Ntest])

    train, valid, test, extra = np.split(
        data_perm, [Ntrain, Ntrain + Nvalid, Ntrain + Nvalid + Ntest]
    )

    assert len(extra) == 0, "Split was inexact {} {} {} {}".format(
        len(train), len(valid), len(test), len(extra)
    )

    train = included_idxs[train]
    valid = included_idxs[valid]
    test = included_idxs[test]

    splits = {"train": train, "valid": valid, "test": test}

    # Lower samples just for testing (opt-in via QM9_SUBSET; full data by default).
    if QM9_SUBSET > 0:
        logging.warning("QM9_SUBSET=%d: truncating every split -- smoke test only!", QM9_SUBSET)
        for key in splits:
            splits[key] = splits[key][:QM9_SUBSET]

    # Cleanup (only files we downloaded ourselves, never the repo's raw copy)
    if downloaded_excluded is not None:
        cleanup_file(downloaded_excluded, cleanup)

    return splits


def get_thermo_dict(gdb9dir, cleanup=True):
    """
    Get dictionary of thermochemical energy to subtract off from
    properties of molecules.

    Probably would be easier just to just precompute this and enter it explicitly.
    """
    # Prefer the copy shipped in RAW_DATA_DIR; only hit figshare if it is absent.
    gdb9_txt_thermo = join(RAW_DATA_DIR, "atomref.txt")
    downloaded_thermo = None
    if not exists(gdb9_txt_thermo):
        gdb9_url_thermo = "https://springernature.figshare.com/ndownloader/files/3195395"
        downloaded_thermo = join(gdb9dir, "atomref.txt")
        logging.info("atomref.txt not found locally, attempting download.")
        try:
            urllib.request.urlretrieve(gdb9_url_thermo, filename=downloaded_thermo)
        except Exception as err:
            logging.warning("Could not download atomref.txt: %s", err)
            return {}
        if not exists(downloaded_thermo) or os.path.getsize(downloaded_thermo) == 0:
            logging.warning("Downloaded atomref.txt is empty (figshare blocked?).")
            cleanup_file(downloaded_thermo, cleanup)
            return {}
        gdb9_txt_thermo = downloaded_thermo

    # Loop over file of thermochemical energies
    therm_targets = ["zpve", "U0", "U", "H", "G", "Cv"]

    # Dictionary that
    id2charge = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}

    # Loop over file of thermochemical energies
    therm_energy = {target: {} for target in therm_targets}
    with open(gdb9_txt_thermo) as f:
        for line in f:
            # If line starts with an element, convert the rest to a list of energies.
            split = line.split()

            # Check charge corresponds to an atom
            if len(split) == 0 or split[0] not in id2charge.keys():
                continue

            # Loop over learning targets with defined thermochemical energy
            for therm_target, split_therm in zip(therm_targets, split[1:]):
                therm_energy[therm_target][id2charge[split[0]]] = float(split_therm)

    # Cleanup file when finished (only if we downloaded it ourselves).
    if downloaded_thermo is not None:
        cleanup_file(downloaded_thermo, cleanup)

    return therm_energy


def add_thermo_targets(data, therm_energy_dict):
    """
    Adds a new molecular property, which is the thermochemical energy.

    Parameters
    ----------
    data : ?????
        QM9 dataset split.
    therm_energy : dict
        Dictionary of thermochemical energies for relevant properties found using :get_thermo_dict:
    """
    # Get the charge and number of charges
    charge_counts = get_unique_charges(data["charges"])

    # Now, loop over the targets with defined thermochemical energy
    for target, target_therm in therm_energy_dict.items():
        thermo = np.zeros(len(data[target]))

        # Loop over each charge, and multiplicity of the charge
        for z, num_z in charge_counts.items():
            if z == 0 or z not in target_therm:
                continue
            # Now add the thermochemical energy per atomic charge * the number of atoms of that type
            thermo += target_therm[z] * num_z

        # Now add the thermochemical energy as a property
        data[target + "_thermo"] = thermo

    return data


def get_unique_charges(charges):
    """
    Get count of each charge for each molecule.
    """
    # Create a dictionary of charges
    charge_counts = {int(z): np.zeros(len(charges), dtype=int) for z in np.unique(charges)}
    print(charge_counts.keys())

    # Loop over molecules, for each molecule get the unique charges
    for idx, mol_charges in enumerate(charges):
        # For each molecule, get the unique charge and multiplicity
        for z, num_z in zip(*np.unique(mol_charges, return_counts=True)):
            # Store the multiplicity of each charge in charge_counts
            charge_counts[int(z)][idx] = num_z

    return charge_counts
