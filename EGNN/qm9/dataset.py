import os
from torch.utils.data import DataLoader
from qm9.data.utils import initialize_datasets
from qm9.args import init_argparse
from qm9.data.collate import collate_fn
import torch

def worker_kwargs(num_workers):
    """DataLoader options for worker processes.

    Python 3.14 defaults to the 'forkserver' start method, which cannot spawn
    workers on this cluster (ConnectionResetError), so ask for 'fork' explicitly.
    None of these change what a batch contains.
    """
    if num_workers <= 0:
        return {}
    # EGNN_NO_PIN=1 disables pinning: the pin thread contends with kernel issue
    # for the GIL, and the per-batch H2D volume here is <1 MB.
    return dict(multiprocessing_context='fork', persistent_workers=True,
                pin_memory=not os.environ.get('EGNN_NO_PIN'), prefetch_factor=4)


def rebuild_dataloaders(dataloaders, batch_size, num_workers, collate):
    """Recreate loaders with a different collate_fn, keeping datasets and order."""
    specs = {split: dl.dataset for split, dl in dataloaders.items()}
    shutdown_dataloaders(dataloaders)
    return {split: DataLoader(ds,
                              batch_size=batch_size,
                              shuffle=(split == 'train'),
                              num_workers=num_workers,
                              collate_fn=collate,
                              **worker_kwargs(num_workers))
            for split, ds in specs.items()}


def shutdown_dataloaders(dataloaders):
    """Stop persistent workers now instead of leaving them to atexit.

    Precompute iterates all three original loaders, so rebuilding without this
    left 24 obsolete workers alive for the rest of the run.  Explicit shutdown
    also avoids the multi-minute interpreter-exit hang seen on Python 3.14.
    """
    for loader in dataloaders.values():
        iterator = getattr(loader, '_iterator', None)
        if iterator is not None:
            iterator._shutdown_workers()
            loader._iterator = None


def retrieve_dataloaders(batch_size, num_workers=1):
    # Initialize dataloader
    args = init_argparse('qm9')
    args, datasets, num_species, charge_scale = initialize_datasets(args, args.datadir, 'qm9',
                                                                    subtract_thermo=args.subtract_thermo,
                                                                    force_download=args.force_download
                                                                    )
    qm9_to_eV = {'U0': 27.2114, 'U': 27.2114, 'G': 27.2114, 'H': 27.2114, 'zpve': 27211.4, 'gap': 27.2114, 'homo': 27.2114,
                 'lumo': 27.2114}

    for dataset in datasets.values():
        dataset.convert_units(qm9_to_eV)


    loader_kwargs = worker_kwargs(num_workers)

    # Construct PyTorch dataloaders from datasets
    dataloaders = {split: DataLoader(dataset,
                                     batch_size=batch_size,
                                     shuffle=args.shuffle if (split == 'train') else False,
                                     num_workers=num_workers,
                                     collate_fn=collate_fn,
                                     **loader_kwargs)
                         for split, dataset in datasets.items()}

    return dataloaders, charge_scale



def batch_stack(props):
    """
    Stack a list of torch.tensors so they are padded to the size of the
    largest tensor along each axis.

    Parameters
    ----------
    props : list of Pytorch Tensors
        Pytorch tensors to stack

    Returns
    -------
    props : Pytorch tensor
        Stacked pytorch tensor.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if not torch.is_tensor(props[0]):
        return torch.tensor(props)
    elif props[0].dim() == 0:
        return torch.stack(props)
    else:
        return torch.nn.utils.rnn.pad_sequence(props, batch_first=True, padding_value=0)


def drop_zeros(props, to_keep):
    """
    Function to drop zeros from batches when the entire dataset is padded to the largest molecule size.

    Parameters
    ----------
    props : Pytorch tensor
        Full Dataset


    Returns
    -------
    props : Pytorch tensor
        The dataset with  only the retained information.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if not torch.is_tensor(props[0]):
        return props
    elif props[0].dim() == 0:
        return props
    else:
        return props[:, to_keep, ...]


if __name__ == '__main__':
    '''
    dataloader = retrieve_dataloaders(batch_size=64)
    for i, batch in enumerate(dataloader['train']):
        print(i)
    '''
