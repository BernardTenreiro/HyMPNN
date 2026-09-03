import os
import torch


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


def collate_fn(batch):
    """
    Collation function that collates datapoints into the batch format for cormorant

    Parameters
    ----------
    batch : list of datapoints
        The data to be collated.

    Returns
    -------
    batch : dict of Pytorch tensors
        The collated data.
    """
    batch = {prop: batch_stack([mol[prop] for mol in batch]) for prop in batch[0].keys()}

    to_keep = (batch['charges'].sum(0) > 0)

    batch = {key: drop_zeros(prop, to_keep) for key, prop in batch.items()}

    atom_mask = batch['charges'] > 0
    batch['atom_mask'] = atom_mask

    #Obtain edges
    batch_size, n_nodes = atom_mask.size()
    edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)

    #mask diagonal
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask

    #edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
    batch['edge_mask'] = edge_mask.view(batch_size * n_nodes * n_nodes, 1)

    # Compressed dense edge index: only the edges the mask keeps.
    #
    # The full index from get_adj_matrix enumerates (b, i, j) in exactly this
    # flattened order, so selecting the kept positions is the same as indexing
    # that full list -- rows stay sorted ascending, which the segment-sum
    # aggregation relies on. Padded atoms are ~53% of the dense edges and
    # contribute exactly zero (edge_feat is multiplied by edge_mask), so
    # dropping them changes only how many zero terms enter the scatter-add.
    if os.environ.get('EGNN_BASELINE'):
        return batch          # A/B switch: skip edge compression

    flat = edge_mask.view(-1)
    keep = flat.nonzero(as_tuple=True)[0]
    b_idx = torch.div(keep, n_nodes * n_nodes, rounding_mode='floor')
    rem = keep - b_idx * (n_nodes * n_nodes)
    i_idx = torch.div(rem, n_nodes, rounding_mode='floor')
    j_idx = rem - i_idx * n_nodes
    batch['dense_rows'] = b_idx * n_nodes + i_idx
    batch['dense_cols'] = b_idx * n_nodes + j_idx

    return batch
