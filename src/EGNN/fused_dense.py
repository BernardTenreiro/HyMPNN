"""CUDA-fused drop-in replacement for E_GCL_mask.

This layer is used by BOTH the standard EGNN baselines and the HyEGNN hybrid,
so the speedup applies to both and the comparison stays symmetric.

The three MLPs stay in cuBLAS; the ~13 small launches around them collapse into
one prologue and one epilogue kernel. The forward aggregation drops the
reference's float atomics in favour of a segmented reduction over the sorted
`row` index.

Note the BACKWARD still uses atomicAdd to scatter into dh/dx (`col` is not
sorted, so only the row half could be made segmented). This layer is therefore
no more run-to-run deterministic than the reference -- measured, not assumed.
"""

import os
import weakref

import torch
from torch import nn

_KERNELS = None


def load_kernels(verbose=False):
    global _KERNELS
    if _KERNELS is None:
        from torch.utils.cpp_extension import load

        here = os.path.dirname(os.path.abspath(__file__))
        _KERNELS = load(
            name="egnn_dense",
            sources=[os.path.join(here, "dense_egcl.cu")],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90,code=sm_90"],
            verbose=verbose,
        )
    return _KERNELS


_OFF_CACHE = {}


def row_offsets(row, num_nodes):
    """Segment boundaries for a sorted `row`. offsets[n]..offsets[n+1] is node n.

    Every layer in a model sees the same `row` for a given batch, so memoise on
    the tensor identity rather than recomputing the searchsorted 5x per step.
    """
    # Allocator addresses are reused across batches, so data_ptr() alone can
    # return a stale offset tensor for different row contents.  Cache by Python
    # tensor identity and keep a weak reference to verify that identity before
    # accepting a hit.  A captured CUDA graph keeps its static row tensor alive,
    # which in turn keeps the matching offsets alive here for every replay.
    key = id(row)
    hit = _OFF_CACHE.get(key)
    if hit is not None and hit[0]() is row and hit[1] == num_nodes:
        return hit[2]
    off = torch.searchsorted(row, torch.arange(num_nodes + 1, device=row.device, dtype=row.dtype))

    def discard(ref, cache_key=key):
        current = _OFF_CACHE.get(cache_key)
        if current is not None and current[0] is ref:
            _OFF_CACHE.pop(cache_key, None)

    _OFF_CACHE[key] = (weakref.ref(row, discard), num_nodes, off)
    return off


class _Prologue(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, x, row, col):
        A = load_kernels().dense_prologue_forward(h, x, row, col)
        ctx.save_for_backward(h, x, row, col)
        return A

    @staticmethod
    def backward(ctx, dA):
        h, x, row, col = ctx.saved_tensors
        dh, dx = load_kernels().dense_prologue_backward(h, x, row, col, dA)
        return dh, dx, None, None


class _Epilogue(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, m, att, emask, row, off):
        B = load_kernels().dense_epilogue_forward(h, m, att, emask, off)
        ctx.save_for_backward(h, m, att, emask, row)
        return B

    @staticmethod
    def backward(ctx, dB):
        h, m, att, emask, row = ctx.saved_tensors
        dh, dm, datt = load_kernels().dense_epilogue_backward(h, m, att, emask, dB, row)
        return dh, dm, datt, None, None, None


class FusedEGCLMask(nn.Module):
    """Same parameters and semantics as E_GCL_mask (attention on, no coord update)."""

    def __init__(
        self, input_nf, output_nf, hidden_nf, act_fn=nn.SiLU(), attention=True, mask_is_ones=False
    ):
        super().__init__()
        self.attention = attention
        self.mask_is_ones = mask_is_ones
        self.hidden_nf = hidden_nf
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_nf * 2 + 1, hidden_nf), act_fn, nn.Linear(hidden_nf, hidden_nf), act_fn
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf), act_fn, nn.Linear(hidden_nf, output_nf)
        )
        if attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    @torch.no_grad()
    def load_from(self, eager):
        self.edge_mlp.load_state_dict(eager.edge_mlp.state_dict())
        self.node_mlp.load_state_dict(eager.node_mlp.state_dict())
        if self.attention:
            self.att_mlp.load_state_dict(eager.att_mlp.state_dict())
        return self

    def forward(
        self,
        h,
        edge_index,
        coord,
        node_mask,
        edge_mask,
        edge_attr=None,
        node_attr=None,
        n_nodes=None,
        off=None,
    ):
        row, col = edge_index
        A = _Prologue.apply(h, coord, row, col)
        m = self.edge_mlp(A)
        att = (
            self.att_mlp(m)
            if self.attention
            else torch.ones(m.size(0), 1, device=m.device, dtype=m.dtype)
        )
        # An all-ones mask (the compressed-edge path) is skipped rather than
        # multiplied: x*1.0 is exact, so this is bitwise-neutral.
        # Skipping an all-ones mask is bitwise-neutral (x*1.0 is exact), but the
        # .all() check is a device sync, so trust the caller's flag instead.
        emask = h.new_empty(0) if self.mask_is_ones or edge_mask is None else edge_mask
        if off is None:
            off = row_offsets(row, h.size(0))
        B = _Epilogue.apply(h, m, att, emask, row, off)
        return h + self.node_mlp(B), coord, edge_attr
