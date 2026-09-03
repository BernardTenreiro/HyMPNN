"""CUDA-fused drop-in replacement for PairwiseSymAsymLayer.

The two MLPs stay in cuBLAS (they are genuine GEMMs and were only ~9% of the
layer); the ~40 small launches around them -- gather, h_s/h_d/abs/radial, the
two concatenations, the gate multiply, both residual updates, h.clone() and two
index_put_ -- collapse into one kernel before the GEMMs and one after.

Numerics follow the eager layer op for op, but the reduction order in the
gradient scatter differs, so agreement is to fp32 rounding rather than bitwise.
That is well inside this model's own run-to-run nondeterminism, which comes
from the atomic scatter_add in the standard EGNN layers.
"""
import os
import torch
from torch import nn

_KERNELS = None


def load_kernels(verbose=False):
    """JIT-build (and cache) the CUDA extension."""
    global _KERNELS
    if _KERNELS is None:
        from torch.utils.cpp_extension import load
        here = os.path.dirname(os.path.abspath(__file__))
        _KERNELS = load(
            name="hyegnn_pairwise",
            sources=[os.path.join(here, "pairwise_sym_asym.cu")],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90,code=sm_90"],
            verbose=verbose,
        )
    return _KERNELS


class _Prologue(torch.autograd.Function):
    """(h, x, rows, cols) -> A = [h_s, |h_d|, r], B = [|h_d|, r]"""

    @staticmethod
    def forward(ctx, h, x, rows, cols):
        A, B = load_kernels().prologue_forward(h.contiguous(), x.contiguous(), rows, cols)
        ctx.save_for_backward(h, x, rows, cols)
        return A, B

    @staticmethod
    def backward(ctx, dA, dB):
        h, x, rows, cols = ctx.saved_tensors
        dh, dx = load_kernels().prologue_backward(h, x, rows, cols, dA, dB)
        return dh, dx, None, None


class _Epilogue(torch.autograd.Function):
    """(h, z_s, gate, rows, cols) -> h with both endpoints of each pair updated."""

    @staticmethod
    def forward(ctx, h, z_s, gate, rows, cols, eager_compat):
        out = load_kernels().epilogue_forward(h.contiguous(), z_s, gate, rows, cols)
        ctx.save_for_backward(h, gate, rows, cols)
        ctx.eager_compat = eager_compat
        return out

    @staticmethod
    def backward(ctx, dout):
        h, gate, rows, cols = ctx.saved_tensors
        dh, dz_s, dgate = load_kernels().epilogue_backward(
            h, gate, dout, rows, cols, ctx.eager_compat)
        return dh, dz_s, dgate, None, None, None


class FusedPairwiseSymAsymLayer(nn.Module):
    """Same parameters and semantics as PairwiseSymAsymLayer, fused kernels."""

    def __init__(self, hidden_nf, act_fn=nn.SiLU(), eager_compat=False):
        super().__init__()
        self.hidden_nf = hidden_nf
        # Default False = full gradient (both endpoints). The coloring now emits
        # one direction per pair, so the eager layer computes the same thing;
        # eager_compat=True only exists to reproduce the pre-fix two-direction
        # behaviour (h_i_new with zero gradient) for old-run comparisons.
        self.eager_compat = eager_compat
        self.f_s = nn.Sequential(
            nn.Linear(2 * hidden_nf + 1, hidden_nf), act_fn,
            nn.Linear(hidden_nf, hidden_nf),
        )
        self.f_d_gate = nn.Sequential(
            nn.Linear(hidden_nf + 1, hidden_nf), act_fn,
            nn.Linear(hidden_nf, hidden_nf), nn.Sigmoid(),
        )

    @torch.no_grad()
    def load_from(self, eager_layer):
        """Copy weights from an eager PairwiseSymAsymLayer."""
        self.f_s.load_state_dict(eager_layer.f_s.state_dict())
        self.f_d_gate.load_state_dict(eager_layer.f_d_gate.state_dict())
        return self

    def forward(self, h, x, rows, cols):
        if rows.size(0) == 0:
            return h
        A, B = _Prologue.apply(h, x, rows, cols)
        z_s = self.f_s(A)
        gate = self.f_d_gate(B)
        return _Epilogue.apply(h, z_s, gate, rows, cols, self.eager_compat)
