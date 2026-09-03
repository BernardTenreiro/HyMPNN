"""Fully fused symmetric/asymmetric layer (nf=64): one kernel per direction with the
MLP weights resident in shared memory. See fused_pairwise_mlp.cu for why.
Weight gradients are the only reduction over pairs and go to cuBLAS as four
small GEMMs on the per-pair pre-activation grads the backward kernel emits."""

import os

import torch
from torch import nn

_K = None


def load_kernels(verbose=False):
    global _K
    if _K is None:
        from torch.utils.cpp_extension import load

        here = os.path.dirname(os.path.abspath(__file__))
        _K = load(
            name="hympnn_pairwise_mlp",
            sources=[os.path.join(here, "fused_pairwise_mlp.cu")],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90,code=sm_90"],
            verbose=verbose,
        )
    return _K


class _Fused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, x, rows, cols, W1s, b1s, W2s, b2s, W1g, b1g, W2g, b2g):
        out, sA, sS1, sHS, sG1, sHG, sGate = load_kernels().fused_forward(
            h, x, rows, cols, W1s, b1s, W2s, b2s, W1g, b1g, W2g, b2g
        )
        ctx.save_for_backward(
            h, x, rows, cols, W1s, b1s, W2s, b2s, W1g, b1g, W2g, b2g, sA, sS1, sHS, sG1, sHG, sGate
        )
        return out

    @staticmethod
    def backward(ctx, dout):
        (
            h,
            x,
            rows,
            cols,
            W1s,
            b1s,
            W2s,
            b2s,
            W1g,
            b1g,
            W2g,
            b2g,
            sA,
            sS1,
            sHS,
            sG1,
            sHG,
            sGate,
        ) = ctx.saved_tensors
        dh, dx, dS1, dZ, dG1, dG2 = load_kernels().fused_backward(
            h, x, rows, cols, W1s, b1s, W2s, b2s, W1g, b1g, W2g, b2g, dout, sS1, sG1, sGate
        )
        nf = h.size(1)
        # weight grads: reductions over pairs -> GEMMs (E is small, these are tiny)
        dW1s = dS1.t() @ sA
        db1s = dS1.sum(0)
        dW2s = dZ.t() @ sHS
        db2s = dZ.sum(0)
        dW1g = dG1.t() @ sA[:, nf:]
        db1g = dG1.sum(0)
        dW2g = dG2.t() @ sHG
        db2g = dG2.sum(0)
        return (dh, dx, None, None, dW1s, db1s, dW2s, db2s, dW1g, db1g, dW2g, db2g)


class FusedSymmetricAsymmetricPairwiseMLP(nn.Module):
    """Fused pairwise MLP specialized for ``hidden_nf == 64``."""

    def __init__(self, hidden_nf, act_fn=nn.SiLU()):
        super().__init__()
        assert hidden_nf == 64, "fully fused kernel is specialised for nf=64"
        self.hidden_nf = hidden_nf
        self.f_s = nn.Sequential(
            nn.Linear(2 * hidden_nf + 1, hidden_nf), act_fn, nn.Linear(hidden_nf, hidden_nf)
        )
        self.f_d_gate = nn.Sequential(
            nn.Linear(hidden_nf + 1, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            nn.Sigmoid(),
        )

    @torch.no_grad()
    def load_from(self, eager):
        self.f_s.load_state_dict(eager.f_s.state_dict())
        self.f_d_gate.load_state_dict(eager.f_d_gate.state_dict())
        return self

    def forward(self, h, x, rows, cols):
        if rows.size(0) == 0:
            return h
        return _Fused.apply(
            h,
            x,
            rows,
            cols,
            self.f_s[0].weight,
            self.f_s[0].bias,
            self.f_s[2].weight,
            self.f_s[2].bias,
            self.f_d_gate[0].weight,
            self.f_d_gate[0].bias,
            self.f_d_gate[2].weight,
            self.f_d_gate[2].bias,
        )
