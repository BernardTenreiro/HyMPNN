// Fused CUDA kernels for PairwiseSymAsymLayer (HyEGNN).
//
// The layer's two MLPs are left to cuBLAS -- they are real GEMMs and the op
// profile showed them at only ~9% of the layer. What dominated was the ~40
// small elementwise/gather/scatter launches around them, each ~2.6 us, i.e. at
// the kernel-launch floor. These two kernels collapse that surrounding work
// into one launch before the GEMMs and one after.
//
// Reference semantics (models_pairwise.py::PairwiseSymAsymLayer.forward):
//     h_s = h_i + h_j ; h_d = h_i - h_j ; r = |x_i - x_j|^2
//     A   = [h_s, |h_d|, r]        -> f_s   -> z_s
//     B   = [|h_d|, r]             -> f_gate-> gate
//     h_i' = h_i + z_s + h_d*gate
//     h_j' = h_j + z_s - h_d*gate
//     nodes outside the matching pass through unchanged.
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

// ---------------------------------------------------------------- prologue
// gather + h_s/h_d/|h_d|/radial + pack the two MLP input matrices.
__global__ void prologue_fwd(const float *__restrict__ h, const float *__restrict__ x,
                             const long *__restrict__ rows, const long *__restrict__ cols,
                             float *__restrict__ A, float *__restrict__ B, int E, int nf) {
    int e = blockIdx.x;
    if (e >= E)
        return;
    long ri = rows[e], ci = cols[e];

    // radial is shared by every feature lane; compute once per block.
    __shared__ float rad;
    if (threadIdx.x == 0) {
        float d0 = x[ri * 3 + 0] - x[ci * 3 + 0];
        float d1 = x[ri * 3 + 1] - x[ci * 3 + 1];
        float d2 = x[ri * 3 + 2] - x[ci * 3 + 2];
        rad = d0 * d0 + d1 * d1 + d2 * d2;
    }
    __syncthreads();

    const int Aw = 2 * nf + 1, Bw = nf + 1;
    for (int k = threadIdx.x; k < nf; k += blockDim.x) {
        float hi = h[ri * nf + k], hj = h[ci * nf + k];
        float hd = hi - hj;
        float ad = fabsf(hd);
        A[(long)e * Aw + k] = hi + hj; // h_s
        A[(long)e * Aw + nf + k] = ad; // |h_d|
        B[(long)e * Bw + k] = ad;
    }
    if (threadIdx.x == 0) {
        A[(long)e * Aw + 2 * nf] = rad;
        B[(long)e * Bw + nf] = rad;
    }
}

__global__ void prologue_bwd(const float *__restrict__ h, const float *__restrict__ x,
                             const long *__restrict__ rows, const long *__restrict__ cols,
                             const float *__restrict__ dA, const float *__restrict__ dB,
                             float *__restrict__ dh, float *__restrict__ dx, int E, int nf) {
    int e = blockIdx.x;
    if (e >= E)
        return;
    long ri = rows[e], ci = cols[e];
    const int Aw = 2 * nf + 1, Bw = nf + 1;

    for (int k = threadIdx.x; k < nf; k += blockDim.x) {
        float hi = h[ri * nf + k], hj = h[ci * nf + k];
        float hd = hi - hj;
        float d_hs = dA[(long)e * Aw + k];
        float d_ad = dA[(long)e * Aw + nf + k] + dB[(long)e * Bw + k];
        float d_hd = d_ad * (hd >= 0.f ? 1.f : -1.f); // d|h_d|/dh_d
        atomicAdd(&dh[ri * nf + k], d_hs + d_hd);
        atomicAdd(&dh[ci * nf + k], d_hs - d_hd);
    }
    if (threadIdx.x == 0) {
        float d_rad = dA[(long)e * Aw + 2 * nf] + dB[(long)e * Bw + nf];
#pragma unroll
        for (int c = 0; c < 3; ++c) {
            float diff = x[ri * 3 + c] - x[ci * 3 + c];
            float g = 2.f * d_rad * diff;
            atomicAdd(&dx[ri * 3 + c], g);
            atomicAdd(&dx[ci * 3 + c], -g);
        }
    }
}

// ---------------------------------------------------------------- epilogue
// z_d = h_d*gate, both residual updates, and the scatter -- replacing
// h.clone() plus two index_put_ plus the elementwise chain.
__global__ void epilogue_fwd(const float *__restrict__ h, const float *__restrict__ z_s,
                             const float *__restrict__ gate, const long *__restrict__ rows,
                             const long *__restrict__ cols, float *__restrict__ out, int E,
                             int nf) {
    int e = blockIdx.x;
    if (e >= E)
        return;
    long ri = rows[e], ci = cols[e];
    for (int k = threadIdx.x; k < nf; k += blockDim.x) {
        float hi = h[ri * nf + k], hj = h[ci * nf + k];
        float zd = (hi - hj) * gate[(long)e * nf + k];
        float zs = z_s[(long)e * nf + k];
        out[ri * nf + k] = hi + zs + zd;
        out[ci * nf + k] = hj + zs - zd;
    }
}

__global__ void epilogue_bwd(const float *__restrict__ h, const float *__restrict__ gate,
                             const float *__restrict__ dout, const long *__restrict__ rows,
                             const long *__restrict__ cols, float *__restrict__ dh,
                             float *__restrict__ dz_s, float *__restrict__ dgate, int E, int nf,
                             int eager_compat) {
    int e = blockIdx.x;
    if (e >= E)
        return;
    long ri = rows[e], ci = cols[e];
    for (int k = threadIdx.x; k < nf; k += blockDim.x) {
        float hi = h[ri * nf + k], hj = h[ci * nf + k];
        float hd = hi - hj;
        float g = gate[(long)e * nf + k];
        float gi = dout[ri * nf + k]; // dL/dh_i'
        float gj = dout[ci * nf + k]; // dL/dh_j'
        // h_i' = h_i + z_s + (h_i-h_j)*g ; h_j' = h_j + z_s - (h_i-h_j)*g
        //
        // eager_compat: the reference layer does
        //     out = h.clone(); out[rows] = h_i_new; out[cols] = h_j_new
        // and the coloring emits BOTH (i,j) and (j,i), so rows and cols hold the
        // same node set and the second index_put_ fully overwrites the first.
        // h_i_new therefore receives no gradient at all in the reference. Matching
        // that keeps this kernel a pure optimisation; setting eager_compat=0 gives
        // the mathematically complete gradient instead (both branches).
        if (eager_compat) {
            atomicAdd(&dh[ri * nf + k], -gj * g);
            atomicAdd(&dh[ci * nf + k], gj * (1.f + g));
            dz_s[(long)e * nf + k] = gj;
            dgate[(long)e * nf + k] = -gj * hd;
        } else {
            atomicAdd(&dh[ri * nf + k], gi * (1.f + g) - gj * g);
            atomicAdd(&dh[ci * nf + k], gj * (1.f + g) - gi * g);
            dz_s[(long)e * nf + k] = gi + gj;
            dgate[(long)e * nf + k] = (gi - gj) * hd;
        }
    }
}

// dh starts as dout, but pair nodes are overwritten in the forward, so the
// pass-through gradient must be cleared there before the pair terms are added.
__global__ void zero_pair_rows(float *__restrict__ dh, const long *__restrict__ rows,
                               const long *__restrict__ cols, int E, int nf) {
    int e = blockIdx.x;
    if (e >= E)
        return;
    for (int k = threadIdx.x; k < nf; k += blockDim.x) {
        dh[rows[e] * nf + k] = 0.f;
        dh[cols[e] * nf + k] = 0.f;
    }
}

inline int threads_for(int nf) {
    return nf < 128 ? 64 : 128;
}

} // namespace

std::vector<torch::Tensor> prologue_forward(torch::Tensor h, torch::Tensor x, torch::Tensor rows,
                                            torch::Tensor cols) {
    int E = rows.size(0), nf = h.size(1);
    auto A = torch::empty({E, 2 * nf + 1}, h.options());
    auto B = torch::empty({E, nf + 1}, h.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (E > 0)
        prologue_fwd<<<E, threads_for(nf), 0, stream>>>(
            h.data_ptr<float>(), x.data_ptr<float>(), rows.data_ptr<long>(), cols.data_ptr<long>(),
            A.data_ptr<float>(), B.data_ptr<float>(), E, nf);
    return {A, B};
}

std::vector<torch::Tensor> prologue_backward(torch::Tensor h, torch::Tensor x, torch::Tensor rows,
                                             torch::Tensor cols, torch::Tensor dA,
                                             torch::Tensor dB) {
    int E = rows.size(0), nf = h.size(1);
    auto dh = torch::zeros_like(h);
    auto dx = torch::zeros_like(x);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (E > 0)
        prologue_bwd<<<E, threads_for(nf), 0, stream>>>(
            h.data_ptr<float>(), x.data_ptr<float>(), rows.data_ptr<long>(), cols.data_ptr<long>(),
            dA.contiguous().data_ptr<float>(), dB.contiguous().data_ptr<float>(),
            dh.data_ptr<float>(), dx.data_ptr<float>(), E, nf);
    return {dh, dx};
}

torch::Tensor epilogue_forward(torch::Tensor h, torch::Tensor z_s, torch::Tensor gate,
                               torch::Tensor rows, torch::Tensor cols) {
    int E = rows.size(0), nf = h.size(1);
    auto out = h.clone();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (E > 0)
        epilogue_fwd<<<E, threads_for(nf), 0, stream>>>(
            h.data_ptr<float>(), z_s.contiguous().data_ptr<float>(),
            gate.contiguous().data_ptr<float>(), rows.data_ptr<long>(), cols.data_ptr<long>(),
            out.data_ptr<float>(), E, nf);
    return out;
}

std::vector<torch::Tensor> epilogue_backward(torch::Tensor h, torch::Tensor gate,
                                             torch::Tensor dout, torch::Tensor rows,
                                             torch::Tensor cols, bool eager_compat) {
    int E = rows.size(0), nf = h.size(1);
    auto dh = dout.contiguous().clone();
    auto dz_s = torch::empty({E, nf}, h.options());
    auto dgate = torch::empty({E, nf}, h.options());
    if (E > 0) {
        int t = threads_for(nf);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        zero_pair_rows<<<E, t, 0, stream>>>(dh.data_ptr<float>(), rows.data_ptr<long>(),
                                            cols.data_ptr<long>(), E, nf);
        epilogue_bwd<<<E, t, 0, stream>>>(h.data_ptr<float>(), gate.contiguous().data_ptr<float>(),
                                          dout.contiguous().data_ptr<float>(),
                                          rows.data_ptr<long>(), cols.data_ptr<long>(),
                                          dh.data_ptr<float>(), dz_s.data_ptr<float>(),
                                          dgate.data_ptr<float>(), E, nf, eager_compat ? 1 : 0);
    }
    return {dh, dz_s, dgate};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("prologue_forward", &prologue_forward);
    m.def("prologue_backward", &prologue_backward);
    m.def("epilogue_forward", &epilogue_forward);
    m.def("epilogue_backward", &epilogue_backward);
}
