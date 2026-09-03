// Fused CUDA kernels for E_GCL_mask -- the dense EGNN layer shared by BOTH the
// standard EGNN baselines and the HyEGNN hybrid. Optimising here speeds up both
// models through the same code path, so it does not bias their comparison.
//
// Reference: MaskedEquivariantGraphConvolution.forward.
//     radial = |x_i - x_j|^2
//     A      = [h_i, h_j, radial]                    -> edge_mlp -> m
//     att    = sigmoid(att_mlp(m))
//     e      = m * att * edge_mask
//     agg[n] = sum_{edges with row==n} e
//     B      = [h, agg]                              -> node_mlp -> out
//     h_new  = h + out
//
// Two kernels replace the ~13 small launches around the three GEMMs:
//   prologue : 4 gathers + sub + pow + sum + cat            -> 1 launch
//   epilogue : 2 muls + expand + scatter_add + cat          -> 1 launch
//
// The aggregation exploits `row` being sorted ascending (guaranteed by
// get_adj_matrix's construction order, and preserved by the compressed-edge
// path in collate_fn). That turns the scatter_add's E*nf float atomics into an
// atomic-free segmented reduction -- faster, and deterministic run to run,
// which the reference version is NOT.
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

// ---------------------------------------------------------------- prologue
// NOTE: K = 2*nf+1 = 257 is alignment-hostile (55.6 vs 99 TFLOP/s on the bare
// GEMM), but splitting radial out as a rank-1 update measured SLOWER end to
// end (4.00 -> 4.79 s/epoch): the two extra (E,nf) elementwise passes cost more
// than the alignment saves. If revisited, pad A to K=264 with a zero column and
// keep ONE GEMM -- do not reassociate into extra passes.
__global__ void prologue_fwd(const float *__restrict__ h, const float *__restrict__ x,
                             const long *__restrict__ row, const long *__restrict__ col,
                             float *__restrict__ A, int E, int nf) {
    int e = blockIdx.x;
    if (e >= E)
        return;
    long ri = row[e], ci = col[e];
    const int Aw = 2 * nf + 1;
    for (int k = threadIdx.x; k < nf; k += blockDim.x) {
        A[(long)e * Aw + k] = h[ri * nf + k];
        A[(long)e * Aw + nf + k] = h[ci * nf + k];
    }
    if (threadIdx.x == 0) {
        float d0 = x[ri * 3 + 0] - x[ci * 3 + 0];
        float d1 = x[ri * 3 + 1] - x[ci * 3 + 1];
        float d2 = x[ri * 3 + 2] - x[ci * 3 + 2];
        A[(long)e * Aw + 2 * nf] = d0 * d0 + d1 * d1 + d2 * d2;
    }
}

__global__ void prologue_bwd(const float *__restrict__ x, const long *__restrict__ row,
                             const long *__restrict__ col, const float *__restrict__ dA,
                             float *__restrict__ dh, float *__restrict__ dx, int E, int nf) {
    int e = blockIdx.x;
    if (e >= E)
        return;
    long ri = row[e], ci = col[e];
    const int Aw = 2 * nf + 1;
    for (int k = threadIdx.x; k < nf; k += blockDim.x) {
        atomicAdd(&dh[ri * nf + k], dA[(long)e * Aw + k]);
        atomicAdd(&dh[ci * nf + k], dA[(long)e * Aw + nf + k]);
    }
    if (threadIdx.x == 0) {
        float d_rad = dA[(long)e * Aw + 2 * nf];
#pragma unroll
        for (int c = 0; c < 3; ++c) {
            float g = 2.f * d_rad * (x[ri * 3 + c] - x[ci * 3 + c]);
            atomicAdd(&dx[ri * 3 + c], g);
            atomicAdd(&dx[ci * 3 + c], -g);
        }
    }
}

// ---------------------------------------------------------------- epilogue
// One block per NODE. `row` is sorted, so each node owns the contiguous edge
// range [off[n], off[n+1]) and the sum needs no atomics at all.
__global__ void epilogue_fwd(const float *__restrict__ h, const float *__restrict__ m,
                             const float *__restrict__ att, const float *__restrict__ emask,
                             const long *__restrict__ off, float *__restrict__ B, int N, int nf,
                             int has_mask) {
    int n = blockIdx.x;
    if (n >= N)
        return;
    const int Bw = 2 * nf;
    long s = off[n], t = off[n + 1];
    for (int k = threadIdx.x; k < nf; k += blockDim.x) {
        B[(long)n * Bw + k] = h[(long)n * nf + k]; // identity half
        float acc = 0.f;
        for (long e = s; e < t; ++e) {
            float w = att[e];
            if (has_mask)
                w *= emask[e];
            acc += m[(long)e * nf + k] * w;
        }
        B[(long)n * Bw + nf + k] = acc;
    }
}

__global__ void epilogue_bwd(const float *__restrict__ m, const float *__restrict__ att,
                             const float *__restrict__ emask, const float *__restrict__ dB,
                             const long *__restrict__ row, float *__restrict__ dh,
                             float *__restrict__ dm, float *__restrict__ datt, int E, int nf,
                             int has_mask, int N) {
    int e = blockIdx.x;
    if (e >= E)
        return;
    long n = row[e];
    const int Bw = 2 * nf;
    float w = att[e];
    if (has_mask)
        w *= emask[e];

    // datt[e] = sum_k dagg[n,k] * m[e,k] * mask[e]  -- reduce across the block
    extern __shared__ float red[];
    float partial = 0.f;
    for (int k = threadIdx.x; k < nf; k += blockDim.x) {
        float dagg = dB[(long)n * Bw + nf + k];
        dm[(long)e * nf + k] = dagg * w;
        partial += dagg * m[(long)e * nf + k];
    }
    red[threadIdx.x] = partial;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        datt[e] = red[0] * (has_mask ? emask[e] : 1.f);
}

// identity half of B: dh += dB[:, :nf]
__global__ void epilogue_bwd_identity(const float *__restrict__ dB, float *__restrict__ dh, int N,
                                      int nf) {
    int n = blockIdx.x;
    if (n >= N)
        return;
    const int Bw = 2 * nf;
    for (int k = threadIdx.x; k < nf; k += blockDim.x)
        dh[(long)n * nf + k] = dB[(long)n * Bw + k];
}

inline int threads_for(int nf) {
    return nf <= 64 ? 64 : 128;
}

} // namespace

torch::Tensor dense_prologue_forward(torch::Tensor h, torch::Tensor x, torch::Tensor row,
                                     torch::Tensor col) {
    int E = row.size(0), nf = h.size(1);
    auto A = torch::empty({E, 2 * nf + 1}, h.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (E > 0)
        prologue_fwd<<<E, threads_for(nf), 0, stream>>>(
            h.contiguous().data_ptr<float>(), x.contiguous().data_ptr<float>(),
            row.data_ptr<long>(), col.data_ptr<long>(), A.data_ptr<float>(), E, nf);
    return A;
}

std::vector<torch::Tensor> dense_prologue_backward(torch::Tensor h, torch::Tensor x,
                                                   torch::Tensor row, torch::Tensor col,
                                                   torch::Tensor dA) {
    int E = row.size(0), nf = h.size(1);
    auto dh = torch::zeros_like(h);
    auto dx = torch::zeros_like(x);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (E > 0)
        prologue_bwd<<<E, threads_for(nf), 0, stream>>>(
            x.contiguous().data_ptr<float>(), row.data_ptr<long>(), col.data_ptr<long>(),
            dA.contiguous().data_ptr<float>(), dh.data_ptr<float>(), dx.data_ptr<float>(), E, nf);
    return {dh, dx};
}

torch::Tensor dense_epilogue_forward(torch::Tensor h, torch::Tensor m, torch::Tensor att,
                                     torch::Tensor emask, torch::Tensor off) {
    int N = h.size(0), nf = h.size(1);
    auto B = torch::empty({N, 2 * nf}, h.options());
    int has_mask = emask.numel() > 0 ? 1 : 0;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    epilogue_fwd<<<N, threads_for(nf), 0, stream>>>(
        h.contiguous().data_ptr<float>(), m.contiguous().data_ptr<float>(),
        att.contiguous().data_ptr<float>(),
        has_mask ? emask.contiguous().data_ptr<float>() : nullptr, off.data_ptr<long>(),
        B.data_ptr<float>(), N, nf, has_mask);
    return B;
}

std::vector<torch::Tensor> dense_epilogue_backward(torch::Tensor h, torch::Tensor m,
                                                   torch::Tensor att, torch::Tensor emask,
                                                   torch::Tensor dB, torch::Tensor row) {
    int E = row.size(0), N = h.size(0), nf = h.size(1);
    auto dh = torch::empty_like(h);
    auto dm = torch::empty({E, nf}, h.options());
    auto datt = torch::empty({E, 1}, h.options());
    int has_mask = emask.numel() > 0 ? 1 : 0;
    int t = threads_for(nf);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    epilogue_bwd_identity<<<N, t, 0, stream>>>(dB.contiguous().data_ptr<float>(),
                                               dh.data_ptr<float>(), N, nf);
    if (E > 0)
        epilogue_bwd<<<E, t, t * sizeof(float), stream>>>(
            m.contiguous().data_ptr<float>(), att.contiguous().data_ptr<float>(),
            has_mask ? emask.contiguous().data_ptr<float>() : nullptr,
            dB.contiguous().data_ptr<float>(), row.data_ptr<long>(), dh.data_ptr<float>(),
            dm.data_ptr<float>(), datt.data_ptr<float>(), E, nf, has_mask, N);
    return {dh, dm, datt};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
    mod.def("dense_prologue_forward", &dense_prologue_forward);
    mod.def("dense_prologue_backward", &dense_prologue_backward);
    mod.def("dense_epilogue_forward", &dense_epilogue_forward);
    mod.def("dense_epilogue_backward", &dense_epilogue_backward);
}
