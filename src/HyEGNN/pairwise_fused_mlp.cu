// Fully fused PairwiseSymAsymLayer: gather + both MLPs + gate + scatter in ONE
// kernel per direction, weights resident in shared memory.
//
// Why: with one direction per pair the layer sees ~578 rows x 64 features
// (~25 MFLOP). Measured in the real loop it still cost 688 us/layer because the
// two MLPs went to cuBLAS as tiny GEMMs and the layer was ~101 launches. The
// arithmetic is irrelevant at this size; launches are the cost. Here the whole
// forward is 1 kernel (+1 clone) and the backward is 1 kernel (+1 clone +4 GEMMs
// +4 bias sums for the weight gradients, which ARE a reduction over all pairs).
//
// Restricted to hidden_nf == 64: blockDim == nf and all weights + activations
// (~85 KB) must fit in shared memory (nf=128 would need ~330 KB > 228 KB).
// Math is the eager layer's, op for op, in fp32:
//   h_s=h_i+h_j  h_d=h_i-h_j  a=|h_d|  r=|x_i-x_j|^2
//   z  = W2s.silu(W1s.[h_s,a,r]+b1s)+b2s
//   g  = sigmoid(W2g.silu(W1g.[a,r]+b1g)+b2g)
//   h_i'=h_i+z+h_d*g   h_j'=h_j+z-h_d*g
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {
constexpr int NF = 64;
constexpr int A_W = 2*NF + 1;   // [h_s, |h_d|, r]
constexpr int B_W = NF + 1;     // [|h_d|, r]  == A[NF..2NF]
constexpr int PAIRS_PER_BLOCK = 4;

// shared-memory layout (floats)
constexpr int OFF_W1S = 0;                          // NF x A_W
constexpr int OFF_B1S = OFF_W1S + NF*A_W;
constexpr int OFF_W2S = OFF_B1S + NF;               // NF x NF
constexpr int OFF_B2S = OFF_W2S + NF*NF;
constexpr int OFF_W1G = OFF_B2S + NF;               // NF x B_W
constexpr int OFF_B1G = OFF_W1G + NF*B_W;
constexpr int OFF_W2G = OFF_B1G + NF;               // NF x NF
constexpr int OFF_B2G = OFF_W2G + NF*NF;
constexpr int OFF_A   = OFF_B2G + NF;               // A_W
constexpr int OFF_V0  = OFF_A + A_W;                // NF scratch
constexpr int OFF_V1  = OFF_V0 + NF;                // NF scratch
constexpr int SMEM_FLOATS = OFF_V1 + NF;
constexpr int SMEM_BYTES = SMEM_FLOATS * sizeof(float);

__device__ __forceinline__ float silu(float v)  { return v / (1.f + __expf(-v)); }
__device__ __forceinline__ float dsilu(float v) { float s = 1.f/(1.f+__expf(-v)); return s*(1.f + v*(1.f-s)); }
__device__ __forceinline__ float sigm(float v)  { return 1.f/(1.f+__expf(-v)); }

__device__ __forceinline__ void load_weights(
    float* sm, const float* W1s, const float* b1s, const float* W2s, const float* b2s,
    const float* W1g, const float* b1g, const float* W2g, const float* b2g) {
  for (int i = threadIdx.x; i < NF*A_W; i += blockDim.x) sm[OFF_W1S+i] = W1s[i];
  for (int i = threadIdx.x; i < NF*NF;  i += blockDim.x) { sm[OFF_W2S+i] = W2s[i]; sm[OFF_W2G+i] = W2g[i]; }
  for (int i = threadIdx.x; i < NF*B_W; i += blockDim.x) sm[OFF_W1G+i] = W1g[i];
  if (threadIdx.x < NF) { sm[OFF_B1S+threadIdx.x]=b1s[threadIdx.x]; sm[OFF_B2S+threadIdx.x]=b2s[threadIdx.x];
                          sm[OFF_B1G+threadIdx.x]=b1g[threadIdx.x]; sm[OFF_B2G+threadIdx.x]=b2g[threadIdx.x]; }
}

// ------------------------------------------------------------------ forward
__global__ void fused_fwd(
    const float* __restrict__ h, const float* __restrict__ x,
    const long* __restrict__ rows, const long* __restrict__ cols,
    const float* W1s, const float* b1s, const float* W2s, const float* b2s,
    const float* W1g, const float* b1g, const float* W2g, const float* b2g,
    float* __restrict__ out,
    float* __restrict__ sA, float* __restrict__ sS1, float* __restrict__ sHS,
    float* __restrict__ sG1, float* __restrict__ sHG, float* __restrict__ sGate,
    int E) {
  extern __shared__ float sm[];
  load_weights(sm, W1s,b1s,W2s,b2s,W1g,b1g,W2g,b2g);
  __syncthreads();
  const int k = threadIdx.x;
  float* A  = sm + OFF_A;
  float* V0 = sm + OFF_V0;
  float* V1 = sm + OFF_V1;

  for (int p = 0; p < PAIRS_PER_BLOCK; ++p) {
    const int e = blockIdx.x * PAIRS_PER_BLOCK + p;
    if (e >= E) break;
    const long ri = rows[e], ci = cols[e];
    const float hi = h[ri*NF + k], hj = h[ci*NF + k];
    const float hd = hi - hj;
    A[k]      = hi + hj;
    A[NF + k] = fabsf(hd);
    if (k == 0) {
      float d0=x[ri*3]-x[ci*3], d1=x[ri*3+1]-x[ci*3+1], d2=x[ri*3+2]-x[ci*3+2];
      A[2*NF] = d0*d0 + d1*d1 + d2*d2;
    }
    __syncthreads();
    // f_s layer 1
    float s1 = sm[OFF_B1S + k];
    #pragma unroll 8
    for (int i = 0; i < A_W; ++i) s1 += sm[OFF_W1S + k*A_W + i] * A[i];
    const float hs = silu(s1);
    V0[k] = hs;
    // gate layer 1 (input is A[NF..2NF], contiguous)
    float g1 = sm[OFF_B1G + k];
    #pragma unroll 8
    for (int i = 0; i < B_W; ++i) g1 += sm[OFF_W1G + k*B_W + i] * A[NF + i];
    const float hg = silu(g1);
    V1[k] = hg;
    __syncthreads();
    // layer 2s
    float z = sm[OFF_B2S + k];
    #pragma unroll 8
    for (int i = 0; i < NF; ++i) z += sm[OFF_W2S + k*NF + i] * V0[i];
    float g2 = sm[OFF_B2G + k];
    #pragma unroll 8
    for (int i = 0; i < NF; ++i) g2 += sm[OFF_W2G + k*NF + i] * V1[i];
    const float gate = sigm(g2);
    const float zd = hd * gate;
    out[ri*NF + k] = hi + z + zd;
    out[ci*NF + k] = hj + z - zd;
    // stash for backward (E is tiny; these are cheap)
    sS1[(long)e*NF+k]=s1; sHS[(long)e*NF+k]=hs; sG1[(long)e*NF+k]=g1; sHG[(long)e*NF+k]=hg; sGate[(long)e*NF+k]=gate;
    for (int i = k; i < A_W; i += NF) sA[(long)e*A_W + i] = A[i];
    __syncthreads();   // A/V0/V1 reused by next pair
  }
}

// ----------------------------------------------------------------- backward
// Data-path gradients. dh must arrive as a copy of dout (pass-through for
// non-pair nodes); pair rows are overwritten with plain stores because rows
// and cols are disjoint and each node appears at most once. Per-pair
// pre-activation grads are written out for the weight-gradient GEMMs.
__global__ void fused_bwd(
    const float* __restrict__ h, const float* __restrict__ x,
    const long* __restrict__ rows, const long* __restrict__ cols,
    const float* W1s, const float* b1s, const float* W2s, const float* b2s,
    const float* W1g, const float* b1g, const float* W2g, const float* b2g,
    const float* __restrict__ dout,
    const float* __restrict__ sS1, const float* __restrict__ sG1, const float* __restrict__ sGate,
    float* __restrict__ dh, float* __restrict__ dx,
    float* __restrict__ dS1, float* __restrict__ dZ, float* __restrict__ dG1, float* __restrict__ dG2,
    int E) {
  extern __shared__ float sm[];
  load_weights(sm, W1s,b1s,W2s,b2s,W1g,b1g,W2g,b2g);
  __syncthreads();
  const int k = threadIdx.x;
  float* V0 = sm + OFF_V0;   // dz / ds1 broadcast
  float* V1 = sm + OFF_V1;   // dg2 / dg1 broadcast
  float* A  = sm + OFF_A;    // dA accumulation (A_W)

  for (int p = 0; p < PAIRS_PER_BLOCK; ++p) {
    const int e = blockIdx.x * PAIRS_PER_BLOCK + p;
    if (e >= E) break;
    const long ri = rows[e], ci = cols[e];
    const float hi = h[ri*NF+k], hj = h[ci*NF+k], hd = hi - hj;
    const float gate = sGate[(long)e*NF+k];
    const float gi = dout[ri*NF+k], gj = dout[ci*NF+k];
    // epilogue grads
    const float dz    = gi + gj;
    const float dgate = (gi - gj) * hd;
    float dhi = gi*(1.f+gate) - gj*gate;
    float dhj = gj*(1.f+gate) - gi*gate;
    const float dg2 = dgate * gate * (1.f - gate);
    V0[k] = dz;  V1[k] = dg2;
    dZ[(long)e*NF+k] = dz;  dG2[(long)e*NF+k] = dg2;
    __syncthreads();
    // layer-2 backward: dhid[k] = sum_o W2[o][k] * dvec[o]
    float dhs = 0.f, dhg = 0.f;
    #pragma unroll 8
    for (int o = 0; o < NF; ++o) { dhs += sm[OFF_W2S + o*NF + k]*V0[o]; dhg += sm[OFF_W2G + o*NF + k]*V1[o]; }
    const float ds1 = dhs * dsilu(sS1[(long)e*NF+k]);
    const float dg1 = dhg * dsilu(sG1[(long)e*NF+k]);
    dS1[(long)e*NF+k] = ds1;  dG1[(long)e*NF+k] = dg1;
    __syncthreads();            // V0/V1 about to be overwritten
    V0[k] = ds1;  V1[k] = dg1;
    __syncthreads();
    // layer-1 backward into A-space: dA[i] = sum_o W1s[o][i]*ds1[o]  (+ W1g for i>=NF)
    float dA_k = 0.f, dA_nk = 0.f, dB_k = 0.f;
    #pragma unroll 8
    for (int o = 0; o < NF; ++o) {
      const float w = sm[OFF_W1S + o*A_W];
      dA_k  += sm[OFF_W1S + o*A_W + k]      * V0[o];
      dA_nk += sm[OFF_W1S + o*A_W + NF + k] * V0[o];
      dB_k  += sm[OFF_W1G + o*B_W + k]      * V1[o];
      (void)w;
    }
    if (k == 0) {
      float dr = 0.f;
      for (int o = 0; o < NF; ++o) dr += sm[OFF_W1S + o*A_W + 2*NF]*V0[o] + sm[OFF_W1G + o*B_W + NF]*V1[o];
      A[2*NF] = dr;
    }
    // combine: A[k]=h_s grad, A[NF+k]=|h_d| grad (from both MLPs)
    const float dhs_in = dA_k;
    const float dabs   = dA_nk + dB_k;
    const float dhd    = dabs * (hd >= 0.f ? 1.f : -1.f);
    dhi += dhs_in + dhd;
    dhj += dhs_in - dhd;
    dh[ri*NF+k] = dhi;          // plain stores: rows/cols disjoint & unique
    dh[ci*NF+k] = dhj;
    __syncthreads();
    if (k == 0) {
      const float dr = A[2*NF];
      #pragma unroll
      for (int c = 0; c < 3; ++c) {
        const float g = 2.f*dr*(x[ri*3+c]-x[ci*3+c]);
        atomicAdd(&dx[ri*3+c],  g);
        atomicAdd(&dx[ci*3+c], -g);
      }
    }
    __syncthreads();
  }
}

bool g_attr_set = false;
void ensure_smem() {
  if (!g_attr_set) {
    cudaFuncSetAttribute(fused_fwd, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    cudaFuncSetAttribute(fused_bwd, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    g_attr_set = true;
  }
}
} // namespace

std::vector<torch::Tensor> fused_forward(
    torch::Tensor h, torch::Tensor x, torch::Tensor rows, torch::Tensor cols,
    torch::Tensor W1s, torch::Tensor b1s, torch::Tensor W2s, torch::Tensor b2s,
    torch::Tensor W1g, torch::Tensor b1g, torch::Tensor W2g, torch::Tensor b2g) {
  TORCH_CHECK(h.size(1) == NF, "fused pairwise MLP kernel requires hidden_nf == 64");
  ensure_smem();
  const int E = rows.size(0);
  auto out = h.clone();
  auto opt = h.options();
  auto sA=torch::empty({E,A_W},opt), sS1=torch::empty({E,NF},opt), sHS=torch::empty({E,NF},opt),
       sG1=torch::empty({E,NF},opt), sHG=torch::empty({E,NF},opt), sGate=torch::empty({E,NF},opt);
  if (E > 0) {
    const int blocks = (E + PAIRS_PER_BLOCK - 1) / PAIRS_PER_BLOCK;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    fused_fwd<<<blocks, NF, SMEM_BYTES, stream>>>(
        h.contiguous().data_ptr<float>(), x.contiguous().data_ptr<float>(),
        rows.data_ptr<long>(), cols.data_ptr<long>(),
        W1s.contiguous().data_ptr<float>(), b1s.contiguous().data_ptr<float>(),
        W2s.contiguous().data_ptr<float>(), b2s.contiguous().data_ptr<float>(),
        W1g.contiguous().data_ptr<float>(), b1g.contiguous().data_ptr<float>(),
        W2g.contiguous().data_ptr<float>(), b2g.contiguous().data_ptr<float>(),
        out.data_ptr<float>(),
        sA.data_ptr<float>(), sS1.data_ptr<float>(), sHS.data_ptr<float>(),
        sG1.data_ptr<float>(), sHG.data_ptr<float>(), sGate.data_ptr<float>(), E);
  }
  return {out, sA, sS1, sHS, sG1, sHG, sGate};
}

std::vector<torch::Tensor> fused_backward(
    torch::Tensor h, torch::Tensor x, torch::Tensor rows, torch::Tensor cols,
    torch::Tensor W1s, torch::Tensor b1s, torch::Tensor W2s, torch::Tensor b2s,
    torch::Tensor W1g, torch::Tensor b1g, torch::Tensor W2g, torch::Tensor b2g,
    torch::Tensor dout, torch::Tensor sS1, torch::Tensor sG1, torch::Tensor sGate) {
  ensure_smem();
  const int E = rows.size(0);
  auto dh = dout.contiguous().clone();
  auto dx = torch::zeros_like(x);
  auto opt = h.options();
  auto dS1=torch::empty({E,NF},opt), dZ=torch::empty({E,NF},opt), dG1=torch::empty({E,NF},opt), dG2=torch::empty({E,NF},opt);
  if (E > 0) {
    const int blocks = (E + PAIRS_PER_BLOCK - 1) / PAIRS_PER_BLOCK;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    fused_bwd<<<blocks, NF, SMEM_BYTES, stream>>>(
        h.contiguous().data_ptr<float>(), x.contiguous().data_ptr<float>(),
        rows.data_ptr<long>(), cols.data_ptr<long>(),
        W1s.contiguous().data_ptr<float>(), b1s.contiguous().data_ptr<float>(),
        W2s.contiguous().data_ptr<float>(), b2s.contiguous().data_ptr<float>(),
        W1g.contiguous().data_ptr<float>(), b1g.contiguous().data_ptr<float>(),
        W2g.contiguous().data_ptr<float>(), b2g.contiguous().data_ptr<float>(),
        dout.contiguous().data_ptr<float>(),
        sS1.data_ptr<float>(), sG1.data_ptr<float>(), sGate.data_ptr<float>(),
        dh.data_ptr<float>(), dx.data_ptr<float>(),
        dS1.data_ptr<float>(), dZ.data_ptr<float>(), dG1.data_ptr<float>(), dG2.data_ptr<float>(), E);
  }
  return {dh, dx, dS1, dZ, dG1, dG2};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_forward",  &fused_forward);
  m.def("fused_backward", &fused_backward);
}
