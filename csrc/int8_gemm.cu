/*
 * INT8 GEMM kernel for nano-vllm
 *
 * Strategy:
 *   - Weight: per-output-channel static quantization (done once at load time)
 *   - Activation: per-token dynamic quantization (done at runtime)
 *   - Compute: tiled INT8 GEMM with __dp4a (4× INT8 → INT32 per instruction)
 *   - Epilogue: dequantize INT32 → FP16/BF16 using x_scale * w_scale
 *
 * Tile sizes: TILE_M=32, TILE_N=32, TILE_K=32
 * Block dim:  (32, 32) = 1024 threads  →  each thread owns one (m, n) output element
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <float.h>

#define TILE 32          // square tile for M / N / K
#define WARP_SIZE 32

// ─────────────────────────────────────────────────────────────────────────────
// 1. Quantize weight  [N, K] fp → [N, K] int8  +  [N] float scale
//    Per-output-channel symmetric: scale[n] = max|w[n,:]| / 127
// ─────────────────────────────────────────────────────────────────────────────
template <typename scalar_t>
__global__ void quantize_weight_kernel(
    const scalar_t* __restrict__ w,   // [N, K]
    int8_t*         __restrict__ wq,  // [N, K]  out
    float*          __restrict__ ws,  // [N]      out
    int N, int K
) {
    int n = blockIdx.x;
    if (n >= N) return;

    const scalar_t* row = w + n * K;

    // ── find max |w| across K, reduction inside one block row ──
    float max_abs = 0.f;
    for (int k = threadIdx.x; k < K; k += blockDim.x)
        max_abs = fmaxf(max_abs, fabsf((float)row[k]));

    // warp reduce
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1)
        max_abs = fmaxf(max_abs, __shfl_xor_sync(0xffffffff, max_abs, mask));

    // block reduce via smem
    __shared__ float smem[WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int wid  = threadIdx.x / WARP_SIZE;
    if (lane == 0) smem[wid] = max_abs;
    __syncthreads();
    if (wid == 0) {
        int n_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        max_abs = (lane < n_warps) ? smem[lane] : 0.f;
        for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1)
            max_abs = fmaxf(max_abs, __shfl_xor_sync(0xffffffff, max_abs, mask));
        if (lane == 0) smem[0] = max_abs;
    }
    __syncthreads();
    max_abs = smem[0];

    float scale     = max_abs / 127.f + 1e-8f;
    float inv_scale = 1.f / scale;
    if (threadIdx.x == 0) ws[n] = scale;

    // quantize
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        float v = (float)row[k] * inv_scale;
        wq[n * K + k] = (int8_t)__float2int_rn(fmaxf(-127.f, fminf(127.f, v)));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Quantize activation  [M, K] fp → [M, K] int8  +  [M] float scale
//    Per-token symmetric: scale[m] = max|x[m,:]| / 127
//    (Same structure as weight quantize, launched with N→M)
// ─────────────────────────────────────────────────────────────────────────────
template <typename scalar_t>
__global__ void quantize_activation_kernel(
    const scalar_t* __restrict__ x,   // [M, K]
    int8_t*         __restrict__ xq,  // [M, K]  out
    float*          __restrict__ xs,  // [M]      out
    int M, int K
) {
    int m = blockIdx.x;
    if (m >= M) return;

    const scalar_t* row = x + m * K;

    float max_abs = 0.f;
    for (int k = threadIdx.x; k < K; k += blockDim.x)
        max_abs = fmaxf(max_abs, fabsf((float)row[k]));

    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1)
        max_abs = fmaxf(max_abs, __shfl_xor_sync(0xffffffff, max_abs, mask));

    __shared__ float smem[WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int wid  = threadIdx.x / WARP_SIZE;
    if (lane == 0) smem[wid] = max_abs;
    __syncthreads();
    if (wid == 0) {
        int n_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        max_abs = (lane < n_warps) ? smem[lane] : 0.f;
        for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1)
            max_abs = fmaxf(max_abs, __shfl_xor_sync(0xffffffff, max_abs, mask));
        if (lane == 0) smem[0] = max_abs;
    }
    __syncthreads();
    max_abs = smem[0];

    float scale     = max_abs / 127.f + 1e-8f;
    float inv_scale = 1.f / scale;
    if (threadIdx.x == 0) xs[m] = scale;

    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        float v = (float)row[k] * inv_scale;
        xq[m * K + k] = (int8_t)__float2int_rn(fmaxf(-127.f, fminf(127.f, v)));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Tiled INT8 GEMM
//    A: [M, K] int8,  B: [N, K] int8  (B is stored row-major, i.e. B^T in math)
//    C: [M, N] int32  (raw accumulator, before dequant)
//
//    Block   : (TILE, TILE) threads  →  gridDim = (ceil(N/TILE), ceil(M/TILE))
//    Each thread computes C[row, col] via tiled dot product along K
//    __dp4a(a, b, c) = c + dot4(a_int8x4, b_int8x4)  (Volta+)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void int8_gemm_kernel(
    const int8_t* __restrict__ A,  // [M, K]
    const int8_t* __restrict__ B,  // [N, K]   ← transposed storage
    int32_t*      __restrict__ C,  // [M, N]   out
    int M, int N, int K
) {
    //
    //  smA[TILE_M][TILE_K]: tile of A, loaded by (ty, tx) → smA[ty][tx]
    //  smB[TILE_N][TILE_K]: tile of B, loaded by (ty, tx) → smB[tx][ty]
    //                        (so smB[col_in_tile][k_in_tile] = B[col_global, k_global])
    //
    __shared__ int8_t smA[TILE][TILE];  // [row_in_tile][k_in_tile]
    __shared__ int8_t smB[TILE][TILE];  // [col_in_tile][k_in_tile]

    int tx  = threadIdx.x;   // column index within tile  [0, TILE)
    int ty  = threadIdx.y;   // row    index within tile  [0, TILE)
    int row = blockIdx.y * TILE + ty;   // global row in A / C
    int col = blockIdx.x * TILE + tx;   // global col in B / C

    int32_t acc = 0;

    // ── sweep over K dimension in tiles of TILE_K ──
    for (int k0 = 0; k0 < K; k0 += TILE) {

        // Load A tile: thread (ty, tx) loads A[row, k0+tx]
        smA[ty][tx] = (row < M && (k0 + tx) < K) ? A[row * K + k0 + tx] : (int8_t)0;

        // Load B tile: thread (ty, tx) loads B[col, k0+ty]  → smB[tx][ty]
        smB[tx][ty] = (col < N && (k0 + ty) < K) ? B[col * K + k0 + ty] : (int8_t)0;

        __syncthreads();

        // ── dot product over TILE_K using __dp4a (4 int8 per instruction) ──
        #pragma unroll
        for (int kk = 0; kk < TILE; kk += 4) {
            // reinterpret 4 consecutive int8 as one int32 for __dp4a
            int a4 = *reinterpret_cast<const int32_t*>(&smA[ty][kk]);
            int b4 = *reinterpret_cast<const int32_t*>(&smB[tx][kk]);
            acc = __dp4a(a4, b4, acc);
        }

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Dequantize + optional bias
//    C_out[m, n] = C_int[m, n] * xs[m] * ws[n]  + bias[n]
// ─────────────────────────────────────────────────────────────────────────────
template <typename scalar_t>
__global__ void dequantize_kernel(
    const int32_t*  __restrict__ C_int,   // [M, N]
    const float*    __restrict__ xs,      // [M]  activation scale
    const float*    __restrict__ ws,      // [N]  weight    scale
    const scalar_t* __restrict__ bias,    // [N]  (nullable)
    scalar_t*       __restrict__ C_out,   // [M, N]  out
    int M, int N, bool has_bias
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    int m = idx / N;
    int n = idx % N;
    float val = (float)C_int[idx] * xs[m] * ws[n];
    if (has_bias) val += (float)bias[n];
    C_out[idx] = (scalar_t)val;
}

// ─────────────────────────────────────────────────────────────────────────────
// Host-side launchers (called from the pybind11 binding)
// ─────────────────────────────────────────────────────────────────────────────
#include <torch/extension.h>

// Launch quantize_weight for fp16 / bf16
torch::Tensor quantize_weight_cuda(torch::Tensor w) {
    // w: [N, K]  fp16 or bf16
    TORCH_CHECK(w.is_cuda());
    TORCH_CHECK(w.dim() == 2);
    int N = w.size(0), K = w.size(1);
    TORCH_CHECK(K % 4 == 0, "K must be a multiple of 4 for __dp4a");

    auto wq = torch::empty({N, K}, w.options().dtype(torch::kInt8));
    auto ws = torch::empty({N},    w.options().dtype(torch::kFloat32));

    int threads = std::min(256, K);
    // round up to nearest warp
    threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        w.scalar_type(), "quantize_weight_cuda", [&] {
            quantize_weight_kernel<scalar_t><<<N, threads>>>(
                w.data_ptr<scalar_t>(),
                wq.data_ptr<int8_t>(),
                ws.data_ptr<float>(),
                N, K
            );
        }
    );
    return wq;  // caller keeps ws separately
}

// Returns {wq [N,K] int8,  ws [N] float}
std::vector<torch::Tensor> quantize_weight_with_scale_cuda(torch::Tensor w) {
    TORCH_CHECK(w.is_cuda());
    TORCH_CHECK(w.dim() == 2);
    int N = w.size(0), K = w.size(1);
    TORCH_CHECK(K % 4 == 0, "K must be a multiple of 4 for __dp4a");

    auto wq = torch::empty({N, K}, w.options().dtype(torch::kInt8));
    auto ws = torch::empty({N},    w.options().dtype(torch::kFloat32));

    int threads = ((std::min(256, K) + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        w.scalar_type(), "quantize_weight_with_scale_cuda", [&] {
            quantize_weight_kernel<scalar_t><<<N, threads>>>(
                w.data_ptr<scalar_t>(),
                wq.data_ptr<int8_t>(),
                ws.data_ptr<float>(),
                N, K
            );
        }
    );
    return {wq, ws};
}

// Exposed for Python: just activation quantization (caller does GEMM via torch._int_mm)
//   x : [M, K]  fp16/bf16
// Returns {xq [M,K] int8,  xs [M] float32}
std::vector<torch::Tensor> quantize_activation_with_scale_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda());
    TORCH_CHECK(x.dim() == 2);
    int M = x.size(0), K = x.size(1);

    auto xq = torch::empty({M, K}, x.options().dtype(torch::kInt8));
    auto xs = torch::empty({M},    x.options().dtype(torch::kFloat32));

    int threads = ((std::min(256, K) + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "quantize_activation_cuda", [&] {
            quantize_activation_kernel<scalar_t><<<M, threads>>>(
                x.data_ptr<scalar_t>(),
                xq.data_ptr<int8_t>(),
                xs.data_ptr<float>(),
                M, K
            );
        }
    );
    return {xq, xs};
}

// Exposed for Python: fused dequantize (int32->fp16/bf16, no float32 intermediate)
//   C_int : [M, N]  int32  (output of torch._int_mm)
//   xs    : [M]     float32  per-token activation scale
//   ws    : [N]     float32  per-channel weight scale
//   bias  : [N]     fp16/bf16 or None
//   ref   : any tensor with target dtype/device (used for options only)
// Returns [M, N] same dtype as ref
torch::Tensor dequantize_output_cuda(
    torch::Tensor C_int,
    torch::Tensor xs,
    torch::Tensor ws,
    c10::optional<torch::Tensor> bias,
    torch::Tensor ref
) {
    TORCH_CHECK(C_int.is_cuda() && xs.is_cuda() && ws.is_cuda());
    int M = C_int.size(0), N = C_int.size(1);
    auto out = torch::empty({M, N}, ref.options());
    bool has_bias = bias.has_value();
    int total   = M * N;
    int threads = 256;
    int blocks  = (total + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        ref.scalar_type(), "dequantize_output_cuda", [&] {
            const scalar_t* bias_ptr = has_bias
                ? bias.value().data_ptr<scalar_t>() : nullptr;
            dequantize_kernel<scalar_t><<<blocks, threads>>>(
                C_int.data_ptr<int32_t>(),
                xs.data_ptr<float>(),
                ws.data_ptr<float>(),
                bias_ptr,
                out.data_ptr<scalar_t>(),
                M, N, has_bias
            );
        }
    );
    return out;
}

// Main entry: linear_int8(x, wq_KN, ws, bias=None)
//   x     : [M, K]  fp16/bf16 (activation, will be quantized inside)
//   wq_KN : [K, N]  int8      (pre-transposed weight, stored column dimension first)
//   ws    : [N]     float32   (per-output-channel weight scale)
//   bias  : [N]     fp16/bf16 or None
// Returns: [M, N] same dtype as x
//
// Pipeline (single C++ call, 3 CUDA kernels):
//   1. quantize_activation_kernel  : x fp16 → xq int8 + xs float32
//   2. at::_int_mm (cublasGemmEx)  : xq[M,K] × wq_KN[K,N] → C_int[M,N] int32
//   3. dequantize_kernel           : C_int → out fp16 (fused with bias + scales)
torch::Tensor linear_int8_cuda(
    torch::Tensor x,
    torch::Tensor wq_KN,    // [K, N] int8  (already transposed)
    torch::Tensor ws,
    c10::optional<torch::Tensor> bias
) {
    TORCH_CHECK(x.is_cuda() && wq_KN.is_cuda() && ws.is_cuda());
    TORCH_CHECK(x.dim() == 2 && wq_KN.dim() == 2);
    int M = x.size(0), K = x.size(1);
    int N = wq_KN.size(1);
    TORCH_CHECK(wq_KN.size(0) == K, "weight K mismatch");

    // ── step 1: quantize activation ──
    auto xq = torch::empty({M, K}, x.options().dtype(torch::kInt8));
    auto xs = torch::empty({M},    x.options().dtype(torch::kFloat32));
    {
        int threads = ((std::min(256, K) + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            x.scalar_type(), "quantize_activation_cuda", [&] {
                quantize_activation_kernel<scalar_t><<<M, threads>>>(
                    x.data_ptr<scalar_t>(),
                    xq.data_ptr<int8_t>(),
                    xs.data_ptr<float>(),
                    M, K
                );
            }
        );
    }

    // ── step 2: INT8 GEMM via cuBLAS (at::_int_mm) ──
    // Uses cublasGemmEx with CUDA_R_8I, INT8 Tensor Core on Ampere+.
    // xq [M,K] contiguous, wq_KN [K,N] contiguous → C_int [M,N] int32
    auto C_int = at::_int_mm(xq, wq_KN);  // [M, N] int32

    // ── step 3: dequantize + optional bias ──
    auto out = torch::empty({M, N}, x.options());
    {
        bool has_bias = bias.has_value();
        int total = M * N;
        int threads = 256;
        int blocks  = (total + threads - 1) / threads;
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            x.scalar_type(), "dequantize_cuda", [&] {
                const scalar_t* bias_ptr = has_bias
                    ? bias.value().data_ptr<scalar_t>() : nullptr;
                dequantize_kernel<scalar_t><<<blocks, threads>>>(
                    C_int.data_ptr<int32_t>(),
                    xs.data_ptr<float>(),
                    ws.data_ptr<float>(),
                    bias_ptr,
                    out.data_ptr<scalar_t>(),
                    M, N, has_bias
                );
            }
        );
    }

    return out;
}
