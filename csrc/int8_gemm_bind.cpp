#include <torch/extension.h>
#include <vector>

// ── Forward declarations (implemented in int8_gemm.cu) ──────────────────────

// quantize_weight_with_scale: fp16/bf16 [N,K] → (int8 [N,K], float [N])
std::vector<torch::Tensor> quantize_weight_with_scale_cuda(torch::Tensor w);
// quantize_activation_with_scale: fp16/bf16 [M,K] → (int8 [M,K], float [M])
std::vector<torch::Tensor> quantize_activation_with_scale_cuda(torch::Tensor x);

// dequantize_output: int32 [M,N] + scales → fp16/bf16 [M,N]  (no float32 intermediate)
torch::Tensor dequantize_output_cuda(
    torch::Tensor C_int, torch::Tensor xs, torch::Tensor ws,
    c10::optional<torch::Tensor> bias, torch::Tensor ref);

// linear_int8: x [M,K] fp16/bf16, wq [K,N] int8 (pre-transposed), ws [N] float → [M,N] fp16/bf16
torch::Tensor linear_int8_cuda(
    torch::Tensor x,
    torch::Tensor wq,
    torch::Tensor ws,
    c10::optional<torch::Tensor> bias
);

// ── pybind11 module ──────────────────────────────────────────────────────────

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "INT8 GEMM kernels for nano-vllm (per-channel weight, per-token activation)";

    m.def(
        "quantize_weight",
        &quantize_weight_with_scale_cuda,
        "Quantize weight tensor [N,K] fp16/bf16 to int8 per output-channel.\n"
        "Returns (wq: int8 [N,K], ws: float32 [N]).",
        py::arg("w")
    );

    m.def(
        "quantize_activation",
        &quantize_activation_with_scale_cuda,
        "Quantize activation tensor [M,K] fp16/bf16 to int8 per token.\n"
        "Returns (xq: int8 [M,K], xs: float32 [M]).",
        py::arg("x")
    );

    m.def(
        "dequantize_output",
        &dequantize_output_cuda,
        "Fused dequantize: int32[M,N] + xs[M] + ws[N] -> fp16/bf16[M,N]. No float32 intermediate.",
        py::arg("C_int"), py::arg("xs"), py::arg("ws"),
        py::arg("bias") = py::none(), py::arg("ref")
    );

    m.def(
        "linear_int8",
        &linear_int8_cuda,
        "INT8 linear forward.\n"
        "  x   : [M, K]  fp16 or bf16\n"
        "  wq  : [N, K]  int8  (pre-quantized weight)\n"
        "  ws  : [N]     float32 (per-channel weight scale)\n"
        "  bias: [N]     fp16/bf16 or None\n"
        "Returns [M, N] with same dtype as x.",
        py::arg("x"), py::arg("wq"), py::arg("ws"),
        py::arg("bias") = py::none()
    );
}
