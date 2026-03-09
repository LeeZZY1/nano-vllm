"""
bench_int8.py  ─  FP16 vs INT8 吞吐量 / 时延 / 显存对比

用法
----
# 修改下方 MODEL_PATH, 然后:
python bench_int8.py

指标
----
1. Throughput      (tok/s)   ← 越高越好
2. Avg latency     (ms/tok)  ← 越低越好
3. GPU memory peak (MB)      ← INT8 应比 FP16 低 ~40-50%
4. （可选）输出一致性校验   ← INT8 相对 FP16 的 token 匹配率

常见问题
--------
* cudaErrorIllegalAddress：通常是 KV cache 块不足导致 slot_mapping 越界。
  解决：减小 NUM_SEQS / MAX_OUTPUT_LEN，或提高 GPU_MEM_UTIL。
"""

import gc
import os
import time
from random import randint, seed

import torch

from nanovllm import LLM, SamplingParams

# ─── 配置 ────────────────────────────────────────────────────────────────────
MODEL_PATH      = os.path.expanduser("/mnt/workspace/.cache/modelscope/models/Qwen/Qwen3-8B")  # ← 改成你的路径
# /mnt/workspace/models/Qwen/Qwen3-0.6B
NUM_SEQS        = 16       # 大模型(>7B)建议 8~16；小模型可以调到 32~64
MAX_INPUT_LEN   = 512      # 单条 prompt token 数上限
MAX_OUTPUT_LEN  = 32     # ≥2 时启用一致性校验；越大统计越准，但耗时更长
MAX_MODEL_LEN   = 2048     # 降低可为 KV cache 留更多空间
GPU_MEM_UTIL    = 0.85     # 适当调高让 KV cache 分配更多块
ENFORCE_EAGER   = True     # 先用 eager 排除 CUDA Graph 干扰；稳定后可改 False


def make_inputs(n_seqs: int):
    seed(42)
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(50, MAX_INPUT_LEN))]
        for _ in range(n_seqs)
    ]
    sampling_params = [
        SamplingParams(
            temperature=0.0,        # greedy → 可做一致性校验
            ignore_eos=True,
            max_tokens=randint(min(50, MAX_OUTPUT_LEN), MAX_OUTPUT_LEN) if MAX_OUTPUT_LEN > 1 else 1,
        )
        for _ in range(n_seqs)
    ]
    return prompt_token_ids, sampling_params


def reset_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def run_bench(use_int8: bool, label: str):
    print(f"\n{'='*55}")
    print(f"  Mode: {label}")
    print(f"{'='*55}")

    reset_gpu()

    llm = LLM(
        MODEL_PATH,
        enforce_eager=ENFORCE_EAGER,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM_UTIL,
        use_int8=use_int8,
    )

    mem_after_load = torch.cuda.memory_allocated() / 1024**2
    print(f"  GPU memory after model load : {mem_after_load:.1f} MB")

    # ── KV cache 健康度检查 ──────────────────────────────────────────
    num_blocks = llm.model_runner.config.num_kvcache_blocks
    block_size = llm.model_runner.config.kvcache_block_size
    total_kv_tokens = num_blocks * block_size
    max_needed = NUM_SEQS * (MAX_INPUT_LEN + MAX_OUTPUT_LEN)
    print(f"  KV cache blocks : {num_blocks}  "
          f"({total_kv_tokens} tokens total, need ~{max_needed} for this bench)")
    if total_kv_tokens < max_needed:
        print(f"  [WARN] KV cache 可能不足，建议降低 NUM_SEQS 或 MAX_OUTPUT_LEN，"
              f"或提高 GPU_MEM_UTIL。继续运行但可能触发 OOM / CUDA 非法地址错误。")
    # ────────────────────────────────────────────────────────────────

    prompt_ids, sampling_params = make_inputs(NUM_SEQS)

    # ── warmup (不计时) ──
    llm.generate(["warmup"], SamplingParams(max_tokens=4))
    reset_gpu()
    torch.cuda.synchronize()

    # ── timed run ──
    t0 = time.perf_counter()
    outputs = llm.generate(prompt_ids, sampling_params, use_tqdm=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    mem_peak = torch.cuda.max_memory_allocated() / 1024**2
    total_out_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_out_tokens / elapsed
    avg_latency_ms = elapsed * 1000 / total_out_tokens

    print(f"  Sequences        : {NUM_SEQS}")
    print(f"  Total out tokens : {total_out_tokens}")
    print(f"  Time             : {elapsed:.2f} s")
    print(f"  Throughput       : {throughput:.1f} tok/s")
    print(f"  Avg latency      : {avg_latency_ms:.3f} ms/tok")
    print(f"  GPU peak memory  : {mem_peak:.1f} MB")
    print(f"  Wall time/seq    : {elapsed * 1000 / NUM_SEQS:.1f} ms")

    result = {
        "label": label,
        "throughput": throughput,
        "avg_latency_ms": avg_latency_ms,
        "gpu_peak_mb": mem_peak,
        "outputs": outputs,
    }
    import atexit
    atexit.unregister(llm.exit)  # 避免进程退出时 atexit 再次调用导致 double-destroy
    llm.exit()   # 显式销毁 dist process group，使下一次 LLM() 可正常初始化
    del llm
    reset_gpu()
    return result


def compare(fp16_res, int8_res):
    print(f"\n{'='*55}")
    print("  Comparison: INT8 vs FP16")
    print(f"{'='*55}")

    thr_gain = (int8_res["throughput"] / fp16_res["throughput"] - 1) * 100
    lat_gain = (1 - int8_res["avg_latency_ms"] / fp16_res["avg_latency_ms"]) * 100
    mem_gain = (1 - int8_res["gpu_peak_mb"]   / fp16_res["gpu_peak_mb"])   * 100

    print(f"  Throughput  : {fp16_res['throughput']:.1f} → {int8_res['throughput']:.1f} tok/s  "
          f"({thr_gain:+.1f}%)")
    print(f"  Avg latency : {fp16_res['avg_latency_ms']:.3f} → {int8_res['avg_latency_ms']:.3f} ms/tok  "
          f"({-lat_gain:+.1f}%)")
    print(f"  GPU peak    : {fp16_res['gpu_peak_mb']:.1f} → {int8_res['gpu_peak_mb']:.1f} MB  "
          f"({mem_gain:+.1f}%)")

    # ── 输出一致性（greedy @ temperature=0.0）──
    # generate() 返回 list[{"text": str, "token_ids": list[int]}]
    fp16_out = fp16_res["outputs"]
    int8_out = int8_res["outputs"]
    if fp16_out and int8_out:
        matched = 0
        total   = 0
        for f, i in zip(fp16_out, int8_out):
            f_ids = f["token_ids"]
            i_ids = i["token_ids"]
            n = min(len(f_ids), len(i_ids))
            matched += sum(a == b for a, b in zip(f_ids[:n], i_ids[:n]))
            total   += n
        pct = matched / total * 100 if total else 0
        print(f"  Token match : {matched}/{total} ({pct:.1f}%)  "
              f"[greedy, temperature=0]")
    print()


if __name__ == "__main__":
    assert os.path.isdir(MODEL_PATH), (
        f"Model path not found: {MODEL_PATH}\n"
        "请修改 bench_int8.py 顶部的 MODEL_PATH 变量"
    )

    fp16_result = run_bench(use_int8=False, label="FP16 (baseline)")
    int8_result = run_bench(use_int8=True,  label="INT8 (quantized)")
    compare(fp16_result, int8_result)
