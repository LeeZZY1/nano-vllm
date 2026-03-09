## 一、Chunked Prefill 实现架构梳理

### 1. 核心数据流

```
用户请求 → LLMEngine.add_request() → Scheduler.add(seq)
                                         ↓
                        ┌────── Scheduler.schedule() ──────┐
                        │  1. running 队列：续做未完成 prefill │
                        │  2. waiting 队列：取新 prefill 序列  │
                        │  3. 混合调度 decode 序列           │
                        └──────────────┬───────────────────┘
                                       ↓
                        ModelRunner.prepare_prefill()
                        → 按 chunk 切分 input_ids / positions
                        → 构建 cu_seqlens_q/k, slot_mapping, context_lens
                                       ↓
                        Attention.forward()
                        → prefill: flash_attn_with_kvcache (逐序列)
                        → decode:  flash_attn_with_kvcache (批量)
                                       ↓
                        ModelRunner.run() → 采样 (跳过 prefill logits)
                                       ↓
                        Scheduler.postprocess()
                        → prefill 序列: prefilled_tokens += chunk_size
                        → decode 序列: append_token + 检查 EOS
```

### 2. Scheduler（同步）的 chunked prefill 关键逻辑

**调度阶段** (scheduler.py)：

| 步骤 | 逻辑 |
|------|------|
| **① running 队列续做 prefill** | 遍历 `running`，找 `prompt_remaining > 0` 的序列，计算 `chunk_size = min(CHUNK_SIZE, remaining)`，调用 `allocate_incremental` 预分配 block |
| **② waiting 队列取新 prefill** | 从 `waiting` 队首取序列，同样切 chunk，分配 block，状态→RUNNING |
| **③ 混合调度 decode** | 当 `enable_chunked_prefill=True` 时，即使有 prefill 也会继续调度 decode 序列（跳过还有 `prompt_remaining > 0` 的），形成混合批次 |

**关键设计**：用 `max_num_prefill_seqs` 限制每批并发 prefill 数（默认 2），防止 decode 批突然膨胀。

**数据准备** (model_runner.py)：

- **Prefill 序列**：取 `seq[start_pos:end_pos]`，其中 `start_pos = cached + prefilled`
- **Decode 序列**（混合批次中）：只取 `seq[-1]`（最后一个 token）
- 构建 `cu_seqlens_q`（query chunk 长度）和 `cu_seqlens_k`（包含历史 KV 的总长度）
- `context_lens[i]` = `end_pos`，告诉注意力层每个序列实际的 KV 覆盖范围

**注意力计算** (attention.py)：

```
if has_kv_cache:
    for i in range(num_prefill_seqs):          # 逐序列处理 prefill
        flash_attn_with_kvcache(q_i, ...)
    if num_decode_tokens > 0:                  # 批量处理 decode
        flash_attn_with_kvcache(q_decode, ...)
    o = torch.cat(output_parts)
else:
    flash_attn_varlen_func(q, k, v, ...)       # warmup / 无 cache
```

**采样** (model_runner.py)：跳过 prefill 序列的 logits（它们不产出 token），只对 decode/完成 prefill 的序列做采样。

### 3. AsyncScheduler 的额外设计

async_scheduler.py 的关键差异：

| 特性 | 说明 |
|------|------|
| `pending_batches` 队列 | 记录已提交但未完成的批次信息（chunk_info, seq_ids） |
| `_get_effective_prefilled_len()` | 计算"实际 prefilled + 所有 pending 中待完成的"，避免重复调度同一 chunk |
| 每批只调度 **1** 个 prefill 序列 | 用 `break` 限制（与 sync 版支持多 prefill 不同） |
| `AsyncModelRunner` | 用后台线程 `run_async()` 非阻塞提交，`wait_for_result()` 同步等待 |

### 4. Block 管理

block_manager.py 重点：

- `allocate_incremental(seq, num_new_tokens)`：增量分配，只分配新 chunk 需要的 block，支持 **prefix cache** 匹配（hash 链）
- `can_allocate_incremental()`：保守估计（假设无 cache 命中），检查空闲 block 是否足够

---

## 二、与 vLLM 高级实现的对比

| 维度 | 当前实现 | vLLM (v0.6+) |
|------|----------|-------------|
| **调度策略** | 简单遍历 running → waiting → decode | 三级调度 `_schedule_running` → `_schedule_swapped` → `_schedule_prefills`，用 `SchedulingBudget` 统一管理 token/seq 预算 |
| **Prefill 并发** | 同步版最多 `max_num_prefill_seqs` 个，异步版只有 1 个 | 由 token budget 动态决定，多个 prefill 可并发 |
| **注意力后端** | 逐 prefill 序列调用 `flash_attn_with_kvcache` + decode 批量处理 | **FlashInfer** 的 `BatchPrefillWithPagedKVCacheWrapper` 原生支持混合批次，一次 kernel 调用处理所有序列 |
| **KV Cache 管理** | 基于 hash 的 prefix cache + 增量分配 | 更完善的 APC（Automatic Prefix Caching）+ 可插拔的 eviction policy + **swap to CPU** 支持 |
| **抢占策略** | 只有 `deallocate + 放回 waiting`（recompute） | recompute / **swap（KV 交换到 CPU）** 两种策略可选 |
| **CUDA 图** | 仅 decode 固定 batch size | decode 使用 CUDA Graph，且支持 **multi-step decode**（单次调度多步 decode） |
| **异步流水线** | 后台线程 + `threading.Thread` | **AsyncLLMEngine** + `asyncio` + 独立 output 处理协程，CPU 调度与 GPU 计算 overlap |
| **Speculative Decoding** | 无 | 与 chunked prefill 协同工作 |
| **Disaggregated Prefill** | 无 | prefill 和 decode 可以在不同 GPU 上执行 |
| **Multi-step Scheduling** | 每次调度一步 | 可一次调度 N 步 decode，减少调度开销 |

---

## 三、下一步可以做的提升

#### 1. **统一注意力 kernel，消除逐序列循环**

当前 attention.py 对 prefill 序列逐个调用 `flash_attn_with_kvcache`，这会产生多次 kernel launch 开销。

#### 2. **Token Budget 调度系统**

当前调度逻辑散落在多个 `if/while` 中，难以扩展。可以引入 vLLM 风格的 `SchedulingBudget`：

这样 `schedule()` 就变为：先给 running decode 序列留配额 → 用剩余配额调度 prefill → 分配剩余给 decode。

#### 3. **decode 混合批次中的 block 分配时序修复**

当前 sync scheduler 在混合批次中对 decode 序列 **不调用** `may_append`（scheduler.py，`if not running_has_prefill` 的保护），但在 `postprocess` 中才分配。这导致 slot_mapping 在 `prepare_prefill` 时可能指向尚未分配的 block。vLLM 在调度阶段就确保 block 分配完成。

### P1：功能完善

#### 4. **Swap-based 抢占**

当前抢占只有 recompute（释放所有 block，放回 waiting 重新 prefill）。添加 swap 策略：这对长序列尤其重要——重新 prefill 一个 4K 序列代价很高，但 swap 只需要 GPU↔CPU 拷贝。

#### 5. **Multi-step Decode**

减少调度开销：一次调度 N 步 decode，ModelRunner 内部循环执行

#### 6. **AsyncScheduler 支持多 prefill 并发**

当前 async_scheduler.py 用 `break` 限制每批只有 1 个 prefill。应该像 sync 版一样支持多个，用 `_get_effective_prefilled_len` 正确追踪即可。

### P2：工程质量

#### 7. **Context 避免全局可变状态**

#### 8. **prepare_prefill 中 slot_mapping 简化**

## 四、总结

当前实现已经覆盖了 chunked prefill 的核心链路：**chunk 切分 → 增量 block 分配 → 混合批次调度 → 分离注意力计算 → 跳过 prefill logits → 增量状态更新**。与 vLLM 的差距主要在：

| 层面 | 关键差距 |
|------|----------|
| **性能** | 注意力逐序列循环（最大瓶颈）、无 multi-step decode |
| **调度** | 无 token budget 系统、无 swap 抢占 |
| **工程** | 全局 Context 状态、异步模式仅支持单 prefill 并发 |
