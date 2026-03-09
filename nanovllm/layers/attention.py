import torch
import triton
import triton.language as tl
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from torch import nn

from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    slot = tl.load(slot_mapping_ptr + idx)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](
        key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D
    )


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        # 将输入拆分为 [seq, heads, head_dim]
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        
        k_cache, v_cache = self.k_cache, self.v_cache

        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if context.is_prefill:
            num_ps = context.num_prefill_seqs  # prefill 序列数量
            has_kv_cache = k_cache.numel() > 0 and v_cache.numel() > 0 and context.block_tables is not None

            if has_kv_cache:
                # 有 KV cache：逐个处理 prefill 序列（各自 chunk 大小不同），批量处理 decode 序列
                output_parts = []

                # ===== Prefill 部分：逐序列处理 =====
                for i in range(num_ps):
                    q_start = context.cu_seqlens_q[i].item()
                    q_end = context.cu_seqlens_q[i + 1].item()
                    q_i = q[q_start:q_end]  # (chunk_size_i, heads, dim)

                    o_i = flash_attn_with_kvcache(
                        q_i.unsqueeze(0),  # (1, chunk_size_i, heads, dim)
                        k_cache, v_cache,
                        block_table=context.block_tables[i:i+1],
                        cache_seqlens=context.context_lens[i:i+1],
                        softmax_scale=self.scale,
                        causal=True
                    ).squeeze(0)  # (chunk_size_i, heads, dim)
                    output_parts.append(o_i)

                # ===== Decode 部分：批量处理 =====
                if context.num_decode_tokens > 0:
                    q_decode = q[context.num_prefill_tokens:]
                    o_decode = flash_attn_with_kvcache(
                        q_decode.unsqueeze(1),  # (num_decode, 1, heads, dim)
                        k_cache, v_cache,
                        block_table=context.block_tables[num_ps:],
                        cache_seqlens=context.context_lens[num_ps:],
                        softmax_scale=self.scale,
                        causal=True
                    ).squeeze(1)  # (num_decode, heads, dim)
                    output_parts.append(o_decode)

                o = torch.cat(output_parts, dim=0)
            else:
                # 无 KV cache（warmup / 无 prefix cache 的首个 chunk）
                # flash_attn_varlen_func 通过 cu_seqlens 天然支持多序列
                o = flash_attn_varlen_func(
                    q, k, v,
                    cu_seqlens_q=context.cu_seqlens_q,
                    cu_seqlens_k=context.cu_seqlens_k,
                    max_seqlen_q=context.max_seqlen_q,
                    max_seqlen_k=context.max_seqlen_k,
                    softmax_scale=self.scale,
                    causal=True
                )

        else:
            # === Pure Decode ===
            o = flash_attn_with_kvcache(
                q.unsqueeze(1),
                k_cache, v_cache,
                cache_seqlens=context.context_lens,
                block_table=context.block_tables,
                softmax_scale=self.scale,
                causal=True
            ).squeeze(1)

        o = o.view(-1, self.num_heads * self.head_dim)
        return o
