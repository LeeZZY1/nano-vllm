from collections import deque
from typing import Dict
from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager

class AsyncScheduler:
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.chunk_size = config.chunked_prefill_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

        # 异步核心：记录已调度但未完成的批次
        self.pending_batches: deque[Dict] = deque()

        # 统计信息
        self.stats = {
            "total_scheduled": 0,
            "pending_batches": 0,
            "max_pending_batches": 0
        }

    def get_stats(self) -> dict:
        """获取调度器统计信息"""
        return {
            **self.stats,
            "waiting": len(self.waiting),
            "running": len(self.running),
            "current_pending": len(self.pending_batches)
        }

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def _get_effective_prefilled_len(self, seq: Sequence) -> int:
        """计算有效 prefilled 长度 = 实际完成 + pending 中待完成"""
        effective = seq.prefilled_tokens
        for batch_info in self.pending_batches:
            chunk_info = batch_info.get('chunk_info', {})
            if seq.seq_id in chunk_info:
                effective += chunk_info[seq.seq_id]
        return effective

    def schedule(self) -> tuple[list[Sequence], bool, int, int]:
        """
        调度下一个批次
        
        关键区别：使用 effective_prefilled_len（包括 pending）
        
        Returns:
            (seqs, is_prefill, num_prefill_tokens, num_decode_tokens)
        """
        CHUNK_SIZE = self.chunk_size
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        has_prefill = False
        batch_chunk_info: Dict = {}

        # 1. running 队列中未完成 prefill 的序列优先
        prefill_seq_found = None
        for seq in list(self.running):
            if num_seqs >= self.max_num_seqs:
                break
            # 【new】使用 effective_prefilled_len，考虑 pending 状态
            effective = self._get_effective_prefilled_len(seq)
            prompt_remaining = seq.num_prompt_tokens - seq.num_cached_tokens - effective
            if prompt_remaining <= 0:
                continue
            this_chunk_size = min(prompt_remaining, CHUNK_SIZE)
            if num_batched_tokens + this_chunk_size > self.max_num_batched_tokens:
                break
            num_seqs += 1
            num_batched_tokens += this_chunk_size
            scheduled_seqs.append(seq)
            has_prefill = True
            prefill_seq_found = seq

            # 【new】这里记录本次调度的序列的 prefill token 数量，供 postprocess 更新状态
            batch_chunk_info[seq.seq_id] = this_chunk_size
            break

        if prefill_seq_found is not None:
            self.running.remove(prefill_seq_found)

        # 2. waiting 队列取新 prefill
        if not has_prefill and self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]

            # 【new】使用 effective_prefilled_len，考虑 pending 状态
            effective = self._get_effective_prefilled_len(seq)
            prompt_remaining = seq.num_prompt_tokens - seq.num_cached_tokens - effective
            this_chunk_size = min(prompt_remaining, CHUNK_SIZE)
            if (this_chunk_size > 0
                    and num_batched_tokens + this_chunk_size <= self.max_num_batched_tokens
                    and self.block_manager.can_allocate(seq)):
                num_seqs += 1
                self.block_manager.allocate(seq)
                seq.status = SequenceStatus.RUNNING

                self.waiting.popleft()
                self.running.append(seq)
                scheduled_seqs.append(seq)
                num_batched_tokens += this_chunk_size
                has_prefill = True

                # 【new】记录本次调度的序列的 prefill token 数量，供 postprocess 更新状态
                batch_chunk_info[seq.seq_id] = this_chunk_size

        # 3. decode 阶段
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                if seq in scheduled_seqs:
                    continue

                # 【new】使用 effective_prefilled_len，考虑 pending 状态
                effective = self._get_effective_prefilled_len(seq)
                prompt_remaining = seq.num_prompt_tokens - seq.num_cached_tokens - effective
                
                if prompt_remaining > 0:
                    self.running.appendleft(seq)
                    continue

                if (self.block_manager.can_append(seq)
                        and num_batched_tokens + 1 <= self.max_num_batched_tokens):
                    if not has_prefill:
                        self.block_manager.may_append(seq)

                    num_batched_tokens += 1
                    num_seqs += 1
                    scheduled_seqs.append(seq)
                else:
                    self.running.appendleft(seq)
                    break

        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))

        # 【new】记录 pending 批次信息，供 postprocess 消费
        self.pending_batches.append({
            'is_prefill': has_prefill,
            'chunk_info': batch_chunk_info,
            'seq_ids': [seq.seq_id for seq in scheduled_seqs]
        })

        # 更新统计
        self.stats["total_scheduled"] += 1
        self.stats["pending_batches"] = len(self.pending_batches)
        self.stats["max_pending_batches"] = max(
            self.stats["max_pending_batches"],
            len(self.pending_batches)
        )

        # 计算 token 数量
        num_prefill_tokens = 0
        num_decode_tokens = 0
        for seq in scheduled_seqs:
            if seq.seq_id in batch_chunk_info:
                num_prefill_tokens += batch_chunk_info[seq.seq_id]
            else:
                num_decode_tokens += 1

        return scheduled_seqs, has_prefill, num_prefill_tokens, num_decode_tokens

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool = False):
        """应用 pending → actual 状态转换"""
        # 从 pending 队列获取批次信息
        if not self.pending_batches:
            raise RuntimeError("postprocess called but no pending batches")
        
        batch_info = self.pending_batches.popleft()
        chunk_info = batch_info['chunk_info']

        # 过滤已取消的序列
        active_seqs = [seq for seq in seqs if not getattr(seq, 'aborted', False)]
        
        if not active_seqs:
            return

        if is_prefill:
            if len(active_seqs) > 1:   # 混合批次
                
                # 第一个序列是prefill序列，后续序列是decode序列
                prefill_seq = active_seqs[0]
                if prefill_seq.seq_id in chunk_info:
                    prefill_seq.prefilled_tokens += chunk_info[prefill_seq.seq_id]

                for seq, token_id in zip(active_seqs[1:], token_ids):
                    if seq.aborted:
                        continue

                    self.block_manager.may_append(seq)
                    seq.append_token(token_id)

                    if (not seq.ignore_eos and token_id == self.eos) \
                            or seq.num_completion_tokens == seq.max_tokens:
                        seq.status = SequenceStatus.FINISHED
                        self.block_manager.deallocate(seq)
                        self.running.remove(seq)
            else:   # 纯 prefill
                prefill_seq = active_seqs[0]
                if prefill_seq.seq_id in chunk_info:
                    prefill_seq.prefilled_tokens += chunk_info[prefill_seq.seq_id]

        else:   # 纯 decode
            for seq, token_id in zip(active_seqs, token_ids):
                if seq.aborted:
                    continue

                seq.append_token(token_id)
                
                if (not seq.ignore_eos and token_id == self.eos) \
                        or seq.num_completion_tokens == seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    self.running.remove(seq)

    def abort_request(self, request_id: str):
        """取消请求"""
        # 从 waiting 队列移除
        self.waiting = deque([
            seq for seq in self.waiting 
            if not (hasattr(seq, 'request_id') and seq.request_id == request_id)
        ])
        
        # 从 running 队列移除并释放资源
        for seq in list(self.running):
            if hasattr(seq, 'request_id') and seq.request_id == request_id:
                seq.aborted = True
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
                break

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)