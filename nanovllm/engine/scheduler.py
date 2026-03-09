from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager

# 实现了continuous batching调度算法
class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs # 单次批量推理允许的最大序列数量
        self.max_num_batched_tokens = config.max_num_batched_tokens # 单次批量推理允许的最大 token 数量
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        # [ ] chunked prefill 的 chunk 大小参数
        self.chunk_size = config.chunked_prefill_size
        self.enable_chunked_prefill = config.enable_chunked_prefill
        self.max_num_prefill_seqs = config.max_num_prefill_seqs

    def is_finished(self):
        # 两个队列都没有序列则表示全部完成
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        # 直接加入等待队列
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0

        # 1. 从 running 队列中收集可调度的 prefill 序列
        running_has_prefill = False
        prefill_seqs_found = []
        num_prefill_seqs = 0
        # 仅在开启 chunked prefill 时限制并发 prefill 数，关闭时不限制
        prefill_seq_limit = self.max_num_prefill_seqs if self.enable_chunked_prefill else self.max_num_seqs

        for seq in list(self.running):
            if num_seqs >= self.max_num_seqs or num_prefill_seqs >= prefill_seq_limit:
                break
            prompt_tokens_left = seq.num_prompt_tokens - seq.prefilled_tokens - seq.num_cached_tokens
            if prompt_tokens_left <= 0:
                continue
            new_chunk_size = min(self.chunk_size, prompt_tokens_left)
            # 如果加入该序列会超出 token 数上限，或者无法分配足够的块，则停止
            if num_batched_tokens + new_chunk_size > self.max_num_batched_tokens:
                break
            if not self.block_manager.can_allocate_incremental(seq, new_chunk_size):
                break
            self.block_manager.allocate_incremental(seq, new_chunk_size)
            num_seqs += 1
            num_prefill_seqs += 1
            num_batched_tokens += new_chunk_size
            scheduled_seqs.append(seq)
            running_has_prefill = True
            prefill_seqs_found.append(seq)
        
        # 从 running 队列中临时移除所有已调度的 prefill 序列，避免在 decode 阶段重复处理
        for seq in prefill_seqs_found:
            self.running.remove(seq)
        
        # 2. 从 waiting 队列中取更多的 prefill 序列
        while self.waiting and num_seqs < self.max_num_seqs and num_prefill_seqs < prefill_seq_limit:
            seq = self.waiting[0]
            prompt_tokens_left = seq.num_prompt_tokens - seq.prefilled_tokens - seq.num_cached_tokens
            new_chunk_size = min(self.chunk_size, prompt_tokens_left)
            if new_chunk_size > 0 and num_batched_tokens + new_chunk_size <= self.max_num_batched_tokens and self.block_manager.can_allocate_incremental(seq, new_chunk_size):
                num_seqs += 1
                num_prefill_seqs += 1
                self.block_manager.allocate_incremental(seq, new_chunk_size)
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                # 不在此处 append 到 self.running，函数末尾 extendleft 会统一添加
                scheduled_seqs.append(seq)
                num_batched_tokens += new_chunk_size
                running_has_prefill = True
            else:
                break

        # decode
        # 原代码默认所有序列都完成了prefill，但是当前是把chunked prefill和decode混合在一起调度的
        # 原代码如果没有需要prefill的序列，则进行解码阶段的调度
        if self.enable_chunked_prefill or not running_has_prefill:
            while self.running and num_seqs < self.max_num_seqs :
                seq = self.running.popleft()
                # 如果无法为该序列追加块，则开始抢占
                # [ ] 关于抢占的知识点
                while not self.block_manager.can_append(seq):
                    if self.running:
                        # 如果有正在运行的序列，从running队列的右侧pop一个seq并抢占资源
                        self.preempt(self.running.pop())
                    else:
                        # 如果没有正在运行的序列，直接将当前序列抢占
                        self.preempt(seq)
                        break
                else:
                    if seq in scheduled_seqs:
                        continue  # 已经调度过该序列，跳过
                    prompt_tokens_left = seq.num_prompt_tokens - seq.prefilled_tokens - seq.num_cached_tokens
                    if prompt_tokens_left > 0:
                        # 还有prefill未完成的chunk，跳过decode调度
                        # 放回running队列
                        self.running.appendleft(seq)
                        continue
                    # BUG 这里can allocate改成can_append了，否则有的时候会出现后续的assert scheduled_seqs断言失败
                    if self.block_manager.can_append(seq) and num_batched_tokens + 1 <= self.max_num_batched_tokens:
                        # 可以为该序列追加块，且不超出token数上限
                        # BUG 只有在纯decode批次才提前分配块，混合批次不提前分配
                        if not running_has_prefill:
                            # 判断是否需要追加块
                            self.block_manager.may_append(seq)
                        num_batched_tokens += 1
                        num_seqs += 1
                        scheduled_seqs.append(seq)
                    else:
                        self.running.appendleft(seq)
                        break
                    
                    
        assert scheduled_seqs
        # 再塞回去
        self.running.extendleft(reversed(scheduled_seqs))
        num_prefill_tokens = 0
        num_decode_tokens = 0
        for seq in scheduled_seqs:
            prompt_tokens_left = seq.num_prompt_tokens - seq.prefilled_tokens - seq.num_cached_tokens
            if prompt_tokens_left > 0:
                new_chunk_size = min(prompt_tokens_left, self.chunk_size)
                num_prefill_tokens += new_chunk_size
            else:
                num_decode_tokens += 1
        return scheduled_seqs, running_has_prefill, num_prefill_tokens, num_decode_tokens

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool = False) -> list[bool]:
        '''
        处理模型输出（支持多 prefill 序列并发）：
        - prefill 序列（prompt_remaining > 0）：更新 prefilled_tokens，不消耗 token_ids
        - decode 序列（prompt_remaining <= 0）：消耗 token_ids，追加生成的 token
        - token_ids 仅包含 decode 序列的输出（prefill 序列的 logits 已被跳过）
        '''
        if is_prefill:
            # 分离 prefill 和 decode 序列（scheduled_seqs 中 prefill 在前，decode 在后）
            prefill_seqs = []
            decode_seqs = []
            for seq in seqs:
                prompt_tokens_left = seq.num_prompt_tokens - seq.prefilled_tokens - seq.num_cached_tokens
                if prompt_tokens_left > 0:
                    prefill_seqs.append(seq)
                else:
                    decode_seqs.append(seq)

            # 更新所有 prefill 序列的 prefilled_tokens
            for seq in prefill_seqs:
                prompt_tokens_left = seq.num_prompt_tokens - seq.prefilled_tokens - seq.num_cached_tokens
                chunk_size = min(self.chunk_size, prompt_tokens_left)
                seq.prefilled_tokens += chunk_size

            # 处理 decode 序列（token_ids 与 decode_seqs 一一对应）
            for seq, token_id in zip(decode_seqs, token_ids):
                self.block_manager.may_append(seq)
                seq.append_token(token_id)
                if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    self.running.remove(seq)
        else:
            # 纯 decode 批次
            for seq, token_id in zip(seqs, token_ids):
                seq.append_token(token_id)
                if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    self.running.remove(seq)

