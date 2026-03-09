# from collections import deque

# from nanovllm.config import Config
# from nanovllm.engine.block_manager import BlockManager
# from nanovllm.engine.sequence import Sequence, SequenceStatus


# class Scheduler:

#     def __init__(self, config: Config):
#         self.max_num_seqs = config.max_num_seqs
#         self.max_num_batched_tokens = config.max_num_batched_tokens
#         self.eos = config.eos
#         self.block_manager = BlockManager(
#             config.num_kvcache_blocks, config.kvcache_block_size
#         )
#         self.waiting: deque[Sequence] = deque()
#         self.running: deque[Sequence] = deque()
#         self.chunk_size = config.chunk_prefill_size  # 从配置读取 chunk 大小

#     def is_finished(self):
#         return not self.waiting and not self.running

#     def add(self, seq: Sequence):
#         self.waiting.append(seq)

# #     def schedule(self) -> tuple[list[Sequence], bool]:
# #         # prefill
# #         scheduled_seqs = []
# #         num_seqs = 0
# #         num_batched_tokens = 0
# #         while self.waiting and num_seqs < self.max_num_seqs:
# #             seq = self.waiting[0]
# #             if num_batched_tokens + len(
# #                 seq
# #             ) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
# #                 break
# #             num_seqs += 1
# #             self.block_manager.allocate(seq)
# #             num_batched_tokens += len(seq) - seq.num_cached_tokens
# #             seq.status = SequenceStatus.RUNNING
# #             self.waiting.popleft()
# #             self.running.append(seq)
# #             scheduled_seqs.append(seq)
# #         if scheduled_seqs:
# #             return scheduled_seqs, True

# #         # decode
# #         while self.running and num_seqs < self.max_num_seqs:
# #             seq = self.running.popleft()
# #             while not self.block_manager.can_append(seq):
# #                 if self.running:
# #                     self.preempt(self.running.pop())
# #                 else:
# #                     self.preempt(seq)
# #                     break
# #             else:
# #                 num_seqs += 1
# #                 self.block_manager.may_append(seq)
# #                 scheduled_seqs.append(seq)
# #         assert scheduled_seqs
# #         self.running.extendleft(reversed(scheduled_seqs))
# #         return scheduled_seqs, False

#     def preempt(self, seq: Sequence):
#         seq.status = SequenceStatus.WAITING
#         self.block_manager.deallocate(seq)
#         self.waiting.appendleft(seq)

#     # def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
#     #     for seq, token_id in zip(seqs, token_ids):
#     #         seq.append_token(token_id)
#     #         if (
#     #             not seq.ignore_eos and token_id == self.eos
#     #         ) or seq.num_completion_tokens == seq.max_tokens:
#     #             seq.status = SequenceStatus.FINISHED
#     #             self.block_manager.deallocate(seq)
#     #             self.running.remove(seq)
                
#     def schedule(self) -> tuple[list[Sequence], bool]:
#         # prefill
#         CHUNK_SIZE = self.chunk_size
#         scheduled_seqs = []
#         num_seqs = 0
#         num_batched_tokens = 0
#         # 标志位：当前 Batch 是否包含 Prefill 任务
#         has_prefill = False

#         # ------------------- 添加chunked prefill逻辑 ------------------
#         # 检查是否running队列中还有prefill未完成的序列
#         prefill_seq_found = None
#         for seq in list(self.running):
#             if num_seqs >= self.max_num_seqs:
#                 break
#             # 计算还需要prefill的长度（基于 prompt）
#             prompt_remaining = seq.num_prompt_tokens - seq.num_cached_tokens - seq.prefilled_len
#             if prompt_remaining <= 0:
#                 continue  # 已经prefill完成，跳过
#             this_chunk_size = min(prompt_remaining, CHUNK_SIZE)
#             # 计算token预算
#             if num_batched_tokens + this_chunk_size > self.max_num_batched_tokens:
#                 break
#             num_seqs += 1
#             num_batched_tokens += this_chunk_size
#             scheduled_seqs.append(seq)
#             has_prefill = True
#             prefill_seq_found = seq
#             break  # 每次只处理一个prefill序列的一个chunk
        
#         # 从 running 队列中临时移除已调度的 prefill 序列，避免在 decode 阶段重复处理
#         if prefill_seq_found is not None:
#             self.running.remove(prefill_seq_found)
#         # ------------------- 添加chunked prefill逻辑 ------------------

#         # 从 waiting 队列添加新的 prefill 请求（如果还没有 prefill）
#         if not has_prefill and self.waiting and num_seqs < self.max_num_seqs:
#             # 取出waiting队列的第一个序列
#             seq = self.waiting[0]
            
#             # ------------------ 添加chunked prefill逻辑 ------------------
#             # 计算本次可以处理的长度（基于 prompt）
#             prompt_remaining = seq.num_prompt_tokens - seq.num_cached_tokens - seq.prefilled_len
#             this_chunk_size = min(prompt_remaining, CHUNK_SIZE)
            
#             # 使用实际的 chunk size 检查预算
#             if this_chunk_size > 0 and num_batched_tokens + this_chunk_size <= self.max_num_batched_tokens and self.block_manager.can_allocate(seq):
#                 # ------------------ 添加chunked prefill逻辑 ------------------
                
#                 num_seqs += 1
#                 self.block_manager.allocate(seq)
                
#                 seq.status = SequenceStatus.RUNNING
#                 self.waiting.popleft()
#                 self.running.append(seq)
#                 scheduled_seqs.append(seq)
#                 num_batched_tokens += this_chunk_size  
#                 has_prefill = True
                
#                 # num_batched_tokens += len(seq)  # 原始代码
#         # if scheduled_seqs:
#         #     # 如果有prefill，直接返回，后续的decode阶段留到下一次调用schedule时处理
#         #     return scheduled_seqs, True

#         # decode
#         while self.running and num_seqs < self.max_num_seqs:
#             seq = self.running.popleft()
#             while not self.block_manager.can_append(seq):
#                 # 如果添加不了新的block块了，启动抢占逻辑
#                 if self.running:
#                     # 如果有正在运行的序列，从running队列的右侧pop一个seq并抢占资源
#                     # 抢占逻辑就在下面的preempt函数里
#                     self.preempt(self.running.pop())
#                 else:
#                     # 如果没有正在运行的序列，直接将当前序列抢占
#                     self.preempt(seq)
#                     break
#             else:
#                 # 正常运行时
#                 # 检查这个 seq 是不是刚才已经作为 Prefill 块加进去了，如果是的话就放最后，继续下一个
#                 if seq in scheduled_seqs:
#                     # self.running.append(seq) # 已经在scheduler队列里了，最后会放进来
#                     continue
#                 # 检查是否是已经完成 Prefill 的 Decode 请求
#                 # remaining_prefill = len(seq) - seq.num_cached_tokens - seq.prefilled_len
#                 # 检查是否是已经完成 Prefill 的 Decode 请求（基于 prompt 长度判断）
#                 prompt_remaining = seq.num_prompt_tokens - seq.num_cached_tokens - seq.prefilled_len
#                 if prompt_remaining > 0:
#                     self.running.appendleft(seq) # 还没 Prefill 完的先不在这里处理
#                     continue

#                 if self.block_manager.can_append(seq) and num_batched_tokens + 1 <= self.max_num_batched_tokens:
#                     # self.block_manager.may_append(seq)
#                     # 只有在纯 decode 批次时才提前 append，混合批次不能提前 append
#                     if not has_prefill:
#                         self.block_manager.may_append(seq)
#                     num_batched_tokens += 1
#                     num_seqs += 1
#                     scheduled_seqs.append(seq)
#                 else:
#                     self.running.appendleft(seq) # 预算不够了，放回队头下次再说
#                     break

#                 # num_seqs += 1
#                 # self.block_manager.may_append(seq)
#                 # scheduled_seqs.append(seq)
#         assert scheduled_seqs
#         # running队列extendleft从队列左侧添加
#         # scheduled_seqs是按顺序添加的，例如[seq1,seq2,seq3]，running里还剩seq4,seq5没被选取到
#         # 如果直接添加，会变成[seq3,seq2,seq1]，但是调度的时候是从左边开始的，调度顺序就反了，
#         # 所以需要reversed(scheduled_seqs)，反转调度队列再添加到running队列的左边
#         self.running.extendleft(reversed(scheduled_seqs))
#         # self.running.extendleft(reversed([s for s in scheduled_seqs if s.status == SequenceStatus.RUNNING]))

#         # return scheduled_seqs, has_prefill
#         # 返回调度结果及元数据
#         # 计算 prefill 和 decode 的 token 数量
#         num_prefill_tokens = 0
#         num_decode_tokens = 0
#         for seq in scheduled_seqs:
#             prompt_remaining = seq.num_prompt_tokens - seq.num_cached_tokens - seq.prefilled_len
#             if prompt_remaining > 0:
#                 # Prefill 序列
#                 chunk_size = min(prompt_remaining, CHUNK_SIZE)
#                 num_prefill_tokens += chunk_size
#             else:
#                 # Decode 序列
#                 num_decode_tokens += 1
        
#         return scheduled_seqs, has_prefill, num_prefill_tokens, num_decode_tokens

#     def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill:bool=False) -> list[bool]:
#         # -------- 添加chunked prefill逻辑 ---------
#         '''
#         处理模型输出：
#         - 如果is_prefill为True，且序列长度超过1，则第一个是prefill的token_ids，后续是decode的token_ids
#         - 如果is_prefill为True且序列长度为1，则全部是prefill的token_ids
#         - 如果is_prefill为False，则全部是decode的token_ids
#         '''
#         # CHUNK_SIZE = 512
#         CHUNK_SIZE = self.chunk_size
#         if is_prefill:
#             if len(seqs) > 1:
#                 # 混合批次，第一个是prefill，后续是decode
#                 # prefill阶段不消耗token_ids
#                 prefill_seq = seqs[0]
#                 # remaining_len = len(prefill_seq) - prefill_seq.num_cached_tokens - prefill_seq.prefilled_len
#                 # prefill_chunk_size = min(remaining_len, CHUNK_SIZE)
#                 # prefill_seq.prefilled_len += prefill_chunk_size
#                 prompt_remaining = prefill_seq.num_prompt_tokens - prefill_seq.num_cached_tokens - prefill_seq.prefilled_len
#                 if prompt_remaining > 0:
#                     # 第一个序列是 prefill，更新 prefilled_len
#                     prefill_chunk_size = min(prompt_remaining, CHUNK_SIZE)
#                     prefill_seq.prefilled_len += prefill_chunk_size
#                     # DEBUG
#                     # print(f"[DEBUG postprocess] Mixed batch: {len(seqs)} seqs, {len(token_ids) if token_ids else 0} tokens")
#                     # print(f"  Prefill seq {prefill_seq.seq_id}: updated prefilled_len to {prefill_seq.prefilled_len}")
#                     # print(f"  Decode seqs: {[s.seq_id for s in seqs[1:]]}")
#                     # print(f"  Token IDs for decode: {token_ids}")


#                 # decode阶段，处理后续序列（从seqs[1:]开始）
#                 # for seq, token_id in zip(seqs[1:], token_ids):
#                 for i, (seq, token_id) in enumerate(zip(seqs[1:], token_ids)):
#                     # print(f"  Processing decode seq {seq.seq_id}: token_id={token_id}, current_completion={seq.num_completion_tokens}")
#                     # 混合批次中的 decode 序列需要先 may_append 再 append_token
#                     self.block_manager.may_append(seq)
#                     seq.append_token(token_id)
#                     # print(f"  After append: seq {seq.seq_id} completion_tokens={seq.num_completion_tokens}")
#                     if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
#                         seq.status = SequenceStatus.FINISHED
#                         self.block_manager.deallocate(seq)
#                         self.running.remove(seq)
#             else:
#                 # 纯prefill，只更新prefilled_len
#                 prefill_seq = seqs[0]
#                 # remaining_len = len(prefill_seq) - prefill_seq.num_cached_tokens - prefill_seq.prefilled_len
#                 prompt_remaining = prefill_seq.num_prompt_tokens - prefill_seq.num_cached_tokens - prefill_seq.prefilled_len
#                 prefill_chunk_size = min(prompt_remaining, CHUNK_SIZE)
#                 prefill_seq.prefilled_len += prefill_chunk_size
#         else:
#             # decode阶段，处理所有序列
#             for seq, token_id in zip(seqs, token_ids):
#                 seq.append_token(token_id)
#                 if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
#                     seq.status = SequenceStatus.FINISHED
#                     self.block_manager.deallocate(seq)
#                     self.running.remove(seq)
#         # -------- 添加chunked prefill逻辑 ---------

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

        # 1. 首先检查当前running队列中是否有prefill序列
        running_has_prefill = False
        # 记录running队列中prefill未完成的序列
        prefill_seq_found = None

        for seq in list(self.running):
            if num_seqs >= self.max_num_seqs:
                break
            prompt_tokens_left = seq.num_prompt_tokens - seq.prefilled_tokens - seq.num_cached_tokens
            if prompt_tokens_left <= 0:
                continue
            new_chunk_size = min(self.chunk_size, prompt_tokens_left)
            # 如果加入该序列会超出token数上限，或者无法分配足够的块，则停止调度
            if num_batched_tokens + new_chunk_size > self.max_num_batched_tokens:
                break
            num_seqs += 1
            num_batched_tokens += new_chunk_size
            scheduled_seqs.append(seq)
            running_has_prefill = True
            prefill_seq_found = seq
            # 每次只插入一个prefill序列的一个chunk
            break
        
        # 从 running 队列中临时移除已调度的 prefill 序列，避免在 decode 阶段重复处理
        if prefill_seq_found is not None:
            self.running.remove(prefill_seq_found)
        
        # 2. 如果running中没有需要prefill的队列，就进waiting取
        if not running_has_prefill and self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]

            prompt_tokens_left = seq.num_prompt_tokens - seq.prefilled_tokens - seq.num_cached_tokens
            new_chunk_size = min(self.chunk_size, prompt_tokens_left)
            # 如果加入该序列会超出token数上限，或者无法分配足够的块，则停止调度
            if new_chunk_size > 0 and num_batched_tokens + new_chunk_size <= self.max_num_batched_tokens and self.block_manager.can_allocate(seq):
                # 加入该序列
                num_seqs += 1
                self.block_manager.allocate(seq)
                
                seq.status = SequenceStatus.RUNNING
                # 调整当前序列到运行队列
                self.waiting.popleft()
                self.running.append(seq)
                # 预填充阶段加入调度列表
                scheduled_seqs.append(seq)
                num_batched_tokens += new_chunk_size
                running_has_prefill = True

        # # 循环处理调度等待队列中的序列
        # # 保证不超过单次批量推理的序列数和token数上限
        # while self.waiting and num_seqs < self.max_num_seqs:
        #     seq = self.waiting[0]
        #     # 如果加入该序列会超出token数上限，或者无法分配足够的块，则停止调度
        #     if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
        #         break
        #     num_seqs += 1
        #     self.block_manager.allocate(seq)
        #     # 这里天然是为chunked prefill设计的
        #     num_batched_tokens += len(seq) - seq.num_cached_tokens
        #     seq.status = SequenceStatus.RUNNING

        #     self.waiting.popleft()
        #     self.running.append(seq)

        #     scheduled_seqs.append(seq)
        
        # # 如果有预填充的序列则直接返回，说明还需要继续进行预填充
        # if scheduled_seqs:
        #     return scheduled_seqs, True

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

                    if self.block_manager.can_allocate(seq) and num_batched_tokens + 1 <= self.max_num_batched_tokens:
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
        处理模型输出：
        - 如果is_prefill为True，且序列长度超过1，则第一个是prefill的token_ids，后续是decode的token_ids
        - 如果is_prefill为True且序列长度为1，则全部是prefill的token_ids
        - 如果is_prefill为False，则全部是decode的token_ids
        '''
        # [ ] 根据is_prefill标志处理不同情况
        if is_prefill:
            # 混合prefill和decode批次
            # 因为有prefill序列，同时总序列数又不止1个，所以肯定有decode在
            if len(seqs) > 1:
                # 混合批次，第一个是prefill，后续是decode
                # 为了不浪费算力，我们把那个正在做 Prefill 的“大胖子”放在 seqs[0]，然后把所有正在排队等着 Decode 的“瘦子”们塞在后面 seqs[1:].
                # prefill阶段不消耗token_ids
                prompt_tokens_left = seqs[0].num_prompt_tokens - seqs[0].prefilled_tokens - seqs[0].num_cached_tokens
                if prompt_tokens_left > 0:
                    new_chunk_size = min(self.chunk_size, prompt_tokens_left)
                    seqs[0].prefilled_tokens += new_chunk_size
                    # decode部分
                    for i, (seq, token_id) in enumerate(zip(seqs[1:], token_ids)):
                        self.block_manager.may_append(seq)
                        seq.append_token(token_id)
                        # 检查是否结束，结束则释放块并从运行队列移除
                        # 注意这里的结束条件包括遇到eos token或者达到最大生成长度
                        if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                            seq.status = SequenceStatus.FINISHED
                            self.block_manager.deallocate(seq)
                            self.running.remove(seq)
            else:
                # 纯prefill，只更新prefilled_len
                prompt_tokens_left = seqs[0].num_prompt_tokens - seqs[0].prefilled_tokens - seqs[0].num_cached_tokens
                prefill_chunk_size = min(self.chunk_size, prompt_tokens_left)
                seqs[0].prefilled_tokens += prefill_chunk_size
        else:
            # 纯decode批次
            for seq, token_id in zip(seqs, token_ids):
                seq.append_token(token_id)
                # 检查是否结束，结束则释放块并从运行队列移除
                # 注意这里的结束条件包括遇到eos token或者达到最大生成长度
                if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    self.running.remove(seq)

