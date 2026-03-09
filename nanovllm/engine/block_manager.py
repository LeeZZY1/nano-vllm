from collections import deque

import numpy as np
import xxhash

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        assert num_blocks > 0
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = (
                self.compute_hash(token_ids, h)
                if len(token_ids) == self.block_size
                else -1
            )
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def can_allocate_incremental(self, seq: Sequence, num_new_tokens: int) -> bool:
        """检查是否有足够的空闲 block 来增量分配 num_new_tokens 个新 token。
        
        保守估计（假设无 prefix cache 命中），实际消耗的空闲 block 数
        与此估计值相同（每个 cache 命中增加 target 但不消耗空闲 block）。
        """
        current_blocks = len(seq.block_table)
        target_tokens = seq.num_cached_tokens + seq.prefilled_tokens + num_new_tokens
        target_blocks = (target_tokens + self.block_size - 1) // self.block_size
        new_blocks_needed = max(0, target_blocks - current_blocks)
        return len(self.free_block_ids) >= new_blocks_needed

    def allocate_incremental(self, seq: Sequence, num_new_tokens: int):
        """增量分配 block：只分配覆盖 num_new_tokens 所需的新 block。
        
        支持首次分配（block_table 为空）和扩展分配（block_table 已有条目）。
        分配过程中进行 prefix cache 匹配；cache 命中会推高 num_cached_tokens
        从而动态增加 target_blocks，但不消耗空闲 block。
        """
        current_blocks = len(seq.block_table)

        # 确定 prefix cache 哈希链的起始状态
        if current_blocks == 0:
            h = -1
            cache_miss = False
        else:
            last_block = self.blocks[seq.block_table[current_blocks - 1]]
            h = last_block.hash
            # 上一个 block 未完整填充（hash=-1），哈希链断裂
            cache_miss = (h == -1)

        i = current_blocks
        while True:
            # 每次迭代重新计算目标（cache 命中会增加 num_cached_tokens）
            target_tokens = seq.num_cached_tokens + seq.prefilled_tokens + num_new_tokens
            target_blocks = (target_tokens + self.block_size - 1) // self.block_size
            if i >= target_blocks:
                break

            token_ids = seq.block(i)
            h = (
                self.compute_hash(token_ids, h)
                if len(token_ids) == self.block_size
                else -1
            )
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)
            i += 1

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.prefilled_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks - 1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1
