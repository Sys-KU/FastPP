from __future__ import annotations

"""Cache for chunked prefill, used when RadixCache is disabled."""

from typing import TYPE_CHECKING, Callable, List, Optional, Tuple
import torch

from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.memory_pool import BaseTokenToKVPool, ReqToTokenPool

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class ChunkCacheEntry:
    def __init__(self, rid, value):
        self.rid = rid
        self.value = value


class ChunkCache(BasePrefixCache):
    def __init__(
        self, req_to_token_pool: ReqToTokenPool, token_to_kv_pool: BaseTokenToKVPool
    ):
        self.disable = True
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool = token_to_kv_pool

        self.reset()

    def reset(self):
        self.entries = {}

    def match_prefix(self, rid: int, key: List[int]) -> Tuple[List[int], int]:
        if rid not in self.entries:
            return [], None

        entry = self.entries[rid]
        max_prefix_len = len(key)
        return entry.value[:max_prefix_len], entry

    def cache_finished_req(self, req: Req, token_ids: Optional[List[int]] = None):
        if token_ids is None:
            token_id_len = len(req.origin_input_ids) + len(req.output_ids) - 1
        else:
            token_id_len = len(token_ids)

        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :token_id_len
        ]
        self.req_to_token_pool.free(req.req_pool_idx)
        self.token_to_kv_pool.free(kv_indices)

        if req.rid in self.entries:
            del self.entries[req.rid]

    def cache_unfinished_req(self, req: Req, token_ids: Optional[List[int]] = None):
        if token_ids is None:
            token_id_len = len(req.fill_ids)
        else:
            token_id_len = len(token_ids)

        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :token_id_len
        ]

        if req.rid not in self.entries:
            self.entries[req.rid] = ChunkCacheEntry(req.rid, kv_indices)

        entry = self.entries[req.rid]
        entry.value = kv_indices
        req.prefix_indices = kv_indices
        req.last_node = entry

    # def move_rebalancing_req(self, req: Req, token_id_len, tree_cache_prev):
    #     # current memory pool
    #     token_to_kv_pool = self.token_to_kv_pool
    #     req_to_token_pool = self.req_to_token_pool
    #
    #     # prev memory pool
    #     token_to_kv_pool_prev = tree_cache_prev.token_to_kv_pool
    #     req_to_token_pool_prev = tree_cache_prev.req_to_token_pool
    #
    #     # token_id_len = len(req.origin_input_ids) + len(req.output_ids) - 1
    #
    #     # Allocate memory
    #     req.req_pool_idx = req_to_token_pool.alloc(1)[0]
    #     # kv_indices = token_to_kv_pool.alloc(token_id_len)
    #     # assert kv_indices is not None and req.req_pool_idx is not None
    #
    #     # Free prev memory pool
    #     # kv_indices_prev = req_to_token_pool_prev.req_to_token[
    #     #     req.req_pool_idx, : token_id_len
    #     # ]
    #     # kv_indices_prev, kv_indices = kv_indices, kv_indices_prev
    #     req_to_token_pool_prev.free(req.req_pool_idx)
    #     # token_to_kv_pool_prev.free(kv_indices_prev)
    #
    #     req.rebalance_cache_loc = kv_indices

    def move_rebalancing_req(self, req: Req, token_id_len, tree_cache_prev):
        # current memory pool
        req_to_token_pool = self.req_to_token_pool
        # prev memory pool
        req_to_token_pool_prev = tree_cache_prev.req_to_token_pool

        kv_indices = req_to_token_pool_prev.req_to_token[
            req.req_pool_idx, : token_id_len
        ]
        req.rebalance_cache_loc = kv_indices
        req_to_token_pool_prev.free(req.req_pool_idx)

        # Allocate
        req.req_pool_idx = req_to_token_pool.alloc(1)[0]

    def move_chunked_req(self, req: Req, tree_cache_prev):
        entry = tree_cache_prev.entries[req.rid]
        assert req.rid not in self.entries
        self.entries[req.rid] = entry

        del tree_cache_prev.entries[req.rid]

    def insert(self):
        raise NotImplementedError()

    def evict(self, num_tokens: int, evict_callback: Callable):
        pass

    def inc_lock_ref(self, node):
        return 0

    def dec_lock_ref(self, node):
        return 0

    def evictable_size(self):
        return 0
