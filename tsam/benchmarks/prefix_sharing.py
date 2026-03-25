"""
Prefix sharing efficiency benchmark — Experiment 3.

Compares throughput of:
  (a) Vanilla vLLM with prefix caching
  (b) TSAM with SHARED prefix pages
  (c) Paper 5 orthogonal projection (per-tenant prefix recomputation)

Target: TSAM within 5% of vanilla prefix-caching throughput.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import numpy as np

from tsam.core.block_manager import TypedBlockManager
from tsam.core.types import PageType, TenantId, SHARED_TENANT_ID
from tsam.attention.paged_attention import tsam_paged_attention


@dataclass
class PrefixSharingResult:
    """Result from prefix sharing throughput test.

    Attributes:
        tokens_per_sec:          Throughput.
        num_tenants:             Number of tenants sharing the prefix.
        prefix_tokens:           Number of tokens in the shared prefix.
        total_tokens:            Total tokens processed.
        elapsed_sec:             Wall-clock time.
        prefix_cache_ratio:      Fraction of KV cache used by shared prefix.
        overhead_vs_vanilla_pct: Throughput overhead vs. vanilla prefix caching.
        defense:                 Defense condition.
    """

    tokens_per_sec: float
    num_tenants: int
    prefix_tokens: int
    total_tokens: int
    elapsed_sec: float
    prefix_cache_ratio: float
    overhead_vs_vanilla_pct: float
    defense: str


class PrefixSharingBenchmark:
    """Benchmarks prefix sharing efficiency under TSAM vs. vanilla vs. Paper 5."""

    def __init__(
        self,
        page_size: int = 16,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        device: str = "cpu",
    ):
        self.page_size = page_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device

    def run_tsam_prefix_sharing(
        self,
        num_tenants: int = 100,
        prefix_tokens: int = 1024,
        private_tokens_per_tenant: int = 256,
        queries_per_tenant: int = 10,
    ) -> PrefixSharingResult:
        """TSAM with SHARED prefix pages — prefix allocated once, shared by all."""
        prefix_blocks = (prefix_tokens + self.page_size - 1) // self.page_size
        private_blocks = (private_tokens_per_tenant + self.page_size - 1) // self.page_size
        total_blocks = prefix_blocks + num_tenants * private_blocks + 1

        bm = TypedBlockManager(
            num_blocks=total_blocks,
            page_size=self.page_size,
            device=self.device,
        )

        # SHARED prefix: allocated once.
        shared_pages = bm.allocate_shared(count=prefix_blocks)
        shared_ids = [p.block_id for p in shared_pages]

        # Private blocks per tenant.
        tenant_blocks: Dict[int, List[int]] = {}
        for t in range(num_tenants):
            pages = bm.allocate_private(tenant=t, count=private_blocks)
            tenant_blocks[t] = [p.block_id for p in pages]

        # KV cache.
        key_cache = torch.randn(
            total_blocks, self.page_size, self.num_kv_heads, self.head_dim,
            device=self.device, dtype=torch.float32,
        )
        value_cache = torch.randn(
            total_blocks, self.page_size, self.num_kv_heads, self.head_dim,
            device=self.device, dtype=torch.float32,
        )

        seq_len_kv = prefix_tokens + private_tokens_per_tenant
        total_tokens = 0
        start_time = time.perf_counter()

        for t in range(num_tenants):
            bt_ids = shared_ids + tenant_blocks[t]
            block_table = torch.tensor(bt_ids, dtype=torch.int32, device=self.device)
            for _ in range(queries_per_tenant):
                query = torch.randn(
                    self.num_heads, self.head_dim,
                    device=self.device, dtype=torch.float32,
                )
                output = tsam_paged_attention(
                    query=query,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    block_table=block_table,
                    page_type=bm.page_type,
                    tenant_id_meta=bm.tenant_id,
                    query_tenant=t,
                    page_size=self.page_size,
                    seq_len_kv=seq_len_kv,
                )
                total_tokens += seq_len_kv

        elapsed = time.perf_counter() - start_time
        tokens_per_sec = total_tokens / elapsed
        prefix_ratio = prefix_blocks / (prefix_blocks + num_tenants * private_blocks)

        return PrefixSharingResult(
            tokens_per_sec=tokens_per_sec,
            num_tenants=num_tenants,
            prefix_tokens=prefix_tokens,
            total_tokens=total_tokens,
            elapsed_sec=elapsed,
            prefix_cache_ratio=prefix_ratio,
            overhead_vs_vanilla_pct=0.0,  # Set by caller.
            defense="TSAM",
        )

    def run_vanilla_prefix_caching(
        self,
        num_tenants: int = 100,
        prefix_tokens: int = 1024,
        private_tokens_per_tenant: int = 256,
        queries_per_tenant: int = 10,
    ) -> PrefixSharingResult:
        """Vanilla prefix caching — shared prefix, no isolation overhead."""
        prefix_blocks = (prefix_tokens + self.page_size - 1) // self.page_size
        private_blocks = (private_tokens_per_tenant + self.page_size - 1) // self.page_size
        total_blocks = prefix_blocks + num_tenants * private_blocks + 1

        key_cache = torch.randn(
            total_blocks, self.page_size, self.num_kv_heads, self.head_dim,
            device=self.device, dtype=torch.float32,
        )
        value_cache = torch.randn(
            total_blocks, self.page_size, self.num_kv_heads, self.head_dim,
            device=self.device, dtype=torch.float32,
        )

        seq_len_kv = prefix_tokens + private_tokens_per_tenant
        scale = 1.0 / math.sqrt(self.head_dim)
        total_tokens = 0
        start_time = time.perf_counter()

        for t in range(num_tenants):
            # Build contiguous KV from shared prefix + private.
            prefix_end = prefix_blocks
            private_start = prefix_blocks + t * private_blocks
            private_end = private_start + private_blocks

            bt_ids = list(range(prefix_end)) + list(range(private_start, private_end))
            keys = key_cache[bt_ids].reshape(-1, self.num_kv_heads, self.head_dim)[:seq_len_kv]
            values = value_cache[bt_ids].reshape(-1, self.num_kv_heads, self.head_dim)[:seq_len_kv]

            num_groups = self.num_heads // self.num_kv_heads
            if num_groups > 1:
                keys = keys.unsqueeze(2).expand(-1, -1, num_groups, -1).reshape(
                    seq_len_kv, self.num_heads, self.head_dim
                )
                values = values.unsqueeze(2).expand(-1, -1, num_groups, -1).reshape(
                    seq_len_kv, self.num_heads, self.head_dim
                )

            for _ in range(queries_per_tenant):
                query = torch.randn(
                    self.num_heads, 1, self.head_dim,
                    device=self.device, dtype=torch.float32,
                )
                k = keys.permute(1, 2, 0)
                scores = torch.bmm(query, k) * scale
                weights = torch.nn.functional.softmax(scores, dim=-1)
                v = values.permute(1, 0, 2)
                output = torch.bmm(weights, v)
                total_tokens += seq_len_kv

        elapsed = time.perf_counter() - start_time
        tokens_per_sec = total_tokens / elapsed
        prefix_ratio = prefix_blocks / (prefix_blocks + num_tenants * private_blocks)

        return PrefixSharingResult(
            tokens_per_sec=tokens_per_sec,
            num_tenants=num_tenants,
            prefix_tokens=prefix_tokens,
            total_tokens=total_tokens,
            elapsed_sec=elapsed,
            prefix_cache_ratio=prefix_ratio,
            overhead_vs_vanilla_pct=0.0,
            defense="vanilla",
        )

    def run_paper5_no_prefix_sharing(
        self,
        num_tenants: int = 100,
        prefix_tokens: int = 1024,
        private_tokens_per_tenant: int = 256,
        queries_per_tenant: int = 10,
    ) -> PrefixSharingResult:
        """Paper 5 orthogonal projection — per-tenant prefix recomputation.

        Each tenant must have its own copy of the prefix KV because the projection
        destroys shareability. This models the throughput collapse.
        """
        prefix_blocks = (prefix_tokens + self.page_size - 1) // self.page_size
        private_blocks = (private_tokens_per_tenant + self.page_size - 1) // self.page_size
        # Each tenant needs its own prefix copy.
        blocks_per_tenant = prefix_blocks + private_blocks
        total_blocks = num_tenants * blocks_per_tenant + 1

        key_cache = torch.randn(
            total_blocks, self.page_size, self.num_kv_heads, self.head_dim,
            device=self.device, dtype=torch.float32,
        )
        value_cache = torch.randn(
            total_blocks, self.page_size, self.num_kv_heads, self.head_dim,
            device=self.device, dtype=torch.float32,
        )

        seq_len_kv = prefix_tokens + private_tokens_per_tenant
        scale = 1.0 / math.sqrt(self.head_dim)
        total_tokens = 0
        start_time = time.perf_counter()

        for t in range(num_tenants):
            start_block = t * blocks_per_tenant
            end_block = start_block + blocks_per_tenant
            keys = key_cache[start_block:end_block].reshape(
                -1, self.num_kv_heads, self.head_dim
            )[:seq_len_kv]
            values = value_cache[start_block:end_block].reshape(
                -1, self.num_kv_heads, self.head_dim
            )[:seq_len_kv]

            num_groups = self.num_heads // self.num_kv_heads
            if num_groups > 1:
                keys = keys.unsqueeze(2).expand(-1, -1, num_groups, -1).reshape(
                    seq_len_kv, self.num_heads, self.head_dim
                )
                values = values.unsqueeze(2).expand(-1, -1, num_groups, -1).reshape(
                    seq_len_kv, self.num_heads, self.head_dim
                )

            for _ in range(queries_per_tenant):
                query = torch.randn(
                    self.num_heads, 1, self.head_dim,
                    device=self.device, dtype=torch.float32,
                )
                k = keys.permute(1, 2, 0)
                scores = torch.bmm(query, k) * scale
                weights = torch.nn.functional.softmax(scores, dim=-1)
                v = values.permute(1, 0, 2)
                output = torch.bmm(weights, v)
                total_tokens += seq_len_kv

        elapsed = time.perf_counter() - start_time
        tokens_per_sec = total_tokens / elapsed

        return PrefixSharingResult(
            tokens_per_sec=tokens_per_sec,
            num_tenants=num_tenants,
            prefix_tokens=prefix_tokens,
            total_tokens=total_tokens,
            elapsed_sec=elapsed,
            prefix_cache_ratio=0.0,  # No sharing.
            overhead_vs_vanilla_pct=0.0,  # Set by caller.
            defense="Paper5_orthogonal",
        )
