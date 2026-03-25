"""
Throughput benchmarking for TSAM — Experiments 2 and 3.

Measures tokens/sec throughput at varying tenant counts (10, 100, 1000, 10000)
and compares TSAM overhead against vanilla (no defense) and Paper 5 orthogonal
projection.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

from tsam.core.block_manager import TypedBlockManager
from tsam.core.types import PageType, TenantId
from tsam.attention.masking import generate_tsam_mask_batched
from tsam.attention.paged_attention import tsam_paged_attention


@dataclass
class ThroughputResult:
    """Result from a throughput benchmark.

    Attributes:
        tokens_per_sec:    Achieved throughput.
        num_tenants:       Number of concurrent tenants.
        total_tokens:      Total tokens processed.
        elapsed_sec:       Wall-clock time.
        p50_latency_ms:    50th percentile per-query latency.
        p99_latency_ms:    99th percentile per-query latency.
        overhead_pct:      Overhead vs. baseline (0 = no overhead).
        defense:           Defense name.
    """

    tokens_per_sec: float
    num_tenants: int
    total_tokens: int
    elapsed_sec: float
    p50_latency_ms: float
    p99_latency_ms: float
    overhead_pct: float
    defense: str


class ThroughputBenchmark:
    """Benchmarks TSAM attention throughput at varying tenant counts.

    This benchmark:
    1. Allocates a KV-cache pool with typed blocks for N tenants.
    2. Runs batched attention with TSAM masking.
    3. Measures tokens/sec and per-query latency.
    """

    def __init__(
        self,
        num_blocks: int = 8192,
        page_size: int = 16,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        device: str = "cpu",
    ):
        self.num_blocks = num_blocks
        self.page_size = page_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device

    def _setup_kv_cache(
        self,
        num_tenants: int,
        blocks_per_tenant: int,
        shared_prefix_blocks: int,
    ) -> Tuple[TypedBlockManager, torch.Tensor, torch.Tensor]:
        """Set up typed KV-cache blocks for benchmarking."""
        total_needed = shared_prefix_blocks + num_tenants * blocks_per_tenant
        actual_num_blocks = max(self.num_blocks, total_needed + 1)

        bm = TypedBlockManager(
            num_blocks=actual_num_blocks,
            page_size=self.page_size,
            device=self.device,
        )

        # Allocate shared prefix blocks.
        shared_pages = bm.allocate_shared(count=shared_prefix_blocks)

        # Allocate private blocks per tenant.
        for t in range(num_tenants):
            bm.allocate_private(tenant=t, count=blocks_per_tenant)

        # Create key and value cache tensors.
        key_cache = torch.randn(
            actual_num_blocks,
            self.page_size,
            self.num_kv_heads,
            self.head_dim,
            device=self.device,
            dtype=torch.float32,
        )
        value_cache = torch.randn(
            actual_num_blocks,
            self.page_size,
            self.num_kv_heads,
            self.head_dim,
            device=self.device,
            dtype=torch.float32,
        )

        return bm, key_cache, value_cache

    def run_tsam(
        self,
        num_tenants: int,
        queries_per_tenant: int = 100,
        seq_len_kv: int = 512,
        shared_prefix_blocks: int = 4,
    ) -> ThroughputResult:
        """Run TSAM throughput benchmark.

        Args:
            num_tenants:          Number of concurrent tenants.
            queries_per_tenant:   Queries per tenant.
            seq_len_kv:           KV sequence length per query.
            shared_prefix_blocks: Number of shared prefix blocks.
        """
        blocks_per_tenant = (seq_len_kv + self.page_size - 1) // self.page_size
        bm, key_cache, value_cache = self._setup_kv_cache(
            num_tenants, blocks_per_tenant, shared_prefix_blocks
        )

        total_queries = num_tenants * queries_per_tenant
        latencies: List[float] = []
        total_tokens = 0

        # Build block tables for each tenant.
        block_tables: Dict[int, torch.Tensor] = {}
        shared_block_ids = list(range(shared_prefix_blocks))
        for t in range(num_tenants):
            tenant_blocks = sorted(bm.get_tenant_blocks(t))
            all_blocks = shared_block_ids + tenant_blocks
            # Pad/truncate to match required blocks.
            needed = blocks_per_tenant + shared_prefix_blocks
            while len(all_blocks) < needed:
                all_blocks.append(all_blocks[-1] if all_blocks else 0)
            block_tables[t] = torch.tensor(
                all_blocks[:needed], dtype=torch.int32, device=self.device
            )

        start_time = time.perf_counter()

        for t in range(num_tenants):
            block_table = block_tables[t]
            for q in range(queries_per_tenant):
                query = torch.randn(
                    1, self.num_heads, self.head_dim,
                    device=self.device, dtype=torch.float32,
                )
                q_start = time.perf_counter()

                output = tsam_paged_attention(
                    query=query.squeeze(0),
                    key_cache=key_cache,
                    value_cache=value_cache,
                    block_table=block_table,
                    page_type=bm.page_type,
                    tenant_id_meta=bm.tenant_id,
                    query_tenant=t,
                    page_size=self.page_size,
                    seq_len_kv=seq_len_kv,
                )

                q_end = time.perf_counter()
                latencies.append((q_end - q_start) * 1000)  # ms
                total_tokens += seq_len_kv

        elapsed = time.perf_counter() - start_time
        tokens_per_sec = total_tokens / elapsed

        latencies_arr = np.array(latencies)
        p50 = float(np.percentile(latencies_arr, 50))
        p99 = float(np.percentile(latencies_arr, 99))

        return ThroughputResult(
            tokens_per_sec=tokens_per_sec,
            num_tenants=num_tenants,
            total_tokens=total_tokens,
            elapsed_sec=elapsed,
            p50_latency_ms=p50,
            p99_latency_ms=p99,
            overhead_pct=0.0,  # Set by caller after comparison.
            defense="TSAM",
        )

    def run_vanilla(
        self,
        num_tenants: int,
        queries_per_tenant: int = 100,
        seq_len_kv: int = 512,
        shared_prefix_blocks: int = 4,
    ) -> ThroughputResult:
        """Run vanilla (no defense) throughput benchmark for comparison."""
        blocks_per_tenant = (seq_len_kv + self.page_size - 1) // self.page_size
        total_blocks = shared_prefix_blocks + num_tenants * blocks_per_tenant + 1
        actual_num_blocks = max(self.num_blocks, total_blocks)

        key_cache = torch.randn(
            actual_num_blocks,
            self.page_size,
            self.num_kv_heads,
            self.head_dim,
            device=self.device,
            dtype=torch.float32,
        )
        value_cache = torch.randn(
            actual_num_blocks,
            self.page_size,
            self.num_kv_heads,
            self.head_dim,
            device=self.device,
            dtype=torch.float32,
        )

        total_queries = num_tenants * queries_per_tenant
        latencies: List[float] = []
        total_tokens = 0

        import math

        start_time = time.perf_counter()

        for t in range(num_tenants):
            for q in range(queries_per_tenant):
                query = torch.randn(
                    self.num_heads, 1, self.head_dim,
                    device=self.device, dtype=torch.float32,
                )
                # Vanilla attention: Q @ K^T, softmax, @ V (no masking).
                # Use first blocks_per_tenant blocks as KV.
                start_block = shared_prefix_blocks + t * blocks_per_tenant
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

                q_start = time.perf_counter()

                scale = 1.0 / math.sqrt(self.head_dim)
                k = keys.permute(1, 2, 0)
                scores = torch.bmm(query, k) * scale
                weights = torch.nn.functional.softmax(scores, dim=-1)
                v = values.permute(1, 0, 2)
                output = torch.bmm(weights, v)

                q_end = time.perf_counter()
                latencies.append((q_end - q_start) * 1000)
                total_tokens += seq_len_kv

        elapsed = time.perf_counter() - start_time
        tokens_per_sec = total_tokens / elapsed

        latencies_arr = np.array(latencies)
        p50 = float(np.percentile(latencies_arr, 50))
        p99 = float(np.percentile(latencies_arr, 99))

        return ThroughputResult(
            tokens_per_sec=tokens_per_sec,
            num_tenants=num_tenants,
            total_tokens=total_tokens,
            elapsed_sec=elapsed,
            p50_latency_ms=p50,
            p99_latency_ms=p99,
            overhead_pct=0.0,
            defense="vanilla",
        )

    @staticmethod
    def compute_overhead(
        vanilla: ThroughputResult, tsam: ThroughputResult
    ) -> float:
        """Compute TSAM overhead percentage vs. vanilla.

        Returns:
            Overhead percentage. Target: < 1%.
        """
        if vanilla.tokens_per_sec <= 0:
            return 0.0
        overhead = (
            1.0 - tsam.tokens_per_sec / vanilla.tokens_per_sec
        ) * 100.0
        return overhead
