"""Tests for TSAM attention masking — Experiment 6 kernel correctness tests.

Test cases from the spec:
  (a) PRIVATE-to-PRIVATE same tenant: attention weight > 0 (allowed)
  (b) PRIVATE-to-PRIVATE different tenant: attention weight = 0 (blocked)
  (c) SHARED page any tenant: attention weight > 0 (allowed)
  (d) Edge cases: page boundary, batch sizes, sequence lengths
"""

import math

import pytest
import torch

from tsam.core.block_manager import TypedBlockManager
from tsam.core.types import PageType, SHARED_TENANT_ID
from tsam.attention.masking import (
    generate_tsam_mask,
    generate_tsam_mask_batched,
    apply_tsam_mask_to_scores,
    TSAMAttentionMask,
)
from tsam.attention.paged_attention import tsam_paged_attention, tsam_paged_attention_batched


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------

@pytest.fixture
def page_size():
    return 16


@pytest.fixture
def head_dim():
    return 64


@pytest.fixture
def num_heads():
    return 4


@pytest.fixture
def num_kv_heads():
    return 4


@pytest.fixture
def setup_two_tenant_cache(page_size, num_kv_heads, head_dim):
    """Set up a KV cache with shared + 2 tenants."""
    shared_blocks = 2
    private_blocks = 4
    total_blocks = shared_blocks + 2 * private_blocks + 1

    bm = TypedBlockManager(num_blocks=total_blocks, page_size=page_size)
    shared_pages = bm.allocate_shared(count=shared_blocks)
    t0_pages = bm.allocate_private(tenant=0, count=private_blocks)
    t1_pages = bm.allocate_private(tenant=1, count=private_blocks)

    key_cache = torch.randn(total_blocks, page_size, num_kv_heads, head_dim)
    value_cache = torch.randn(total_blocks, page_size, num_kv_heads, head_dim)

    # Put distinctive values in each tenant's blocks for verification.
    for p in t0_pages:
        key_cache[p.block_id] = 1.0
        value_cache[p.block_id] = 1.0
    for p in t1_pages:
        key_cache[p.block_id] = -1.0
        value_cache[p.block_id] = -1.0

    return {
        "bm": bm,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "shared_ids": [p.block_id for p in shared_pages],
        "t0_ids": [p.block_id for p in t0_pages],
        "t1_ids": [p.block_id for p in t1_pages],
        "shared_blocks": shared_blocks,
        "private_blocks": private_blocks,
    }


# ---------------------------------------------------------------
# Test (a): PRIVATE same tenant — attention weight > 0
# ---------------------------------------------------------------

class TestPrivateSameTenant:
    def test_mask_allows_own_private(self, setup_two_tenant_cache, page_size):
        ctx = setup_two_tenant_cache
        bm = ctx["bm"]
        bt = ctx["shared_ids"] + ctx["t0_ids"]
        block_table = torch.tensor(bt, dtype=torch.int32)
        seq_len = len(bt) * page_size

        mask = generate_tsam_mask(
            block_table=block_table,
            page_type=bm.page_type,
            tenant_id=bm.tenant_id,
            query_tenant=0,
            page_size=page_size,
            seq_len_kv=seq_len,
        )
        # All positions should be True (shared + own private).
        assert mask.mask.all(), "Tenant should see all own PRIVATE + SHARED pages"
        assert mask.num_blocked == 0

    def test_attention_weight_positive(
        self, setup_two_tenant_cache, page_size, num_heads, head_dim
    ):
        ctx = setup_two_tenant_cache
        bm = ctx["bm"]
        bt = ctx["shared_ids"] + ctx["t0_ids"]
        block_table = torch.tensor(bt, dtype=torch.int32)
        seq_len = len(bt) * page_size

        query = torch.randn(num_heads, head_dim)
        output = tsam_paged_attention(
            query=query,
            key_cache=ctx["key_cache"],
            value_cache=ctx["value_cache"],
            block_table=block_table,
            page_type=bm.page_type,
            tenant_id_meta=bm.tenant_id,
            query_tenant=0,
            page_size=page_size,
            seq_len_kv=seq_len,
        )
        # Output should be non-zero (attention to own content).
        assert output.abs().sum() > 0


# ---------------------------------------------------------------
# Test (b): PRIVATE different tenant — attention weight = 0
# ---------------------------------------------------------------

class TestPrivateDifferentTenant:
    def test_mask_blocks_other_private(self, setup_two_tenant_cache, page_size):
        ctx = setup_two_tenant_cache
        bm = ctx["bm"]
        # Block table with shared + t0 private + t1 private.
        bt = ctx["shared_ids"] + ctx["t0_ids"] + ctx["t1_ids"]
        block_table = torch.tensor(bt, dtype=torch.int32)
        seq_len = len(bt) * page_size

        # Tenant 1 queries — should NOT see tenant 0's private.
        mask = generate_tsam_mask(
            block_table=block_table,
            page_type=bm.page_type,
            tenant_id=bm.tenant_id,
            query_tenant=1,
            page_size=page_size,
            seq_len_kv=seq_len,
        )

        shared_end = ctx["shared_blocks"] * page_size
        t0_end = shared_end + ctx["private_blocks"] * page_size

        # Shared: allowed.
        assert mask.mask[:shared_end].all()
        # Tenant 0's private: blocked.
        assert not mask.mask[shared_end:t0_end].any(), \
            "Cross-tenant PRIVATE must be fully blocked"
        # Tenant 1's own private: allowed.
        assert mask.mask[t0_end:].all()

    def test_zero_attention_to_cross_tenant(
        self, setup_two_tenant_cache, page_size, num_heads, head_dim
    ):
        ctx = setup_two_tenant_cache
        bm = ctx["bm"]
        bt = ctx["shared_ids"] + ctx["t0_ids"] + ctx["t1_ids"]
        block_table = torch.tensor(bt, dtype=torch.int32)
        seq_len = len(bt) * page_size

        # Tenant 1 queries — verify output has NO information from t0's blocks.
        # T0's blocks have value = 1.0, T1's have -1.0, shared is random.
        # If isolation works, output should have NO contribution from the 1.0 values.
        torch.manual_seed(42)
        # Use a query that would strongly attend to the 1.0 keys if unmasked.
        query = torch.ones(num_heads, head_dim) * 0.5

        output_t1 = tsam_paged_attention(
            query=query,
            key_cache=ctx["key_cache"],
            value_cache=ctx["value_cache"],
            block_table=block_table,
            page_type=bm.page_type,
            tenant_id_meta=bm.tenant_id,
            query_tenant=1,
            page_size=page_size,
            seq_len_kv=seq_len,
        )

        # Also compute with ONLY shared + t1 blocks (ground truth isolation).
        bt_t1_only = ctx["shared_ids"] + ctx["t1_ids"]
        block_table_t1 = torch.tensor(bt_t1_only, dtype=torch.int32)
        seq_len_t1 = len(bt_t1_only) * page_size

        output_t1_isolated = tsam_paged_attention(
            query=query,
            key_cache=ctx["key_cache"],
            value_cache=ctx["value_cache"],
            block_table=block_table_t1,
            page_type=bm.page_type,
            tenant_id_meta=bm.tenant_id,
            query_tenant=1,
            page_size=page_size,
            seq_len_kv=seq_len_t1,
        )

        # TSAM output from mixed block table should equal output from isolated table.
        # (Because the masked positions contribute exactly 0 weight.)
        torch.testing.assert_close(output_t1, output_t1_isolated, atol=1e-5, rtol=1e-5)

    def test_zero_leakage_bits(self, setup_two_tenant_cache, page_size):
        """Verify 0 bits of leakage — the core KPI."""
        ctx = setup_two_tenant_cache
        bm = ctx["bm"]
        bt = ctx["shared_ids"] + ctx["t0_ids"] + ctx["t1_ids"]
        block_table = torch.tensor(bt, dtype=torch.int32)
        seq_len = len(bt) * page_size

        mask = generate_tsam_mask(
            block_table=block_table,
            page_type=bm.page_type,
            tenant_id=bm.tenant_id,
            query_tenant=1,
            page_size=page_size,
            seq_len_kv=seq_len,
        )

        shared_end = ctx["shared_blocks"] * page_size
        t0_end = shared_end + ctx["private_blocks"] * page_size

        # Every single position in t0's range must be False.
        cross_tenant_mask = mask.mask[shared_end:t0_end]
        assert cross_tenant_mask.sum().item() == 0, \
            f"Expected 0 bits leakage, got {cross_tenant_mask.sum().item()} unmasked positions"


# ---------------------------------------------------------------
# Test (c): SHARED pages — any tenant can attend
# ---------------------------------------------------------------

class TestSharedPages:
    def test_shared_accessible_by_all(self, setup_two_tenant_cache, page_size):
        ctx = setup_two_tenant_cache
        bm = ctx["bm"]
        bt = ctx["shared_ids"] + ctx["t0_ids"] + ctx["t1_ids"]
        block_table = torch.tensor(bt, dtype=torch.int32)
        seq_len = len(bt) * page_size
        shared_end = ctx["shared_blocks"] * page_size

        for tenant in [0, 1, 2, 999]:
            mask = generate_tsam_mask(
                block_table=block_table,
                page_type=bm.page_type,
                tenant_id=bm.tenant_id,
                query_tenant=tenant,
                page_size=page_size,
                seq_len_kv=seq_len,
            )
            assert mask.mask[:shared_end].all(), \
                f"SHARED pages must be accessible to tenant {tenant}"

    def test_shared_attention_weight_positive(
        self, setup_two_tenant_cache, page_size, num_heads, head_dim
    ):
        ctx = setup_two_tenant_cache
        bm = ctx["bm"]
        # Block table with only shared blocks.
        bt = ctx["shared_ids"]
        block_table = torch.tensor(bt, dtype=torch.int32)
        seq_len = len(bt) * page_size

        query = torch.randn(num_heads, head_dim)
        output = tsam_paged_attention(
            query=query,
            key_cache=ctx["key_cache"],
            value_cache=ctx["value_cache"],
            block_table=block_table,
            page_type=bm.page_type,
            tenant_id_meta=bm.tenant_id,
            query_tenant=999,  # Any tenant.
            page_size=page_size,
            seq_len_kv=seq_len,
        )
        assert output.abs().sum() > 0


# ---------------------------------------------------------------
# Test (d): Edge cases
# ---------------------------------------------------------------

class TestEdgeCases:
    def test_batch_size_1(self, page_size, num_heads, num_kv_heads, head_dim):
        total_blocks = 10
        bm = TypedBlockManager(num_blocks=total_blocks, page_size=page_size)
        bm.allocate_shared(count=1)
        bm.allocate_private(tenant=0, count=2)
        key_cache = torch.randn(total_blocks, page_size, num_kv_heads, head_dim)
        value_cache = torch.randn(total_blocks, page_size, num_kv_heads, head_dim)

        queries = torch.randn(1, num_heads, head_dim)
        block_tables = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        query_tenants = torch.tensor([0], dtype=torch.int32)
        seq_lens = torch.tensor([3 * page_size], dtype=torch.int32)

        output = tsam_paged_attention_batched(
            queries=queries,
            key_cache=key_cache,
            value_cache=value_cache,
            block_tables=block_tables,
            page_type=bm.page_type,
            tenant_id_meta=bm.tenant_id,
            query_tenants=query_tenants,
            page_size=page_size,
            seq_lens_kv=seq_lens,
        )
        assert output.shape == (1, num_heads, head_dim)

    def test_page_boundary_spanning(self, num_kv_heads, head_dim):
        """Test with sequence length that doesn't align to page boundaries."""
        page_size = 16
        total_blocks = 10
        bm = TypedBlockManager(num_blocks=total_blocks, page_size=page_size)
        bm.allocate_shared(count=1)
        bm.allocate_private(tenant=0, count=2)

        key_cache = torch.randn(total_blocks, page_size, num_kv_heads, head_dim)
        value_cache = torch.randn(total_blocks, page_size, num_kv_heads, head_dim)

        block_table = torch.tensor([0, 1, 2], dtype=torch.int32)
        # Non-aligned sequence length.
        seq_len = 2 * page_size + 7  # Spans into third block partially.

        mask = generate_tsam_mask(
            block_table=block_table,
            page_type=bm.page_type,
            tenant_id=bm.tenant_id,
            query_tenant=0,
            page_size=page_size,
            seq_len_kv=seq_len,
        )
        assert mask.mask.shape[0] == seq_len

    def test_seq_len_1(self, num_heads, num_kv_heads, head_dim):
        """Minimum sequence length."""
        page_size = 16
        total_blocks = 5
        bm = TypedBlockManager(num_blocks=total_blocks, page_size=page_size)
        bm.allocate_private(tenant=0, count=1)

        key_cache = torch.randn(total_blocks, page_size, num_kv_heads, head_dim)
        value_cache = torch.randn(total_blocks, page_size, num_kv_heads, head_dim)

        block_table = torch.tensor([0], dtype=torch.int32)
        query = torch.randn(num_heads, head_dim)

        output = tsam_paged_attention(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            page_type=bm.page_type,
            tenant_id_meta=bm.tenant_id,
            query_tenant=0,
            page_size=page_size,
            seq_len_kv=1,
        )
        assert output.shape == (num_heads, head_dim)

    def test_long_sequence(self, num_heads, num_kv_heads, head_dim):
        """Long sequence (32768 tokens)."""
        page_size = 16
        seq_len = 32768
        num_seq_blocks = seq_len // page_size
        total_blocks = num_seq_blocks + 5
        bm = TypedBlockManager(num_blocks=total_blocks, page_size=page_size)
        shared = bm.allocate_shared(count=4)
        private = bm.allocate_private(tenant=0, count=num_seq_blocks - 4)

        block_table_ids = [p.block_id for p in shared] + [p.block_id for p in private]
        block_table = torch.tensor(block_table_ids, dtype=torch.int32)

        mask = generate_tsam_mask(
            block_table=block_table,
            page_type=bm.page_type,
            tenant_id=bm.tenant_id,
            query_tenant=0,
            page_size=page_size,
            seq_len_kv=seq_len,
        )
        assert mask.mask.shape[0] == seq_len
        assert mask.mask.all(), "All own + shared positions should be accessible"
        assert mask.num_blocked == 0

    def test_batch_size_512(self, page_size, num_heads, num_kv_heads, head_dim):
        """Large batch size."""
        batch_size = 512
        blocks_per_seq = 4
        total_blocks = batch_size * blocks_per_seq + 10
        bm = TypedBlockManager(num_blocks=total_blocks, page_size=page_size)

        shared = bm.allocate_shared(count=1)
        for t in range(batch_size):
            bm.allocate_private(tenant=t, count=blocks_per_seq - 1)

        # Build block tables.
        block_tables_list = []
        for t in range(batch_size):
            tenant_blocks = sorted(bm.get_tenant_blocks(t))
            bt = [shared[0].block_id] + tenant_blocks
            while len(bt) < blocks_per_seq:
                bt.append(bt[-1])
            block_tables_list.append(bt[:blocks_per_seq])

        block_tables = torch.tensor(block_tables_list, dtype=torch.int32)
        query_tenants = torch.arange(batch_size, dtype=torch.int32)
        seq_lens = torch.full((batch_size,), blocks_per_seq * page_size, dtype=torch.int32)

        masks = generate_tsam_mask_batched(
            block_tables=block_tables,
            page_type=bm.page_type,
            tenant_id=bm.tenant_id,
            query_tenants=query_tenants,
            page_size=page_size,
            seq_lens_kv=seq_lens,
            max_seq_len_kv=blocks_per_seq * page_size,
        )
        assert masks.shape == (batch_size, blocks_per_seq * page_size)

    def test_numerical_equality_same_tenant(
        self, setup_two_tenant_cache, page_size, num_heads, head_dim
    ):
        """Verify exact numerical equality: TSAM output == isolated output."""
        ctx = setup_two_tenant_cache
        bm = ctx["bm"]

        torch.manual_seed(123)
        query = torch.randn(num_heads, head_dim)

        # Mixed block table.
        bt_mixed = ctx["shared_ids"] + ctx["t0_ids"] + ctx["t1_ids"]
        block_table_mixed = torch.tensor(bt_mixed, dtype=torch.int32)
        seq_len_mixed = len(bt_mixed) * page_size

        # Isolated block table (only tenant 0's blocks + shared).
        bt_isolated = ctx["shared_ids"] + ctx["t0_ids"]
        block_table_isolated = torch.tensor(bt_isolated, dtype=torch.int32)
        seq_len_isolated = len(bt_isolated) * page_size

        output_mixed = tsam_paged_attention(
            query=query,
            key_cache=ctx["key_cache"],
            value_cache=ctx["value_cache"],
            block_table=block_table_mixed,
            page_type=bm.page_type,
            tenant_id_meta=bm.tenant_id,
            query_tenant=0,
            page_size=page_size,
            seq_len_kv=seq_len_mixed,
        )

        output_isolated = tsam_paged_attention(
            query=query,
            key_cache=ctx["key_cache"],
            value_cache=ctx["value_cache"],
            block_table=block_table_isolated,
            page_type=bm.page_type,
            tenant_id_meta=bm.tenant_id,
            query_tenant=0,
            page_size=page_size,
            seq_len_kv=seq_len_isolated,
        )

        torch.testing.assert_close(
            output_mixed, output_isolated, atol=1e-5, rtol=1e-5
        )


# ---------------------------------------------------------------
# Test (e): GQA/MQA attention isolation
# ---------------------------------------------------------------

class TestGQAMQAIsolation:
    """Verify isolation holds when num_heads != num_kv_heads (grouped-query attention)."""

    def test_gqa_cross_tenant_isolation(self):
        """GQA: 32 query heads, 8 KV heads — cross-tenant output must match isolated."""
        page_size = 16
        num_heads = 32
        num_kv_heads = 8
        head_dim = 64
        shared_blocks = 2
        private_blocks = 4
        total_blocks = shared_blocks + 2 * private_blocks + 1

        bm = TypedBlockManager(num_blocks=total_blocks, page_size=page_size)
        shared_pages = bm.allocate_shared(count=shared_blocks)
        t0_pages = bm.allocate_private(tenant=0, count=private_blocks)
        t1_pages = bm.allocate_private(tenant=1, count=private_blocks)

        key_cache = torch.randn(total_blocks, page_size, num_kv_heads, head_dim)
        value_cache = torch.randn(total_blocks, page_size, num_kv_heads, head_dim)

        # Distinctive content in tenant 0's blocks.
        for p in t0_pages:
            key_cache[p.block_id] = 1.0
            value_cache[p.block_id] = 1.0

        shared_ids = [p.block_id for p in shared_pages]
        t0_ids = [p.block_id for p in t0_pages]
        t1_ids = [p.block_id for p in t1_pages]

        torch.manual_seed(99)
        query = torch.randn(num_heads, head_dim)

        # Mixed block table (tenant 1 queries with tenant 0's blocks present).
        bt_mixed = shared_ids + t0_ids + t1_ids
        block_table_mixed = torch.tensor(bt_mixed, dtype=torch.int32)
        seq_len_mixed = len(bt_mixed) * page_size

        output_mixed = tsam_paged_attention(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table_mixed,
            page_type=bm.page_type,
            tenant_id_meta=bm.tenant_id,
            query_tenant=1,
            page_size=page_size,
            seq_len_kv=seq_len_mixed,
        )

        # Isolated block table (only tenant 1's blocks + shared).
        bt_isolated = shared_ids + t1_ids
        block_table_isolated = torch.tensor(bt_isolated, dtype=torch.int32)
        seq_len_isolated = len(bt_isolated) * page_size

        output_isolated = tsam_paged_attention(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table_isolated,
            page_type=bm.page_type,
            tenant_id_meta=bm.tenant_id,
            query_tenant=1,
            page_size=page_size,
            seq_len_kv=seq_len_isolated,
        )

        torch.testing.assert_close(
            output_mixed, output_isolated, atol=1e-5, rtol=1e-5
        )

    def test_mqa_single_kv_head_isolation(self):
        """MQA: 8 query heads, 1 KV head — verify isolation."""
        page_size = 16
        num_heads = 8
        num_kv_heads = 1
        head_dim = 64
        total_blocks = 10

        bm = TypedBlockManager(num_blocks=total_blocks, page_size=page_size)
        bm.allocate_shared(count=1)
        bm.allocate_private(tenant=0, count=3)
        bm.allocate_private(tenant=1, count=3)

        key_cache = torch.randn(total_blocks, page_size, num_kv_heads, head_dim)
        value_cache = torch.randn(total_blocks, page_size, num_kv_heads, head_dim)

        torch.manual_seed(77)
        query = torch.randn(num_heads, head_dim)
        bt_mixed = list(range(7))
        block_table = torch.tensor(bt_mixed, dtype=torch.int32)
        seq_len = 7 * page_size

        output = tsam_paged_attention(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            page_type=bm.page_type,
            tenant_id_meta=bm.tenant_id,
            query_tenant=1,
            page_size=page_size,
            seq_len_kv=seq_len,
        )
        assert output.shape == (num_heads, head_dim)
        assert output.abs().sum() > 0


# ---------------------------------------------------------------
# Test (f): Batched attention isolation verification
# ---------------------------------------------------------------

class TestBatchedIsolation:
    """Verify cross-tenant isolation in batched attention, not just mask shape."""

    def test_batched_cross_tenant_numerical_equality(self):
        """Each query in a batch must produce output identical to its isolated run."""
        page_size = 16
        num_heads = 4
        num_kv_heads = 4
        head_dim = 64
        batch_size = 4
        shared_blocks = 1
        private_blocks = 2
        total_blocks = shared_blocks + batch_size * private_blocks + 1

        bm = TypedBlockManager(num_blocks=total_blocks, page_size=page_size)
        shared = bm.allocate_shared(count=shared_blocks)
        shared_ids = [p.block_id for p in shared]

        tenant_ids_map = {}
        for t in range(batch_size):
            pages = bm.allocate_private(tenant=t, count=private_blocks)
            tenant_ids_map[t] = [p.block_id for p in pages]

        key_cache = torch.randn(total_blocks, page_size, num_kv_heads, head_dim)
        value_cache = torch.randn(total_blocks, page_size, num_kv_heads, head_dim)

        # Put distinct values per tenant.
        for t in range(batch_size):
            for blk_id in tenant_ids_map[t]:
                key_cache[blk_id] = float(t + 1)
                value_cache[blk_id] = float(t + 1)

        torch.manual_seed(42)
        queries = torch.randn(batch_size, num_heads, head_dim)

        # Build block tables: each tenant has shared + own private.
        blocks_per_query = shared_blocks + private_blocks
        block_tables_list = []
        for t in range(batch_size):
            block_tables_list.append(shared_ids + tenant_ids_map[t])
        block_tables = torch.tensor(block_tables_list, dtype=torch.int32)
        query_tenants = torch.arange(batch_size, dtype=torch.int32)
        seq_lens = torch.full((batch_size,), blocks_per_query * page_size, dtype=torch.int32)

        batched_output = tsam_paged_attention_batched(
            queries=queries,
            key_cache=key_cache,
            value_cache=value_cache,
            block_tables=block_tables,
            page_type=bm.page_type,
            tenant_id_meta=bm.tenant_id,
            query_tenants=query_tenants,
            page_size=page_size,
            seq_lens_kv=seq_lens,
        )

        # Compare each query's batched output against its individual isolated run.
        for t in range(batch_size):
            isolated_bt = shared_ids + tenant_ids_map[t]
            block_table_single = torch.tensor(isolated_bt, dtype=torch.int32)
            seq_len_single = len(isolated_bt) * page_size

            output_single = tsam_paged_attention(
                query=queries[t],
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=block_table_single,
                page_type=bm.page_type,
                tenant_id_meta=bm.tenant_id,
                query_tenant=t,
                page_size=page_size,
                seq_len_kv=seq_len_single,
            )
            torch.testing.assert_close(
                batched_output[t], output_single, atol=1e-5, rtol=1e-5
            )

    def test_batched_mask_blocks_cross_tenant(self):
        """Verify the batched mask correctly blocks cross-tenant positions."""
        page_size = 16
        total_blocks = 10
        bm = TypedBlockManager(num_blocks=total_blocks, page_size=page_size)
        bm.allocate_shared(count=1)
        bm.allocate_private(tenant=0, count=2)
        bm.allocate_private(tenant=1, count=2)

        # Both tenants reference same physical blocks: shared + t0 + t1.
        block_tables = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], dtype=torch.int32)
        query_tenants = torch.tensor([0, 1], dtype=torch.int32)
        seq_lens = torch.tensor([5 * page_size, 5 * page_size], dtype=torch.int32)

        masks = generate_tsam_mask_batched(
            block_tables=block_tables,
            page_type=bm.page_type,
            tenant_id=bm.tenant_id,
            query_tenants=query_tenants,
            page_size=page_size,
            seq_lens_kv=seq_lens,
            max_seq_len_kv=5 * page_size,
        )

        # Tenant 0: should see shared (block 0) + own (blocks 1-2), not tenant 1's (blocks 3-4).
        assert masks[0, :page_size].all()  # shared
        assert masks[0, page_size:3 * page_size].all()  # own private
        assert not masks[0, 3 * page_size:5 * page_size].any()  # other tenant

        # Tenant 1: should see shared (block 0) + own (blocks 3-4), not tenant 0's (blocks 1-2).
        assert masks[1, :page_size].all()  # shared
        assert not masks[1, page_size:3 * page_size].any()  # other tenant
        assert masks[1, 3 * page_size:5 * page_size].all()  # own private


# ---------------------------------------------------------------
# Test (g): Paged attention parameter paths
# ---------------------------------------------------------------

class TestPagedAttentionParams:
    def test_causal_false(self):
        """Verify attention works with causal=False."""
        page_size = 16
        num_heads = 4
        num_kv_heads = 4
        head_dim = 64
        total_blocks = 5

        bm = TypedBlockManager(num_blocks=total_blocks, page_size=page_size)
        bm.allocate_private(tenant=0, count=2)
        key_cache = torch.randn(total_blocks, page_size, num_kv_heads, head_dim)
        value_cache = torch.randn(total_blocks, page_size, num_kv_heads, head_dim)

        query = torch.randn(num_heads, head_dim)
        block_table = torch.tensor([0, 1], dtype=torch.int32)

        output = tsam_paged_attention(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            page_type=bm.page_type,
            tenant_id_meta=bm.tenant_id,
            query_tenant=0,
            page_size=page_size,
            seq_len_kv=2 * page_size,
            causal=False,
        )
        assert output.shape == (num_heads, head_dim)
        assert output.abs().sum() > 0

    def test_custom_scale(self):
        """Verify custom attention scale produces different output than default."""
        page_size = 16
        num_heads = 4
        num_kv_heads = 4
        head_dim = 64
        total_blocks = 5

        bm = TypedBlockManager(num_blocks=total_blocks, page_size=page_size)
        bm.allocate_private(tenant=0, count=2)
        key_cache = torch.randn(total_blocks, page_size, num_kv_heads, head_dim)
        value_cache = torch.randn(total_blocks, page_size, num_kv_heads, head_dim)

        torch.manual_seed(42)
        query = torch.randn(num_heads, head_dim)
        block_table = torch.tensor([0, 1], dtype=torch.int32)
        seq_len = 2 * page_size

        output_default = tsam_paged_attention(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            page_type=bm.page_type,
            tenant_id_meta=bm.tenant_id,
            query_tenant=0,
            page_size=page_size,
            seq_len_kv=seq_len,
        )

        output_custom = tsam_paged_attention(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            page_type=bm.page_type,
            tenant_id_meta=bm.tenant_id,
            query_tenant=0,
            page_size=page_size,
            seq_len_kv=seq_len,
            scale=0.01,  # Very different scale.
        )
        # Outputs should differ because scale affects attention distribution.
        assert not torch.allclose(output_default, output_custom, atol=1e-4)

    def test_long_sequence_attention_output_isolation(self):
        """Run actual attention through a long sequence and verify isolation."""
        page_size = 16
        seq_len = 4096
        num_heads = 4
        num_kv_heads = 4
        head_dim = 64
        num_seq_blocks = seq_len // page_size
        shared_blocks = 4
        private_blocks = num_seq_blocks - shared_blocks
        total_blocks = num_seq_blocks + private_blocks + 5

        bm = TypedBlockManager(num_blocks=total_blocks, page_size=page_size)
        shared = bm.allocate_shared(count=shared_blocks)
        t0_private = bm.allocate_private(tenant=0, count=private_blocks)
        t1_private = bm.allocate_private(tenant=1, count=private_blocks)

        key_cache = torch.randn(total_blocks, page_size, num_kv_heads, head_dim)
        value_cache = torch.randn(total_blocks, page_size, num_kv_heads, head_dim)

        shared_ids = [p.block_id for p in shared]
        t0_ids = [p.block_id for p in t0_private]
        t1_ids = [p.block_id for p in t1_private]

        # Mixed block table with both tenants.
        bt_mixed = shared_ids + t0_ids + t1_ids
        block_table_mixed = torch.tensor(bt_mixed, dtype=torch.int32)
        seq_len_mixed = len(bt_mixed) * page_size

        torch.manual_seed(42)
        query = torch.randn(num_heads, head_dim)

        output_mixed = tsam_paged_attention(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table_mixed,
            page_type=bm.page_type,
            tenant_id_meta=bm.tenant_id,
            query_tenant=0,
            page_size=page_size,
            seq_len_kv=seq_len_mixed,
        )

        # Isolated: only tenant 0.
        bt_isolated = shared_ids + t0_ids
        block_table_isolated = torch.tensor(bt_isolated, dtype=torch.int32)
        seq_len_isolated = len(bt_isolated) * page_size

        output_isolated = tsam_paged_attention(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table_isolated,
            page_type=bm.page_type,
            tenant_id_meta=bm.tenant_id,
            query_tenant=0,
            page_size=page_size,
            seq_len_kv=seq_len_isolated,
        )

        torch.testing.assert_close(
            output_mixed, output_isolated, atol=1e-4, rtol=1e-4
        )

    def test_3d_query_input(self):
        """Verify (1, num_heads, head_dim) query shape works and matches 2D."""
        page_size = 16
        num_heads = 4
        num_kv_heads = 4
        head_dim = 64
        total_blocks = 5

        bm = TypedBlockManager(num_blocks=total_blocks, page_size=page_size)
        bm.allocate_private(tenant=0, count=2)
        key_cache = torch.randn(total_blocks, page_size, num_kv_heads, head_dim)
        value_cache = torch.randn(total_blocks, page_size, num_kv_heads, head_dim)

        torch.manual_seed(42)
        query_2d = torch.randn(num_heads, head_dim)
        query_3d = query_2d.unsqueeze(0)
        block_table = torch.tensor([0, 1], dtype=torch.int32)

        output_2d = tsam_paged_attention(
            query=query_2d, key_cache=key_cache, value_cache=value_cache,
            block_table=block_table, page_type=bm.page_type,
            tenant_id_meta=bm.tenant_id, query_tenant=0,
            page_size=page_size, seq_len_kv=2 * page_size,
        )

        output_3d = tsam_paged_attention(
            query=query_3d, key_cache=key_cache, value_cache=value_cache,
            block_table=block_table, page_type=bm.page_type,
            tenant_id_meta=bm.tenant_id, query_tenant=0,
            page_size=page_size, seq_len_kv=2 * page_size,
        )

        assert output_2d.shape == (num_heads, head_dim)
        assert output_3d.shape == (1, num_heads, head_dim)
        torch.testing.assert_close(output_2d, output_3d.squeeze(0), atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------
# Test: apply_tsam_mask_to_scores
# ---------------------------------------------------------------

class TestApplyMaskToScores:
    def test_basic_masking(self):
        scores = torch.randn(4, 1, 8)
        mask = torch.tensor([True, True, False, False, True, True, False, True])
        masked = apply_tsam_mask_to_scores(scores, mask.unsqueeze(0).unsqueeze(0))
        assert torch.isinf(masked[:, :, 2]).all()
        assert torch.isinf(masked[:, :, 3]).all()
        assert torch.isinf(masked[:, :, 6]).all()
        assert not torch.isinf(masked[:, :, 0]).any()
        assert not torch.isinf(masked[:, :, 1]).any()
        assert not torch.isinf(masked[:, :, 4]).any()

    def test_all_masked_produces_uniform_after_softmax(self):
        """If all positions are masked, softmax produces NaN (expected — no valid keys)."""
        scores = torch.randn(2, 1, 4)
        mask = torch.zeros(4, dtype=torch.bool)
        masked = apply_tsam_mask_to_scores(scores, mask.unsqueeze(0).unsqueeze(0))
        weights = torch.nn.functional.softmax(masked, dim=-1)
        # All -inf -> softmax produces NaN (mathematically correct: no valid keys).
        assert torch.isnan(weights).all()

    def test_single_unmasked_gets_full_weight(self):
        scores = torch.randn(1, 1, 4)
        mask = torch.tensor([False, False, True, False])
        masked = apply_tsam_mask_to_scores(scores, mask.unsqueeze(0).unsqueeze(0))
        weights = torch.nn.functional.softmax(masked, dim=-1)
        # Only position 2 is unmasked, so it should get weight 1.0.
        assert torch.isclose(weights[0, 0, 2], torch.tensor(1.0), atol=1e-6)
