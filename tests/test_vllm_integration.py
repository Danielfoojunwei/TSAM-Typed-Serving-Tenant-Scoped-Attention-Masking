"""Tests for vLLM integration layer."""

import pytest
import torch

from tsam.integration.vllm_patch import (
    TSAMBlockTableExtension,
    TSAMAttentionWrapper,
    generate_vllm_kernel_patch,
)
from tsam.core.types import PageType, SHARED_TENANT_ID


class TestTSAMBlockTableExtension:
    def test_init(self):
        ext = TSAMBlockTableExtension(num_blocks=100, device="cpu")
        assert ext.page_type.shape == (100,)
        assert ext.tenant_id.shape == (100,)

    def test_mark_private(self):
        ext = TSAMBlockTableExtension(num_blocks=10, device="cpu")
        ext.mark_private(3, tenant=7)
        assert ext.page_type[3].item() == PageType.PRIVATE
        assert ext.tenant_id[3].item() == 7

    def test_mark_shared(self):
        ext = TSAMBlockTableExtension(num_blocks=10, device="cpu")
        ext.mark_shared(5)
        assert ext.page_type[5].item() == PageType.SHARED
        assert ext.tenant_id[5].item() == SHARED_TENANT_ID

    def test_mark_private_batch(self):
        ext = TSAMBlockTableExtension(num_blocks=10, device="cpu")
        block_ids = torch.tensor([1, 2, 3])
        ext.mark_private_batch(block_ids, tenant=5)
        for i in [1, 2, 3]:
            assert ext.page_type[i].item() == PageType.PRIVATE
            assert ext.tenant_id[i].item() == 5

    def test_generate_mask(self):
        ext = TSAMBlockTableExtension(num_blocks=10, device="cpu")
        ext.mark_shared(0)
        ext.mark_private(1, tenant=0)
        ext.mark_private(2, tenant=1)

        block_table = torch.tensor([0, 1, 2], dtype=torch.int32)
        page_size = 16
        seq_len = 3 * page_size

        mask = ext.generate_mask(block_table, query_tenant=0, page_size=page_size, seq_len_kv=seq_len)
        # Shared (block 0): all True.
        assert mask[:page_size].all()
        # Own private (block 1): all True.
        assert mask[page_size:2 * page_size].all()
        # Other private (block 2): all False.
        assert not mask[2 * page_size:].any()

    def test_generate_mask_batched(self):
        ext = TSAMBlockTableExtension(num_blocks=10, device="cpu")
        ext.mark_shared(0)
        ext.mark_private(1, tenant=0)
        ext.mark_private(2, tenant=1)

        block_tables = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.int32)
        query_tenants = torch.tensor([0, 1], dtype=torch.int32)
        page_size = 16
        seq_lens = torch.tensor([48, 48], dtype=torch.int32)

        masks = ext.generate_mask_batched(
            block_tables, query_tenants, page_size, seq_lens, max_seq_len_kv=48
        )
        assert masks.shape == (2, 48)
        # Tenant 0 sees block 0 and 1, not 2.
        assert masks[0, :page_size].all()
        assert masks[0, page_size:2 * page_size].all()
        assert not masks[0, 2 * page_size:].any()
        # Tenant 1 sees block 0 and 2, not 1.
        assert masks[1, :page_size].all()
        assert not masks[1, page_size:2 * page_size].any()
        assert masks[1, 2 * page_size:].all()


class TestKernelPatch:
    def test_patch_contains_tsam_mask(self):
        patch = generate_vllm_kernel_patch()
        assert "page_type" in patch
        assert "tenant_id" in patch
        assert "current_tenant_id" in patch
        assert "TSAM TYPE MASK" in patch
        assert "-FLT_MAX" in patch

    def test_patch_is_unified_diff(self):
        patch = generate_vllm_kernel_patch()
        assert patch.startswith("--- a/")
        assert "+++ b/" in patch
        assert "@@" in patch
