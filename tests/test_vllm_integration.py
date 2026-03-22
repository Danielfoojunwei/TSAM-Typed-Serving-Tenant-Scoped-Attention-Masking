"""Tests for vLLM integration layer."""

import pytest
import torch

from tsam.integration.vllm_patch import (
    TSAMBlockTableExtension,
    TSAMAttentionWrapper,
    generate_vllm_kernel_patch,
    patch_vllm_block_manager,
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

    def test_mark_shared_batch(self):
        ext = TSAMBlockTableExtension(num_blocks=10, device="cpu")
        block_ids = torch.tensor([6, 7, 8])
        ext.mark_shared_batch(block_ids)
        for i in [6, 7, 8]:
            assert ext.page_type[i].item() == PageType.SHARED
            assert ext.tenant_id[i].item() == SHARED_TENANT_ID

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


class TestTSAMAttentionWrapper:
    def test_wrapper_calls_backend_forward(self):
        """Verify wrapper passes TSAM mask to backend and calls forward."""
        ext = TSAMBlockTableExtension(num_blocks=10, device="cpu")
        ext.mark_shared(0)
        ext.mark_private(1, tenant=0)

        call_log = {}

        class MockBackend:
            def forward(self, **kwargs):
                call_log["called"] = True
                call_log["attn_mask"] = kwargs.get("attn_mask")
                # Return dummy output.
                batch_size = kwargs["query"].shape[0]
                num_heads = kwargs["query"].shape[1]
                head_dim = kwargs["query"].shape[2]
                return torch.zeros(batch_size, num_heads, head_dim)

        wrapper = TSAMAttentionWrapper(
            backend=MockBackend(),
            tsam_ext=ext,
            page_size=16,
        )

        query = torch.randn(1, 4, 64)
        key = torch.randn(1, 4, 64)
        value = torch.randn(1, 4, 64)
        block_tables = torch.tensor([[0, 1]], dtype=torch.int32)
        query_tenants = torch.tensor([0], dtype=torch.int32)
        seq_lens = torch.tensor([32], dtype=torch.int32)

        output = wrapper.forward(
            query=query, key=key, value=value,
            block_tables=block_tables,
            query_tenants=query_tenants,
            seq_lens_kv=seq_lens,
            max_seq_len_kv=32,
        )

        assert call_log["called"] is True
        assert call_log["attn_mask"] is not None
        assert call_log["attn_mask"].shape == (1, 32)

    def test_wrapper_merges_existing_mask(self):
        """Verify wrapper AND-merges existing attn_mask with TSAM mask."""
        ext = TSAMBlockTableExtension(num_blocks=10, device="cpu")
        ext.mark_shared(0)
        ext.mark_private(1, tenant=0)

        received_mask = {}

        class MockBackend:
            def forward(self, **kwargs):
                received_mask["attn_mask"] = kwargs.get("attn_mask")
                return torch.zeros(1, 4, 64)

        wrapper = TSAMAttentionWrapper(
            backend=MockBackend(), tsam_ext=ext, page_size=16,
        )

        # Pass an existing mask that blocks the first position.
        existing = torch.ones(1, 32, dtype=torch.bool)
        existing[0, 0] = False

        wrapper.forward(
            query=torch.randn(1, 4, 64),
            key=torch.randn(1, 4, 64),
            value=torch.randn(1, 4, 64),
            block_tables=torch.tensor([[0, 1]], dtype=torch.int32),
            query_tenants=torch.tensor([0], dtype=torch.int32),
            seq_lens_kv=torch.tensor([32], dtype=torch.int32),
            max_seq_len_kv=32,
            attn_mask=existing,
        )

        merged = received_mask["attn_mask"]
        # Position 0 should be False (from existing mask) even though TSAM allows it.
        assert merged[0, 0].item() is False

    def test_wrapper_raises_on_no_forward(self):
        """Wrapper should raise if backend lacks forward()."""
        ext = TSAMBlockTableExtension(num_blocks=10, device="cpu")

        class NoForwardBackend:
            pass

        # The object has no forward method, but Python objects always have hasattr
        # return False for missing attributes. The wrapper checks hasattr.
        # Actually the wrapper does `if hasattr(self.backend, "forward")` which is True
        # for objects with a forward method. For objects without, it raises AttributeError.
        wrapper = TSAMAttentionWrapper(
            backend=NoForwardBackend(), tsam_ext=ext, page_size=16,
        )

        with pytest.raises(AttributeError, match="does not have a forward"):
            wrapper.forward(
                query=torch.randn(1, 4, 64),
                key=torch.randn(1, 4, 64),
                value=torch.randn(1, 4, 64),
                block_tables=torch.tensor([[0]], dtype=torch.int32),
                query_tenants=torch.tensor([0], dtype=torch.int32),
                seq_lens_kv=torch.tensor([16], dtype=torch.int32),
                max_seq_len_kv=16,
            )


class TestPatchVllmBlockManager:
    def test_patch_tracks_private_allocations(self):
        """Monkey-patched allocate tracks PRIVATE metadata."""
        ext = TSAMBlockTableExtension(num_blocks=10, device="cpu")

        class FakeBlockManager:
            def allocate(self):
                return 3

        bm = FakeBlockManager()
        patch_vllm_block_manager(bm, ext)

        bm.allocate(tenant_id=7, shared=False)
        assert ext.page_type[3].item() == PageType.PRIVATE
        assert ext.tenant_id[3].item() == 7

    def test_patch_tracks_shared_allocations(self):
        """Monkey-patched allocate tracks SHARED metadata."""
        ext = TSAMBlockTableExtension(num_blocks=10, device="cpu")

        class FakeBlockManager:
            def allocate(self):
                return 5

        bm = FakeBlockManager()
        patch_vllm_block_manager(bm, ext)

        bm.allocate(shared=True)
        assert ext.page_type[5].item() == PageType.SHARED
        assert ext.tenant_id[5].item() == SHARED_TENANT_ID

    def test_patch_attaches_tsam_ext(self):
        """Verify the extension is attached to the block manager."""
        ext = TSAMBlockTableExtension(num_blocks=10, device="cpu")

        class FakeBlockManager:
            def allocate(self):
                return 0

        bm = FakeBlockManager()
        patch_vllm_block_manager(bm, ext)
        assert bm.tsam_ext is ext


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

    def test_patch_blocks_private_pages(self):
        """Verify the patch logic correctly masks PRIVATE pages."""
        patch = generate_vllm_kernel_patch()
        # Must check page_type == 0 (PRIVATE) AND tenant_id != current_tenant_id.
        assert "page_type[physical_block_id] == 0" in patch
        assert "tenant_id[physical_block_id] != current_tenant_id" in patch
