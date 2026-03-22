"""Tests for TypedBlockManager."""

import pytest
import torch
from tsam.core.block_manager import TypedBlockManager
from tsam.core.types import PageType, SHARED_TENANT_ID


class TestTypedBlockManager:
    def test_init(self):
        bm = TypedBlockManager(num_blocks=100, page_size=16)
        assert bm.num_blocks == 100
        assert bm.num_free_blocks == 100
        assert bm.page_type.shape == (100,)
        assert bm.tenant_id.shape == (100,)

    def test_allocate_private(self):
        bm = TypedBlockManager(num_blocks=100)
        pages = bm.allocate_private(tenant=0, count=3)
        assert len(pages) == 3
        assert bm.num_free_blocks == 97
        for p in pages:
            assert p.page_type == PageType.PRIVATE
            assert p.tenant_id == 0
            assert bm.page_type[p.block_id].item() == PageType.PRIVATE
            assert bm.tenant_id[p.block_id].item() == 0

    def test_allocate_shared(self):
        bm = TypedBlockManager(num_blocks=100)
        pages = bm.allocate_shared(count=2)
        assert len(pages) == 2
        assert bm.num_free_blocks == 98
        for p in pages:
            assert p.page_type == PageType.SHARED
            assert p.tenant_id == SHARED_TENANT_ID
            assert bm.page_type[p.block_id].item() == PageType.SHARED
            assert bm.tenant_id[p.block_id].item() == SHARED_TENANT_ID

    def test_free(self):
        bm = TypedBlockManager(num_blocks=10)
        pages = bm.allocate_private(tenant=0, count=3)
        assert bm.num_free_blocks == 7
        bm.free(pages[0].block_id)
        assert bm.num_free_blocks == 8

    def test_free_tenant(self):
        bm = TypedBlockManager(num_blocks=100)
        bm.allocate_private(tenant=0, count=5)
        bm.allocate_private(tenant=1, count=3)
        freed = bm.free_tenant(0)
        assert freed == 5
        assert bm.num_free_blocks == 100 - 3

    def test_free_nonexistent_raises(self):
        bm = TypedBlockManager(num_blocks=10)
        with pytest.raises(ValueError, match="not allocated"):
            bm.free(99)

    def test_allocation_exhaustion(self):
        bm = TypedBlockManager(num_blocks=5)
        bm.allocate_private(tenant=0, count=5)
        with pytest.raises(RuntimeError, match="Cannot allocate"):
            bm.allocate_private(tenant=1, count=1)

    def test_get_tenant_blocks(self):
        bm = TypedBlockManager(num_blocks=100)
        pages = bm.allocate_private(tenant=7, count=4)
        blocks = bm.get_tenant_blocks(7)
        assert blocks == {p.block_id for p in pages}

    def test_build_access_mask_private(self):
        bm = TypedBlockManager(num_blocks=10)
        bm.allocate_private(tenant=0, count=3)
        bm.allocate_private(tenant=1, count=3)
        mask_0 = bm.build_access_mask(query_tenant=0)
        mask_1 = bm.build_access_mask(query_tenant=1)
        # Tenant 0 should see its own blocks but not tenant 1's.
        for blk_id in bm.get_tenant_blocks(0):
            assert mask_0[blk_id].item() is True
        for blk_id in bm.get_tenant_blocks(1):
            assert mask_0[blk_id].item() is False

    def test_build_access_mask_shared(self):
        bm = TypedBlockManager(num_blocks=10)
        shared = bm.allocate_shared(count=2)
        bm.allocate_private(tenant=0, count=2)
        mask = bm.build_access_mask(query_tenant=0)
        # Shared blocks accessible to any tenant.
        for p in shared:
            assert mask[p.block_id].item() is True

    def test_ref_unref_shared(self):
        bm = TypedBlockManager(num_blocks=10)
        shared = bm.allocate_shared(count=1)
        blk = shared[0].block_id
        bm.ref_shared(blk)
        bm.ref_shared(blk)
        assert bm.get_page(blk) is not None
        bm.unref_shared(blk)  # refcount 1
        assert bm.get_page(blk) is not None
        bm.unref_shared(blk)  # refcount 0 -> freed
        assert bm.get_page(blk) is None

    def test_multiple_tenants_isolation(self):
        bm = TypedBlockManager(num_blocks=1000)
        shared = bm.allocate_shared(count=4)
        for t in range(50):
            bm.allocate_private(tenant=t, count=5)

        for t in range(50):
            mask = bm.build_access_mask(query_tenant=t)
            # Own blocks accessible.
            for blk in bm.get_tenant_blocks(t):
                assert mask[blk].item() is True
            # Other tenants' blocks not accessible.
            for other_t in range(50):
                if other_t == t:
                    continue
                for blk in bm.get_tenant_blocks(other_t):
                    assert mask[blk].item() is False
            # Shared blocks accessible.
            for p in shared:
                assert mask[p.block_id].item() is True
