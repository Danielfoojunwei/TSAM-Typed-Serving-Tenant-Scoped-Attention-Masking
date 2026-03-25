"""Tests for TSAM core types."""

import pytest
from tsam.core.types import PageType, TypedPage, SHARED_TENANT_ID


class TestPageType:
    def test_private_value(self):
        assert PageType.PRIVATE == 0

    def test_shared_value(self):
        assert PageType.SHARED == 1

    def test_int_enum(self):
        assert int(PageType.PRIVATE) == 0
        assert int(PageType.SHARED) == 1


class TestTypedPage:
    def test_private_page_creation(self):
        page = TypedPage(block_id=0, tenant_id=1, page_type=PageType.PRIVATE)
        assert page.block_id == 0
        assert page.tenant_id == 1
        assert page.page_type == PageType.PRIVATE
        assert page.access_list == frozenset({1})

    def test_shared_page_creation(self):
        page = TypedPage(
            block_id=5, tenant_id=SHARED_TENANT_ID, page_type=PageType.SHARED
        )
        assert page.block_id == 5
        assert page.tenant_id == SHARED_TENANT_ID
        assert page.page_type == PageType.SHARED
        assert page.access_list is None

    def test_private_page_rejects_shared_tenant(self):
        with pytest.raises(ValueError, match="PRIVATE page cannot have SHARED_TENANT_ID"):
            TypedPage(
                block_id=0,
                tenant_id=SHARED_TENANT_ID,
                page_type=PageType.PRIVATE,
            )

    def test_shared_page_rejects_specific_tenant(self):
        with pytest.raises(ValueError, match="SHARED page must have SHARED_TENANT_ID"):
            TypedPage(block_id=0, tenant_id=1, page_type=PageType.SHARED)

    def test_private_page_rejects_wrong_access_list(self):
        with pytest.raises(ValueError, match="PRIVATE page access_list"):
            TypedPage(
                block_id=0,
                tenant_id=1,
                page_type=PageType.PRIVATE,
                access_list=frozenset({1, 2}),
            )

    def test_shared_page_rejects_explicit_access_list(self):
        with pytest.raises(ValueError, match="SHARED page access_list must be None"):
            TypedPage(
                block_id=0,
                tenant_id=SHARED_TENANT_ID,
                page_type=PageType.SHARED,
                access_list=frozenset({1}),
            )

    def test_can_attend_private_same_tenant(self):
        page = TypedPage(block_id=0, tenant_id=1, page_type=PageType.PRIVATE)
        assert page.can_attend(1) is True

    def test_can_attend_private_different_tenant(self):
        page = TypedPage(block_id=0, tenant_id=1, page_type=PageType.PRIVATE)
        assert page.can_attend(2) is False

    def test_can_attend_shared_any_tenant(self):
        page = TypedPage(
            block_id=0, tenant_id=SHARED_TENANT_ID, page_type=PageType.SHARED
        )
        assert page.can_attend(0) is True
        assert page.can_attend(1) is True
        assert page.can_attend(999) is True

    def test_frozen(self):
        page = TypedPage(block_id=0, tenant_id=1, page_type=PageType.PRIVATE)
        with pytest.raises(AttributeError):
            page.block_id = 5
