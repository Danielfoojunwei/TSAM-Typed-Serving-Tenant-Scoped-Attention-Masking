"""
Core types for TSAM typed KV-cache page structure.

Each KV-cache page is augmented with metadata:
  page.tenant_id   ∈ {0..N_tenants} ∪ {SHARED}
  page.page_type   ∈ {PRIVATE, SHARED}
  page.access_list = {tenant_ids allowed to attend to this page}
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import FrozenSet, Set, Union

# Sentinel value for SHARED tenant — no single owner.
SHARED_TENANT_ID: int = -1


class PageType(enum.IntEnum):
    """KV-cache page type. Encoded as uint8 in the block table metadata."""

    PRIVATE = 0
    SHARED = 1


# Alias for clarity in signatures.
TenantId = int


@dataclass(frozen=True)
class TypedPage:
    """Metadata for a single typed KV-cache page (block).

    Attributes:
        block_id:    Physical block index in the KV-cache pool.
        tenant_id:   Owning tenant, or SHARED_TENANT_ID for shared pages.
        page_type:   PRIVATE or SHARED.
        access_list: Set of tenant IDs that may attend to this page.
                     For PRIVATE pages: {tenant_id}.
                     For SHARED pages: represents ALL tenants (we use None
                     as a sentinel for "universal access" in the fast path).
    """

    block_id: int
    tenant_id: TenantId
    page_type: PageType
    access_list: Union[FrozenSet[TenantId], None] = None  # None => universal

    def __post_init__(self):
        # Enforce invariants.
        if self.page_type == PageType.PRIVATE:
            if self.tenant_id == SHARED_TENANT_ID:
                raise ValueError("PRIVATE page cannot have SHARED_TENANT_ID")
            if self.access_list is not None and self.access_list != frozenset(
                {self.tenant_id}
            ):
                raise ValueError(
                    "PRIVATE page access_list must be {tenant_id} or None (auto-set)"
                )
            # Auto-set access_list for PRIVATE pages.
            if self.access_list is None:
                object.__setattr__(
                    self, "access_list", frozenset({self.tenant_id})
                )
        elif self.page_type == PageType.SHARED:
            if self.tenant_id != SHARED_TENANT_ID:
                raise ValueError("SHARED page must have SHARED_TENANT_ID")
            # SHARED pages: access_list=None means universal access.
            if self.access_list is not None:
                raise ValueError(
                    "SHARED page access_list must be None (universal)"
                )

    def can_attend(self, query_tenant: TenantId) -> bool:
        """Return True if a query from *query_tenant* may attend to this page.

        This is the core isolation predicate:
          - SHARED pages: always True (universal access).
          - PRIVATE pages: True iff query_tenant == page.tenant_id.
        """
        if self.page_type == PageType.SHARED:
            return True
        # PRIVATE page — check membership.
        assert self.access_list is not None
        return query_tenant in self.access_list
