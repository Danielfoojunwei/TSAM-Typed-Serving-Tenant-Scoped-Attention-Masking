"""
TypedBlockManager — extends the standard block manager with TSAM page metadata.

Mirrors vLLM's BlockManager interface, adding:
  - page_type tensor:  uint8, shape (num_blocks,)
  - tenant_id tensor:  int32, shape (num_blocks,)

Allocation paths:
  - allocate_private(tenant_id): returns PRIVATE block(s) owned by tenant.
  - allocate_shared():           returns SHARED block(s) for system prompts / prefixes.
  - free(block_id):              returns block to the free pool.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import torch

from tsam.core.types import PageType, TenantId, TypedPage, SHARED_TENANT_ID


@dataclass
class TypedBlockManager:
    """Manages a pool of typed KV-cache blocks.

    Parameters:
        num_blocks: Total number of physical KV-cache blocks.
        page_size:  Number of tokens per block (default 16, matching vLLM).
        device:     Torch device for metadata tensors.
    """

    num_blocks: int
    page_size: int = 16
    device: str = "cpu"

    # Metadata tensors (materialised in __post_init__).
    page_type: torch.Tensor = field(init=False)
    tenant_id: torch.Tensor = field(init=False)

    # Internal bookkeeping.
    _free_blocks: deque = field(init=False, repr=False)
    _allocated: Dict[int, TypedPage] = field(init=False, repr=False)
    _tenant_blocks: Dict[TenantId, Set[int]] = field(init=False, repr=False)
    _shared_prefix_refcount: Dict[int, int] = field(init=False, repr=False)
    _lock: threading.Lock = field(init=False, repr=False)

    def __post_init__(self):
        self.page_type = torch.zeros(
            self.num_blocks, dtype=torch.uint8, device=self.device
        )
        self.tenant_id = torch.full(
            (self.num_blocks,), SHARED_TENANT_ID, dtype=torch.int32, device=self.device
        )
        self._free_blocks = deque(range(self.num_blocks))
        self._allocated = {}
        self._tenant_blocks = {}
        self._shared_prefix_refcount = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------

    def allocate_private(self, tenant: TenantId, count: int = 1) -> List[TypedPage]:
        """Allocate *count* PRIVATE blocks for *tenant*."""
        with self._lock:
            if len(self._free_blocks) < count:
                raise RuntimeError(
                    f"Cannot allocate {count} blocks: only {len(self._free_blocks)} free"
                )
            pages: List[TypedPage] = []
            for _ in range(count):
                blk = self._free_blocks.popleft()
                page = TypedPage(
                    block_id=blk,
                    tenant_id=tenant,
                    page_type=PageType.PRIVATE,
                )
                self._register(page)
                pages.append(page)
            return pages

    def allocate_shared(self, count: int = 1) -> List[TypedPage]:
        """Allocate *count* SHARED blocks (system prompts, common prefixes)."""
        with self._lock:
            if len(self._free_blocks) < count:
                raise RuntimeError(
                    f"Cannot allocate {count} blocks: only {len(self._free_blocks)} free"
                )
            pages: List[TypedPage] = []
            for _ in range(count):
                blk = self._free_blocks.popleft()
                page = TypedPage(
                    block_id=blk,
                    tenant_id=SHARED_TENANT_ID,
                    page_type=PageType.SHARED,
                )
                self._register(page)
                self._shared_prefix_refcount[blk] = 0
                pages.append(page)
            return pages

    def ref_shared(self, block_id: int) -> None:
        """Increment reference count for a SHARED block (prefix sharing)."""
        with self._lock:
            if block_id not in self._shared_prefix_refcount:
                raise ValueError(f"Block {block_id} is not a SHARED block")
            self._shared_prefix_refcount[block_id] += 1

    def unref_shared(self, block_id: int) -> None:
        """Decrement reference count for a SHARED block; free if zero."""
        with self._lock:
            if block_id not in self._shared_prefix_refcount:
                raise ValueError(f"Block {block_id} is not a SHARED block")
            self._shared_prefix_refcount[block_id] -= 1
            if self._shared_prefix_refcount[block_id] <= 0:
                self._free_block_internal(block_id)

    # ------------------------------------------------------------------
    # Deallocation
    # ------------------------------------------------------------------

    def free(self, block_id: int) -> None:
        """Return a block to the free pool (thread-safe)."""
        with self._lock:
            self._free_block_internal(block_id)

    def _free_block_internal(self, block_id: int) -> None:
        """Internal free — caller must hold self._lock."""
        if block_id not in self._allocated:
            raise ValueError(f"Block {block_id} is not allocated")
        page = self._allocated.pop(block_id)
        # Update metadata tensors.
        self.page_type[block_id] = 0
        self.tenant_id[block_id] = SHARED_TENANT_ID
        # Remove from tenant tracking.
        tid = page.tenant_id
        if tid in self._tenant_blocks:
            self._tenant_blocks[tid].discard(block_id)
        # Remove refcount entry if shared.
        self._shared_prefix_refcount.pop(block_id, None)
        self._free_blocks.append(block_id)

    def free_tenant(self, tenant: TenantId) -> int:
        """Free all PRIVATE blocks owned by *tenant*. Returns count freed."""
        with self._lock:
            blocks = list(self._tenant_blocks.get(tenant, set()))
            for blk in blocks:
                if blk in self._allocated and self._allocated[blk].page_type == PageType.PRIVATE:
                    self._free_block_internal(blk)
            return len(blocks)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_blocks)

    def get_page(self, block_id: int) -> Optional[TypedPage]:
        return self._allocated.get(block_id)

    def get_tenant_blocks(self, tenant: TenantId) -> Set[int]:
        return set(self._tenant_blocks.get(tenant, set()))

    def build_access_mask(self, query_tenant: TenantId) -> torch.Tensor:
        """Build a boolean mask of shape (num_blocks,) indicating which blocks
        *query_tenant* may attend to.

        True  = allowed (SHARED or own PRIVATE).
        False = blocked (other tenant's PRIVATE).
        """
        # SHARED pages (page_type == 1) are always accessible.
        shared_mask = self.page_type == PageType.SHARED
        # Own PRIVATE pages.
        own_mask = (self.page_type == PageType.PRIVATE) & (
            self.tenant_id == query_tenant
        )
        # Unallocated blocks (page_type == 0, tenant_id == SHARED) should not
        # appear in any active block table, but mask them out for safety.
        return shared_mask | own_mask

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _register(self, page: TypedPage) -> None:
        blk = page.block_id
        self._allocated[blk] = page
        self.page_type[blk] = page.page_type.value
        self.tenant_id[blk] = page.tenant_id
        self._tenant_blocks.setdefault(page.tenant_id, set()).add(blk)
