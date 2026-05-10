"""
Typed page lifecycle state machine for TSAM.

Closes the "shared pages might leak" objection raised in the NeurIPS-readiness
review by enforcing a typed transition policy on KV-cache pages:

    UNALLOCATED
        |
        v
    SHARED_PREFIX  --(copy-on-write)-->  PRIVATE_CONTINUATION(t)
        |                                      |
        |                                      v
        +------------(free)----------> FREED -> UNALLOCATED (after zeroize)

Rules enforced:

  - SHARED_PREFIX admission requires the page's content hash to be a member of
    a configured allow-list of approved prefix hashes (system prompts, etc.).
    A page cannot be silently allocated as SHARED.
  - Once any tenant-private token would append to a SHARED_PREFIX page, the
    page enters copy-on-write: a fresh block is allocated, contents are copied
    to the new block, and the new block is marked PRIVATE_CONTINUATION(t).
    The original SHARED page is left untouched, preserving sharing for other
    tenants.
  - Free is defensive: the KV slot for the freed block is zeroized in the
    backing buffer before the block returns to the free pool. This prevents a
    re-allocated block from inheriting victim content.
  - Invariants are checked after every state transition via assert_invariants.

Integration: this module is composition-only. TypedBlockManager holds an
optional PageLifecycle instance gated by enable_lifecycle. When disabled,
behavior is unchanged from the pre-lifecycle implementation; when enabled,
allocation and free paths route through the lifecycle to enforce the rules
above. The default for new TypedBlockManager instances remains False so that
existing tests and benchmark code continue to pass without modification.

This file is CPU-only and does not import torch.cuda. All metadata tensors
inherit the device of the KV-cache buffer passed in.
"""

from __future__ import annotations

import enum
import hashlib
import threading
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Optional, Set

import torch

from tsam.core.types import SHARED_TENANT_ID, TenantId


class PageState(enum.IntEnum):
    """States in the typed page lifecycle.

    Encoded as uint8 in the per-block state tensor.
    """

    UNALLOCATED = 0
    SHARED_PREFIX = 1
    PRIVATE_CONTINUATION = 2
    FREED = 3  # transient: about to return to UNALLOCATED after zeroize


class LifecycleViolation(RuntimeError):
    """Raised when an operation would violate a lifecycle invariant.

    Concrete subclasses (UnapprovedSharedHash, IllegalTransition, ...) carry
    enough context for callers to fail closed with a clear diagnostic.
    """


class UnapprovedSharedHash(LifecycleViolation):
    """Attempted to admit a SHARED page whose content hash is not on the allow-list."""


class IllegalTransition(LifecycleViolation):
    """Attempted a state transition that the lifecycle rules forbid."""


class StaleBlock(LifecycleViolation):
    """Attempted to operate on a block that has been freed but not yet zeroized."""


@dataclass
class PageLifecycle:
    """Tracks per-block state and enforces COW + zeroize-on-free.

    Args:
        num_blocks: Total physical block count.
        page_size: Tokens per block.
        kv_buffer: The KV-cache tensor that lifecycle owns zeroizing access to.
            Expected shape (num_blocks, page_size, *) where the trailing dims
            are head/feature axes; the trailing layout is opaque to lifecycle.
            If None, zeroize is a no-op (used by tests that don't model KV).
        approved_prefix_hashes: Initial allow-list of SHA-256 hex digests that
            may legally back a SHARED_PREFIX page. Pass {} to start empty and
            register with admit_prefix_hash.
        device: Device for metadata tensors. Inherits from kv_buffer if given.
    """

    num_blocks: int
    page_size: int = 16
    kv_buffer: Optional[torch.Tensor] = None
    approved_prefix_hashes: FrozenSet[str] = frozenset()
    device: str = "cpu"

    _state: torch.Tensor = field(init=False, repr=False)
    _owner: torch.Tensor = field(init=False, repr=False)
    _origin_block: Dict[int, int] = field(init=False, repr=False)
    _hash_index: Dict[int, str] = field(init=False, repr=False)
    _allow: Set[str] = field(init=False, repr=False)
    _lock: threading.Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.kv_buffer is not None:
            self.device = str(self.kv_buffer.device)
        self._state = torch.zeros(
            self.num_blocks, dtype=torch.uint8, device=self.device
        )
        self._owner = torch.full(
            (self.num_blocks,),
            SHARED_TENANT_ID,
            dtype=torch.int32,
            device=self.device,
        )
        self._origin_block = {}
        self._hash_index = {}
        self._allow = set(self.approved_prefix_hashes)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Admission policy
    # ------------------------------------------------------------------

    def admit_prefix_hash(self, prefix_hash: str) -> None:
        """Register a new content hash as an allowed SHARED_PREFIX backing.

        The hash must be a SHA-256 hex digest of the canonical shared-prefix
        token sequence. Registration should be performed at deployment time by
        the operator, not by tenant code.
        """
        with self._lock:
            self._allow.add(prefix_hash)

    @staticmethod
    def hash_prefix(tokens: bytes) -> str:
        """Compute the canonical SHA-256 hex digest of a shared-prefix token sequence."""
        return hashlib.sha256(tokens).hexdigest()

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def admit_shared(self, block_id: int, prefix_hash: str) -> None:
        """Transition UNALLOCATED -> SHARED_PREFIX, gated on allow-list.

        Raises UnapprovedSharedHash if prefix_hash is not registered.
        Raises IllegalTransition if the block is not currently UNALLOCATED.
        """
        with self._lock:
            self._require_state(block_id, PageState.UNALLOCATED)
            if prefix_hash not in self._allow:
                raise UnapprovedSharedHash(
                    f"prefix_hash {prefix_hash[:16]}... is not in the approved "
                    "allow-list; refusing to admit block as SHARED"
                )
            self._state[block_id] = PageState.SHARED_PREFIX
            self._owner[block_id] = SHARED_TENANT_ID
            self._hash_index[block_id] = prefix_hash
            self._assert_invariants_locked()

    def admit_private(self, block_id: int, tenant_id: TenantId) -> None:
        """Transition UNALLOCATED -> PRIVATE_CONTINUATION(tenant_id).

        Use this for fresh per-tenant pages that did not derive from a shared
        prefix (e.g., the tenant's first private message in a new session).
        """
        if tenant_id == SHARED_TENANT_ID:
            raise IllegalTransition(
                f"private page must have tenant_id != SHARED_TENANT_ID; "
                f"got {tenant_id}"
            )
        with self._lock:
            self._require_state(block_id, PageState.UNALLOCATED)
            self._state[block_id] = PageState.PRIVATE_CONTINUATION
            self._owner[block_id] = tenant_id
            self._assert_invariants_locked()

    def transition_to_private(
        self,
        source_block_id: int,
        tenant_id: TenantId,
        free_dest_block_id: int,
    ) -> int:
        """Copy-on-write a SHARED_PREFIX page into a fresh PRIVATE_CONTINUATION page.

        The source SHARED_PREFIX block is left untouched (still shared with
        other tenants). The contents are copied into free_dest_block_id, which
        must currently be UNALLOCATED, and that destination is then marked
        PRIVATE_CONTINUATION(tenant_id). The destination block_id is returned
        so the caller can patch the tenant's block table to reference it.

        Raises IllegalTransition if source is not SHARED_PREFIX or destination
        is not UNALLOCATED.
        """
        if tenant_id == SHARED_TENANT_ID:
            raise IllegalTransition(
                f"COW destination must have tenant_id != SHARED_TENANT_ID; "
                f"got {tenant_id}"
            )
        with self._lock:
            self._require_state(source_block_id, PageState.SHARED_PREFIX)
            self._require_state(free_dest_block_id, PageState.UNALLOCATED)

            if self.kv_buffer is not None:
                self.kv_buffer[free_dest_block_id].copy_(
                    self.kv_buffer[source_block_id]
                )

            self._state[free_dest_block_id] = PageState.PRIVATE_CONTINUATION
            self._owner[free_dest_block_id] = tenant_id
            self._origin_block[free_dest_block_id] = source_block_id
            self._assert_invariants_locked()
            return free_dest_block_id

    def free(self, block_id: int) -> None:
        """Free a block: zeroize the KV slot, then return it to UNALLOCATED.

        Zeroization is unconditional (covers private and shared alike) and
        defensive — even though SHARED pages contain no tenant-private data,
        zeroizing on free prevents block-reuse confusion if the page was
        mistakenly admitted under an attacker-controlled hash.

        Raises IllegalTransition if the block is already UNALLOCATED.
        """
        with self._lock:
            current = int(self._state[block_id].item())
            if current == int(PageState.UNALLOCATED):
                raise IllegalTransition(
                    f"block {block_id} is already UNALLOCATED; double free"
                )
            self._state[block_id] = PageState.FREED
            if self.kv_buffer is not None:
                self.kv_buffer[block_id].zero_()
            self._state[block_id] = PageState.UNALLOCATED
            self._owner[block_id] = SHARED_TENANT_ID
            self._origin_block.pop(block_id, None)
            self._hash_index.pop(block_id, None)
            self._assert_invariants_locked()

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def state_of(self, block_id: int) -> PageState:
        """Return the current PageState of a block."""
        return PageState(int(self._state[block_id].item()))

    def owner_of(self, block_id: int) -> TenantId:
        """Return the owning tenant ID, or SHARED_TENANT_ID if shared/unallocated."""
        return int(self._owner[block_id].item())

    def origin_of(self, block_id: int) -> Optional[int]:
        """If block_id was COW'd from a shared block, return the source; else None."""
        return self._origin_block.get(block_id)

    def assert_invariants(self) -> None:
        """Public invariant check — raises LifecycleViolation if anything is off."""
        with self._lock:
            self._assert_invariants_locked()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _require_state(self, block_id: int, expected: PageState) -> None:
        actual = int(self._state[block_id].item())
        if actual != int(expected):
            raise IllegalTransition(
                f"block {block_id}: expected {expected.name}, "
                f"found {PageState(actual).name}"
            )

    def _assert_invariants_locked(self) -> None:
        # FREED is transient: should never be observed at rest after a transition.
        freed = (self._state == int(PageState.FREED)).any().item()
        if bool(freed):
            raise LifecycleViolation(
                "invariant violated: a block is in transient FREED state at rest"
            )
        # SHARED_PREFIX must have SHARED_TENANT_ID owner.
        shared_mask = self._state == int(PageState.SHARED_PREFIX)
        if shared_mask.any():
            owners = self._owner[shared_mask]
            if not bool((owners == SHARED_TENANT_ID).all().item()):
                raise LifecycleViolation(
                    "invariant violated: SHARED_PREFIX block has non-shared owner"
                )
        # PRIVATE_CONTINUATION must have a real tenant id.
        private_mask = self._state == int(PageState.PRIVATE_CONTINUATION)
        if private_mask.any():
            owners = self._owner[private_mask]
            if bool((owners == SHARED_TENANT_ID).any().item()):
                raise LifecycleViolation(
                    "invariant violated: PRIVATE_CONTINUATION block has SHARED owner"
                )
        # SHARED_PREFIX blocks must have a registered hash.
        for blk_id in range(self.num_blocks):
            st = int(self._state[blk_id].item())
            if st == int(PageState.SHARED_PREFIX) and blk_id not in self._hash_index:
                raise LifecycleViolation(
                    f"invariant violated: SHARED_PREFIX block {blk_id} has no "
                    "recorded prefix hash"
                )
