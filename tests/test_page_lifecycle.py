"""
Unit tests for tsam.core.page_lifecycle.

These tests are CPU-only and use a small synthetic KV buffer so zeroize
behavior can be observed directly. They do not import GPU code or initialize
CUDA contexts.

Coverage:
  - admission requires hash registration
  - state machine refuses illegal transitions
  - copy-on-write produces a fresh PRIVATE_CONTINUATION without disturbing source
  - free zeroizes the KV slot before returning to UNALLOCATED
  - invariants catch direct tensor tampering
"""

from __future__ import annotations

import hashlib

import pytest
import torch

from tsam.core.page_lifecycle import (
    IllegalTransition,
    LifecycleViolation,
    PageLifecycle,
    PageState,
    UnapprovedSharedHash,
)


def _make_lifecycle(num_blocks: int = 8, page_size: int = 4, head_dim: int = 2):
    kv = torch.randn(num_blocks, page_size, head_dim, dtype=torch.float32)
    return PageLifecycle(num_blocks=num_blocks, page_size=page_size, kv_buffer=kv), kv


def test_unapproved_shared_hash_is_refused():
    lc, _ = _make_lifecycle()
    rogue_hash = hashlib.sha256(b"attacker-controlled").hexdigest()
    with pytest.raises(UnapprovedSharedHash):
        lc.admit_shared(block_id=0, prefix_hash=rogue_hash)


def test_approved_shared_hash_is_admitted():
    from tsam.core.page_lifecycle import PageLifecycle, PageState

    lc, _ = _make_lifecycle()
    h = hashlib.sha256(b"You are a helpful assistant.").hexdigest()
    lc.admit_prefix_hash(h)
    lc.admit_shared(block_id=0, prefix_hash=h)
    assert lc.state_of(0) == PageState.SHARED_PREFIX


def test_admit_private_requires_real_tenant():
    from tsam.core.page_lifecycle import IllegalTransition

    lc, _ = _make_lifecycle()
    with pytest.raises(IllegalTransition):
        lc.admit_private(block_id=0, tenant_id=-1)  # SHARED_TENANT_ID is reserved


def test_double_admit_is_refused():
    from tsam.core.page_lifecycle import IllegalTransition

    lc, _ = _make_lifecycle()
    lc.admit_private(block_id=0, tenant_id=42)
    with pytest.raises(IllegalTransition):
        lc.admit_private(block_id=0, tenant_id=42)


def test_cow_copies_content_and_does_not_disturb_source():
    from tsam.core.page_lifecycle import PageState

    lc, kv = _make_lifecycle()

    h = hashlib.sha256(b"system prompt").hexdigest()
    lc.admit_prefix_hash(h)
    lc.admit_shared(block_id=0, prefix_hash=h)

    source_snapshot = kv[0].clone()

    new_id = lc.transition_to_private(
        source_block_id=0, tenant_id=7, free_dest_block_id=1
    )

    assert new_id == 1
    assert lc.state_of(1) == PageState.PRIVATE_CONTINUATION
    assert lc.owner_of(1) == 7
    assert lc.origin_of(1) == 0

    assert lc.state_of(0) == PageState.SHARED_PREFIX
    assert torch.equal(kv[0], source_snapshot)
    assert torch.equal(kv[1], source_snapshot)


def test_free_zeroizes_kv_slot():
    lc, kv = _make_lifecycle()
    lc.admit_private(block_id=2, tenant_id=99)
    assert kv[2].abs().sum().item() > 0  # has random content
    lc.free(block_id=2)
    assert torch.all(kv[2] == 0)


def test_free_unallocated_is_refused():
    from tsam.core.page_lifecycle import IllegalTransition

    lc, _ = _make_lifecycle()
    with pytest.raises(IllegalTransition):
        lc.free(block_id=0)


def test_invariants_catch_direct_tampering():
    from tsam.core.page_lifecycle import LifecycleViolation, PageState

    lc, _ = _make_lifecycle()
    lc.admit_private(block_id=0, tenant_id=5)
    # Bypass the API and tamper directly with the metadata tensor.
    lc._state[0] = int(PageState.SHARED_PREFIX)  # type: ignore[index]
    with pytest.raises(LifecycleViolation):
        lc.assert_invariants()


def test_freed_block_can_be_re_admitted():
    from tsam.core.page_lifecycle import PageState

    lc, kv = _make_lifecycle()
    lc.admit_private(block_id=3, tenant_id=11)
    lc.free(block_id=3)

    h = hashlib.sha256(b"reused prompt").hexdigest()
    lc.admit_prefix_hash(h)
    lc.admit_shared(block_id=3, prefix_hash=h)
    assert lc.state_of(3) == PageState.SHARED_PREFIX
    # KV slot was zeroized, so reuse cannot leak prior tenant content.
    assert torch.all(kv[3] == 0)
