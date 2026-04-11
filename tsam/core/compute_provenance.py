"""
Compute provenance tracking for shared prefix safety.

Tracks HOW each KV-cache block's content was computed, not just who owns it.
This addresses a critical gap: TSAM's Theorem 2 (Shared Prefix Safety) assumes
shared KV entries are "deterministic functions of the system prompt and model
weights." If shared blocks are computed in a batch containing private data,
the shared KV entries may be contaminated via residual connections.

Usage:
    provenance = ComputeProvenanceTracker(num_blocks=8192)

    # When computing shared prefix KV in an isolated batch:
    provenance.mark_clean(block_id=42)

    # When a block is computed in a batch containing private data:
    provenance.mark_contaminated(block_id=99, reason="computed with tenant 3 private data")

    # Before allowing a block to be accessed as SHARED:
    if not provenance.is_clean(block_id):
        raise SecurityError("Cannot share contaminated block")
"""

from __future__ import annotations

import enum
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional

from tsam.core.types import TenantId


class Provenance(enum.Enum):
    """Compute provenance of a KV-cache block's content."""

    UNKNOWN = "UNKNOWN"       # Block allocated but provenance not yet tracked.
    CLEAN = "CLEAN"           # Computed from public inputs only (system prompt + weights).
    CONTAMINATED = "CONTAMINATED"  # Computed in a batch containing private data.


@dataclass
class ProvenanceRecord:
    """Provenance record for a single block."""

    status: Provenance = Provenance.UNKNOWN
    reason: Optional[str] = None
    contaminating_tenants: Optional[set] = None


class ComputeProvenanceTracker:
    """Tracks compute provenance of KV-cache blocks.

    This ensures that blocks marked as SHARED were computed in a clean
    environment (no private data in the same batch), which is a prerequisite
    for Theorem 2's safety guarantee.
    """

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self._records: Dict[int, ProvenanceRecord] = {}
        self._lock = threading.Lock()

    def mark_clean(self, block_id: int) -> None:
        """Mark a block as cleanly computed (public inputs only)."""
        with self._lock:
            self._records[block_id] = ProvenanceRecord(status=Provenance.CLEAN)

    def mark_contaminated(
        self,
        block_id: int,
        reason: str = "",
        tenant_ids: Optional[set] = None,
    ) -> None:
        """Mark a block as contaminated (computed with private data present)."""
        with self._lock:
            self._records[block_id] = ProvenanceRecord(
                status=Provenance.CONTAMINATED,
                reason=reason,
                contaminating_tenants=tenant_ids,
            )

    def is_clean(self, block_id: int) -> bool:
        """Check if a block has verified clean provenance."""
        with self._lock:
            record = self._records.get(block_id)
            return record is not None and record.status == Provenance.CLEAN

    def is_contaminated(self, block_id: int) -> bool:
        """Check if a block is known to be contaminated."""
        with self._lock:
            record = self._records.get(block_id)
            return record is not None and record.status == Provenance.CONTAMINATED

    def get_status(self, block_id: int) -> Provenance:
        """Get the provenance status of a block."""
        with self._lock:
            record = self._records.get(block_id)
            return record.status if record else Provenance.UNKNOWN

    def assert_clean_for_sharing(self, block_id: int) -> None:
        """Assert that a block is safe to use as SHARED.

        Raises:
            ValueError: If the block is contaminated or has unknown provenance.
        """
        status = self.get_status(block_id)
        if status == Provenance.CONTAMINATED:
            record = self._records.get(block_id)
            reason = record.reason if record else "unknown"
            raise ValueError(
                f"Block {block_id} cannot be shared: contaminated ({reason}). "
                f"Theorem 2 requires shared blocks to be computed from "
                f"public inputs only."
            )
        if status == Provenance.UNKNOWN:
            raise ValueError(
                f"Block {block_id} has unknown provenance. "
                f"Mark it as CLEAN via mark_clean() before sharing. "
                f"Shared blocks must have verified clean provenance for "
                f"Theorem 2 to hold."
            )

    def clear(self, block_id: int) -> None:
        """Remove provenance record (e.g., when block is freed)."""
        with self._lock:
            self._records.pop(block_id, None)

    def validate_shared_blocks(self, shared_block_ids: list) -> bool:
        """Validate that all shared blocks have clean provenance.

        Returns True if all blocks are clean, False otherwise.
        """
        for block_id in shared_block_ids:
            if not self.is_clean(block_id):
                return False
        return True
