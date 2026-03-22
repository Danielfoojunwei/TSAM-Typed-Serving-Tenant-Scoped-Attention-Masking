"""
TSAM: Typed Serving — Tenant-Scoped Attention Masking

Formal information-flow isolation for multi-tenant LLM serving at unlimited scale.
Hard attention masking over typed KV-cache pages achieves perfect tenant isolation
(I(output(q_i); KV_private(t_j)) = 0 for all i != j) while eliminating the
dimensionality constraint that limits orthogonal projection schemes to 16 tenants.
"""

__version__ = "0.1.0"

from tsam.core.types import PageType, TenantId, TypedPage, SHARED_TENANT_ID
from tsam.core.block_manager import TypedBlockManager
from tsam.attention.masking import TSAMAttentionMask, generate_tsam_mask
from tsam.dai.timing_dfa import TimingDFAState, TimingDFA

__all__ = [
    "PageType",
    "TenantId",
    "TypedPage",
    "SHARED_TENANT_ID",
    "TypedBlockManager",
    "TSAMAttentionMask",
    "generate_tsam_mask",
    "TimingDFAState",
    "TimingDFA",
]
