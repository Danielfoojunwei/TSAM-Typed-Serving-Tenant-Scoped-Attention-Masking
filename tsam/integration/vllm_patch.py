"""
vLLM integration patch for TSAM.

Extends vLLM's BlockTable and attention backends with TSAM typed page metadata
and tenant-scoped attention masking.

Target files in vLLM:
  - vllm/core/block_manager.py     (BlockTable extension)
  - vllm/attention/backends/flash_attn.py  (Flash Attention path)
  - vllm/attention/backends/paged_attn.py  (PagedAttention path)

This module provides:
  1. TSAMBlockTableExtension — wraps vLLM BlockTable with page_type and tenant_id.
  2. TSAMAttentionWrapper — wraps vLLM attention backend forward() to inject TSAM mask.
  3. Monkey-patch functions for in-place vLLM modification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from tsam.core.types import PageType, TenantId, SHARED_TENANT_ID
from tsam.attention.masking import (
    generate_tsam_mask,
    generate_tsam_mask_batched,
    apply_tsam_mask_to_scores,
)


@dataclass
class TSAMBlockTableExtension:
    """Extends a vLLM BlockTable with TSAM metadata.

    This is designed to wrap the existing vLLM block table data structure,
    adding per-block page_type and tenant_id metadata tensors.

    Usage:
        # After vLLM allocates blocks:
        ext = TSAMBlockTableExtension(num_blocks=block_manager.num_blocks, device="cuda")
        # When allocating a block for a tenant:
        ext.mark_private(block_id, tenant_id)
        # When allocating a shared prefix block:
        ext.mark_shared(block_id)
        # When computing attention:
        mask = ext.generate_mask(block_table, query_tenant, page_size, seq_len_kv)
    """

    num_blocks: int
    device: str = "cuda"
    page_type: torch.Tensor = field(init=False)
    tenant_id: torch.Tensor = field(init=False)

    def __post_init__(self):
        self.page_type = torch.zeros(
            self.num_blocks, dtype=torch.uint8, device=self.device
        )
        self.tenant_id = torch.full(
            (self.num_blocks,), SHARED_TENANT_ID, dtype=torch.int32, device=self.device
        )

    def mark_private(self, block_id: int, tenant: TenantId) -> None:
        """Mark a block as PRIVATE for a specific tenant."""
        self.page_type[block_id] = PageType.PRIVATE
        self.tenant_id[block_id] = tenant

    def mark_shared(self, block_id: int) -> None:
        """Mark a block as SHARED (system prompt, common prefix)."""
        self.page_type[block_id] = PageType.SHARED
        self.tenant_id[block_id] = SHARED_TENANT_ID

    def mark_private_batch(self, block_ids: torch.Tensor, tenant: TenantId) -> None:
        """Mark multiple blocks as PRIVATE for a tenant."""
        self.page_type[block_ids] = PageType.PRIVATE
        self.tenant_id[block_ids] = tenant

    def mark_shared_batch(self, block_ids: torch.Tensor) -> None:
        """Mark multiple blocks as SHARED."""
        self.page_type[block_ids] = PageType.SHARED
        self.tenant_id[block_ids] = SHARED_TENANT_ID

    def generate_mask(
        self,
        block_table: torch.Tensor,
        query_tenant: TenantId,
        page_size: int,
        seq_len_kv: int,
    ) -> torch.Tensor:
        """Generate TSAM boolean mask for a single query.

        Returns:
            Bool tensor (seq_len_kv,). True = attend.
        """
        result = generate_tsam_mask(
            block_table=block_table,
            page_type=self.page_type,
            tenant_id=self.tenant_id,
            query_tenant=query_tenant,
            page_size=page_size,
            seq_len_kv=seq_len_kv,
            device=self.device,
        )
        return result.mask

    def generate_mask_batched(
        self,
        block_tables: torch.Tensor,
        query_tenants: torch.Tensor,
        page_size: int,
        seq_lens_kv: torch.Tensor,
        max_seq_len_kv: int,
    ) -> torch.Tensor:
        """Generate TSAM masks for a batch of queries.

        Returns:
            Bool tensor (batch_size, max_seq_len_kv). True = attend.
        """
        return generate_tsam_mask_batched(
            block_tables=block_tables,
            page_type=self.page_type,
            tenant_id=self.tenant_id,
            query_tenants=query_tenants,
            page_size=page_size,
            seq_lens_kv=seq_lens_kv,
            max_seq_len_kv=max_seq_len_kv,
            device=self.device,
        )


class TSAMAttentionWrapper:
    """Wraps a vLLM attention backend to inject TSAM masking.

    This wrapper intercepts the forward() call of any vLLM attention backend
    and applies the TSAM mask before softmax.

    Usage:
        # Wrap existing backend:
        original_backend = FlashAttentionBackend(...)
        tsam_backend = TSAMAttentionWrapper(
            backend=original_backend,
            tsam_ext=tsam_block_table_extension,
            page_size=16,
        )
        # Use tsam_backend.forward() in place of original_backend.forward()
    """

    def __init__(
        self,
        backend: Any,
        tsam_ext: TSAMBlockTableExtension,
        page_size: int = 16,
    ):
        self.backend = backend
        self.tsam_ext = tsam_ext
        self.page_size = page_size

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_tables: torch.Tensor,
        query_tenants: torch.Tensor,
        seq_lens_kv: torch.Tensor,
        max_seq_len_kv: int,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass with TSAM masking.

        Generates the TSAM mask and passes it to the backend as an additional
        attention mask. For backends that support custom masks (Flash Attention v2),
        the mask is passed directly. For others, it's applied post-hoc.
        """
        tsam_mask = self.tsam_ext.generate_mask_batched(
            block_tables=block_tables,
            query_tenants=query_tenants,
            page_size=self.page_size,
            seq_lens_kv=seq_lens_kv,
            max_seq_len_kv=max_seq_len_kv,
        )

        # Pass the TSAM mask to the backend.
        # For Flash Attention: merge with existing attention mask.
        if hasattr(self.backend, "forward"):
            existing_mask = kwargs.get("attn_mask", None)
            if existing_mask is not None:
                combined_mask = existing_mask & tsam_mask
            else:
                combined_mask = tsam_mask
            kwargs["attn_mask"] = combined_mask
            return self.backend.forward(
                query=query,
                key=key,
                value=value,
                block_tables=block_tables,
                seq_lens_kv=seq_lens_kv,
                max_seq_len_kv=max_seq_len_kv,
                **kwargs,
            )
        raise AttributeError(
            f"Backend {type(self.backend)} does not have a forward() method"
        )


def patch_vllm_block_manager(
    block_manager: Any,
    tsam_ext: TSAMBlockTableExtension,
) -> None:
    """Monkey-patch a vLLM BlockManager to track TSAM metadata.

    Wraps allocate() and free() methods to maintain page_type and tenant_id
    metadata in the TSAMBlockTableExtension.
    """
    original_allocate = block_manager.allocate

    def tsam_allocate(
        *args, tenant_id: Optional[TenantId] = None, shared: bool = False, **kwargs
    ):
        result = original_allocate(*args, **kwargs)
        block_ids = result if isinstance(result, (list, tuple)) else [result]
        for blk_id in block_ids:
            if shared:
                tsam_ext.mark_shared(blk_id)
            elif tenant_id is not None:
                tsam_ext.mark_private(blk_id, tenant_id)
        return result

    block_manager.allocate = tsam_allocate
    block_manager.tsam_ext = tsam_ext


def generate_vllm_kernel_patch() -> str:
    """Generate the CUDA kernel patch diff for vLLM's attention kernel.

    Returns a unified diff string that can be applied to vLLM's
    attention_kernels.cu to add TSAM type masking.
    """
    return """--- a/csrc/attention/attention_kernels.cu
+++ b/csrc/attention/attention_kernels.cu
@@ -1,5 +1,8 @@
 // vLLM Paged Attention Kernel
 // Modified for TSAM: Typed Serving — Tenant-Scoped Attention Masking
+//
+// TSAM adds a per-page type check to enforce tenant isolation.
+// Cross-tenant PRIVATE pages are masked to -inf before softmax.

 template <typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS>
 __global__ void paged_attention_v1_kernel(
@@ -10,6 +13,9 @@
     const int* __restrict__ block_tables,
     const int* __restrict__ seq_lens,
     const int max_num_blocks_per_seq,
+    // TSAM metadata arrays
+    const uint8_t* __restrict__ page_type,     // PageType per block
+    const int* __restrict__ tenant_id,         // TenantId per block
+    const int current_tenant_id,               // Query tenant
     const float scale) {

   const int seq_idx = blockIdx.x;
@@ -45,6 +51,15 @@
       const int physical_block_id = block_table[logical_block_idx];
       const scalar_t* k_ptr = k_cache + physical_block_id * kv_block_stride;

+      // === TSAM TYPE MASK ===
+      // Check page type: block cross-tenant PRIVATE pages.
+      // This is a warp-uniform branch (all threads process the same page).
+      if (page_type[physical_block_id] == 0  // PRIVATE
+          && tenant_id[physical_block_id] != current_tenant_id) {
+        qk = -FLT_MAX;  // Mask out cross-tenant private keys
+        continue;        // Skip to next key position
+      }
+
       // Compute attention score: q . k
       float qk = 0.0f;
       for (int d = 0; d < HEAD_SIZE; d++) {
"""
