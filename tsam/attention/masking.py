"""
TSAM Attention Masking — core isolation mechanism.

Generates and applies the type-based attention mask that enforces:
  I(output(q_i); KV_private(t_j)) = 0  for all i != j

The mask is a 1D boolean tensor over the KV sequence dimension. For each key
position j, the mask is True (attend) or False (block) based on the page
metadata of the block containing position j.

Mask generation rule (from Section 3.2 of the spec):
  for j in 0..num_keys:
      page_j = block_table[j // page_size]
      if page_type[page_j] == PRIVATE and tenant_id[page_j] != current_tenant:
          mask[j] = False   (block cross-tenant private keys)
      else:
          mask[j] = True    (allow own private + all shared)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from tsam.core.types import PageType, TenantId


@dataclass
class TSAMAttentionMask:
    """Pre-computed TSAM mask for a single query sequence.

    Attributes:
        mask:          Bool tensor, shape (seq_len_kv,). True = attend, False = block.
        query_tenant:  The tenant that owns the query.
        num_shared:    Count of SHARED key positions (for diagnostics).
        num_own:       Count of own PRIVATE key positions.
        num_blocked:   Count of blocked cross-tenant PRIVATE key positions.
    """

    mask: torch.Tensor
    query_tenant: TenantId
    num_shared: int
    num_own: int
    num_blocked: int

    @property
    def seq_len_kv(self) -> int:
        return self.mask.shape[0]


def generate_tsam_mask(
    block_table: torch.Tensor,
    page_type: torch.Tensor,
    tenant_id: torch.Tensor,
    query_tenant: TenantId,
    page_size: int,
    seq_len_kv: int,
    device: Optional[str] = None,
) -> TSAMAttentionMask:
    """Generate the TSAM attention mask for a query from *query_tenant*.

    Args:
        block_table: int32 tensor, shape (num_blocks_in_sequence,).
                     Maps logical block index to physical block ID.
        page_type:   uint8 tensor, shape (total_physical_blocks,).
                     PageType metadata for every physical block.
        tenant_id:   int32 tensor, shape (total_physical_blocks,).
                     TenantId metadata for every physical block.
        query_tenant: The tenant issuing the query.
        page_size:   Tokens per block.
        seq_len_kv:  Total number of KV positions to mask.
        device:      Target device (defaults to block_table's device).

    Returns:
        TSAMAttentionMask with the computed boolean mask.
    """
    if device is None:
        device = block_table.device

    # Map each key position to its physical block ID.
    # position j -> logical block j // page_size -> physical block via block_table.
    num_positions = seq_len_kv
    position_indices = torch.arange(num_positions, device=device)
    logical_block_indices = position_indices // page_size

    # Clamp to block_table length (handles sequences shorter than full blocks).
    logical_block_indices = logical_block_indices.clamp(max=block_table.shape[0] - 1)
    physical_block_ids = block_table[logical_block_indices]

    # Look up metadata for each position's physical block.
    pos_page_type = page_type[physical_block_ids]  # uint8
    pos_tenant_id = tenant_id[physical_block_ids]  # int32

    # Apply TSAM masking rule:
    #   block iff page_type == PRIVATE AND tenant_id != query_tenant
    is_private = pos_page_type == PageType.PRIVATE
    is_other_tenant = pos_tenant_id != query_tenant
    blocked = is_private & is_other_tenant

    mask = ~blocked  # True = attend, False = block

    num_shared = int((pos_page_type == PageType.SHARED).sum().item())
    num_own = int((is_private & ~is_other_tenant).sum().item())
    num_blocked_count = int(blocked.sum().item())

    return TSAMAttentionMask(
        mask=mask,
        query_tenant=query_tenant,
        num_shared=num_shared,
        num_own=num_own,
        num_blocked=num_blocked_count,
    )


def generate_tsam_mask_batched(
    block_tables: torch.Tensor,
    page_type: torch.Tensor,
    tenant_id: torch.Tensor,
    query_tenants: torch.Tensor,
    page_size: int,
    seq_lens_kv: torch.Tensor,
    max_seq_len_kv: int,
    device: Optional[str] = None,
) -> torch.Tensor:
    """Generate TSAM masks for a batch of queries.

    Args:
        block_tables:   int32 tensor, shape (batch_size, max_blocks_per_seq).
        page_type:      uint8 tensor, shape (total_physical_blocks,).
        tenant_id:      int32 tensor, shape (total_physical_blocks,).
        query_tenants:  int32 tensor, shape (batch_size,).
        page_size:      Tokens per block.
        seq_lens_kv:    int32 tensor, shape (batch_size,). Actual KV length per query.
        max_seq_len_kv: Maximum KV length across the batch (for padding).
        device:         Target device.

    Returns:
        Bool tensor, shape (batch_size, max_seq_len_kv). True = attend.
    """
    if device is None:
        device = block_tables.device

    batch_size = block_tables.shape[0]
    position_indices = torch.arange(max_seq_len_kv, device=device)
    logical_block_indices = position_indices // page_size  # (max_seq_len_kv,)

    # Clamp to block table width.
    max_blocks = block_tables.shape[1]
    logical_block_indices = logical_block_indices.clamp(max=max_blocks - 1)

    # Expand for batch: (batch_size, max_seq_len_kv)
    logical_expanded = logical_block_indices.unsqueeze(0).expand(batch_size, -1)

    # Gather physical block IDs: block_tables[b, logical_expanded[b, j]]
    physical_block_ids = torch.gather(block_tables, 1, logical_expanded)

    # Look up metadata.
    pos_page_type = page_type[physical_block_ids]   # (batch, max_seq_len_kv)
    pos_tenant_id = tenant_id[physical_block_ids]    # (batch, max_seq_len_kv)

    # TSAM rule.
    query_tenants_expanded = query_tenants.unsqueeze(1).expand(-1, max_seq_len_kv)
    is_private = pos_page_type == PageType.PRIVATE
    is_other_tenant = pos_tenant_id != query_tenants_expanded
    blocked = is_private & is_other_tenant

    # Also mask out positions beyond each sequence's actual KV length.
    seq_mask = position_indices.unsqueeze(0) < seq_lens_kv.unsqueeze(1)

    mask = (~blocked) & seq_mask
    return mask


def apply_tsam_mask_to_scores(
    attention_scores: torch.Tensor,
    tsam_mask: torch.Tensor,
    mask_value: float = -float("inf"),
) -> torch.Tensor:
    """Apply TSAM boolean mask to raw attention scores (before softmax).

    Args:
        attention_scores: float tensor, shape (..., seq_len_kv).
        tsam_mask:        bool tensor, broadcastable to attention_scores shape.
                          True = attend, False = block.
        mask_value:       Value to fill blocked positions (default -inf).

    Returns:
        Masked attention scores. Blocked positions set to mask_value.
    """
    return attention_scores.masked_fill(~tsam_mask, mask_value)
