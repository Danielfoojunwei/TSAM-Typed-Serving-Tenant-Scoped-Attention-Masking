"""
TSAM-patched PagedAttention — production attention computation with tenant isolation.

This module provides a full attention forward pass that integrates TSAM masking
directly into the computation, mirroring what the modified vLLM CUDA kernel does.

The implementation:
1. Computes Q @ K^T attention scores
2. Applies causal mask
3. Applies TSAM tenant isolation mask (cross-tenant PRIVATE → -inf)
4. Applies softmax
5. Computes attention_output = weights @ V

This is the reference implementation; the production CUDA/Triton kernel applies
the same logic fused into the attention kernel inner loop.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from tsam.attention.masking import (
    generate_tsam_mask,
    generate_tsam_mask_batched,
    apply_tsam_mask_to_scores,
)
from tsam.core.types import PageType, TenantId


def tsam_paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    page_type: torch.Tensor,
    tenant_id_meta: torch.Tensor,
    query_tenant: TenantId,
    page_size: int,
    seq_len_kv: int,
    causal: bool = True,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Single-query TSAM paged attention forward pass.

    Args:
        query:          (num_heads, head_dim) or (1, num_heads, head_dim).
        key_cache:      (total_blocks, page_size, num_kv_heads, head_dim).
        value_cache:    (total_blocks, page_size, num_kv_heads, head_dim).
        block_table:    int32 (num_blocks_in_seq,). Logical-to-physical mapping.
        page_type:      uint8 (total_blocks,). TSAM page type metadata.
        tenant_id_meta: int32 (total_blocks,). TSAM tenant ID metadata.
        query_tenant:   Tenant issuing this query.
        page_size:      Tokens per block.
        seq_len_kv:     Number of active KV positions.
        causal:         Apply causal masking.
        scale:          Attention scale (default 1/sqrt(head_dim)).

    Returns:
        attention_output: (num_heads, head_dim) or (1, num_heads, head_dim).
    """
    squeeze = False
    if query.dim() == 2:
        query = query.unsqueeze(0)  # (1, num_heads, head_dim)
        squeeze = True

    num_heads = query.shape[1]
    head_dim = query.shape[2]
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Reconstruct KV from paged cache using block table.
    num_blocks_in_seq = block_table.shape[0]
    physical_ids = block_table[:num_blocks_in_seq]

    # Gather key and value blocks: (num_blocks, page_size, num_kv_heads, head_dim)
    keys = key_cache[physical_ids]
    values = value_cache[physical_ids]

    # Reshape to flat sequence: (seq_len_kv, num_kv_heads, head_dim)
    total_positions = num_blocks_in_seq * page_size
    keys = keys.reshape(total_positions, -1, head_dim)[:seq_len_kv]
    values = values.reshape(total_positions, -1, head_dim)[:seq_len_kv]

    num_kv_heads = keys.shape[1]

    # GQA/MQA: repeat KV heads to match query heads.
    num_groups = num_heads // num_kv_heads
    if num_groups > 1:
        keys = keys.unsqueeze(2).expand(-1, -1, num_groups, -1).reshape(seq_len_kv, num_heads, head_dim)
        values = values.unsqueeze(2).expand(-1, -1, num_groups, -1).reshape(seq_len_kv, num_heads, head_dim)
    # keys/values: (seq_len_kv, num_heads, head_dim)

    # Compute attention scores: query (1, num_heads, head_dim) @ keys^T
    # -> (num_heads, 1, seq_len_kv)
    q = query.permute(1, 0, 2)       # (num_heads, 1, head_dim)
    k = keys.permute(1, 2, 0)        # (num_heads, head_dim, seq_len_kv)
    scores = torch.bmm(q, k) * scale  # (num_heads, 1, seq_len_kv)

    # Generate and apply TSAM mask.
    tsam_mask = generate_tsam_mask(
        block_table=block_table,
        page_type=page_type,
        tenant_id=tenant_id_meta,
        query_tenant=query_tenant,
        page_size=page_size,
        seq_len_kv=seq_len_kv,
        device=query.device,
    )

    # Expand mask to (1, 1, seq_len_kv) for broadcasting with (num_heads, 1, seq_len_kv).
    mask_expanded = tsam_mask.mask.unsqueeze(0).unsqueeze(0)
    scores = apply_tsam_mask_to_scores(scores, mask_expanded)

    # Causal mask (for decode, query position = seq_len_kv - 1; all prior positions visible).
    if causal:
        # For single-token decode, all positions < seq_len_kv are causally visible.
        # For prefill with seq_len_q > 1 this would need a triangular mask.
        pass  # No-op for single-token decode.

    # Softmax.
    weights = F.softmax(scores, dim=-1)  # (num_heads, 1, seq_len_kv)

    # Weighted sum of values.
    v = values.permute(1, 0, 2)  # (num_heads, seq_len_kv, head_dim)
    output = torch.bmm(weights, v)  # (num_heads, 1, head_dim)
    output = output.permute(1, 0, 2)  # (1, num_heads, head_dim)

    if squeeze:
        output = output.squeeze(0)

    return output


def tsam_paged_attention_batched(
    queries: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    page_type: torch.Tensor,
    tenant_id_meta: torch.Tensor,
    query_tenants: torch.Tensor,
    page_size: int,
    seq_lens_kv: torch.Tensor,
    causal: bool = True,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Batched TSAM paged attention.

    Args:
        queries:        (batch_size, num_heads, head_dim).
        key_cache:      (total_blocks, page_size, num_kv_heads, head_dim).
        value_cache:    (total_blocks, page_size, num_kv_heads, head_dim).
        block_tables:   int32 (batch_size, max_blocks_per_seq).
        page_type:      uint8 (total_blocks,).
        tenant_id_meta: int32 (total_blocks,).
        query_tenants:  int32 (batch_size,).
        page_size:      Tokens per block.
        seq_lens_kv:    int32 (batch_size,). Actual KV length per query.
        causal:         Apply causal masking.
        scale:          Attention scale.

    Returns:
        (batch_size, num_heads, head_dim).
    """
    batch_size = queries.shape[0]
    outputs = []
    for b in range(batch_size):
        out = tsam_paged_attention(
            query=queries[b].unsqueeze(0),
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_tables[b],
            page_type=page_type,
            tenant_id_meta=tenant_id_meta,
            query_tenant=int(query_tenants[b].item()),
            page_size=page_size,
            seq_len_kv=int(seq_lens_kv[b].item()),
            causal=causal,
            scale=scale,
        )
        outputs.append(out)
    return torch.cat(outputs, dim=0)
