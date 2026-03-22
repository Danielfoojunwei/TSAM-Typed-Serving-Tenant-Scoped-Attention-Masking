"""
TSAM Triton Flash Attention kernel — production kernel for GPU execution.

This implements the TSAM-modified Flash Attention kernel using Triton, applying
the tenant isolation mask directly within the tiled attention computation.

The mask is applied as an additional 1D boolean mask over the KV sequence
dimension, alongside the causal mask, before the softmax operation.

For vLLM integration: this kernel replaces the standard Flash Attention forward
pass when TSAM is enabled. The mask_k argument is generated from block table
metadata on the host side.
"""

from __future__ import annotations

from typing import Optional

import torch

# Triton import — this is optional; the module gracefully degrades if Triton
# is not available (e.g., on CPU-only machines).
try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _tsam_flash_attention_fwd_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        Out_ptr,
        Mask_ptr,
        stride_qb,
        stride_qh,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vd,
        stride_ob,
        stride_oh,
        stride_od,
        stride_mask_b,
        seq_len_q,
        seq_len_kv,
        head_dim: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        scale,
        causal: tl.constexpr,
    ):
        """TSAM Flash Attention forward kernel.

        Computes:
            O = softmax(Q @ K^T * scale + tsam_mask + causal_mask) @ V

        where tsam_mask sets blocked positions to -inf.
        """
        pid_batch_head = tl.program_id(0)
        pid_m = tl.program_id(1)

        # Batch and head indices.
        # Assume Q/K/V are laid out as (batch*num_heads, seq_len, head_dim).
        bh = pid_batch_head

        # Offsets for the M block of queries.
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, head_dim)

        # Load Q block: (BLOCK_M, head_dim).
        q_ptrs = Q_ptr + bh * stride_qb + offs_m[:, None] * stride_qh + offs_d[None, :] * stride_qd
        q_mask = offs_m[:, None] < seq_len_q
        q = tl.load(q_ptrs, mask=q_mask, other=0.0)

        # Running softmax accumulators.
        m_i = tl.full([BLOCK_M], value=float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        o_i = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)

        # Iterate over KV blocks.
        for start_n in range(0, seq_len_kv, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)

            # Load K block: (BLOCK_N, head_dim).
            k_ptrs = K_ptr + bh * stride_kb + offs_n[:, None] * stride_kh + offs_d[None, :] * stride_kd
            k_mask = offs_n[:, None] < seq_len_kv
            k = tl.load(k_ptrs, mask=k_mask, other=0.0)

            # Compute attention scores: (BLOCK_M, BLOCK_N).
            scores = tl.dot(q, tl.trans(k)) * scale

            # === TSAM MASK APPLICATION ===
            # Load per-position mask: shape (BLOCK_N,), True = attend.
            mask_ptrs = Mask_ptr + bh * stride_mask_b + offs_n
            tsam_mask = tl.load(mask_ptrs, mask=offs_n < seq_len_kv, other=False)
            # Broadcast to (BLOCK_M, BLOCK_N) and apply.
            tsam_mask_2d = tsam_mask[None, :]
            scores = tl.where(tsam_mask_2d, scores, float("-inf"))

            # Causal mask.
            if causal:
                causal_mask = offs_m[:, None] >= offs_n[None, :]
                scores = tl.where(causal_mask, scores, float("-inf"))

            # KV boundary mask.
            kv_mask = offs_n[None, :] < seq_len_kv
            scores = tl.where(kv_mask, scores, float("-inf"))

            # Online softmax update.
            m_ij = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            beta = tl.exp(m_ij - m_new)
            l_i = l_i * alpha + tl.sum(tl.exp(scores - m_ij[:, None]) * beta[:, None], axis=1)
            o_i = o_i * alpha[:, None]

            # Compute attention weights for this block.
            p = tl.exp(scores - m_new[:, None])

            # Load V block: (BLOCK_N, head_dim).
            v_ptrs = V_ptr + bh * stride_vb + offs_n[:, None] * stride_vh + offs_d[None, :] * stride_vd
            v = tl.load(v_ptrs, mask=offs_n[:, None] < seq_len_kv, other=0.0)

            # Accumulate output.
            o_i += tl.dot(p.to(v.dtype), v)
            m_i = m_new

        # Normalize output.
        o_i = o_i / l_i[:, None]

        # Store output.
        out_ptrs = Out_ptr + bh * stride_ob + offs_m[:, None] * stride_oh + offs_d[None, :] * stride_od
        out_mask = offs_m[:, None] < seq_len_q
        tl.store(out_ptrs, o_i.to(Out_ptr.dtype.element_ty), mask=out_mask)


def tsam_flash_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    tsam_mask: torch.Tensor,
    causal: bool = True,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Launch the TSAM Flash Attention Triton kernel.

    Args:
        query:     (batch*num_heads, seq_len_q, head_dim) float16/bfloat16.
        key:       (batch*num_heads, seq_len_kv, head_dim) float16/bfloat16.
        value:     (batch*num_heads, seq_len_kv, head_dim) float16/bfloat16.
        tsam_mask: (batch*num_heads, seq_len_kv) bool. True = attend.
        causal:    Apply causal masking.
        scale:     Attention scale (default 1/sqrt(head_dim)).

    Returns:
        (batch*num_heads, seq_len_q, head_dim).
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError(
            "Triton is not available. Install with: pip install triton>=2.3.0"
        )

    bh, seq_len_q, head_dim = query.shape
    _, seq_len_kv, _ = key.shape

    if scale is None:
        import math
        scale = 1.0 / math.sqrt(head_dim)

    output = torch.empty_like(query)

    BLOCK_M = 64
    BLOCK_N = 64
    # Pad head_dim to power of 2 for Triton constexpr.
    head_dim_padded = triton.next_power_of_2(head_dim)

    grid = (bh, triton.cdiv(seq_len_q, BLOCK_M))

    _tsam_flash_attention_fwd_kernel[grid](
        query,
        key,
        value,
        output,
        tsam_mask,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        tsam_mask.stride(0),
        seq_len_q,
        seq_len_kv,
        head_dim=head_dim_padded,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        scale=scale,
        causal=causal,
    )

    return output
