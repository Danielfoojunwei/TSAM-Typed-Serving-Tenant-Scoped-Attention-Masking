"""
TSAMTransformer — a minimal multi-layer transformer with TSAM isolation at every layer.

This module implements a complete transformer stack where:
  1. Each attention layer enforces TSAM masking independently
  2. MLP, LayerNorm, and residual connections are shared across tenants
  3. The end-to-end output can be measured for cross-tenant information flow

The key theoretical question: does per-layer attention isolation guarantee
end-to-end noninterference through the full model?

Answer: YES, under precisely characterized conditions (Theorem 3 in our analysis).
The conditions are:
  C1. Shared prefix KV is computed from public inputs only (clean provenance).
  C2. TSAM masking is applied at every attention layer (no unmasked layers).
  C3. Model weights are frozen and shared (no per-tenant weight modification).

Under C1-C3, the residual stream for tenant t_i at every layer depends only on:
  - The shared prefix tokens (public)
  - Tenant t_i's own private tokens
  - The (shared) model weights
and is structurally independent of any other tenant's private data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsam.core.types import PageType, TenantId, SHARED_TENANT_ID
from tsam.core.block_manager import TypedBlockManager
from tsam.attention.masking import generate_tsam_mask, apply_tsam_mask_to_scores


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TSAMTransformerConfig:
    """Configuration for a TSAM-isolated multi-layer transformer."""

    num_layers: int = 6
    d_model: int = 256
    num_heads: int = 4
    d_ff: int = 1024          # Feed-forward hidden dimension.
    dropout: float = 0.0      # Disabled for deterministic isolation testing.
    page_size: int = 16
    max_seq_len: int = 512
    vocab_size: int = 1000    # Small vocab for testing.

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads


# ---------------------------------------------------------------------------
# TSAM-Isolated Attention Layer
# ---------------------------------------------------------------------------

class TSAMMultiHeadAttention(nn.Module):
    """Multi-head attention with TSAM tenant isolation.

    This is a standard MHA layer that accepts TSAM metadata and applies
    per-tenant masking before softmax at every forward pass.
    """

    def __init__(self, config: TSAMTransformerConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        tsam_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with optional TSAM masking.

        Args:
            x: (batch_size, seq_len, d_model)
            tsam_mask: (batch_size, seq_len) bool. True=attend, False=block.
                       If None, no tenant masking is applied (vanilla mode).

        Returns:
            (batch_size, seq_len, d_model)
        """
        B, S, D = x.shape
        H = self.num_heads
        Dh = self.head_dim

        # Project Q, K, V.
        q = self.q_proj(x).reshape(B, S, H, Dh).permute(0, 2, 1, 3)  # (B, H, S, Dh)
        k = self.k_proj(x).reshape(B, S, H, Dh).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, S, H, Dh).permute(0, 2, 1, 3)

        # Attention scores: (B, H, S, S).
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Causal mask.
        causal = torch.tril(torch.ones(S, S, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal.unsqueeze(0).unsqueeze(0), -float("inf"))

        # TSAM mask: block cross-tenant private positions.
        if tsam_mask is not None:
            # tsam_mask: (B, S) -> (B, 1, 1, S) for broadcasting over (B, H, S_q, S_kv).
            mask_4d = tsam_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask_4d, -float("inf"))

        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)  # (B, H, S, Dh)
        out = out.permute(0, 2, 1, 3).reshape(B, S, D)

        return self.o_proj(out)


# ---------------------------------------------------------------------------
# Transformer Layer (Attention + MLP + LayerNorm + Residual)
# ---------------------------------------------------------------------------

class TSAMTransformerLayer(nn.Module):
    """Single transformer layer with TSAM-isolated attention."""

    def __init__(self, config: TSAMTransformerConfig):
        super().__init__()
        self.attention = TSAMMultiHeadAttention(config)
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff, bias=False),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model, bias=False),
        )

    def forward(
        self,
        x: torch.Tensor,
        tsam_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through one transformer layer.

        The information flow is:
            h = x + Attention(LayerNorm(x), tsam_mask)     [attention sublayer]
            out = h + MLP(LayerNorm(h))                     [feed-forward sublayer]

        Under TSAM, the attention sublayer is isolated. The MLP and LayerNorm
        operate element-wise on each sequence position independently, so they
        do NOT introduce cross-sequence information flow within a batch.

        Critical insight: LayerNorm is applied per-sequence (not cross-batch),
        and MLP is applied per-position. Neither introduces cross-tenant leakage.
        """
        # Pre-norm attention with residual.
        h = x + self.attention(self.ln1(x), tsam_mask=tsam_mask)
        # Pre-norm MLP with residual.
        out = h + self.mlp(self.ln2(h))
        return out


# ---------------------------------------------------------------------------
# Full Multi-Layer Transformer
# ---------------------------------------------------------------------------

class TSAMTransformer(nn.Module):
    """Multi-layer transformer with TSAM isolation at every attention layer.

    This is the model used to empirically test whether per-layer attention
    masking guarantees end-to-end noninterference through the full stack.
    """

    def __init__(self, config: TSAMTransformerConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.layers = nn.ModuleList(
            [TSAMTransformerLayer(config) for _ in range(config.num_layers)]
        )
        self.ln_final = nn.LayerNorm(config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        tsam_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the full transformer stack.

        Args:
            input_ids: (batch_size, seq_len) long tensor.
            tsam_mask: (batch_size, seq_len) bool tensor.
                       True=attend (shared or own private).
                       False=block (cross-tenant private).

        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        B, S = input_ids.shape
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        for layer in self.layers:
            x = layer(x, tsam_mask=tsam_mask)

        x = self.ln_final(x)
        logits = self.output_proj(x)
        return logits

    def get_hidden_states(
        self,
        input_ids: torch.Tensor,
        tsam_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Forward pass returning hidden states at every layer.

        Returns a list of (batch_size, seq_len, d_model) tensors,
        one per layer plus the final output.
        """
        B, S = input_ids.shape
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        hidden_states = [x.detach().clone()]

        for layer in self.layers:
            x = layer(x, tsam_mask=tsam_mask)
            hidden_states.append(x.detach().clone())

        return hidden_states


# ---------------------------------------------------------------------------
# Multi-Tenant Batch Construction
# ---------------------------------------------------------------------------

@dataclass
class MultiTenantBatch:
    """A batch of sequences from multiple tenants with TSAM metadata.

    Each sequence has: [shared_prefix | private_tokens]
    The TSAM mask ensures tenant i can only see the shared prefix and
    its own private tokens, not other tenants' private tokens.
    """

    input_ids: torch.Tensor          # (batch_size, seq_len) long
    tsam_mask: torch.Tensor          # (batch_size, seq_len) bool
    tenant_ids: List[TenantId]       # Which tenant owns each batch element
    shared_len: int                  # Length of shared prefix
    private_lens: List[int]          # Length of each tenant's private region
    seq_len: int                     # Total sequence length


def build_multi_tenant_batch(
    num_tenants: int,
    shared_prefix: torch.Tensor,
    private_sequences: Dict[TenantId, torch.Tensor],
    max_seq_len: int,
    device: str = "cpu",
) -> MultiTenantBatch:
    """Construct a multi-tenant batch with TSAM masks.

    Args:
        num_tenants: Number of tenants.
        shared_prefix: (shared_len,) long tensor — common system prompt tokens.
        private_sequences: {tenant_id: (private_len,) long tensor} — per-tenant private tokens.
        max_seq_len: Pad all sequences to this length.
        device: Target device.

    Returns:
        MultiTenantBatch with input_ids and tsam_mask ready for the model.
    """
    shared_len = shared_prefix.shape[0]
    batch_input_ids = []
    batch_masks = []
    tenant_ids = []
    private_lens = []

    # Each batch element is: [shared_prefix | own_private | other_tenant_privates | padding]
    # The TSAM mask blocks other tenants' private regions.
    all_tenant_ids = sorted(private_sequences.keys())

    for owner_tid in all_tenant_ids:
        # Build the full sequence: shared + all privates (owner first, then others).
        tokens = [shared_prefix]
        mask_parts = [torch.ones(shared_len, dtype=torch.bool, device=device)]  # Shared: visible

        # Owner's private tokens (visible).
        own_private = private_sequences[owner_tid]
        tokens.append(own_private)
        mask_parts.append(torch.ones(own_private.shape[0], dtype=torch.bool, device=device))
        private_lens.append(own_private.shape[0])

        # Other tenants' private tokens (blocked by TSAM).
        for other_tid in all_tenant_ids:
            if other_tid == owner_tid:
                continue
            other_private = private_sequences[other_tid]
            tokens.append(other_private)
            mask_parts.append(torch.zeros(other_private.shape[0], dtype=torch.bool, device=device))

        seq = torch.cat(tokens)
        mask = torch.cat(mask_parts)

        # Pad to max_seq_len.
        pad_len = max_seq_len - seq.shape[0]
        if pad_len > 0:
            seq = F.pad(seq, (0, pad_len), value=0)
            mask = F.pad(mask, (0, pad_len), value=False)
        else:
            seq = seq[:max_seq_len]
            mask = mask[:max_seq_len]

        batch_input_ids.append(seq)
        batch_masks.append(mask)
        tenant_ids.append(owner_tid)

    return MultiTenantBatch(
        input_ids=torch.stack(batch_input_ids).to(device),
        tsam_mask=torch.stack(batch_masks).to(device),
        tenant_ids=tenant_ids,
        shared_len=shared_len,
        private_lens=private_lens,
        seq_len=max_seq_len,
    )
