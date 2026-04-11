"""
Scalability benchmarking — Experiment 2.

Validates TSAM isolation at 10, 100, 1000, and 10000 concurrent tenants
and documents Paper 5 orthogonal projection failure beyond 16 tenants.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import numpy as np

from tsam.core.block_manager import TypedBlockManager
from tsam.core.types import PageType, TenantId, SHARED_TENANT_ID
from tsam.attention.masking import generate_tsam_mask


@dataclass
class ScalabilityResult:
    """Result from a scalability test at a specific tenant count.

    Attributes:
        num_tenants:           Number of concurrent tenants tested.
        leakage_bits_per_query: Measured leakage (should be 0.0 for TSAM).
        all_masks_correct:     Whether all TSAM masks were correctly generated.
        cross_tenant_weights:  Max attention weight to cross-tenant PRIVATE keys.
        defense:               Defense name.
        failure_mode:          Description of failure (for Paper 5 at N > 16).
    """

    num_tenants: int
    leakage_bits_per_query: float
    all_masks_correct: bool
    cross_tenant_weights: float
    defense: str
    failure_mode: Optional[str] = None
    sampling_note: Optional[str] = None


class Paper5OrthogonalProjection:
    """Paper 5 orthogonal projection isolation scheme.

    Implements the orthogonal subspace projection from KV-Cache Isolation Paper 5.
    Each tenant gets a d_sub-dimensional subspace of the d_model-dimensional space.
    The projection ensures cross-tenant attention score = 0.

    Critical limitation: requires d_sub * N_tenants <= d_model.
    For d_model=4096, d_sub=256: max 16 tenants.
    """

    def __init__(self, d_model: int = 4096, d_sub: int = 256):
        self.d_model = d_model
        self.d_sub = d_sub
        self.max_tenants = d_model // d_sub
        self._projection_matrices: Dict[int, torch.Tensor] = {}

    def can_support_tenants(self, n_tenants: int) -> bool:
        """Check if the scheme can support n_tenants."""
        return n_tenants <= self.max_tenants

    def get_failure_mode(self, n_tenants: int) -> str:
        """Describe the failure mode for n_tenants."""
        return (
            f"Dimensionality violation: {n_tenants} tenants require "
            f"{n_tenants * self.d_sub} dimensions, but d_model={self.d_model}. "
            f"Maximum supported: {self.max_tenants} tenants with d_sub={self.d_sub}."
        )

    def allocate_subspace(self, tenant_id: int) -> Optional[torch.Tensor]:
        """Allocate an orthogonal subspace projection matrix for a tenant.

        Uses QR decomposition of a random matrix to generate a proper random
        orthogonal basis (not trivial identity submatrices), ensuring a fair
        comparison that matches the methodology of the cited paper.

        Returns:
            Projection matrix of shape (d_model, d_model), or None if
            dimensionality is exhausted.
        """
        if len(self._projection_matrices) >= self.max_tenants:
            return None

        offset = len(self._projection_matrices) * self.d_sub
        if offset + self.d_sub > self.d_model:
            return None

        # Generate a full orthogonal basis via QR decomposition of a random
        # matrix (seeded for reproducibility), then extract the slice for this
        # tenant. This produces a proper random orthogonal subspace rather
        # than the trivial identity subspace used previously.
        if not hasattr(self, "_orthogonal_basis"):
            # Generate once: a d_model x d_model orthogonal matrix Q.
            gen = torch.Generator()
            gen.manual_seed(0)  # Reproducible.
            random_matrix = torch.randn(self.d_model, self.d_model, generator=gen)
            Q, _ = torch.linalg.qr(random_matrix)
            self._orthogonal_basis = Q

        # Tenant's basis: columns [offset : offset + d_sub] of Q.
        U = self._orthogonal_basis[:, offset : offset + self.d_sub]  # (d_model, d_sub)
        # Projection matrix P_i = U_i @ U_i^T.
        P = U @ U.T  # (d_model, d_model)

        self._projection_matrices[tenant_id] = P
        return P

    def project_kv(
        self, kv: torch.Tensor, tenant_id: int
    ) -> Optional[torch.Tensor]:
        """Project KV vectors into tenant's subspace.

        Args:
            kv: (..., d_model) tensor.
            tenant_id: Owning tenant.

        Returns:
            Projected KV, or None if tenant has no allocated subspace.
        """
        P = self._projection_matrices.get(tenant_id)
        if P is None:
            return None
        return kv @ P

    def verify_isolation(
        self, kv_i: torch.Tensor, kv_j: torch.Tensor, tenant_i: int, tenant_j: int
    ) -> float:
        """Verify cross-tenant attention score is zero.

        Returns max absolute attention score between projected KVs.
        """
        proj_i = self.project_kv(kv_i, tenant_i)
        proj_j = self.project_kv(kv_j, tenant_j)
        if proj_i is None or proj_j is None:
            return float("nan")
        # Cross-tenant attention score.
        score = torch.abs(proj_i @ proj_j.T).max().item()
        return score


class ScalabilityBenchmark:
    """Tests TSAM isolation at varying tenant counts and documents Paper 5 failure."""

    def __init__(
        self,
        page_size: int = 16,
        head_dim: int = 128,
        d_model: int = 4096,
        d_sub: int = 256,
        device: str = "cpu",
    ):
        self.page_size = page_size
        self.head_dim = head_dim
        self.d_model = d_model
        self.d_sub = d_sub
        self.device = device

    def test_tsam_at_scale(
        self,
        num_tenants: int,
        blocks_per_tenant: int = 4,
        shared_blocks: int = 2,
    ) -> ScalabilityResult:
        """Test TSAM isolation correctness at a given tenant count.

        Creates a TypedBlockManager with N tenants, generates masks for
        cross-tenant queries, and verifies zero leakage.
        """
        total_blocks = shared_blocks + num_tenants * blocks_per_tenant + 1
        bm = TypedBlockManager(
            num_blocks=total_blocks,
            page_size=self.page_size,
            device=self.device,
        )

        # Allocate shared prefix.
        shared_pages = bm.allocate_shared(count=shared_blocks)
        shared_ids = [p.block_id for p in shared_pages]

        # Allocate private blocks per tenant.
        tenant_block_ids: Dict[int, List[int]] = {}
        for t in range(num_tenants):
            pages = bm.allocate_private(tenant=t, count=blocks_per_tenant)
            tenant_block_ids[t] = [p.block_id for p in pages]

        # Test: for each query tenant, verify mask blocks all other tenants' PRIVATE.
        all_correct = True
        max_cross_weight = 0.0
        seq_len_kv = (shared_blocks + blocks_per_tenant) * self.page_size

        # For num_tenants <= 200, test ALL tenants (no sampling).
        # For num_tenants > 200, use a reproducible random sample of at least 200.
        sampling_note: Optional[str] = None
        if num_tenants <= 200:
            test_tenants = list(range(num_tenants))
        else:
            sample_size = max(200, num_tenants // 10)
            sample_size = min(sample_size, num_tenants)
            rng = random.Random(42)  # Documented seed for reproducibility
            test_tenants = sorted(rng.sample(range(num_tenants), sample_size))
            sampling_note = (
                f"Sampled {sample_size} of {num_tenants} tenants "
                f"(random seed=42) for mask correctness testing."
            )

        for query_t in test_tenants:
            # Build block table: shared + own private.
            bt_ids = shared_ids + tenant_block_ids[query_t]
            block_table = torch.tensor(bt_ids, dtype=torch.int32, device=self.device)

            mask_result = generate_tsam_mask(
                block_table=block_table,
                page_type=bm.page_type,
                tenant_id=bm.tenant_id,
                query_tenant=query_t,
                page_size=self.page_size,
                seq_len_kv=seq_len_kv,
                device=self.device,
            )

            # Verify: all shared positions are True.
            shared_positions = shared_blocks * self.page_size
            if not mask_result.mask[:shared_positions].all():
                all_correct = False

            # Verify: own private positions are True.
            if not mask_result.mask[shared_positions:].all():
                all_correct = False

            # Verify: mask has correct counts.
            expected_blocked = 0  # Own block table only contains own + shared.
            if mask_result.num_blocked != expected_blocked:
                all_correct = False

        # Now test cross-tenant isolation explicitly with random pairs.
        # Test at least 10 random (victim, attacker) pairs in addition to the
        # deterministic pair (0, 1).  Uses a documented seed for reproducibility.
        if num_tenants >= 2:
            pair_rng = random.Random(123)  # Documented seed for pair selection
            # Always include the canonical (0, 1) pair, plus random pairs.
            cross_pairs = [(0, 1)]
            all_tenant_ids = list(range(num_tenants))
            num_random_pairs = max(10, min(50, num_tenants // 10))
            for _ in range(num_random_pairs):
                a, b = pair_rng.sample(all_tenant_ids, 2)
                cross_pairs.append((a, b))
            # Deduplicate while preserving order.
            seen_pairs = set()
            unique_pairs = []
            for pair in cross_pairs:
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    unique_pairs.append(pair)
            cross_pairs = unique_pairs

            for t_victim, t_attacker in cross_pairs:
                # Block table with victim's private blocks (attacker shouldn't see them).
                mixed_bt = shared_ids + tenant_block_ids[t_victim] + tenant_block_ids[t_attacker]
                mixed_block_table = torch.tensor(
                    mixed_bt, dtype=torch.int32, device=self.device
                )
                mixed_seq_len = len(mixed_bt) * self.page_size

                mask_result = generate_tsam_mask(
                    block_table=mixed_block_table,
                    page_type=bm.page_type,
                    tenant_id=bm.tenant_id,
                    query_tenant=t_attacker,
                    page_size=self.page_size,
                    seq_len_kv=mixed_seq_len,
                    device=self.device,
                )

                # Victim's private positions should be blocked.
                victim_start = shared_blocks * self.page_size
                victim_end = victim_start + blocks_per_tenant * self.page_size
                victim_mask = mask_result.mask[victim_start:victim_end]

                if victim_mask.any():
                    all_correct = False
                    max_cross_weight = 1.0  # Indicates leakage.

                # Attacker's own positions should be allowed.
                attacker_start = victim_end
                attacker_end = attacker_start + blocks_per_tenant * self.page_size
                attacker_mask = mask_result.mask[attacker_start:attacker_end]
                if not attacker_mask.all():
                    all_correct = False

        leakage = 0.0 if all_correct else float("inf")

        return ScalabilityResult(
            num_tenants=num_tenants,
            leakage_bits_per_query=leakage,
            all_masks_correct=all_correct,
            cross_tenant_weights=max_cross_weight,
            defense="TSAM",
        )

    def test_paper5_at_scale(self, num_tenants: int) -> ScalabilityResult:
        """Test Paper 5 orthogonal projection at a given tenant count.

        Documents the failure mode when N > max_tenants.
        """
        paper5 = Paper5OrthogonalProjection(
            d_model=self.d_model, d_sub=self.d_sub
        )

        if not paper5.can_support_tenants(num_tenants):
            return ScalabilityResult(
                num_tenants=num_tenants,
                leakage_bits_per_query=float("inf"),
                all_masks_correct=False,
                cross_tenant_weights=float("nan"),
                defense="Paper5_orthogonal",
                failure_mode=paper5.get_failure_mode(num_tenants),
            )

        # For supported tenant counts, verify isolation works.
        all_correct = True
        max_cross_score = 0.0

        for t in range(num_tenants):
            P = paper5.allocate_subspace(t)
            if P is None:
                all_correct = False
                break

        # Test cross-tenant isolation.
        # Note: orthogonal projection achieves ε-approximate isolation (not exact).
        # With random orthogonal bases and float32 arithmetic, cross-tenant scores
        # of ~1e-5 are expected due to numerical precision limits of QR decomposition.
        # This is a fundamental limitation of the approach vs. TSAM's exact zero.
        if all_correct and num_tenants >= 2:
            kv_0 = torch.randn(1, self.d_model)
            kv_1 = torch.randn(1, self.d_model)
            score = paper5.verify_isolation(kv_0, kv_1, 0, 1)
            max_cross_score = score
            if score > 1e-4:  # Realistic threshold for float32 orthogonal projection.
                all_correct = False

        return ScalabilityResult(
            num_tenants=num_tenants,
            leakage_bits_per_query=0.0 if all_correct else float("inf"),
            all_masks_correct=all_correct,
            cross_tenant_weights=max_cross_score,
            defense="Paper5_orthogonal",
        )

    def run_full_scalability_test(
        self, tenant_counts: Optional[List[int]] = None
    ) -> Dict[str, List[ScalabilityResult]]:
        """Run Experiment 2: scalability at [10, 100, 1000, 10000] tenants.

        Returns:
            Dict with "TSAM" and "Paper5" keys, each mapping to a list of results.
        """
        if tenant_counts is None:
            tenant_counts = [10, 100, 1000, 10000]

        results: Dict[str, List[ScalabilityResult]] = {"TSAM": [], "Paper5": []}

        for n in tenant_counts:
            tsam_result = self.test_tsam_at_scale(n)
            results["TSAM"].append(tsam_result)

            paper5_result = self.test_paper5_at_scale(n)
            results["Paper5"].append(paper5_result)

        return results
