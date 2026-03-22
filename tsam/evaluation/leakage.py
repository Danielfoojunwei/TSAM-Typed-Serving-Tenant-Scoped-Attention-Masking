"""
Leakage evaluation — mutual information estimation for TSAM isolation verification.

Implements the measurement protocol from Experiment 1:
  - Bits of private prompt reconstructed per query
  - Mutual information I(output(q_i); KV_private(t_j)) estimation
  - Comparison across defense conditions: no defense, KV-Cloak, TSAM

Uses k-nearest-neighbour MI estimator (KSG estimator) for continuous variables
and direct reconstruction-based measurement for discrete prompt tokens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import KDTree
from scipy.special import digamma


@dataclass
class LeakageResult:
    """Result from a leakage measurement experiment.

    Attributes:
        bits_per_query:       Average bits of private information leaked per query.
        total_bits_leaked:    Total bits leaked across all attack queries.
        num_queries:          Number of attack queries evaluated.
        reconstruction_rate:  Fraction of private tokens correctly reconstructed.
        mi_estimate:          Estimated mutual information (nats, converted to bits).
        mi_std:               Standard deviation of MI estimate.
        defense:              Name of the defense evaluated.
    """

    bits_per_query: float
    total_bits_leaked: float
    num_queries: int
    reconstruction_rate: float
    mi_estimate: float
    mi_std: float
    defense: str


class MutualInformationEstimator:
    """KSG (Kraskov-Stögbauer-Grassberger) mutual information estimator.

    Estimates I(X; Y) from paired samples using k-nearest-neighbour distances.
    This is used to estimate I(output(q_i); KV_private(t_j)) — the core
    isolation metric.
    """

    def __init__(self, k: int = 3):
        """
        Args:
            k: Number of nearest neighbours for KSG estimator.
        """
        self.k = k

    def estimate(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[float, float]:
        """Estimate mutual information I(X; Y) in nats.

        Args:
            x: (n_samples, d_x) array.
            y: (n_samples, d_y) array.

        Returns:
            (mi_estimate_nats, mi_std_nats) — MI estimate and bootstrap std.
        """
        n = x.shape[0]
        if n < self.k + 1:
            return 0.0, 0.0

        x = np.ascontiguousarray(x, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)

        # Joint space.
        xy = np.hstack([x, y])

        # Build KD trees.
        tree_xy = KDTree(xy)
        tree_x = KDTree(x)
        tree_y = KDTree(y)

        # For each point, find distance to k-th nearest neighbour in joint space.
        # query_ball_point-based approach for numerical stability.
        nn_dists, _ = tree_xy.query(xy, k=self.k + 1)  # +1 because point itself
        epsilon = nn_dists[:, -1]  # distance to k-th neighbour

        # Count neighbours within epsilon in marginal spaces.
        nx = np.zeros(n)
        ny = np.zeros(n)
        for i in range(n):
            eps_i = epsilon[i]
            # Use max-norm ball (Chebyshev distance).
            nx[i] = len(tree_x.query_ball_point(x[i], eps_i, p=np.inf)) - 1
            ny[i] = len(tree_y.query_ball_point(y[i], eps_i, p=np.inf)) - 1

        # KSG estimator formula.
        mi = digamma(self.k) - np.mean(digamma(nx + 1) + digamma(ny + 1)) + digamma(n)

        # Bootstrap standard deviation.
        n_bootstrap = 50
        bootstrap_mis = []
        rng = np.random.default_rng(42)
        for _ in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            x_b, y_b = x[idx], y[idx]
            try:
                mi_b, _ = self._estimate_single(x_b, y_b)
                bootstrap_mis.append(mi_b)
            except Exception:
                continue
        mi_std = float(np.std(bootstrap_mis)) if bootstrap_mis else 0.0

        return max(0.0, float(mi)), mi_std

    def _estimate_single(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[float, float]:
        """Single MI estimate without bootstrap (used internally)."""
        n = x.shape[0]
        xy = np.hstack([x, y])
        tree_xy = KDTree(xy)
        tree_x = KDTree(x)
        tree_y = KDTree(y)
        nn_dists, _ = tree_xy.query(xy, k=self.k + 1)
        epsilon = nn_dists[:, -1]
        nx = np.zeros(n)
        ny = np.zeros(n)
        for i in range(n):
            nx[i] = len(tree_x.query_ball_point(x[i], epsilon[i], p=np.inf)) - 1
            ny[i] = len(tree_y.query_ball_point(y[i], epsilon[i], p=np.inf)) - 1
        mi = digamma(self.k) - np.mean(digamma(nx + 1) + digamma(ny + 1)) + digamma(n)
        return max(0.0, float(mi)), 0.0

    def estimate_bits(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[float, float]:
        """Estimate MI in bits (nats / ln(2))."""
        mi_nats, mi_std_nats = self.estimate(x, y)
        return mi_nats / math.log(2), mi_std_nats / math.log(2)


class LeakageEstimator:
    """End-to-end leakage estimator for TSAM evaluation.

    Measures information leakage from private prompts through attention outputs,
    using both reconstruction-based and MI-based metrics.
    """

    def __init__(
        self,
        mi_estimator: Optional[MutualInformationEstimator] = None,
        vocab_size: int = 32000,
    ):
        self.mi_estimator = mi_estimator or MutualInformationEstimator(k=3)
        self.vocab_size = vocab_size
        self.bits_per_token = math.log2(vocab_size)

    def measure_reconstruction_leakage(
        self,
        private_tokens: np.ndarray,
        reconstructed_tokens: np.ndarray,
    ) -> float:
        """Measure bits leaked via token reconstruction.

        Args:
            private_tokens:       (num_tokens,) int array — ground truth private prompt.
            reconstructed_tokens: (num_tokens,) int array — attacker's reconstruction.

        Returns:
            Bits leaked = (fraction correct) * log2(vocab_size) * num_tokens.
        """
        if len(private_tokens) == 0:
            return 0.0
        correct = np.sum(private_tokens == reconstructed_tokens)
        fraction_correct = correct / len(private_tokens)
        bits_leaked = fraction_correct * self.bits_per_token * len(private_tokens)
        return float(bits_leaked)

    def measure_output_leakage(
        self,
        query_outputs: np.ndarray,
        private_kv_content: np.ndarray,
    ) -> Tuple[float, float]:
        """Measure MI between query outputs and private KV content.

        Args:
            query_outputs:      (n_queries, d_output) — output embeddings from attack queries.
            private_kv_content: (n_queries, d_private) — corresponding private KV representations.

        Returns:
            (mi_bits, mi_std_bits).
        """
        return self.mi_estimator.estimate_bits(query_outputs, private_kv_content)

    def run_leakage_experiment(
        self,
        attack_fn,
        private_prompt_tokens: np.ndarray,
        num_attack_queries: int,
        defense_name: str,
    ) -> LeakageResult:
        """Run a complete leakage experiment.

        Args:
            attack_fn:            Callable(query_idx) -> (reconstructed_tokens, output_embedding).
                                  The attack function that attempts to reconstruct the private prompt.
            private_prompt_tokens: Ground truth private prompt tokens.
            num_attack_queries:   Number of attack queries to issue.
            defense_name:         Name of the defense being evaluated.

        Returns:
            LeakageResult with all metrics.
        """
        all_reconstructed = []
        all_outputs = []
        total_bits = 0.0

        for q_idx in range(num_attack_queries):
            reconstructed, output_emb = attack_fn(q_idx)
            all_reconstructed.append(reconstructed)
            all_outputs.append(output_emb)
            bits = self.measure_reconstruction_leakage(
                private_prompt_tokens, reconstructed
            )
            total_bits += bits

        # Aggregate reconstruction rate.
        all_reconstructed_arr = np.array(all_reconstructed)
        # Best reconstruction across all queries.
        best_match_per_token = np.zeros(len(private_prompt_tokens), dtype=bool)
        for recon in all_reconstructed:
            best_match_per_token |= recon == private_prompt_tokens
        reconstruction_rate = float(best_match_per_token.mean())

        # MI estimate from output embeddings.
        outputs_arr = np.array(all_outputs)
        # Create pseudo-representation of private content for MI estimation.
        private_rep = np.tile(
            private_prompt_tokens.astype(np.float64),
            (num_attack_queries, 1),
        )
        # Truncate/pad to match dimensions.
        min_d = min(outputs_arr.shape[1], private_rep.shape[1])
        mi_bits, mi_std = self.mi_estimator.estimate_bits(
            outputs_arr[:, :min_d], private_rep[:, :min_d]
        )

        return LeakageResult(
            bits_per_query=total_bits / max(num_attack_queries, 1),
            total_bits_leaked=total_bits,
            num_queries=num_attack_queries,
            reconstruction_rate=reconstruction_rate,
            mi_estimate=mi_bits,
            mi_std=mi_std,
            defense=defense_name,
        )
