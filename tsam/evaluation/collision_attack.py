"""
Collision Attack Simulator — implements the cache-presence inference attack
from "Shadow in the Cache" (Luo et al., NDSS 2026).

The Collision attack infers whether a specific prompt is cached by measuring
response time variance. If a prompt is cached (KV-cache hit), response time
is lower than for uncached prompts. By issuing probe queries and measuring
timing, an attacker can determine co-tenant cache presence.

This module provides:
1. CollisionAttackSimulator — generates probe sequences and measures leakage.
2. TimingOracle — interface for measuring response times from the serving system.
3. Metrics — bits of cache-presence information per query.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CollisionAttackResult:
    """Result from a Collision attack experiment.

    Attributes:
        bits_per_query:         Bits of cache-presence information per probe query.
        total_bits:             Total information bits recovered.
        num_probes:             Number of probe queries issued.
        detection_accuracy:     Binary accuracy of cache-hit vs. miss classification.
        true_positive_rate:     Rate of correctly detecting cache hits.
        false_positive_rate:    Rate of incorrectly classifying misses as hits.
        timing_mean_hit_ms:     Mean response time for cache hits.
        timing_mean_miss_ms:    Mean response time for cache misses.
        timing_sigma_hit_ms:    Std of response time for cache hits.
        timing_sigma_miss_ms:   Std of response time for cache misses.
        defense:                Defense condition name.
    """

    bits_per_query: float
    total_bits: float
    num_probes: int
    detection_accuracy: float
    true_positive_rate: float
    false_positive_rate: float
    timing_mean_hit_ms: float
    timing_mean_miss_ms: float
    timing_sigma_hit_ms: float
    timing_sigma_miss_ms: float
    defense: str


class CollisionAttackSimulator:
    """Simulates the Collision-type cache-presence inference attack.

    The attacker:
    1. Sends a probe query that would benefit from a cached prefix.
    2. Measures response time.
    3. Classifies as cache-hit or cache-miss based on a timing threshold.
    4. Repeats to estimate cache presence with high confidence.

    Information leakage is measured as the mutual information between
    the binary cache-presence variable and the attacker's timing observations.
    """

    def __init__(
        self,
        timing_oracle: Callable[[str, int], float],
        hit_threshold_ms: Optional[float] = None,
        num_calibration_probes: int = 50,
    ):
        """
        Args:
            timing_oracle:          Callable(prompt, tenant_id) -> response_time_ms.
                                    Measures actual response time from the serving system.
            hit_threshold_ms:       Timing threshold for hit/miss classification.
                                    If None, auto-calibrated from initial probes.
            num_calibration_probes: Probes used for threshold calibration.
        """
        self.timing_oracle = timing_oracle
        self.hit_threshold_ms = hit_threshold_ms
        self.num_calibration_probes = num_calibration_probes

    def calibrate_threshold(
        self,
        known_hit_prompt: str,
        known_miss_prompt: str,
        tenant_id: int,
    ) -> float:
        """Calibrate the hit/miss timing threshold from known samples.

        Args:
            known_hit_prompt:  A prompt known to produce a cache hit.
            known_miss_prompt: A prompt known to produce a cache miss.
            tenant_id:         Tenant issuing the probes.

        Returns:
            Calibrated threshold in ms.
        """
        hit_times = []
        miss_times = []
        for _ in range(self.num_calibration_probes):
            hit_times.append(self.timing_oracle(known_hit_prompt, tenant_id))
            miss_times.append(self.timing_oracle(known_miss_prompt, tenant_id))

        hit_mean = np.mean(hit_times)
        miss_mean = np.mean(miss_times)
        # Threshold at midpoint (optimal for equal-variance Gaussian).
        self.hit_threshold_ms = float((hit_mean + miss_mean) / 2.0)
        return self.hit_threshold_ms

    def run_attack(
        self,
        probe_prompts: List[str],
        ground_truth_cached: List[bool],
        attacker_tenant_id: int,
        defense_name: str = "none",
    ) -> CollisionAttackResult:
        """Execute the Collision attack.

        Args:
            probe_prompts:       Prompts to probe (each may or may not be cached).
            ground_truth_cached: Whether each prompt is actually in the cache.
            attacker_tenant_id:  Tenant ID of the attacker.
            defense_name:        Name of the defense being evaluated.

        Returns:
            CollisionAttackResult with detailed metrics.
        """
        if self.hit_threshold_ms is None:
            raise ValueError(
                "Threshold not calibrated. Call calibrate_threshold() first."
            )

        num_probes = len(probe_prompts)
        timings = []
        predictions = []

        for prompt in probe_prompts:
            t = self.timing_oracle(prompt, attacker_tenant_id)
            timings.append(t)
            predictions.append(t < self.hit_threshold_ms)  # Fast = cache hit

        timings_arr = np.array(timings)
        preds_arr = np.array(predictions)
        truth_arr = np.array(ground_truth_cached)

        # Classification metrics.
        correct = preds_arr == truth_arr
        accuracy = float(correct.mean())

        tp = float(((preds_arr == True) & (truth_arr == True)).sum())
        fp = float(((preds_arr == True) & (truth_arr == False)).sum())
        fn = float(((preds_arr == False) & (truth_arr == True)).sum())
        tn = float(((preds_arr == False) & (truth_arr == False)).sum())

        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)

        # Timing statistics per class.
        hit_mask = truth_arr == True
        miss_mask = truth_arr == False

        hit_times = timings_arr[hit_mask] if hit_mask.any() else np.array([0.0])
        miss_times = timings_arr[miss_mask] if miss_mask.any() else np.array([0.0])

        # Information leakage in bits.
        # I(CachePresence; Timing) estimated via binary channel capacity.
        # H(CachePresence) = 1 bit (binary).
        # H(CachePresence | Timing) = H(error rate).
        if accuracy >= 1.0:
            bits_per_query = 1.0  # Perfect classification = 1 bit per query
        elif accuracy <= 0.5:
            bits_per_query = 0.0  # Random guessing = 0 bits
        else:
            # Binary symmetric channel: C = 1 - H(p_error)
            p_error = 1.0 - accuracy
            h_error = -p_error * math.log2(max(p_error, 1e-10)) - (
                1 - p_error
            ) * math.log2(max(1 - p_error, 1e-10))
            bits_per_query = max(0.0, 1.0 - h_error)

        return CollisionAttackResult(
            bits_per_query=bits_per_query,
            total_bits=bits_per_query * num_probes,
            num_probes=num_probes,
            detection_accuracy=accuracy,
            true_positive_rate=tpr,
            false_positive_rate=fpr,
            timing_mean_hit_ms=float(hit_times.mean()),
            timing_mean_miss_ms=float(miss_times.mean()),
            timing_sigma_hit_ms=float(hit_times.std()),
            timing_sigma_miss_ms=float(miss_times.std()),
            defense=defense_name,
        )

    @staticmethod
    def compute_timing_leakage_reduction(
        no_defense_result: CollisionAttackResult,
        defended_result: CollisionAttackResult,
    ) -> float:
        """Compute the percentage reduction in timing leakage.

        Returns:
            Reduction percentage (0-100). Target: >= 95%.
        """
        if no_defense_result.bits_per_query <= 0:
            return 100.0  # No leakage to begin with.
        reduction = 1.0 - (
            defended_result.bits_per_query / no_defense_result.bits_per_query
        )
        return max(0.0, min(100.0, reduction * 100.0))
