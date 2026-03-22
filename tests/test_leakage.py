"""Tests for leakage evaluation and MI estimation."""

import numpy as np
import pytest

from tsam.evaluation.leakage import (
    LeakageEstimator,
    LeakageResult,
    MutualInformationEstimator,
)


class TestMutualInformationEstimator:
    def test_independent_variables_low_mi(self):
        """MI between independent variables should be near zero."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, (200, 3))
        y = rng.normal(0, 1, (200, 3))
        mi_est = MutualInformationEstimator(k=3)
        mi, _ = mi_est.estimate_bits(x, y)
        assert mi < 0.5, f"MI between independent vars should be ~0, got {mi}"

    def test_identical_variables_high_mi(self):
        """MI between identical variables should be high."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, (200, 3))
        y = x + rng.normal(0, 0.01, (200, 3))  # Nearly identical.
        mi_est = MutualInformationEstimator(k=3)
        mi, _ = mi_est.estimate_bits(x, y)
        assert mi > 1.0, f"MI between nearly identical vars should be high, got {mi}"

    def test_small_sample(self):
        """Should handle small sample sizes gracefully."""
        x = np.array([[1.0, 2.0]])
        y = np.array([[3.0, 4.0]])
        mi_est = MutualInformationEstimator(k=3)
        mi, std = mi_est.estimate_bits(x, y)
        assert mi == 0.0  # Too few samples.


class TestLeakageEstimator:
    def test_perfect_reconstruction(self):
        est = LeakageEstimator(vocab_size=32000)
        private = np.array([1, 2, 3, 4, 5])
        recon = np.array([1, 2, 3, 4, 5])
        bits = est.measure_reconstruction_leakage(private, recon)
        expected = 5 * np.log2(32000)
        assert abs(bits - expected) < 0.01

    def test_no_reconstruction(self):
        est = LeakageEstimator(vocab_size=32000)
        private = np.array([1, 2, 3, 4, 5])
        recon = np.array([10, 20, 30, 40, 50])
        bits = est.measure_reconstruction_leakage(private, recon)
        assert bits == 0.0

    def test_partial_reconstruction(self):
        est = LeakageEstimator(vocab_size=32000)
        private = np.array([1, 2, 3, 4, 5])
        recon = np.array([1, 20, 3, 40, 5])  # 3/5 correct.
        bits = est.measure_reconstruction_leakage(private, recon)
        expected = 3.0 / 5.0 * np.log2(32000) * 5
        assert abs(bits - expected) < 0.01

    def test_empty_prompt(self):
        est = LeakageEstimator()
        bits = est.measure_reconstruction_leakage(np.array([]), np.array([]))
        assert bits == 0.0


class TestCollisionAttack:
    def test_collision_result_fields(self):
        from tsam.evaluation.collision_attack import CollisionAttackResult

        result = CollisionAttackResult(
            bits_per_query=0.5,
            total_bits=50.0,
            num_probes=100,
            detection_accuracy=0.75,
            true_positive_rate=0.8,
            false_positive_rate=0.3,
            timing_mean_hit_ms=30.0,
            timing_mean_miss_ms=80.0,
            timing_sigma_hit_ms=5.0,
            timing_sigma_miss_ms=10.0,
            defense="test",
        )
        assert result.bits_per_query == 0.5
        assert result.num_probes == 100

    def test_leakage_reduction_computation(self):
        from tsam.evaluation.collision_attack import (
            CollisionAttackSimulator,
            CollisionAttackResult,
        )

        no_defense = CollisionAttackResult(
            bits_per_query=0.8, total_bits=80.0, num_probes=100,
            detection_accuracy=0.9, true_positive_rate=0.9, false_positive_rate=0.1,
            timing_mean_hit_ms=30.0, timing_mean_miss_ms=80.0,
            timing_sigma_hit_ms=5.0, timing_sigma_miss_ms=10.0,
            defense="none",
        )
        defended = CollisionAttackResult(
            bits_per_query=0.04, total_bits=4.0, num_probes=100,
            detection_accuracy=0.52, true_positive_rate=0.52, false_positive_rate=0.48,
            timing_mean_hit_ms=55.0, timing_mean_miss_ms=60.0,
            timing_sigma_hit_ms=10.0, timing_sigma_miss_ms=10.0,
            defense="tsam_dfa",
        )
        reduction = CollisionAttackSimulator.compute_timing_leakage_reduction(
            no_defense, defended
        )
        assert reduction >= 90.0  # 0.04/0.8 = 5% remaining = 95% reduction.
