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

    def test_estimate_nats_non_negative(self):
        """MI estimate in nats should be non-negative."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, (100, 2))
        y = rng.normal(0, 1, (100, 2))
        mi_est = MutualInformationEstimator(k=3)
        mi_nats, _ = mi_est.estimate(x, y)
        assert mi_nats >= 0.0

    def test_bootstrap_std_returned(self):
        """Verify bootstrap standard deviation is computed and non-negative."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, (100, 3))
        y = x + rng.normal(0, 0.5, (100, 3))
        mi_est = MutualInformationEstimator(k=3)
        mi, std = mi_est.estimate_bits(x, y)
        assert std >= 0.0


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

    def test_measure_output_leakage_independent(self):
        """MI-based leakage between independent outputs and private content should be low."""
        rng = np.random.default_rng(42)
        outputs = rng.normal(0, 1, (50, 10))
        private = rng.normal(0, 1, (50, 10))
        est = LeakageEstimator()
        mi_bits, mi_std = est.measure_output_leakage(outputs, private)
        assert mi_bits < 1.0, f"Independent output/private MI should be low, got {mi_bits}"

    def test_measure_output_leakage_correlated(self):
        """MI-based leakage between correlated outputs and private content should be high."""
        rng = np.random.default_rng(42)
        # Use enough samples and sufficient dimensionality for KSG to detect correlation.
        private = rng.normal(0, 1, (500, 3))
        outputs = private + rng.normal(0, 0.1, (500, 3))  # Strong correlation.
        est = LeakageEstimator()
        mi_bits, mi_std = est.measure_output_leakage(outputs, private)
        assert mi_bits > 0.5, f"Correlated output/private MI should be high, got {mi_bits}"

    def test_run_leakage_experiment_zero_leakage(self):
        """End-to-end experiment where attacker never reconstructs correctly."""
        rng = np.random.default_rng(42)
        private_tokens = np.array([100, 200, 300, 400, 500])

        def attack_fn(q_idx):
            # Attacker always guesses wrong tokens and produces random embeddings.
            wrong_tokens = np.array([0, 0, 0, 0, 0])
            random_embedding = rng.normal(0, 1, 5)
            return wrong_tokens, random_embedding

        est = LeakageEstimator(vocab_size=32000)
        result = est.run_leakage_experiment(
            attack_fn=attack_fn,
            private_prompt_tokens=private_tokens,
            num_attack_queries=20,
            defense_name="test_defense",
        )

        assert isinstance(result, LeakageResult)
        assert result.bits_per_query == 0.0
        assert result.reconstruction_rate == 0.0
        assert result.num_queries == 20
        assert result.defense == "test_defense"

    def test_run_leakage_experiment_perfect_leakage(self):
        """End-to-end experiment where attacker perfectly reconstructs."""
        private_tokens = np.array([100, 200, 300])

        def attack_fn(q_idx):
            return private_tokens.copy(), private_tokens.astype(np.float64)

        est = LeakageEstimator(vocab_size=32000)
        result = est.run_leakage_experiment(
            attack_fn=attack_fn,
            private_prompt_tokens=private_tokens,
            num_attack_queries=10,
            defense_name="no_defense",
        )

        expected_bits_per_query = np.log2(32000) * 3
        assert abs(result.bits_per_query - expected_bits_per_query) < 0.01
        assert result.reconstruction_rate == 1.0


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

    def test_calibrate_threshold(self):
        """Verify threshold calibration produces a value between hit and miss means."""
        from tsam.evaluation.collision_attack import CollisionAttackSimulator

        rng = np.random.default_rng(42)

        def oracle(prompt: str, tenant: int) -> float:
            if prompt == "hit":
                return max(1.0, rng.normal(30.0, 5.0))
            return max(1.0, rng.normal(80.0, 10.0))

        sim = CollisionAttackSimulator(timing_oracle=oracle, num_calibration_probes=20)
        threshold = sim.calibrate_threshold("hit", "miss", tenant_id=0)
        assert 20.0 < threshold < 100.0, f"Threshold {threshold} out of expected range"

    def test_run_attack_end_to_end(self):
        """Full attack: calibrate, probe, measure bits."""
        from tsam.evaluation.collision_attack import CollisionAttackSimulator

        rng = np.random.default_rng(42)
        cached = set(range(50))

        def oracle(prompt: str, tenant: int) -> float:
            idx = int(prompt)
            if idx in cached:
                return max(1.0, rng.normal(30.0, 5.0))
            return max(1.0, rng.normal(80.0, 10.0))

        sim = CollisionAttackSimulator(timing_oracle=oracle, num_calibration_probes=30)
        sim.calibrate_threshold("0", "99", tenant_id=0)

        probes = [str(i) for i in range(100)]
        truth = [i in cached for i in range(100)]

        result = sim.run_attack(probes, truth, attacker_tenant_id=0, defense_name="no_defense")

        assert result.num_probes == 100
        assert result.detection_accuracy > 0.8, (
            f"Attack accuracy {result.detection_accuracy} too low for clear timing gap"
        )
        assert result.bits_per_query > 0.0
        assert result.defense == "no_defense"

    def test_run_attack_requires_calibration(self):
        """Attack should raise if threshold not calibrated."""
        from tsam.evaluation.collision_attack import CollisionAttackSimulator

        sim = CollisionAttackSimulator(
            timing_oracle=lambda p, t: 50.0, num_calibration_probes=10
        )
        with pytest.raises(ValueError, match="Threshold not calibrated"):
            sim.run_attack(["a"], [True], attacker_tenant_id=0)
