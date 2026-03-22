"""Tests for scalability benchmarks — Experiment 2."""

import pytest
from tsam.benchmarks.scalability import (
    ScalabilityBenchmark,
    Paper5OrthogonalProjection,
)


class TestPaper5OrthogonalProjection:
    def test_max_16_tenants(self):
        p5 = Paper5OrthogonalProjection(d_model=4096, d_sub=256)
        assert p5.max_tenants == 16

    def test_supports_16(self):
        p5 = Paper5OrthogonalProjection(d_model=4096, d_sub=256)
        assert p5.can_support_tenants(16) is True

    def test_fails_at_17(self):
        p5 = Paper5OrthogonalProjection(d_model=4096, d_sub=256)
        assert p5.can_support_tenants(17) is False

    def test_fails_at_100(self):
        p5 = Paper5OrthogonalProjection(d_model=4096, d_sub=256)
        assert p5.can_support_tenants(100) is False

    def test_isolation_within_limit(self):
        p5 = Paper5OrthogonalProjection(d_model=4096, d_sub=256)
        for t in range(16):
            P = p5.allocate_subspace(t)
            assert P is not None
        import torch
        kv_0 = torch.randn(1, 4096)
        kv_1 = torch.randn(1, 4096)
        score = p5.verify_isolation(kv_0, kv_1, 0, 1)
        assert score < 1e-5

    def test_allocation_fails_beyond_limit(self):
        p5 = Paper5OrthogonalProjection(d_model=4096, d_sub=256)
        for t in range(16):
            assert p5.allocate_subspace(t) is not None
        assert p5.allocate_subspace(16) is None

    def test_failure_mode_message(self):
        p5 = Paper5OrthogonalProjection(d_model=4096, d_sub=256)
        msg = p5.get_failure_mode(100)
        assert "100 tenants" in msg
        assert "4096" in msg
        assert "Maximum supported: 16" in msg


class TestScalabilityBenchmark:
    def test_tsam_10_tenants(self):
        bench = ScalabilityBenchmark()
        result = bench.test_tsam_at_scale(10)
        assert result.all_masks_correct is True
        assert result.leakage_bits_per_query == 0.0
        assert result.defense == "TSAM"

    def test_tsam_100_tenants(self):
        bench = ScalabilityBenchmark()
        result = bench.test_tsam_at_scale(100)
        assert result.all_masks_correct is True
        assert result.leakage_bits_per_query == 0.0

    def test_tsam_1000_tenants(self):
        bench = ScalabilityBenchmark()
        result = bench.test_tsam_at_scale(1000)
        assert result.all_masks_correct is True
        assert result.leakage_bits_per_query == 0.0

    def test_paper5_fails_at_100(self):
        bench = ScalabilityBenchmark()
        result = bench.test_paper5_at_scale(100)
        assert result.all_masks_correct is False
        assert result.failure_mode is not None
        assert "Dimensionality violation" in result.failure_mode

    def test_paper5_passes_at_10(self):
        bench = ScalabilityBenchmark()
        result = bench.test_paper5_at_scale(10)
        assert result.all_masks_correct is True
        assert result.leakage_bits_per_query == 0.0
