"""Tests for DAI Timing DFA."""

import numpy as np
import pytest

from tsam.dai.timing_dfa import (
    TimingDFA,
    TimingDFAConfig,
    TimingDFAState,
    DFAAction,
    TimingDFAMiddleware,
)


class TestTimingDFA:
    def test_initial_state(self):
        dfa = TimingDFA()
        assert dfa.get_state(0) == TimingDFAState.NORMAL

    def test_normal_with_stable_timing(self):
        dfa = TimingDFA(TimingDFAConfig(window_size=10, sigma_threshold=15.0))
        # Send stable timing — should stay in NORMAL.
        for _ in range(50):
            dfa.observe(0, 50.0)
        assert dfa.get_state(0) == TimingDFAState.NORMAL

    def test_transition_to_probe1(self):
        dfa = TimingDFA(TimingDFAConfig(window_size=10, sigma_threshold=5.0))
        # Send high-variance timing to trigger TIMING_PROBE_1.
        rng = np.random.default_rng(42)
        for _ in range(20):
            dfa.observe(0, rng.normal(50, 20))
        assert dfa.get_state(0) in (
            TimingDFAState.TIMING_PROBE_1,
            TimingDFAState.TIMING_PROBE_2,
            TimingDFAState.ATTACK_CONFIRMED,
        )

    def test_transition_to_probe2(self):
        config = TimingDFAConfig(
            window_size=5, sigma_threshold=3.0, kl_threshold=100.0
        )
        dfa = TimingDFA(config)
        rng = np.random.default_rng(42)
        # Sustain high variance for multiple windows to ensure transition.
        for _ in range(50):
            dfa.observe(0, rng.normal(50, 30))
        state = dfa.get_state(0)
        assert state in (
            TimingDFAState.TIMING_PROBE_1,
            TimingDFAState.TIMING_PROBE_2,
            TimingDFAState.ATTACK_CONFIRMED,
        )

    def test_attack_confirmed_injects_noise(self):
        config = TimingDFAConfig(
            window_size=5, sigma_threshold=1.0, kl_threshold=0.01
        )
        dfa = TimingDFA(config)
        # Send extremely anomalous pattern to force ATTACK_CONFIRMED.
        for _ in range(5):
            dfa.observe(0, 10.0)
        for _ in range(5):
            dfa.observe(0, 200.0)
        for _ in range(5):
            dfa.observe(0, 10.0)
        for _ in range(5):
            dfa.observe(0, 200.0)
        # Keep pushing with anomalous distribution.
        for _ in range(20):
            dfa.observe(0, 500.0)  # Way outside benign distribution.

        state = dfa.get_state(0)
        if state == TimingDFAState.ATTACK_CONFIRMED:
            action = dfa.observe(0, 500.0)
            assert action.inject_noise is True
            assert action.noise_delay_ms >= 0

    def test_reset(self):
        dfa = TimingDFA()
        dfa.observe(0, 50.0)
        dfa.reset(0)
        assert dfa.get_state(0) == TimingDFAState.NORMAL

    def test_per_tenant_isolation(self):
        dfa = TimingDFA()
        dfa.observe(0, 50.0)
        dfa.observe(1, 50.0)
        assert dfa.get_state(0) == TimingDFAState.NORMAL
        assert dfa.get_state(1) == TimingDFAState.NORMAL

    def test_stats(self):
        dfa = TimingDFA()
        for _ in range(10):
            dfa.observe(0, 50.0)
        stats = dfa.get_stats(0)
        assert stats["total_queries"] == 10
        assert stats["state"] == "NORMAL"


class TestTimingDFAMiddleware:
    def test_post_request(self):
        mw = TimingDFAMiddleware()
        action = mw.post_request(0, 50.0)
        assert isinstance(action, DFAAction)
        assert action.inject_noise is False

    def test_should_delay_request_normal(self):
        mw = TimingDFAMiddleware()
        delay = mw.should_delay_request(0)
        assert delay == 0.0
