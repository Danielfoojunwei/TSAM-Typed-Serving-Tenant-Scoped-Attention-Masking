"""
DAI Timing DFA — Deterministic Finite Automaton for Collision-type timing attacks.

Detects when a tenant is performing timing-based cache-presence inference by
monitoring query-response timing patterns. Integrated as a serving middleware
that intercepts per-tenant request/response pairs.

DFA States:
  NORMAL → TIMING_PROBE_1 → TIMING_PROBE_2 → ATTACK_CONFIRMED

Transitions:
  NORMAL → TIMING_PROBE_1:
    σ(response_time over last K queries) > τ_threshold
  TIMING_PROBE_1 → TIMING_PROBE_2:
    σ remains elevated over next K queries
  TIMING_PROBE_2 → ATTACK_CONFIRMED:
    query distribution shifts to cache-presence inference pattern
    (KL divergence from benign distribution > threshold)

Actions on ATTACK_CONFIRMED:
  - Rate-limit tenant
  - Inject Gaussian timing noise (σ=5ms)
"""

from __future__ import annotations

import enum
import math
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

from tsam.core.types import TenantId


class TimingDFAState(enum.Enum):
    """DFA states for timing attack detection."""

    NORMAL = "NORMAL"
    TIMING_PROBE_1 = "TIMING_PROBE_1"
    TIMING_PROBE_2 = "TIMING_PROBE_2"
    ATTACK_CONFIRMED = "ATTACK_CONFIRMED"


@dataclass
class TimingDFAConfig:
    """Configuration for the timing attack DFA.

    Attributes:
        window_size:           K — number of queries in the observation window.
        sigma_threshold:       τ — response time std threshold for probe detection.
        kl_threshold:          KL divergence threshold for distribution shift detection.
        noise_sigma_ms:        Gaussian noise σ (milliseconds) injected on attack.
        rate_limit_factor:     Multiplier for inter-request delay on attack (e.g., 2.0 = 2x).
        cooldown_queries:      Queries after which ATTACK_CONFIRMED drops back to NORMAL
                               if behavior normalizes.
        benign_distribution:   Reference benign timing distribution (histogram bins).
        benign_bin_edges:      Bin edges for the benign distribution histogram.
    """

    window_size: int = 20
    sigma_threshold: float = 15.0  # milliseconds
    kl_threshold: float = 2.0     # nats
    noise_sigma_ms: float = 5.0
    rate_limit_factor: float = 2.0
    cooldown_queries: int = 100
    benign_distribution: Optional[np.ndarray] = None
    benign_bin_edges: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.benign_distribution is None:
            # Default benign distribution: approximate normal around 50ms mean.
            self.benign_bin_edges = np.array(
                [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 500],
                dtype=np.float64,
            )
            # Typical serving latency distribution (roughly log-normal).
            raw = np.array(
                [0.02, 0.05, 0.10, 0.15, 0.20, 0.18, 0.12, 0.08, 0.04, 0.03, 0.02, 0.005, 0.005],
                dtype=np.float64,
            )
            self.benign_distribution = raw / raw.sum()


@dataclass
class TenantTimingState:
    """Per-tenant DFA state and timing history."""

    state: TimingDFAState = TimingDFAState.NORMAL
    response_times_ms: Deque[float] = field(default_factory=lambda: deque(maxlen=200))
    queries_in_state: int = 0
    attack_confirmed_at: Optional[float] = None
    total_queries: int = 0
    noise_injected_count: int = 0
    rate_limited_count: int = 0


class TimingDFA:
    """Per-tenant timing attack DFA.

    Usage:
        dfa = TimingDFA(config)
        # On each request completion:
        action = dfa.observe(tenant_id, response_time_ms)
        if action.inject_noise:
            time.sleep(action.noise_delay_ms / 1000.0)
    """

    def __init__(self, config: Optional[TimingDFAConfig] = None):
        self.config = config or TimingDFAConfig()
        self._tenants: Dict[TenantId, TenantTimingState] = {}
        self._lock = threading.Lock()

    def _get_tenant(self, tenant_id: TenantId) -> TenantTimingState:
        if tenant_id not in self._tenants:
            self._tenants[tenant_id] = TenantTimingState()
        return self._tenants[tenant_id]

    def observe(self, tenant_id: TenantId, response_time_ms: float) -> "DFAAction":
        """Record a query-response observation and advance the DFA.

        Args:
            tenant_id:        The tenant whose response was observed.
            response_time_ms: Response latency in milliseconds.

        Returns:
            DFAAction indicating whether noise/rate-limiting should be applied.
        """
        with self._lock:
            ts = self._get_tenant(tenant_id)
            ts.response_times_ms.append(response_time_ms)
            ts.total_queries += 1
            ts.queries_in_state += 1

            action = DFAAction()

            if ts.state == TimingDFAState.NORMAL:
                self._check_normal_to_probe1(ts)

            elif ts.state == TimingDFAState.TIMING_PROBE_1:
                self._check_probe1_to_probe2(ts)

            elif ts.state == TimingDFAState.TIMING_PROBE_2:
                self._check_probe2_to_confirmed(ts)

            elif ts.state == TimingDFAState.ATTACK_CONFIRMED:
                action = self._handle_attack_confirmed(ts)

            return action

    def get_state(self, tenant_id: TenantId) -> TimingDFAState:
        """Get the current DFA state for a tenant."""
        with self._lock:
            return self._get_tenant(tenant_id).state

    def get_stats(self, tenant_id: TenantId) -> Dict:
        """Get statistics for a tenant."""
        with self._lock:
            ts = self._get_tenant(tenant_id)
            return {
                "state": ts.state.value,
                "total_queries": ts.total_queries,
                "queries_in_state": ts.queries_in_state,
                "noise_injected_count": ts.noise_injected_count,
                "rate_limited_count": ts.rate_limited_count,
                "current_sigma_ms": self._compute_sigma(ts),
            }

    def reset(self, tenant_id: TenantId) -> None:
        """Reset a tenant's DFA state."""
        with self._lock:
            if tenant_id in self._tenants:
                del self._tenants[tenant_id]

    # ------------------------------------------------------------------
    # DFA transition logic
    # ------------------------------------------------------------------

    def _compute_sigma(self, ts: TenantTimingState) -> float:
        """Compute std of recent response times."""
        if len(ts.response_times_ms) < self.config.window_size:
            return 0.0
        recent = list(ts.response_times_ms)[-self.config.window_size :]
        return float(np.std(recent))

    def _compute_kl_divergence(self, ts: TenantTimingState) -> float:
        """Compute KL divergence of recent timing distribution vs. benign."""
        if len(ts.response_times_ms) < self.config.window_size:
            return 0.0
        recent = np.array(list(ts.response_times_ms)[-self.config.window_size :])
        # Compute empirical histogram using benign bin edges.
        observed, _ = np.histogram(recent, bins=self.config.benign_bin_edges)
        # Normalize to probability distribution with Laplace smoothing.
        eps = 1e-10
        p = observed.astype(np.float64) + eps
        p = p / p.sum()
        q = self.config.benign_distribution + eps
        q = q / q.sum()
        # KL(P || Q).
        kl = float(np.sum(p * np.log(p / q)))
        return kl

    def _check_normal_to_probe1(self, ts: TenantTimingState) -> None:
        sigma = self._compute_sigma(ts)
        if sigma > self.config.sigma_threshold:
            ts.state = TimingDFAState.TIMING_PROBE_1
            ts.queries_in_state = 0

    def _check_probe1_to_probe2(self, ts: TenantTimingState) -> None:
        if ts.queries_in_state < self.config.window_size:
            return
        sigma = self._compute_sigma(ts)
        if sigma > self.config.sigma_threshold:
            ts.state = TimingDFAState.TIMING_PROBE_2
            ts.queries_in_state = 0
        else:
            # Sigma normalized — go back to NORMAL.
            ts.state = TimingDFAState.NORMAL
            ts.queries_in_state = 0

    def _check_probe2_to_confirmed(self, ts: TenantTimingState) -> None:
        if ts.queries_in_state < self.config.window_size:
            return
        kl = self._compute_kl_divergence(ts)
        if kl > self.config.kl_threshold:
            ts.state = TimingDFAState.ATTACK_CONFIRMED
            ts.attack_confirmed_at = time.monotonic()
            ts.queries_in_state = 0
        else:
            # Distribution normalized — go back to NORMAL.
            ts.state = TimingDFAState.NORMAL
            ts.queries_in_state = 0

    def _handle_attack_confirmed(self, ts: TenantTimingState) -> "DFAAction":
        action = DFAAction(
            inject_noise=True,
            noise_delay_ms=abs(np.random.normal(0, self.config.noise_sigma_ms)),
            rate_limit=True,
            rate_limit_factor=self.config.rate_limit_factor,
        )
        ts.noise_injected_count += 1
        ts.rate_limited_count += 1

        # Cooldown: if enough queries have passed, check if behavior normalized.
        if ts.queries_in_state >= self.config.cooldown_queries:
            sigma = self._compute_sigma(ts)
            kl = self._compute_kl_divergence(ts)
            if sigma <= self.config.sigma_threshold and kl <= self.config.kl_threshold:
                ts.state = TimingDFAState.NORMAL
                ts.queries_in_state = 0
                ts.attack_confirmed_at = None

        return action


@dataclass
class DFAAction:
    """Action returned by the DFA on each observation.

    The serving middleware should apply these actions to the tenant's
    subsequent requests.
    """

    inject_noise: bool = False
    noise_delay_ms: float = 0.0
    rate_limit: bool = False
    rate_limit_factor: float = 1.0


class TimingDFAMiddleware:
    """Serving middleware that wraps request handling with DFA-based timing defense.

    Usage:
        middleware = TimingDFAMiddleware(dfa)

        # In request handler:
        async def handle_request(request):
            tenant_id = request.tenant_id
            start = time.monotonic()
            response = await process_request(request)
            elapsed_ms = (time.monotonic() - start) * 1000
            action = middleware.post_request(tenant_id, elapsed_ms)
            if action.inject_noise:
                await asyncio.sleep(action.noise_delay_ms / 1000.0)
            return response
    """

    def __init__(self, dfa: Optional[TimingDFA] = None):
        self.dfa = dfa or TimingDFA()

    def post_request(self, tenant_id: TenantId, response_time_ms: float) -> DFAAction:
        """Called after each request completes. Returns action for this tenant."""
        return self.dfa.observe(tenant_id, response_time_ms)

    def should_delay_request(self, tenant_id: TenantId) -> float:
        """Returns additional delay in ms if tenant is rate-limited, else 0."""
        state = self.dfa.get_state(tenant_id)
        if state == TimingDFAState.ATTACK_CONFIRMED:
            return self.dfa.config.noise_sigma_ms * self.dfa.config.rate_limit_factor
        return 0.0
