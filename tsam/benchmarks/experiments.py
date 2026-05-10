"""
Experiment runners — orchestrates the full experimental protocol (Experiments 1-5).

Each experiment function runs the complete protocol, collects results,
and returns structured output for analysis and plotting.

IMPORTANT: Experiment 1 measures *actual* mutual information between
attention outputs and private KV content using the KSG estimator, not
merely boolean mask correctness.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from tsam.benchmarks.scalability import ScalabilityBenchmark, ScalabilityResult
from tsam.benchmarks.throughput import ThroughputBenchmark, ThroughputResult
from tsam.benchmarks.prefix_sharing import PrefixSharingBenchmark, PrefixSharingResult
from tsam.evaluation.collision_attack import CollisionAttackSimulator, CollisionAttackResult
from tsam.evaluation.leakage import MutualInformationEstimator, LeakageEstimator
from tsam.dai.timing_dfa import TimingDFA, TimingDFAConfig, TimingDFAMiddleware


@dataclass
class ExperimentSuite:
    """Results from the full experimental protocol."""

    experiment_1_isolation: Optional[Dict[str, Any]] = None
    experiment_2_scalability: Optional[Dict[str, Any]] = None
    experiment_3_prefix: Optional[Dict[str, Any]] = None
    experiment_4_timing: Optional[Dict[str, Any]] = None
    experiment_5_quality: Optional[Dict[str, Any]] = None

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)


def run_experiment_1_isolation(
    device: str = "cpu",
    num_attack_queries: int = 100,
) -> Dict[str, Any]:
    """Experiment 1 — Isolation Correctness Verification.

    Measures I(output(q_i); KV_private(t_j)) using the KSG mutual information
    estimator on actual attention outputs — not merely boolean mask checks.

    Protocol:
      1. Set up a mixed block table with shared + victim-private + attacker-private.
      2. Run num_attack_queries with random queries from the attacker.
      3. Collect TSAM outputs and vanilla (no-mask) outputs.
      4. Measure MI between each set of outputs and the victim's private KV.
      5. TSAM MI should be ~0; vanilla MI should be significantly > 0.
      6. Also verify mask correctness as a sanity check (not the primary metric).
    """
    from tsam.core.block_manager import TypedBlockManager
    from tsam.attention.masking import generate_tsam_mask
    from tsam.attention.paged_attention import tsam_paged_attention

    page_size = 16
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    private_blocks_per_tenant = 8  # 128 tokens
    shared_blocks = 4  # 64 tokens

    total_blocks = shared_blocks + 2 * private_blocks_per_tenant + 1
    bm = TypedBlockManager(num_blocks=total_blocks, page_size=page_size, device=device)

    shared_pages = bm.allocate_shared(count=shared_blocks)
    shared_ids = [p.block_id for p in shared_pages]

    # Tenant A (victim) — private prompt.
    victim_pages = bm.allocate_private(tenant=0, count=private_blocks_per_tenant)
    victim_ids = [p.block_id for p in victim_pages]

    # Tenant B (attacker) — private context.
    attacker_pages = bm.allocate_private(tenant=1, count=private_blocks_per_tenant)
    attacker_ids = [p.block_id for p in attacker_pages]

    # Attacker's block table includes shared + victim + attacker blocks
    # (worst case: attacker can address victim's physical blocks).
    mixed_bt = shared_ids + victim_ids + attacker_ids
    block_table = torch.tensor(mixed_bt, dtype=torch.int32, device=device)
    seq_len_kv = len(mixed_bt) * page_size

    # Collect outputs from both TSAM-protected and vanilla (unmasked) attention.
    #
    # Methodology: For each query, we compute the output twice with two
    # different victim KV contents (KV_A and KV_B). If the defense works,
    # the output should be identical regardless of the victim's private KV.
    # If it doesn't work (vanilla), the output changes.
    #
    # Metrics:
    #   1. Functional independence: ||output(KV_A) - output(KV_B)||
    #      Should be 0.0 for TSAM, > 0 for vanilla.
    #   2. MI estimation: I(output; victim_kv) via KSG estimator.
    #   3. Mask correctness: sanity check on boolean mask.
    tsam_outputs: List[np.ndarray] = []
    vanilla_outputs: List[np.ndarray] = []
    victim_kv_samples: List[np.ndarray] = []
    tsam_diffs: List[float] = []     # ||output_A - output_B|| under TSAM
    vanilla_diffs: List[float] = []  # ||output_A - output_B|| without TSAM
    mask_all_correct = True

    for q in range(num_attack_queries):
        # Base KV cache (shared + attacker content fixed).
        key_cache = torch.randn(
            total_blocks, page_size, num_kv_heads, head_dim, device=device
        )
        value_cache = torch.randn(
            total_blocks, page_size, num_kv_heads, head_dim, device=device
        )

        # Record the victim's private KV (version A) for MI estimation.
        victim_kv_flat = []
        for blk in victim_ids:
            victim_kv_flat.append(key_cache[blk].detach().cpu().numpy().flatten())
        victim_kv_samples.append(np.concatenate(victim_kv_flat))

        # Same query used for all comparisons in this trial.
        query = torch.randn(num_heads, head_dim, device=device)

        # --- Compute outputs with victim KV version A ---
        output_tsam_a = tsam_paged_attention(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            page_type=bm.page_type,
            tenant_id_meta=bm.tenant_id,
            query_tenant=1,  # Attacker
            page_size=page_size,
            seq_len_kv=seq_len_kv,
        )
        tsam_outputs.append(output_tsam_a.detach().cpu().numpy().flatten())

        # Vanilla output version A (no TSAM masking — attacker sees everything).
        scale = 1.0 / (head_dim ** 0.5)
        all_keys = key_cache[block_table].reshape(-1, num_kv_heads, head_dim)[:seq_len_kv]
        all_values = value_cache[block_table].reshape(-1, num_kv_heads, head_dim)[:seq_len_kv]
        num_groups = num_heads // num_kv_heads
        if num_groups > 1:
            all_keys = all_keys.unsqueeze(2).expand(-1, -1, num_groups, -1).reshape(
                seq_len_kv, num_heads, head_dim
            )
            all_values = all_values.unsqueeze(2).expand(-1, -1, num_groups, -1).reshape(
                seq_len_kv, num_heads, head_dim
            )
        q_expanded = query.unsqueeze(1)  # (num_heads, 1, head_dim)
        k_t = all_keys.permute(1, 2, 0)
        scores_vanilla_a = torch.bmm(q_expanded, k_t) * scale
        weights_vanilla_a = torch.nn.functional.softmax(scores_vanilla_a, dim=-1)
        v_t = all_values.permute(1, 0, 2)
        output_vanilla_a = torch.bmm(weights_vanilla_a, v_t).squeeze(1)
        vanilla_outputs.append(output_vanilla_a.detach().cpu().numpy().flatten())

        # --- Compute outputs with victim KV version B (different random content) ---
        # Replace only the victim's blocks with fresh random data.
        key_cache_b = key_cache.clone()
        value_cache_b = value_cache.clone()
        for blk in victim_ids:
            key_cache_b[blk] = torch.randn(page_size, num_kv_heads, head_dim, device=device)
            value_cache_b[blk] = torch.randn(page_size, num_kv_heads, head_dim, device=device)

        output_tsam_b = tsam_paged_attention(
            query=query,
            key_cache=key_cache_b,
            value_cache=value_cache_b,
            block_table=block_table,
            page_type=bm.page_type,
            tenant_id_meta=bm.tenant_id,
            query_tenant=1,
            page_size=page_size,
            seq_len_kv=seq_len_kv,
        )

        all_keys_b = key_cache_b[block_table].reshape(-1, num_kv_heads, head_dim)[:seq_len_kv]
        all_values_b = value_cache_b[block_table].reshape(-1, num_kv_heads, head_dim)[:seq_len_kv]
        if num_groups > 1:
            all_keys_b = all_keys_b.unsqueeze(2).expand(-1, -1, num_groups, -1).reshape(
                seq_len_kv, num_heads, head_dim
            )
            all_values_b = all_values_b.unsqueeze(2).expand(-1, -1, num_groups, -1).reshape(
                seq_len_kv, num_heads, head_dim
            )
        scores_vanilla_b = torch.bmm(q_expanded, all_keys_b.permute(1, 2, 0)) * scale
        weights_vanilla_b = torch.nn.functional.softmax(scores_vanilla_b, dim=-1)
        output_vanilla_b = torch.bmm(weights_vanilla_b, all_values_b.permute(1, 0, 2)).squeeze(1)

        # Functional independence: measure how much the output changes
        # when the victim's private KV changes (A -> B).
        tsam_diff = float(torch.norm(output_tsam_a - output_tsam_b).item())
        vanilla_diff = float(torch.norm(output_vanilla_a - output_vanilla_b).item())
        tsam_diffs.append(tsam_diff)
        vanilla_diffs.append(vanilla_diff)

        # --- Sanity check: verify mask blocks victim positions ---
        victim_start = shared_blocks * page_size
        victim_end = victim_start + private_blocks_per_tenant * page_size
        mask_result = generate_tsam_mask(
            block_table=block_table,
            page_type=bm.page_type,
            tenant_id=bm.tenant_id,
            query_tenant=1,
            page_size=page_size,
            seq_len_kv=seq_len_kv,
            device=device,
        )
        victim_mask = mask_result.mask[victim_start:victim_end]
        if victim_mask.any():
            mask_all_correct = False

    # --- Mutual Information measurement ---
    # Each query used a different random victim KV, so victim_kv_samples
    # varies across queries. If the output depends on the victim's KV,
    # MI(output; victim_kv) > 0. If TSAM blocks it, MI should be ~0.
    tsam_outputs_arr = np.array(tsam_outputs)         # (num_queries, d_out)
    vanilla_outputs_arr = np.array(vanilla_outputs)    # (num_queries, d_out)
    victim_kv_arr = np.array(victim_kv_samples)        # (num_queries, d_private)

    # KSG estimator works best in low dimensions. Use PCA-like truncation:
    # take only the first D dimensions from each array to keep estimation
    # tractable and numerically stable. D = min(32, actual dims).
    d_trunc = min(32, tsam_outputs_arr.shape[1], victim_kv_arr.shape[1])
    tsam_for_mi = tsam_outputs_arr[:, :d_trunc]
    vanilla_for_mi = vanilla_outputs_arr[:, :d_trunc]
    private_for_mi = victim_kv_arr[:, :d_trunc]

    mi_estimator = MutualInformationEstimator(k=3)

    # MI(TSAM_output; victim_private) — should be ~0.
    mi_tsam_bits, mi_tsam_std = mi_estimator.estimate_bits(
        tsam_for_mi, private_for_mi
    )

    # MI(vanilla_output; victim_private) — should be significantly > 0.
    mi_vanilla_bits, mi_vanilla_std = mi_estimator.estimate_bits(
        vanilla_for_mi, private_for_mi
    )

    # --- Functional Independence summary ---
    mean_tsam_diff = float(np.mean(tsam_diffs))
    mean_vanilla_diff = float(np.mean(vanilla_diffs))
    max_tsam_diff = float(np.max(tsam_diffs))

    results = {
        "defense": "TSAM",
        "num_attack_queries": num_attack_queries,
        # Primary metric 1: Functional independence test.
        # If TSAM works, changing the victim's KV should have ZERO effect
        # on the attacker's output. Vanilla output should change significantly.
        "tsam_mean_output_diff": mean_tsam_diff,
        "tsam_max_output_diff": max_tsam_diff,
        "vanilla_mean_output_diff": mean_vanilla_diff,
        "functional_independence": mean_tsam_diff == 0.0,
        # Primary metric 2: MI-based leakage measurement.
        "tsam_mi_bits": mi_tsam_bits,
        "tsam_mi_std_bits": mi_tsam_std,
        "vanilla_mi_bits": mi_vanilla_bits,
        "vanilla_mi_std_bits": mi_vanilla_std,
        "mi_reduction_pct": (
            (1.0 - mi_tsam_bits / max(mi_vanilla_bits, 1e-10)) * 100.0
            if mi_vanilla_bits > 0 else 100.0
        ),
        # Sanity check: boolean mask correctness (necessary but not sufficient).
        "mask_all_correct": mask_all_correct,
        # Overall pass requires functional independence AND mask correctness.
        "pass": mask_all_correct and max_tsam_diff == 0.0,
    }

    return results


def run_experiment_2_scalability(
    device: str = "cpu",
    tenant_counts: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Experiment 2 — Scalability to Unlimited Tenants."""
    bench = ScalabilityBenchmark(device=device)
    results = bench.run_full_scalability_test(tenant_counts)

    output = {"TSAM": [], "Paper5": []}
    for key in ["TSAM", "Paper5"]:
        for r in results[key]:
            output[key].append(asdict(r))

    return output


def run_experiment_3_prefix_sharing(
    device: str = "cpu",
    num_tenants: int = 100,
    prefix_tokens: int = 1024,
    queries_per_tenant: int = 10,
) -> Dict[str, Any]:
    """Experiment 3 — Prefix Sharing Efficiency."""
    bench = PrefixSharingBenchmark(device=device)

    vanilla = bench.run_vanilla_prefix_caching(
        num_tenants=num_tenants,
        prefix_tokens=prefix_tokens,
        queries_per_tenant=queries_per_tenant,
    )
    tsam = bench.run_tsam_prefix_sharing(
        num_tenants=num_tenants,
        prefix_tokens=prefix_tokens,
        queries_per_tenant=queries_per_tenant,
    )
    paper5 = bench.run_paper5_no_prefix_sharing(
        num_tenants=num_tenants,
        prefix_tokens=prefix_tokens,
        queries_per_tenant=queries_per_tenant,
    )

    # Compute overhead.
    if vanilla.tokens_per_sec > 0:
        tsam_overhead = (1.0 - tsam.tokens_per_sec / vanilla.tokens_per_sec) * 100
        paper5_overhead = (1.0 - paper5.tokens_per_sec / vanilla.tokens_per_sec) * 100
    else:
        tsam_overhead = 0.0
        paper5_overhead = 0.0

    return {
        "vanilla": asdict(vanilla),
        "tsam": {**asdict(tsam), "overhead_vs_vanilla_pct": tsam_overhead},
        "paper5": {**asdict(paper5), "overhead_vs_vanilla_pct": paper5_overhead},
    }


def run_experiment_4_timing_defense(
    num_probes: int = 1000,
) -> Dict[str, Any]:
    """Experiment 4 — Timing Attack Defense.

    Measures Collision attack information leakage reduction under DAI timing DFA.
    """
    rng = np.random.default_rng(42)

    # Timing oracle: simulates cache hit/miss timing.
    base_hit_time = 30.0   # ms
    base_miss_time = 80.0  # ms
    hit_sigma = 5.0
    miss_sigma = 10.0

    cache_contents = set(range(0, 500))  # Prompts 0-499 are cached.

    # (a) No defense oracle.
    def oracle_no_defense(prompt_idx: int, _tenant: int) -> float:
        if prompt_idx in cache_contents:
            return max(1.0, rng.normal(base_hit_time, hit_sigma))
        return max(1.0, rng.normal(base_miss_time, miss_sigma))

    # (b) Noise-only oracle (Gaussian noise, no DFA).
    noise_sigma = 5.0

    def oracle_noise_only(prompt_idx: int, _tenant: int) -> float:
        base = oracle_no_defense(prompt_idx, _tenant)
        return max(1.0, base + rng.normal(0, noise_sigma))

    # (c) TSAM + DAI DFA oracle.
    dfa = TimingDFA(TimingDFAConfig(
        window_size=20,
        sigma_threshold=15.0,
        kl_threshold=2.0,
        noise_sigma_ms=5.0,
    ))

    def oracle_tsam_dfa(prompt_idx: int, tenant: int) -> float:
        base = oracle_no_defense(prompt_idx, tenant)
        action = dfa.observe(tenant, base)
        if action.inject_noise:
            base += action.noise_delay_ms
        return max(1.0, base)

    # Generate probe sequence.
    probe_indices = rng.integers(0, 1000, size=num_probes).tolist()
    ground_truth = [idx in cache_contents for idx in probe_indices]

    # Run attacks under each condition.
    results = {}
    for name, oracle_fn in [
        ("no_defense", oracle_no_defense),
        ("noise_only", oracle_noise_only),
        ("tsam_dfa", oracle_tsam_dfa),
    ]:
        # Wrap oracle to accept string prompt.
        def timing_oracle(prompt: str, tenant: int, _fn=oracle_fn) -> float:
            idx = int(prompt)
            return _fn(idx, tenant)

        attacker = CollisionAttackSimulator(
            timing_oracle=timing_oracle,
            num_calibration_probes=50,
        )

        # Calibrate.
        attacker.calibrate_threshold(
            known_hit_prompt="0",    # Known cached.
            known_miss_prompt="999", # Known not cached.
            tenant_id=99,
        )

        # Run attack.
        result = attacker.run_attack(
            probe_prompts=[str(idx) for idx in probe_indices],
            ground_truth_cached=ground_truth,
            attacker_tenant_id=99,
            defense_name=name,
        )
        results[name] = asdict(result)

    # Compute reduction.
    no_defense_bits = results["no_defense"]["bits_per_query"]
    for key in ["noise_only", "tsam_dfa"]:
        if no_defense_bits > 0:
            reduction = (1.0 - results[key]["bits_per_query"] / no_defense_bits) * 100
        else:
            reduction = 100.0
        results[key]["leakage_reduction_pct"] = reduction

    return results
