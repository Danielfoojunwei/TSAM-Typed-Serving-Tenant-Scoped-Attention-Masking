"""
Real-model end-to-end noninterference experiments.

Runs TSAM isolation experiments on real pretrained models (Qwen2.5-1.5B,
GPT-2, etc.) to answer: does per-layer attention masking guarantee
end-to-end noninterference through a FULL production transformer stack?

This is NOT a toy experiment. We hook TSAM masking into every attention
layer of a real 28-layer, 1.5B-parameter GQA model with pretrained weights,
and measure whether cross-tenant information leaks through:
  - Residual connections
  - LayerNorm statistics
  - MLP transformations
  - GQA head expansion
  - 28 layers of depth

Key insight: we DON'T need to modify the model architecture. We construct
multi-tenant input sequences where different positions belong to different
tenants, and inject a TSAM attention mask that blocks cross-tenant positions.
The model's own attention mechanism is used — we just control what it can see.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class RealModelLeakageMetrics:
    """Per-layer leakage measurement on a real model."""

    layer_index: int
    fi_l2: float           # ||h_A - h_B||_2 at attacker's positions
    fi_linf: float         # ||h_A - h_B||_inf
    fi_relative: float     # ||h_A - h_B|| / ||h_A|| (relative change)
    is_zero: bool          # Bitwise identical?


@dataclass
class RealModelExperimentResult:
    """Full experiment result on a real model."""

    model_name: str
    num_layers: int
    d_model: int
    num_heads: int
    num_kv_heads: int
    total_params_m: float
    condition: str                    # "tsam", "vanilla"
    num_trials: int

    per_layer_fi_l2: List[float]      # Mean FI L2 at each layer
    per_layer_fi_linf: List[float]
    per_layer_fi_relative: List[float]
    per_layer_all_zero: List[bool]

    final_logit_fi_l2: float
    final_logit_fi_linf: float

    probing_accuracy: float
    probing_baseline: float           # 1/num_classes

    end_to_end_isolated: bool         # All layers zero + logits zero
    wall_time_sec: float


def _build_tsam_attention_mask(
    seq_len: int,
    shared_range: Tuple[int, int],
    own_range: Tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """Build a TSAM-style causal attention mask for HuggingFace models.

    Creates a (1, 1, seq_len, seq_len) float mask where:
      - Positions in shared_range and own_range: 0.0 (attend)
      - All other positions: -inf (blocked)
      - Combined with causal masking (upper triangle = -inf)

    This is the format expected by HuggingFace's attention_mask parameter.
    """
    # Start with causal mask.
    mask = torch.full((seq_len, seq_len), -float("inf"), device=device)
    mask = torch.triu(mask, diagonal=1)  # Upper triangle = -inf (causal)

    # Now block cross-tenant positions in the KV dimension (columns).
    # For each query position, which KV positions can it attend to?
    visible = torch.zeros(seq_len, dtype=torch.bool, device=device)
    visible[shared_range[0]:shared_range[1]] = True
    visible[own_range[0]:own_range[1]] = True

    # Block non-visible KV positions across ALL query positions.
    blocked_kv = ~visible  # Positions that should be blocked
    mask[:, blocked_kv] = -float("inf")

    # Re-apply causal mask (query can't attend to future positions).
    causal = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    mask[causal] = -float("inf")

    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, S, S)


def run_real_model_experiment(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    num_trials: int = 30,
    shared_len: int = 20,
    private_len: int = 20,
    num_probe_classes: int = 4,
    condition: str = "tsam",
    device: str = "cpu",
) -> RealModelExperimentResult:
    """Run end-to-end noninterference experiment on a real pretrained model.

    Protocol:
      1. Load real model + tokenizer.
      2. For each trial:
         a. Construct input: [shared_prefix | victim_private | attacker_private]
         b. Build TSAM mask that blocks victim_private for the attacker.
         c. Run model TWICE with different victim_private content (A vs B).
         d. Compare attacker's hidden states: should be IDENTICAL under TSAM.
      3. Collect per-layer metrics + final logit metrics.
      4. Run probing attack on attacker's final hidden states.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        output_hidden_states=True,
    )
    model.eval()
    model.to(device)

    cfg = model.config
    num_layers = cfg.num_hidden_layers
    d_model = cfg.hidden_size
    num_heads = cfg.num_attention_heads
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6

    seq_len = shared_len + private_len * 2  # shared + victim + attacker

    # Token ranges in the sequence.
    shared_range = (0, shared_len)
    victim_range = (shared_len, shared_len + private_len)
    attacker_range = (shared_len + private_len, seq_len)

    # Generate fixed shared prefix and attacker tokens.
    rng = torch.Generator()
    rng.manual_seed(42)
    vocab_size = cfg.vocab_size

    shared_tokens = torch.randint(100, vocab_size - 100, (shared_len,), generator=rng)
    attacker_tokens = torch.randint(100, vocab_size - 100, (private_len,), generator=rng)

    # Generate distinct victim patterns.
    victim_patterns = []
    for c in range(num_probe_classes):
        pattern = torch.randint(100, vocab_size - 100, (private_len,), generator=rng)
        victim_patterns.append(pattern)

    # Build TSAM mask for the attacker.
    # The attacker can see: shared_range + attacker_range. Blocked: victim_range.
    if condition == "tsam":
        tsam_mask = _build_tsam_attention_mask(
            seq_len=seq_len,
            shared_range=shared_range,
            own_range=attacker_range,
            device=device,
        )
    else:
        tsam_mask = None  # Vanilla: full causal attention, no tenant blocking.

    # Collect per-layer FI measurements.
    layer_fi_l2_all: List[List[float]] = [[] for _ in range(num_layers + 1)]
    layer_fi_linf_all: List[List[float]] = [[] for _ in range(num_layers + 1)]
    layer_fi_rel_all: List[List[float]] = [[] for _ in range(num_layers + 1)]

    logit_fi_l2_all: List[float] = []
    logit_fi_linf_all: List[float] = []

    probe_features: List[np.ndarray] = []
    probe_labels: List[int] = []

    with torch.no_grad():
        for trial in range(num_trials):
            class_a = trial % num_probe_classes
            class_b = (trial + num_probe_classes // 2) % num_probe_classes
            if class_b == class_a:
                class_b = (class_a + 1) % num_probe_classes

            victim_a = victim_patterns[class_a]
            victim_b = victim_patterns[class_b]

            # Build input sequences.
            input_a = torch.cat([shared_tokens, victim_a, attacker_tokens]).unsqueeze(0).to(device)
            input_b = torch.cat([shared_tokens, victim_b, attacker_tokens]).unsqueeze(0).to(device)

            # Forward pass A.
            if tsam_mask is not None:
                out_a = model(input_ids=input_a, attention_mask=tsam_mask)
            else:
                out_a = model(input_ids=input_a)

            # Forward pass B.
            if tsam_mask is not None:
                out_b = model(input_ids=input_b, attention_mask=tsam_mask)
            else:
                out_b = model(input_ids=input_b)

            # Extract attacker's hidden states at each layer.
            # hidden_states[0] = embeddings, hidden_states[1..L] = layer outputs
            hidden_a = out_a.hidden_states  # Tuple of (1, seq_len, d_model)
            hidden_b = out_b.hidden_states

            for layer_idx in range(len(hidden_a)):
                # Attacker's positions.
                ha = hidden_a[layer_idx][0, attacker_range[0]:attacker_range[1]]
                hb = hidden_b[layer_idx][0, attacker_range[0]:attacker_range[1]]

                diff = ha - hb
                fi_l2 = float(torch.norm(diff, p=2).item())
                fi_linf = float(torch.norm(diff, p=float("inf")).item())
                ha_norm = float(torch.norm(ha, p=2).item())
                fi_rel = fi_l2 / max(ha_norm, 1e-10)

                layer_fi_l2_all[layer_idx].append(fi_l2)
                layer_fi_linf_all[layer_idx].append(fi_linf)
                layer_fi_rel_all[layer_idx].append(fi_rel)

            # Final logits at attacker positions.
            logits_a = out_a.logits[0, attacker_range[0]:attacker_range[1]]
            logits_b = out_b.logits[0, attacker_range[0]:attacker_range[1]]
            diff_logits = logits_a - logits_b
            logit_fi_l2_all.append(float(torch.norm(diff_logits, p=2).item()))
            logit_fi_linf_all.append(float(torch.norm(diff_logits, p=float("inf")).item()))

            # Collect features for probing.
            # Use last hidden state at attacker positions as features.
            last_hidden = hidden_a[-1][0, attacker_range[0]:attacker_range[1]]
            probe_features.append(last_hidden.cpu().numpy().flatten())
            probe_labels.append(class_a)

    # Aggregate per-layer metrics.
    per_layer_fi_l2 = [float(np.mean(x)) for x in layer_fi_l2_all]
    per_layer_fi_linf = [float(np.mean(x)) for x in layer_fi_linf_all]
    per_layer_fi_rel = [float(np.mean(x)) for x in layer_fi_rel_all]
    per_layer_zero = [bool(np.all(np.array(x) == 0.0)) for x in layer_fi_l2_all]

    # Probing attack.
    X = np.array(probe_features)
    y = np.array(probe_labels)
    n = len(y)
    n_train = max(num_probe_classes + 1, int(0.7 * n))
    n_test = n - n_train

    if n_test > 0:
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]

        Y_train = np.zeros((n_train, num_probe_classes))
        Y_train[np.arange(n_train), y_train] = 1.0
        lam = 1e-2
        XtX = X_train.T @ X_train + lam * np.eye(X_train.shape[1])
        XtY = X_train.T @ Y_train
        W = np.linalg.solve(XtX, XtY)
        preds = X_test @ W
        probe_acc = float((preds.argmax(axis=1) == y_test).mean())
    else:
        probe_acc = 1.0 / num_probe_classes

    probe_baseline = 1.0 / num_probe_classes

    final_fi_l2 = float(np.mean(logit_fi_l2_all))
    final_fi_linf = float(np.mean(logit_fi_linf_all))
    all_zero = all(per_layer_zero)
    e2e = all_zero and final_fi_l2 == 0.0

    wall_time = time.time() - t0

    return RealModelExperimentResult(
        model_name=model_name,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        total_params_m=total_params,
        condition=condition,
        num_trials=num_trials,
        per_layer_fi_l2=per_layer_fi_l2,
        per_layer_fi_linf=per_layer_fi_linf,
        per_layer_fi_relative=per_layer_fi_rel,
        per_layer_all_zero=per_layer_zero,
        final_logit_fi_l2=final_fi_l2,
        final_logit_fi_linf=final_fi_linf,
        probing_accuracy=probe_acc,
        probing_baseline=probe_baseline,
        end_to_end_isolated=e2e,
        wall_time_sec=wall_time,
    )


def run_full_experiment_suite(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    num_trials: int = 30,
    device: str = "cpu",
) -> Dict[str, RealModelExperimentResult]:
    """Run both TSAM and vanilla conditions on the same model.

    Returns:
        {"tsam": result, "vanilla": result}
    """
    results = {}
    for condition in ["tsam", "vanilla"]:
        print(f"\n{'='*60}")
        print(f"Running {condition.upper()} condition on {model_name}")
        print(f"{'='*60}")
        result = run_real_model_experiment(
            model_name=model_name,
            num_trials=num_trials,
            condition=condition,
            device=device,
        )
        results[condition] = result

        # Print summary.
        print(f"\nCondition: {condition}")
        print(f"  End-to-end isolated: {result.end_to_end_isolated}")
        print(f"  Final logit FI L2:   {result.final_logit_fi_l2:.6f}")
        print(f"  Probing accuracy:    {result.probing_accuracy:.3f} (baseline: {result.probing_baseline:.3f})")
        print(f"  Wall time:           {result.wall_time_sec:.1f}s")
        print(f"  Per-layer FI L2:")
        for i, fi in enumerate(result.per_layer_fi_l2):
            zero = "ZERO" if result.per_layer_all_zero[i] else f"{fi:.6f}"
            label = "embed" if i == 0 else f"layer {i}"
            print(f"    {label:>10}: {zero}")

    return results
