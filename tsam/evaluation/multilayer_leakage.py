"""
Multi-layer end-to-end noninterference measurement.

This module answers the central question: does per-layer TSAM attention masking
guarantee zero cross-tenant information flow through a full L-layer transformer?

We measure three complementary metrics at every layer and at the final output:

  1. Functional Independence (FI): ||output(KV_A) - output(KV_B)|| where A, B
     differ only in the victim's private tokens. FI = 0 iff output is
     structurally independent of the victim's data.

  2. Probing Accuracy (PA): A linear classifier trained to predict which of K
     victim private prompts was used, given the attacker's hidden states.
     PA = 1/K (random chance) iff no leakage.

  3. Cosine Leakage (CL): max cosine similarity between the attacker's output
     perturbation (when victim data changes) and the victim's data vector.
     CL ≈ 0 iff no directional leakage.

Experiment matrix:
  - Vary L in {1, 2, 4, 8, 12, 24}
  - Conditions: TSAM (masked), vanilla (unmasked), isolated (single-tenant)
  - Report metrics at every layer and at the final logits
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from tsam.model.transformer import (
    TSAMTransformer,
    TSAMTransformerConfig,
    MultiTenantBatch,
    build_multi_tenant_batch,
)


@dataclass
class LayerLeakageMetrics:
    """Leakage metrics at a single layer."""

    layer_index: int
    functional_independence_l2: float   # ||output_A - output_B||_2
    functional_independence_linf: float  # ||output_A - output_B||_inf
    cosine_leakage: float               # Max cosine similarity to victim data direction
    is_zero: bool                       # Whether FI is exactly 0.0


@dataclass
class EndToEndLeakageResult:
    """Complete end-to-end leakage measurement across all layers."""

    config: Dict                        # Model configuration
    condition: str                      # "tsam", "vanilla", or "isolated"
    num_trials: int
    num_layers: int

    # Per-layer metrics (averaged over trials).
    layer_metrics: List[LayerLeakageMetrics]

    # Final output metrics.
    final_fi_l2: float                  # FI at the logit level
    final_fi_linf: float
    final_probing_accuracy: float       # Linear probe accuracy on logits
    final_probing_baseline: float       # Random chance (1/num_classes)

    # Summary.
    all_layers_zero: bool               # Whether FI = 0 at ALL layers
    end_to_end_isolated: bool           # Whether the model is end-to-end noninterfering


def measure_functional_independence(
    model: TSAMTransformer,
    shared_prefix: torch.Tensor,
    private_a: torch.Tensor,
    private_b: torch.Tensor,
    attacker_private: torch.Tensor,
    max_seq_len: int,
    use_tsam: bool = True,
    device: str = "cpu",
) -> Tuple[List[LayerLeakageMetrics], torch.Tensor, torch.Tensor]:
    """Measure functional independence at every layer.

    Runs the model twice: once with victim's private data = A, once with B.
    The attacker's data stays the same. If TSAM works, the attacker's hidden
    states should be identical regardless of which victim data is used.

    Args:
        model: TSAMTransformer instance.
        shared_prefix: (shared_len,) tokens.
        private_a: (private_len,) victim's private tokens (version A).
        private_b: (private_len,) victim's private tokens (version B).
        attacker_private: (private_len,) attacker's private tokens.
        max_seq_len: Pad sequences to this length.
        use_tsam: If True, apply TSAM masking. If False, vanilla (no masking).
        device: Target device.

    Returns:
        (layer_metrics, logits_a, logits_b) where logits are the attacker's
        output logits under victim data A and B respectively.
    """
    model.eval()

    with torch.no_grad():
        # Build batch A: victim has private_a.
        batch_a = build_multi_tenant_batch(
            num_tenants=2,
            shared_prefix=shared_prefix,
            private_sequences={0: private_a, 1: attacker_private},
            max_seq_len=max_seq_len,
            device=device,
        )
        # Build batch B: victim has private_b (different content).
        batch_b = build_multi_tenant_batch(
            num_tenants=2,
            shared_prefix=shared_prefix,
            private_sequences={0: private_b, 1: attacker_private},
            max_seq_len=max_seq_len,
            device=device,
        )

        mask_a = batch_a.tsam_mask if use_tsam else None
        mask_b = batch_b.tsam_mask if use_tsam else None

        # Get hidden states at every layer.
        hidden_a = model.get_hidden_states(batch_a.input_ids, tsam_mask=mask_a)
        hidden_b = model.get_hidden_states(batch_b.input_ids, tsam_mask=mask_b)

        # Get final logits.
        logits_a = model(batch_a.input_ids, tsam_mask=mask_a)
        logits_b = model(batch_b.input_ids, tsam_mask=mask_b)

    # The attacker is batch element 1.
    attacker_idx = 1

    layer_metrics = []
    for layer_idx in range(len(hidden_a)):
        h_a = hidden_a[layer_idx][attacker_idx]  # (seq_len, d_model)
        h_b = hidden_b[layer_idx][attacker_idx]

        diff = h_a - h_b
        fi_l2 = float(torch.norm(diff, p=2).item())
        fi_linf = float(torch.norm(diff, p=float("inf")).item())

        # Cosine leakage: project the diff onto the victim's data direction.
        victim_h_a = hidden_a[layer_idx][0]  # Victim's hidden state
        victim_diff = victim_h_a.flatten()
        attacker_diff = diff.flatten()

        if victim_diff.norm() > 0 and attacker_diff.norm() > 0:
            cos = float(F.cosine_similarity(
                attacker_diff.unsqueeze(0),
                victim_diff.unsqueeze(0),
            ).item())
        else:
            cos = 0.0

        layer_metrics.append(LayerLeakageMetrics(
            layer_index=layer_idx,
            functional_independence_l2=fi_l2,
            functional_independence_linf=fi_linf,
            cosine_leakage=abs(cos),
            is_zero=(fi_l2 == 0.0),
        ))

    return layer_metrics, logits_a[attacker_idx], logits_b[attacker_idx]


def run_end_to_end_experiment(
    num_layers: int = 6,
    d_model: int = 256,
    num_heads: int = 4,
    num_trials: int = 50,
    shared_len: int = 16,
    private_len: int = 16,
    num_probe_classes: int = 4,
    vocab_size: int = 1000,
    device: str = "cpu",
    condition: str = "tsam",
) -> EndToEndLeakageResult:
    """Run the full end-to-end noninterference experiment.

    Args:
        num_layers: Number of transformer layers.
        d_model: Model dimension.
        num_heads: Number of attention heads.
        num_trials: Number of independent trials.
        shared_len: Length of shared prefix.
        private_len: Length of each tenant's private region.
        num_probe_classes: Number of distinct victim patterns for probing attack.
        vocab_size: Vocabulary size.
        device: Target device.
        condition: "tsam" (masked), "vanilla" (unmasked), or "isolated" (single tenant).

    Returns:
        EndToEndLeakageResult with per-layer and end-to-end metrics.
    """
    config = TSAMTransformerConfig(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_model * 4,
        vocab_size=vocab_size,
        max_seq_len=shared_len + private_len * 2 + 16,  # Room for 2 tenants + padding
        page_size=16,
    )
    max_seq_len = config.max_seq_len

    model = TSAMTransformer(config).to(device)
    model.eval()

    use_tsam = (condition == "tsam")

    # Create distinct victim patterns for probing.
    rng = torch.Generator()
    rng.manual_seed(42)
    victim_patterns = [
        torch.randint(0, vocab_size, (private_len,), generator=rng)
        for _ in range(num_probe_classes)
    ]

    # Collect per-layer FI measurements and probing data.
    all_layer_fi_l2: List[List[float]] = [[] for _ in range(num_layers + 1)]
    all_layer_fi_linf: List[List[float]] = [[] for _ in range(num_layers + 1)]
    all_layer_cos: List[List[float]] = [[] for _ in range(num_layers + 1)]

    # For probing attack.
    probe_logits_tsam: List[np.ndarray] = []
    probe_logits_vanilla: List[np.ndarray] = []
    probe_labels: List[int] = []

    shared_prefix = torch.randint(0, vocab_size, (shared_len,))
    attacker_private = torch.randint(0, vocab_size, (private_len,))

    for trial in range(num_trials):
        # Pick two different victim patterns.
        class_a = trial % num_probe_classes
        class_b = (trial + 1) % num_probe_classes
        private_a = victim_patterns[class_a]
        private_b = victim_patterns[class_b]

        layer_metrics, logits_a, logits_b = measure_functional_independence(
            model=model,
            shared_prefix=shared_prefix,
            private_a=private_a,
            private_b=private_b,
            attacker_private=attacker_private,
            max_seq_len=max_seq_len,
            use_tsam=use_tsam,
            device=device,
        )

        for lm in layer_metrics:
            idx = lm.layer_index
            all_layer_fi_l2[idx].append(lm.functional_independence_l2)
            all_layer_fi_linf[idx].append(lm.functional_independence_linf)
            all_layer_cos[idx].append(lm.cosine_leakage)

        # Collect logits for probing (use class_a as the label).
        probe_logits_tsam.append(logits_a.cpu().numpy().flatten())
        probe_labels.append(class_a)

    # Aggregate per-layer metrics.
    layer_results = []
    for idx in range(num_layers + 1):
        fi_l2_arr = np.array(all_layer_fi_l2[idx])
        fi_linf_arr = np.array(all_layer_fi_linf[idx])
        cos_arr = np.array(all_layer_cos[idx])

        layer_results.append(LayerLeakageMetrics(
            layer_index=idx,
            functional_independence_l2=float(fi_l2_arr.mean()),
            functional_independence_linf=float(fi_linf_arr.mean()),
            cosine_leakage=float(cos_arr.mean()),
            is_zero=bool(np.all(fi_l2_arr == 0.0)),
        ))

    # Final output metrics.
    final_fi_l2 = layer_results[-1].functional_independence_l2
    final_fi_linf = layer_results[-1].functional_independence_linf

    # Probing attack on final logits.
    X = np.array(probe_logits_tsam)
    y = np.array(probe_labels)
    n = len(y)
    n_train = max(1, int(0.7 * n))

    if n_train > num_probe_classes and n - n_train > 0:
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]

        # One-hot + ridge regression.
        Y_train = np.zeros((len(y_train), num_probe_classes))
        Y_train[np.arange(len(y_train)), y_train] = 1.0
        lam = 1e-3
        XtX = X_train.T @ X_train + lam * np.eye(X_train.shape[1])
        XtY = X_train.T @ Y_train
        W = np.linalg.solve(XtX, XtY)
        preds = X_test @ W
        probe_accuracy = float((preds.argmax(axis=1) == y_test).mean())
    else:
        probe_accuracy = 1.0 / num_probe_classes

    probe_baseline = 1.0 / num_probe_classes

    all_layers_zero = all(lm.is_zero for lm in layer_results)
    end_to_end = all_layers_zero and final_fi_l2 == 0.0

    return EndToEndLeakageResult(
        config={
            "num_layers": num_layers,
            "d_model": d_model,
            "num_heads": num_heads,
            "condition": condition,
        },
        condition=condition,
        num_trials=num_trials,
        num_layers=num_layers,
        layer_metrics=layer_results,
        final_fi_l2=final_fi_l2,
        final_fi_linf=final_fi_linf,
        final_probing_accuracy=probe_accuracy,
        final_probing_baseline=probe_baseline,
        all_layers_zero=all_layers_zero,
        end_to_end_isolated=end_to_end,
    )


def run_layer_depth_sweep(
    layer_counts: Optional[List[int]] = None,
    num_trials: int = 50,
    device: str = "cpu",
) -> Dict[str, List[EndToEndLeakageResult]]:
    """Sweep over transformer depths to characterize how isolation scales with L.

    This is the canonical experiment: for each depth L, measure whether TSAM
    maintains zero leakage across L layers, and how vanilla leakage grows.

    Returns:
        {"tsam": [...], "vanilla": [...]} mapping condition to results per depth.
    """
    if layer_counts is None:
        layer_counts = [1, 2, 4, 8, 12]

    results: Dict[str, List[EndToEndLeakageResult]] = {"tsam": [], "vanilla": []}

    for L in layer_counts:
        for condition in ["tsam", "vanilla"]:
            result = run_end_to_end_experiment(
                num_layers=L,
                num_trials=num_trials,
                condition=condition,
                device=device,
            )
            results[condition].append(result)

    return results
