"""
End-to-end noninterference experiment — canonical implementation.

Empirically verifies Theorem 3 (End-to-End Noninterference): per-layer
TSAM attention masking guarantees zero cross-tenant information flow
through the full L-layer transformer stack.

Experiments:
  1. Functional Independence (FI): measures ||output_A - output_B|| at
     attacker positions when only the victim's private tokens change.
  2. Probing Attack: trains a linear classifier to extract victim identity
     from the attacker's output logits.

Both are measured across depths L in {1, 2, 4, 8, 12, 24} under two
conditions: TSAM (masked) and vanilla (unmasked causal attention).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np
import torch

from tsam.model.transformer import TSAMTransformer, TSAMTransformerConfig


@dataclass
class E2EResult:
    """Result from one (depth, condition) experiment."""

    num_layers: int
    condition: str                  # "tsam" or "vanilla"
    d_model: int
    num_heads: int
    num_trials: int

    # Functional independence.
    fi_mean: float                  # Mean ||logits_A - logits_B||_2
    fi_max: float                   # Max across all trials
    fi_all_exact_zero: bool         # Whether ALL trials produced bitwise 0.0

    # Probing attack.
    probe_accuracy: float           # Linear probe test accuracy
    probe_baseline: float           # Random chance (1/num_classes)
    probe_num_classes: int


def run_functional_independence(
    num_layers: int,
    condition: str = "tsam",
    num_trials: int = 50,
    d_model: int = 256,
    num_heads: int = 4,
    shared_len: int = 16,
    private_len: int = 16,
    vocab_size: int = 1000,
) -> Dict:
    """Measure functional independence at a given depth."""
    config = TSAMTransformerConfig(
        num_layers=num_layers, d_model=d_model, num_heads=num_heads,
        d_ff=d_model * 4, vocab_size=vocab_size,
        max_seq_len=shared_len + private_len * 2 + 16,
    )
    model = TSAMTransformer(config)
    model.eval()

    seq_len = shared_len + private_len * 2
    shared = torch.randint(10, vocab_size - 10, (shared_len,))
    attacker_tokens = torch.randint(10, vocab_size - 10, (private_len,))

    diffs = []
    for _ in range(num_trials):
        victim_a = torch.randint(10, vocab_size - 10, (private_len,))
        victim_b = torch.randint(10, vocab_size - 10, (private_len,))

        # Worst-case layout: [shared | victim | attacker]
        input_a = torch.cat([shared, victim_a, attacker_tokens]).unsqueeze(0)
        input_b = torch.cat([shared, victim_b, attacker_tokens]).unsqueeze(0)

        tsam_mask = torch.ones(1, seq_len, dtype=torch.bool)
        tsam_mask[0, shared_len:shared_len + private_len] = False
        mask = tsam_mask if condition == "tsam" else None

        with torch.no_grad():
            out_a = model(input_a, tsam_mask=mask)
            out_b = model(input_b, tsam_mask=mask)

        logits_a = out_a[0, shared_len + private_len:]
        logits_b = out_b[0, shared_len + private_len:]
        diffs.append(float(torch.norm(logits_a - logits_b).item()))

    return {
        "fi_mean": float(np.mean(diffs)),
        "fi_max": float(np.max(diffs)),
        "fi_all_exact_zero": all(d == 0.0 for d in diffs),
    }


def run_probing_attack(
    num_layers: int,
    condition: str = "tsam",
    num_samples: int = 200,
    num_classes: int = 8,
    d_model: int = 256,
    num_heads: int = 4,
    shared_len: int = 16,
    private_len: int = 16,
    vocab_size: int = 1000,
) -> Dict:
    """Run linear probing attack at a given depth."""
    config = TSAMTransformerConfig(
        num_layers=num_layers, d_model=d_model, num_heads=num_heads,
        d_ff=d_model * 4, vocab_size=vocab_size,
        max_seq_len=shared_len + private_len * 2 + 16,
    )
    model = TSAMTransformer(config)
    model.eval()

    seq_len = shared_len + private_len * 2
    shared = torch.randint(10, vocab_size - 10, (shared_len,))
    attacker_tokens = torch.randint(10, vocab_size - 10, (private_len,))

    rng = torch.Generator().manual_seed(42)
    victim_patterns = [
        torch.randint(10, vocab_size - 10, (private_len,), generator=rng)
        for _ in range(num_classes)
    ]

    features, labels = [], []
    for trial in range(num_samples):
        cls = trial % num_classes
        inp = torch.cat([shared, victim_patterns[cls], attacker_tokens]).unsqueeze(0)

        tsam_mask = torch.ones(1, seq_len, dtype=torch.bool)
        tsam_mask[0, shared_len:shared_len + private_len] = False
        mask = tsam_mask if condition == "tsam" else None

        with torch.no_grad():
            logits = model(inp, tsam_mask=mask)
        features.append(logits[0, shared_len + private_len:].flatten().numpy())
        labels.append(cls)

    X, y = np.array(features), np.array(labels)
    n_train = int(0.7 * len(y))
    X_tr, X_te = X[:n_train], X[n_train:]
    y_tr, y_te = y[:n_train], y[n_train:]

    Y_tr = np.zeros((n_train, num_classes))
    Y_tr[np.arange(n_train), y_tr] = 1.0
    W = np.linalg.solve(
        X_tr.T @ X_tr + 0.01 * np.eye(X_tr.shape[1]),
        X_tr.T @ Y_tr,
    )
    acc = float(np.mean((X_te @ W).argmax(axis=1) == y_te))

    return {
        "probe_accuracy": acc,
        "probe_baseline": 1.0 / num_classes,
        "probe_num_classes": num_classes,
    }


def run_full_e2e_suite(
    layer_counts: Optional[List[int]] = None,
    num_fi_trials: int = 50,
    num_probe_samples: int = 200,
) -> List[E2EResult]:
    """Run the canonical end-to-end noninterference experiment suite."""
    if layer_counts is None:
        layer_counts = [1, 2, 4, 8, 12, 24]

    results = []
    for L in layer_counts:
        for cond in ["tsam", "vanilla"]:
            fi = run_functional_independence(L, cond, num_fi_trials)
            probe = run_probing_attack(L, cond, num_probe_samples)

            result = E2EResult(
                num_layers=L,
                condition=cond,
                d_model=256,
                num_heads=4,
                num_trials=num_fi_trials,
                fi_mean=fi["fi_mean"],
                fi_max=fi["fi_max"],
                fi_all_exact_zero=fi["fi_all_exact_zero"],
                probe_accuracy=probe["probe_accuracy"],
                probe_baseline=probe["probe_baseline"],
                probe_num_classes=probe["probe_num_classes"],
            )
            results.append(result)

            z = "EXACT ZERO" if result.fi_all_exact_zero else f"{result.fi_max:.4f}"
            print(
                f"L={L:2d} {cond:>7s}: FI={z:>12s}  "
                f"probe={result.probe_accuracy:.1%} (baseline={result.probe_baseline:.1%})"
            )

    return results


if __name__ == "__main__":
    results = run_full_e2e_suite()
    output = [asdict(r) for r in results]
    print("\n" + json.dumps(output, indent=2))
