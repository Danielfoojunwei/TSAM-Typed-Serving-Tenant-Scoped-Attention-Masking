"""
Probing attack evaluation — demonstrates TSAM security via adversarial probing.

A linear probing attack trains a simple model to predict victim's private KV
content from the attacker's attention outputs. If TSAM works, the probe should
achieve no better than random chance. If TSAM is absent (vanilla), the probe
should achieve non-trivial accuracy.

This provides a concrete adversarial evaluation that goes beyond boolean mask
checks or abstract MI estimation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from tsam.core.block_manager import TypedBlockManager
from tsam.attention.paged_attention import tsam_paged_attention


@dataclass
class ProbingAttackResult:
    """Result from a probing attack evaluation.

    Attributes:
        tsam_probe_accuracy:     Probe accuracy under TSAM (should be ~random).
        vanilla_probe_accuracy:  Probe accuracy without TSAM (should be > random).
        random_baseline:         Expected accuracy of random guessing (1/num_classes).
        num_train_samples:       Number of training samples for the probe.
        num_test_samples:        Number of test samples.
        tsam_leaks:              Whether TSAM shows above-chance probe accuracy.
        defense:                 Defense name.
    """

    tsam_probe_accuracy: float
    vanilla_probe_accuracy: float
    random_baseline: float
    num_train_samples: int
    num_test_samples: int
    tsam_leaks: bool
    defense: str


def run_probing_attack(
    device: str = "cpu",
    num_train: int = 200,
    num_test: int = 100,
    num_classes: int = 4,
) -> ProbingAttackResult:
    """Run a linear probing attack to detect information leakage.

    Protocol:
      1. Create num_classes distinct victim KV patterns (labeled 0..num_classes-1).
      2. For each sample, randomly assign a victim pattern and compute the
         attacker's attention output (with and without TSAM).
      3. Train a linear probe on (output, label) pairs.
      4. Measure probe accuracy on held-out test set.
      5. If TSAM works: accuracy ~= 1/num_classes (random chance).
         If vanilla: accuracy >> 1/num_classes.
    """
    page_size = 16
    num_heads = 8
    num_kv_heads = 4
    head_dim = 64
    private_blocks = 4
    shared_blocks = 2

    total_blocks = shared_blocks + 2 * private_blocks + 1
    bm = TypedBlockManager(num_blocks=total_blocks, page_size=page_size, device=device)

    shared_pages = bm.allocate_shared(count=shared_blocks)
    shared_ids = [p.block_id for p in shared_pages]
    victim_pages = bm.allocate_private(tenant=0, count=private_blocks)
    victim_ids = [p.block_id for p in victim_pages]
    attacker_pages = bm.allocate_private(tenant=1, count=private_blocks)
    attacker_ids = [p.block_id for p in attacker_pages]

    mixed_bt = shared_ids + victim_ids + attacker_ids
    block_table = torch.tensor(mixed_bt, dtype=torch.int32, device=device)
    seq_len_kv = len(mixed_bt) * page_size

    # Create distinct victim KV patterns for each class.
    victim_patterns: List[torch.Tensor] = []
    for c in range(num_classes):
        pattern = torch.randn(private_blocks, page_size, num_kv_heads, head_dim, device=device)
        pattern = pattern * (c + 1)  # Scale to make patterns distinguishable.
        victim_patterns.append(pattern)

    def collect_samples(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Collect (tsam_output, vanilla_output, label) samples."""
        tsam_outs = []
        vanilla_outs = []
        labels = []

        for _ in range(n):
            # Random class for this sample.
            label = np.random.randint(0, num_classes)
            labels.append(label)

            # Build KV cache with chosen victim pattern.
            key_cache = torch.randn(
                total_blocks, page_size, num_kv_heads, head_dim, device=device
            )
            value_cache = torch.randn(
                total_blocks, page_size, num_kv_heads, head_dim, device=device
            )
            for i, blk in enumerate(victim_ids):
                key_cache[blk] = victim_patterns[label][i]
                value_cache[blk] = victim_patterns[label][i]

            query = torch.randn(num_heads, head_dim, device=device)

            # TSAM output.
            out_tsam = tsam_paged_attention(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=block_table,
                page_type=bm.page_type,
                tenant_id_meta=bm.tenant_id,
                query_tenant=1,
                page_size=page_size,
                seq_len_kv=seq_len_kv,
            )
            tsam_outs.append(out_tsam.detach().cpu().numpy().flatten())

            # Vanilla output (no masking).
            scale = 1.0 / math.sqrt(head_dim)
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
            q_exp = query.unsqueeze(1)
            k_t = all_keys.permute(1, 2, 0)
            scores = torch.bmm(q_exp, k_t) * scale
            weights = torch.nn.functional.softmax(scores, dim=-1)
            v_t = all_values.permute(1, 0, 2)
            out_vanilla = torch.bmm(weights, v_t).squeeze(1)
            vanilla_outs.append(out_vanilla.detach().cpu().numpy().flatten())

        return np.array(tsam_outs), np.array(vanilla_outs), np.array(labels)

    # Collect train and test data.
    tsam_train, vanilla_train, labels_train = collect_samples(num_train)
    tsam_test, vanilla_test, labels_test = collect_samples(num_test)

    # Train linear probes using closed-form least squares (one-hot targets).
    def train_and_eval_probe(X_train, y_train, X_test, y_test, n_classes):
        """Train a linear probe and return test accuracy."""
        # One-hot encode targets.
        Y_train = np.zeros((len(y_train), n_classes))
        Y_train[np.arange(len(y_train)), y_train] = 1.0

        # Least squares: W = (X^T X + lambda I)^-1 X^T Y (ridge regression).
        lam = 1e-3
        XtX = X_train.T @ X_train + lam * np.eye(X_train.shape[1])
        XtY = X_train.T @ Y_train
        W = np.linalg.solve(XtX, XtY)

        # Predict.
        preds = X_test @ W
        pred_labels = preds.argmax(axis=1)
        accuracy = float((pred_labels == y_test).mean())
        return accuracy

    tsam_accuracy = train_and_eval_probe(
        tsam_train, labels_train, tsam_test, labels_test, num_classes
    )
    vanilla_accuracy = train_and_eval_probe(
        vanilla_train, labels_train, vanilla_test, labels_test, num_classes
    )

    random_baseline = 1.0 / num_classes
    # TSAM leaks if probe accuracy is significantly above random chance.
    # Use a threshold of 1.5x random chance as a generous margin.
    tsam_leaks = tsam_accuracy > random_baseline * 1.5

    return ProbingAttackResult(
        tsam_probe_accuracy=tsam_accuracy,
        vanilla_probe_accuracy=vanilla_accuracy,
        random_baseline=random_baseline,
        num_train_samples=num_train,
        num_test_samples=num_test,
        tsam_leaks=tsam_leaks,
        defense="TSAM",
    )
