# TSAM: Typed Serving — Tenant-Scoped Attention Masking

Formal information-flow isolation for multi-tenant LLM serving at unlimited scale.

## Core Contribution

Hard attention masking over typed KV-cache pages achieves **perfect tenant isolation**:

```
I(output(q_i); KV_private(t_j)) = 0  for all i ≠ j
```

This eliminates the dimensionality constraint that limits orthogonal projection schemes (Paper 5) to 16 tenants. TSAM scales to unlimited concurrent tenants with <1% throughput overhead.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  KV-Cache Pool                   │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐  │
│  │SHARED│ │SHARED│ │PRIV  │ │PRIV  │ │PRIV  │  │
│  │ sys  │ │prefix│ │ t=0  │ │ t=1  │ │ t=2  │  │
│  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘  │
│     ↑        ↑        ↑        ↑        ↑      │
│     │        │        │        │        │       │
│  page_type: SHARED  SHARED  PRIVATE  PRIVATE    │
│  tenant_id:  -1      -1       0        1        │
└─────────────────────────────────────────────────┘

Attention Masking Rule (per key position j):
  if page_type[j] == PRIVATE and tenant_id[j] != query_tenant:
      attention_logit[j] = -∞   // blocked
  else:
      attention_logit[j] = q·k   // allowed
```

## Installation

```bash
pip install -e .          # Core (PyTorch, NumPy, SciPy)
pip install -e ".[dev]"   # + pytest
pip install -e ".[all]"   # Everything including Triton and vLLM
```

## Running Tests

```bash
pytest tests/ -v
```

### Kernel Correctness Tests (Experiment 6)

```bash
pytest tests/test_attention_masking.py -v
```

Covers:
- **(a)** PRIVATE→PRIVATE same tenant: attention weight > 0 (allowed)
- **(b)** PRIVATE→PRIVATE different tenant: attention weight = 0 (blocked)
- **(c)** SHARED page any tenant: attention weight > 0 (allowed)
- **(d)** Edge cases: page boundaries, batch size 1/512, seq len 1/32768

## Module Structure

```
tsam/
├── core/
│   ├── types.py           # PageType, TypedPage, TenantId
│   └── block_manager.py   # TypedBlockManager with metadata tensors
├── attention/
│   ├── masking.py          # TSAM mask generation (single + batched)
│   ├── paged_attention.py  # Full attention with TSAM isolation
│   └── triton_kernel.py    # Triton Flash Attention with TSAM mask
├── dai/
│   └── timing_dfa.py       # Collision attack timing DFA + middleware
├── evaluation/
│   ├── leakage.py          # MI estimation (KSG), reconstruction leakage
│   └── collision_attack.py # Collision attack simulator
├── benchmarks/
│   ├── throughput.py       # Throughput benchmark (Exp 2)
│   ├── scalability.py      # Scalability + Paper 5 comparison (Exp 2)
│   ├── prefix_sharing.py   # Prefix efficiency (Exp 3)
│   └── experiments.py      # Full experiment orchestrator (Exp 1-5)
└── integration/
    └── vllm_patch.py       # vLLM BlockTable extension + kernel patch
```

## Running Experiments

```python
from tsam.benchmarks.experiments import (
    run_experiment_1_isolation,
    run_experiment_2_scalability,
    run_experiment_3_prefix_sharing,
    run_experiment_4_timing_defense,
)

# Experiment 1: Isolation correctness (0 bits leakage)
results_1 = run_experiment_1_isolation()

# Experiment 2: Scalability to 10K tenants
results_2 = run_experiment_2_scalability()

# Experiment 3: Prefix sharing efficiency
results_3 = run_experiment_3_prefix_sharing()

# Experiment 4: Timing attack defense
results_4 = run_experiment_4_timing_defense()
```

## vLLM Integration

```python
from tsam.integration.vllm_patch import TSAMBlockTableExtension, TSAMAttentionWrapper

# Extend vLLM's block table with TSAM metadata
ext = TSAMBlockTableExtension(num_blocks=8192, device="cuda")

# Mark blocks during allocation
ext.mark_shared(block_id=0)           # System prompt
ext.mark_private(block_id=5, tenant=0) # Tenant 0's private KV

# Generate mask for attention
mask = ext.generate_mask(block_table, query_tenant=0, page_size=16, seq_len_kv=512)
```

## Key Results

| KPI | Target | Method |
|-----|--------|--------|
| Leakage bits/query | 0.0 at all N | TSAM attention mask |
| Scalability | Unlimited tenants | Type metadata (2 bytes/page) |
| Paper 5 limit | 16 tenants (d=4096) | Dimensionality violation |
| Throughput overhead | <1% | Single conditional per key |
| Prefix sharing | Within 5% of vanilla | SHARED page type |
| Timing defense | ≥95% reduction | DAI timing DFA |

## Formal Guarantees

**Theorem 1 (Perfect Tenant Isolation):** Under TSAM, for any query q from tenant t_i and all PRIVATE KV-cache entries from tenant t_j ≠ t_i: I(output(q); KV_private(t_j)) = 0.

**Theorem 2 (Shared Prefix Safety):** SHARED pages do not create cross-tenant private data leakage.

**Theorem 3 (Timing Information Bound):** Under TSAM + DAI timing DFA, per-query timing information leakage is bounded by I_timing ≤ I_natural_variance + ε_FNR.

## Paper

The full NeurIPS paper is available at [`paper/main.tex`](paper/main.tex). To compile:

```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```
