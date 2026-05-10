# TSAM: Tenant-Scoped Attention Masking for Multi-Tenant LLM Serving

Reference implementation and deployment guide. The companion paper draft
(`paper/main.tex`) frames TSAM as a research contribution to information-flow
control for shared LLM serving; this README is the engineering counterpart and
keeps a deliberately practical tone.

TSAM is a lightweight extension to [PagedAttention](https://arxiv.org/abs/2309.06180)
that isolates tenants in shared LLM serving by masking cross-tenant KV-cache
entries in the attention kernel. The novel piece is *not* the mask itself but
the integration of page-level information-flow typing
([Denning 1976](https://dl.acm.org/doi/10.1145/360051.360056),
[Sabelfeld and Myers 2003](https://www.cs.cornell.edu/andru/papers/jsac/sm-jsac03.pdf))
with PagedAttention's KV-cache abstraction, which decouples tenant scalability
from model dimensionality while preserving prefix sharing.

## What It Does

When multiple tenants share a single LLM instance (e.g., via vLLM), their requests are batched together. Without isolation, tenant A's query can attend to tenant B's private KV-cache entries through the attention mechanism.

TSAM prevents this by:

1. **Tagging** each KV-cache page as `SHARED` or `PRIVATE(tenant_id)`
2. **Masking** cross-tenant private pages to `-inf` before softmax in every attention layer
3. **Preserving** shared prefix pages (system prompts) as visible to all tenants

The result: `exp(-inf) = 0`, so cross-tenant private entries get exactly zero attention weight. This is not deep. It is IEEE 754 arithmetic.

## When to Use This

- You serve multiple tenants from a single GPU with vLLM or similar
- Tenants have private prompts that must not leak to other tenants
- You need prefix sharing (system prompts) to remain efficient
- You need a formal argument for compliance (HIPAA, SOC 2, etc.)

## When NOT to Use This

- You need protection against a malicious serving operator (TSAM trusts the operator)
- You need GPU-level side-channel protection (TSAM doesn't address cache timing, DRAM patterns, etc.)
- You use architectures with cross-position mixing outside attention (e.g., sequence-level normalization, 1D convolutions)
- You use per-tenant LoRA adapters in the masked attention path

---

## Quick Start

```bash
pip install -e .
pytest tests/ -v  # 117 tests, ~10s
```

### Basic usage

```python
from tsam.core.block_manager import TypedBlockManager
from tsam.attention.paged_attention import tsam_paged_attention

# Allocate typed blocks
bm = TypedBlockManager(num_blocks=1024, page_size=16)
shared = bm.allocate_shared(count=4)       # System prompt
private = bm.allocate_private(tenant=0, count=8)  # Tenant 0's data

# Run attention with tenant isolation
output = tsam_paged_attention(
    query=query,
    key_cache=key_cache,
    value_cache=value_cache,
    block_table=block_table,
    page_type=bm.page_type,
    tenant_id_meta=bm.tenant_id,
    query_tenant=0,
    page_size=16,
    seq_len_kv=seq_len,
)
```

---

## How It Works

### The Masking Rule

For each KV position `j` and query tenant `t_i`:

```
mask[j] = True   if page_type[j] == SHARED
mask[j] = True   if page_type[j] == PRIVATE and tenant_id[j] == t_i
mask[j] = False  otherwise (cross-tenant private)
```

Blocked positions get score `-inf` before softmax. `exp(-inf) = 0` exactly (IEEE 754). The attention output is therefore a function of only the shared and own-private KV entries.

### Metadata Overhead

Per KV-cache page: 5 bytes (`page_type`: 1 byte uint8, `tenant_id`: 4 bytes int32). For a typical 80GB KV cache with 16-token pages, this is 0.06% overhead.

### Prefix Sharing

SHARED pages are visible to all tenants. A system prompt cached once can be reused by all tenants without duplication. This is the main advantage over orthogonal projection approaches, which require per-tenant copies.

---

## Deployment Checklist

TSAM provides isolation **if and only if** all four conditions hold:

| # | Condition | What to check | If violated |
|---|-----------|---------------|-------------|
| C1 | **Masking at every attention layer** | TSAM mask applied at all L layers | One unmasked layer = full leakage |
| C2 | **Clean shared prefix** | Shared KV computed without private data in batch | Contaminated prefix leaks to all tenants |
| C3 | **Position-local non-attention ops** | No cross-position mixing (standard for GPT/LLaMA/Qwen/Gemma) | Cross-position ops = cross-tenant leakage |
| C4 | **Frozen shared weights** | No per-tenant adapters in masked attention | Adapter weights encode tenant identity |

**C2 is the most commonly violated in production.** vLLM's prefix caching computes shared KV during batches that may include private data. Use the `ComputeProvenanceTracker` (in `tsam/core/compute_provenance.py`) to enforce clean provenance.

---

## What TSAM Does NOT Protect Against

Be explicit with your security team about these:

- **GPU memory side-channels**: L2 cache timing, DRAM row buffer conflicts, CUDA scheduling patterns. These require hardware/OS mitigations, not software masking.
- **Metadata leakage**: Block allocation patterns reveal sequence lengths and usage timing. TSAM does not obfuscate allocation metadata.
- **Malicious operator**: TSAM trusts the serving infrastructure to correctly tag pages and apply masking. A malicious operator can trivially bypass it.
- **Non-attention channels**: If your architecture has cross-position operations outside attention (e.g., sequence-level batch normalization, cross-attention to external memory), TSAM does not cover those paths.
- **Output-semantic channels**: TSAM prevents information flow through attention weights. It does not prevent an adversary from inferring information through the model's generated text (prompt injection, etc.).
- **Training data extraction**: TSAM is an inference-time mechanism. It does not address memorization in model weights.

---

## Empirical Validation

### Functional Independence Test

We measure whether changing the victim's private tokens affects the attacker's output. Setup: `[shared_prefix | victim_private | attacker_private]` with TSAM mask blocking victim positions for the attacker. 50 trials per configuration.

| Depth (layers) | TSAM | Vanilla (no masking) |
|----------------|------|---------------------|
| 1 | **0.0 (exact)** | 1.88 |
| 2 | **0.0 (exact)** | 2.70 |
| 4 | **0.0 (exact)** | 3.69 |
| 8 | **0.0 (exact)** | 5.08 |
| 12 | **0.0 (exact)** | 6.28 |
| 24 | **0.0 (exact)** | 8.07 |

Under TSAM, the attacker's output is **bitwise identical** regardless of what the victim's private data contains. Without masking, leakage grows monotonically with depth.

### Probing Attack

A linear classifier trained to predict which of 8 victim prompts was used, given the attacker's output logits. 200 samples, 70/30 train/test split.

| Depth | TSAM | Vanilla | Random baseline |
|-------|------|---------|-----------------|
| 1 | 11.7% | **100%** | 12.5% |
| 4 | 11.7% | **100%** | 12.5% |
| 12 | 11.7% | **100%** | 12.5% |
| 24 | 11.7% | **100%** | 12.5% |

Under TSAM: random chance. Without TSAM: perfect extraction at every depth.

### Scalability

TSAM mask correctness verified at 10, 100, 1,000, and 10,000 concurrent tenants with randomized cross-tenant pair testing. No failures. Orthogonal projection approaches fail at 17 tenants (hard dimensionality limit: `d_model / d_sub`).

---

## Comparison with Alternatives

| Approach | Max Tenants | Prefix Sharing | Isolation Type | Overhead |
|----------|-------------|----------------|----------------|----------|
| Process isolation | Unlimited | No | Trivial | 5-10x cost |
| Orthogonal projection | ~16 | No | Approximate (epsilon) | Moderate |
| Ad hoc attention masks | Unlimited | Yes | No formal argument | Low |
| **TSAM** | **Unlimited** | **Yes** | **Structural zero (per-layer)** | **5 bytes/page** |

TSAM's advantage over ad hoc masks: a clear correctness argument (the deployment checklist above) and open-source implementation with tests.

TSAM's advantage over orthogonal projection: no tenant limit, prefix sharing preserved, exact isolation instead of epsilon-approximate.

---

## Repository Structure

```
tsam/
  core/
    types.py                  # PageType enum, TypedPage dataclass
    block_manager.py          # Thread-safe typed block allocation
    compute_provenance.py     # Shared prefix clean-compute tracking
  attention/
    masking.py                # TSAM mask generation (single + batched)
    paged_attention.py        # Reference attention with TSAM (vectorized)
    triton_kernel.py          # Triton Flash Attention kernel (prototype)
  model/
    transformer.py            # Multi-layer TSAM-compliant transformer
  evaluation/
    e2e_noninterference.py    # End-to-end isolation experiments
    leakage.py                # MI estimation (KSG estimator)
    collision_attack.py       # Timing side-channel simulation
  benchmarks/
    experiments.py            # Experiment orchestration
    probing_attack.py         # Linear probing adversarial evaluation
    scalability.py            # Multi-tenant scalability tests
    throughput.py             # Throughput measurement
    prefix_sharing.py         # Prefix sharing efficiency
  dai/
    timing_dfa.py             # Timing attack detection (DFA-based)
  integration/
    vllm_patch.py             # vLLM integration (prototype)
tests/                        # 117 tests
paper/                        # LaTeX source (technical report)
```

---

## Running Experiments

```bash
# Functional independence + probing attack across depths
python -m tsam.evaluation.e2e_noninterference

# Isolation correctness (MI-based + functional independence)
python -c "from tsam.benchmarks.experiments import run_experiment_1_isolation; print(run_experiment_1_isolation())"

# Scalability to 10,000 tenants
python -c "from tsam.benchmarks.experiments import run_experiment_2_scalability; print(run_experiment_2_scalability())"

# Probing attack
python -c "from tsam.benchmarks.probing_attack import run_probing_attack; print(run_probing_attack())"
```

---

## Known Limitations and Status

| Component | Status | Notes |
|-----------|--------|-------|
| Core masking | Production-ready | Thread-safe, tested, vectorized |
| Block manager | Production-ready | Thread-safe allocation/deallocation |
| Triton kernel | Prototype | Not GPU-validated, untested on real hardware |
| vLLM integration | Prototype | Skeleton only, not tested against real vLLM |
| Timing defense (DAI) | Prototype | Hardcoded thresholds, bypassable by adaptive attacker |
| Throughput benchmarks | Synthetic only | Not measured on real GPU kernels or real vLLM |

The core isolation mechanism is solid. The integration with production serving systems is not yet validated. Contributions welcome.

---

## Paper

A NeurIPS-style draft is in [`paper/main.tex`](paper/main.tex). Build with
`cd paper && make pdf` (requires `latexmk` or `pdflatex` + `bibtex`).

The draft elevates the engineering framing of this README to a research
contribution: page-typed information-flow control for PagedAttention, with a
formal **Attention-Channel Noninterference under Correct Page Typing** theorem
(Theorem 1) and an end-to-end extension under explicit conditions C1–C4
(Theorem 2). The C1–C4 conditions in the paper are the same deployment
checklist as in this README, lifted into the paper's
**Definition: TSAM-Compliant Transformer**.

The paper is explicit about empirical gaps: production GPU throughput numbers
on Llama-2 with vLLM continuous batching, a real adversarial co-tenant attack
harness, and a compiled CUDA patch are all catalogued as deferred work in
`paper/main.tex` §Limitations and the runbook below.

## License

Apache 2.0
