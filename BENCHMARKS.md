# BENCHMARKS — Reproducibility runbook

This file specifies the exact command sequence that produces every empirical
table in `paper/main.tex`. Treat it as a contract: a result is in the paper
only when the corresponding command is in this file *and* the run output is
archived under `artifacts/runs/<date>_<machine>_<experiment>/`.

## Currently runnable (CPU, no GPU contact)

These commands run on any machine with `pip install -e .` completed. They are
the basis for the headline result in §6 of the paper (24-layer end-to-end
noninterference).

```bash
# Functional independence + linear probing across depths {1,2,4,8,12,24}.
# Produces Tables 1 and 2 in paper §6.
python -m tsam.evaluation.e2e_noninterference

# Single-layer isolation correctness (Experiment 1, paper §8).
python -c "from tsam.benchmarks.experiments import run_experiment_1_isolation; \
           print(run_experiment_1_isolation())"

# Tenant scalability check (Experiment 2, paper §8).
python -c "from tsam.benchmarks.experiments import run_experiment_2_scalability; \
           print(run_experiment_2_scalability())"

# Linear probing attack standalone (Experiment 6 area).
python -c "from tsam.benchmarks.probing_attack import run_probing_attack; \
           print(run_probing_attack())"

# Full pytest suite (CPU, ~10s for the existing 117 tests + new lifecycle suite).
pytest tests/ -v -n auto
```

## Deferred to a GPU-available session

Each command below requires: a GPU free of training jobs, vLLM installed and
pinned to a known commit, a compiled TSAM CUDA extension, and downloaded
Llama-2-7B/13B weights via Hugging Face. None of this is runnable from the
present codebase as-is; each line below is a runbook entry describing the
target command, not a working script.

```bash
# Production throughput, Llama-2-7B / Llama-2-13B, A100/H100 with vLLM
# continuous batching. Fills Table tab:throughput-todo in paper §8.5.
python benchmarks/real_vllm_bench.py \
    --model meta-llama/Llama-2-7b-hf \
    --tenants 100 \
    --batch-size 128 \
    --tsam on \
    --hardware a100

# Real adversarial co-tenant harness: block-table confusion, shared-prefix
# poisoning, stale-block reuse. Targets §Limitations of paper.
python benchmarks/adversarial_serving.py \
    --scenarios block_table_confusion,shared_prefix_poison,stale_reuse \
    --model meta-llama/Llama-2-7b-hf \
    --report artifacts/runs/$(date +%Y-%m-%d)_$(hostname)_adversarial/

# Timing-defense ROC/AUC under real probe traces (DAI evaluation).
python benchmarks/timing_roc.py \
    --defense dai \
    --baseline none,noise,constant_padding \
    --probes 10000 \
    --report artifacts/runs/$(date +%Y-%m-%d)_$(hostname)_dai_roc/
```

When a GPU run lands, capture the output, hardware (`nvidia-smi --query-gpu=name --format=csv,noheader`), CUDA version, vLLM commit, model weights hash, and the exact command into `artifacts/runs/<run-id>/` and update the corresponding paper table in the same commit.

## Notes on running tests during active GPU training

`pytest tests/` is CPU-only by design (every test hardcodes `device="cpu"`),
but `import torch` itself can lazily initialize a CUDA context if any
torch.cuda call is made during import resolution. Per the project's GPU-safety
convention: if `nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader`
shows a meaningful utilization spike from a training job, defer the pytest
run until the job completes.
