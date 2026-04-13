# AgentRL

AgentRL is a lightweight single-GPU framework for verifier-based GRPO style post-training of language models.

It is built for small-scale efficiency experiments: shared-weight LoRA policy/reference training, multiple rollout paths, profiler-friendly runtime metrics, and simple user-defined `BaseEnvironment` / `BaseVerifier` APIs.

## Installation

Core install:

```bash
pip install -e .
```

Install benchmark extras for GSM8K and example scripts that depend on `datasets`:

```bash
pip install -e ".[benchmark]"
```

## Systems Overview

AgentRL currently focuses on single-device post-training efficiency:

- shared-weight LoRA policy/reference layout
- standard rollout and continuous batching
- chunked prefill in both rollout paths
- persistent KV decode for cache-capable models in continuous batching
- CPU-backed trajectory buffering and replay
- checkpoint saving during training plus a final adapter alias
- metrics logging, profiler utilities, debugger hooks, and VRAM headroom reporting
- pluggable text environments and deterministic verifiers

The training objective is GRPO style policy optimization with a frozen reference implemented through shared base weights plus trainable LoRA adapters.

## Minimal Usage

```python
from agentrl import GRPOConfig, GRPOTrainer
from examples.math_env import MathEnvironment, MathVerifier

config = GRPOConfig(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    group_size=8,
    batch_size=4,
    max_new_tokens=128,
    steps=100,
    output_dir="./checkpoints",
)

trainer = GRPOTrainer(
    config=config,
    environment=MathEnvironment(split="smoke"),
    verifier=MathVerifier(),
)

trainer.train()
```

CLI example:

```bash
python -m examples.train_math \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --steps 5 \
  --split smoke
```

## Profiling

AgentRL can export Chrome tracing files for the first few training steps.

Config fields:

- `profile_steps`: number of initial steps to profile
- `profile_dir`: output directory for exported traces

Example:

```python
config = GRPOConfig(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    steps=10,
    profile_steps=2,
    profile_dir="./profiles",
)
```

This writes files such as:

```text
./profiles/step_000000_chrome_trace.json
./profiles/step_000001_chrome_trace.json
```

Open them in Chrome via:

```text
chrome://tracing
```

## Runtime Metrics

Per-step metrics now include runtime-oriented fields such as:

- `generation_time_ms`, `training_time_ms`
- `prefill_time_ms`, `decode_time_ms`
- `tokens_per_second`
- `prefill_tokens_per_second`, `decode_tokens_per_second`
- `padding_ratio`, `generation_padding_ratio`, `sequence_padding_ratio`
- `cache_reuse_effectiveness`
- `generation_peak_vram_mb`, `rollout_peak_vram_mb`
- `generation_runtime_headroom_mb`, `rollout_runtime_headroom_mb`
- `learning_rate`, `mean_token_kl`, `beta`

These are written to `metrics.jsonl` and summarized by the systems benchmark script.

## Systems Benchmark

`examples/benchmark_systems.py` measures runtime behavior, not task accuracy. It runs a fixed synthetic workload and writes a `summary.json` with step-time, throughput, padding, cache reuse, and VRAM metrics.

Single-run example:

```bash
python -m examples.benchmark_systems \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --steps 5 \
  --batch-size 1 \
  --group-size 4 \
  --max-new-tokens 64 \
  --split easy \
  --output-dir ./systems_benchmark
```

Comparison example:

```bash
python -m examples.benchmark_systems \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --steps 5 \
  --batch-size 1 \
  --group-size 4 \
  --max-new-tokens 64 \
  --split easy \
  --output-dir ./systems_benchmark_compare \
  --compare-standard-vs-continuous
```

The generated summary includes fields such as:

- `mean_step_time_ms`
- `mean_generation_fraction`
- `mean_training_fraction`
- `mean_tokens_per_second`
- `mean_prefill_tokens_per_second`
- `mean_decode_tokens_per_second`
- `mean_padding_ratio`
- `mean_cache_reuse_effectiveness`
- `peak_vram_mb`
- `rollout_peak_vram_mb`
- `min_rollout_runtime_headroom_mb`

## GSM8K Workflow

The current GSM8K path is:

1. bootstrap a LoRA adapter with rationale-based supervised targets derived from GSM8K solutions
2. run diagnostic evaluation to measure `pass@1` and `pass@k` before RL
3. run GRPO with a strict binary verifier only if the bootstrap model already produces some correct trajectories
4. evaluate the saved GRPO adapter with strict exact match

The GSM8K verifier is intentionally strict:

- reward is `1.0` only if the last non-empty line is exactly `Final answer: <integer>` and the integer matches the gold answer
- reward is `0.0` otherwise
- there is no partial reward for formatting, near misses, or fallback integer extraction

Bootstrap is rationale-based rather than answer-only. The diagnostic stage is there to answer a practical question before spending RL compute: does the bootstrap adapter produce any correct strict trajectories at all?

### Recommended GSM8K Commands

Bootstrap SFT:

```bash
python -m examples.bootstrap_gsm8k_subset \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --epochs 3 \
  --batch-size 4 \
  --subset-size 128 \
  --max-question-words 45 \
  --curriculum easy \
  --adapter-dir ./bootstrap_gsm8k_adapter_15b
```

Diagnostic evaluation before RL:

```bash
python -m examples.eval_gsm8k_subset \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --init-adapter-path ./bootstrap_gsm8k_adapter_15b \
  --output-dir ./eval_gsm8k_subset_15b_diag \
  --split train \
  --subset-size 128 \
  --max-question-words 45 \
  --curriculum easy \
  --max-new-tokens 96 \
  --num-samples 8
```

GRPO training:

```bash
python -m examples.benchmark_gsm8k_subset \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --steps 10 \
  --batch-size 1 \
  --group-size 4 \
  --max-new-tokens 128 \
  --subset-size 128 \
  --max-question-words 45 \
  --curriculum easy \
  --reward-mode strict \
  --init-adapter-path ./bootstrap_gsm8k_adapter_15b \
  --split train \
  --output-dir ./checkpoints_gsm8k_subset_15b_rl
```

Strict final eval:

```bash
python -m examples.eval_gsm8k_subset \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --init-adapter-path ./checkpoints_gsm8k_subset_15b_rl/checkpoint_final \
  --output-dir ./eval_gsm8k_subset_15b_rl_final \
  --split train \
  --subset-size 128 \
  --max-question-words 45 \
  --curriculum easy \
  --max-new-tokens 128
```

### How To Interpret GSM8K Diagnostics

- if `pass@k` is near zero, RL likely has no signal
- if `pass@k` is nontrivial but `pass@1` is lower, RL may help sharpen the policy toward trajectories the model can already sample but not yet select reliably
- if many responses hit `max_new_tokens`, increase decoding budget before concluding the model is too weak

The diagnostic evaluator writes both a compact `summary.json` and a `predictions.jsonl` file with raw sampled responses, parsed terminal predictions, per-sample rewards, and response lengths for inspection.

## Benchmark Snapshot (Accuracy)

The following result is on a filtered easy-curriculum training subset. It is a project benchmark result, not a broad GSM8K SOTA claim.

| Model | Setup | Subset | Metric | Value |
|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | Rationale SFT bootstrap | GSM8K filtered train easy subset | pass@1 | 0.5156 |
| Qwen2.5-1.5B-Instruct | Rationale SFT bootstrap | GSM8K filtered train easy subset | pass@8 | 0.7734 |
| Qwen2.5-1.5B-Instruct | Bootstrap + strict GRPO | GSM8K filtered train easy subset | pass@1 | 0.6875 |

## Systems Benchmark (Throughput / VRAM)

The systems benchmark measures runtime, batching efficiency, cache reuse, and memory headroom. It is not an accuracy benchmark.

Use `examples/benchmark_systems.py` when you want to compare:

- standard rollout vs continuous batching
- prefill vs decode bottlenecks
- padding waste under different batching choices
- VRAM headroom on a fixed workload

## Real GSM8K Status

The current story is:

- cold-start strict RL on GSM8K did not work on small models
- answer-only SFT plus shaped reward produced superficially better rewards, but much of that signal came from formatting behavior rather than robust problem solving
- rationale-based SFT bootstrap plus a strict binary verifier produced a meaningful bootstrap policy on the filtered GSM8K subset
- on `Qwen/Qwen2.5-1.5B-Instruct`, bootstrap diagnostics showed:
  - `pass@1 = 0.5156`
  - `pass@8 = 0.7734`
  - `fraction_with_any_correct = 0.7734`
- after strict binary GRPO, strict greedy evaluation reached:
  - `pass@1 = 0.6875`

## Synthetic Task Ladder

- `smoke`: ultra-easy arithmetic for the first non-degenerate reward batch
- `easy`: slightly harder synthetic arithmetic before moving to benchmark-style tasks
- `train` / `eval`: stricter arithmetic splits with more room for policy improvement
- `gsm8k_subset`: filtered real GSM8K examples for benchmark-style experiments

## Single-GPU Playbook

- `batch_size * group_size` is the main VRAM dial
- enable gradient checkpointing only when needed
- continuous batching helps most when sequence lengths vary across active samples
- persistent KV decode is useful, but it is not yet a full paged-KV runtime
- larger `max_new_tokens` can be necessary for real reasoning tasks
- if many prompt groups are all-correct or all-wrong, increase prompt batch size so each optimizer step sees more than one prompt group

## Current Constraints

- `use_lora=True` is required
- the framework is single-device only
- continuous batching is not yet a full paged-KV or vLLM-style runtime
- `top_p` support may still be limited depending on the code path
- the strongest benchmark comparison still comes from explicit bootstrap-vs-post-RL evaluation on saved adapters
- the config surface already includes experimental flags for async rollout workers, async trajectory copy, and vLLM rollout, but those paths are reserved and currently raise `NotImplementedError` when enabled

## Security Note

- replay trajectories are loaded with `torch.load(..., weights_only=False)`; do not load untrusted trajectory files
