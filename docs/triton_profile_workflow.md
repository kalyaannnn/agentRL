# Profile-Guided Triton Workflow

This workflow is for the CUDA machine, not the local CPU-only development
environment. The goal is to decide the kernel target from evidence, then
benchmark the optional Triton path against the PyTorch reference.

## 1. Baseline Runtime Comparison

```bash
python -m examples.benchmark_systems \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --task tool-use \
  --split easy \
  --steps 5 \
  --batch-size 1 \
  --group-size 4 \
  --max-new-tokens 64 \
  --max-episode-steps 4 \
  --output-dir ./systems_benchmark_baseline \
  --compare-runtime-modes
```

Inspect `comparison.json` and the per-mode `summary.json` files. Use
`generation_time_ms` versus `training_time_ms` as the first branch:

- high generation time points to continuous batching, decode, or KV handling
- high training time points to the GRPO objective, backward pass, or optimizer

## 2. PyTorch Profiler Trace

```bash
python -m examples.benchmark_systems \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --task tool-use \
  --split easy \
  --steps 3 \
  --batch-size 1 \
  --group-size 4 \
  --max-new-tokens 64 \
  --max-episode-steps 4 \
  --output-dir ./systems_benchmark_profiled \
  --profile-steps 1 \
  --profile-dir ./profiles/agentrl_tool_use \
  --compare-runtime-modes
```

Open the exported Chrome trace and identify the largest repo-owned operation in
the dominant phase. If the largest cost is inside external model attention
kernels, record that and optimize the next repo-owned path.

## 3. Nsight Systems Timeline

```bash
nsys profile -o agentrl_tool_use_baseline \
  python -m examples.benchmark_systems \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --task tool-use \
  --split easy \
  --steps 3 \
  --batch-size 1 \
  --group-size 4 \
  --max-new-tokens 64 \
  --max-episode-steps 4 \
  --output-dir ./systems_benchmark_nsys \
  --compare-runtime-modes
```

Use Nsight Systems for the CUDA timeline: kernel launch gaps, CPU/GPU overlap,
memory copies, and synchronization points. Use Nsight Compute only after a
specific kernel needs occupancy or memory-throughput analysis.

## 4. Triton Rerun

The first optional Triton path covers the sampled-token GRPO objective. It is
used only when CUDA, Triton, and a supported tensor layout are available; all
other environments fall back to the PyTorch reference and report the fallback
reason.

```bash
python -m examples.benchmark_systems \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --task tool-use \
  --split easy \
  --steps 5 \
  --batch-size 1 \
  --group-size 4 \
  --max-new-tokens 64 \
  --max-episode-steps 4 \
  --output-dir ./systems_benchmark_triton \
  --compare-runtime-modes \
  --use-triton-kernels
```

## Result Table

| Metric | Baseline | Triton |
| --- | ---: | ---: |
| Mean step time ms |  |  |
| Mean generation time ms |  |  |
| Mean training time ms |  |  |
| Triton kernel used rate | 0.0 |  |
| Tokens/sec |  |  |
| Peak VRAM MB |  |  |
| Mean reward |  |  |

The interview narrative should state the measured bottleneck, the kernel target,
the numerical parity result, and the before/after runtime delta.
