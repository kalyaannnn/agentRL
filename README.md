# AgentRL

AgentRL is a single-GPU rollout, runtime, and post-training stack for verifier-trained agents. It is intentionally scoped as a research-grade systems project: small enough to inspect, but complete enough to show how rollout efficiency, task abstraction, and verifier-based RL fit together.

The project demonstrates three things:

- **Single-GPU agent systems**: grouped multi-turn rollouts, continuous batching, chunked prefill, KV-aware admission, runtime headroom tracking, and rollout telemetry on one GPU.
- **Post-training workflow**: supervised bootstrap, LoRA adapter reuse, verifier-based GRPO, checkpointing, and strict evaluation.
- **Task scaffold**: first-class deterministic tool-agent tasks for multi-turn workloads, low-level custom environments, and a lightweight BYOD helper for simple single-turn tasks.

The intended story is **not cold-start RL**. Sparse verifier reward is usually too weak for serious tasks unless the starting policy can already sample useful trajectories. AgentRL is designed around a more realistic loop:

1. bootstrap a task-specific adapter with supervised data
2. diagnose whether the bootstrap policy samples useful trajectories
3. continue with verifier-based GRPO from that adapter
4. evaluate the saved RL adapter strictly

**Hardware:** one CUDA GPU, or CPU for small smoke tests. AgentRL does not provide multi-node or multi-GPU orchestration.

## Project Highlights

- **Inspectable runtime behavior:** benchmark outputs expose throughput, padding waste, KV pressure, cache reuse, VRAM headroom, scheduler deferrals, and bottleneck diagnosis.
- **Agent workload support:** the default systems benchmark uses a deterministic multi-turn tool-use task with growing transcript context and uneven episode completion.
- **Practical post-training shape:** the recommended workflow is `SFT bootstrap -> diagnostic eval -> GRPO -> strict final eval`, with adapter reuse throughout.
- **Pluggable task surface:** users can start with the high-level multi-turn tool-agent scaffold, a high-level single-turn BYOD helper, or the low-level environment/verifier contracts.

## Architecture

AgentRL is best understood as three connected layers.

| Layer | What it demonstrates |
| --- | --- |
| Runtime | Standard and continuous batching for multi-turn grouped rollouts, chunked prefill, persistent-KV decode where supported, paged-KV continuous mode, scheduler/controller signals, and bottleneck diagnosis |
| Training | TRL-compatible clipped GRPO path, verifier-driven reward, SFT bootstrap, LoRA adapter reuse, checkpoints, and final adapter aliases |
| Tasks | `AgentTaskRecord` / `ToolSpec` / `make_tool_agent_task` for deterministic multi-turn tool agents, `BaseEnvironment` / `BaseVerifier` for custom tasks, and `BYODRecord` / `make_single_turn_task` for single-turn `prompt -> response -> verify` workflows |

The public demo shape should preserve all three: systems proof on multi-turn agent workloads, post-training proof, and task portability. See [docs/open_source_demo.md](docs/open_source_demo.md) for the full walkthrough.

## Quick Start

```bash
pip install -e .
pip install -e ".[benchmark]"
```

Optional development extras:

```bash
pip install -e ".[dev]"
```

Run the smoke-sized GRPO example:

```bash
python -m examples.train_math
```

Run the systems comparison used for the public demo:

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
  --output-dir ./systems_benchmark_compare \
  --compare-runtime-modes
```

## Minimal GRPO Example

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

## Recommended Workflow

For nontrivial tasks, use the bootstrap-first loop:

1. Build or load task records.
2. Train a supervised LoRA adapter with task-specific targets.
3. Run diagnostic evaluation to check `pass@1`, `pass@k`, response length, and verifier behavior.
4. Run GRPO initialized from the bootstrap adapter only if the policy has nonzero useful signal.
5. Evaluate the saved RL adapter with the strict verifier.

This matters because high strict-eval rewards from cold-start RL are usually not a reliable public story. AgentRL is built around a practical post-training workflow: bootstrap first, diagnose before RL, use shaped reward during RL only when useful, and keep final evaluation strict.

## Systems Benchmark

Use the systems benchmark to compare rollout implementations, runtime bottlenecks, VRAM pressure, cache reuse, scheduler KV pressure, and task reward on the same workload.

The default workload is the bundled multi-turn tool-use task. It grows transcript context across turns, finishes sampled episodes at different times, and makes continuous batching, padding waste, KV pressure, and runtime headroom visible on one GPU.

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
  --output-dir ./systems_benchmark_compare \
  --compare-runtime-modes
```

This compares:

- standard rollout
- legacy continuous batching
- paged-KV continuous batching

Optional speculative decoding runs can be included with `--include-speculative`.

The benchmark writes per-mode summaries plus `comparison.json`. The main signals to inspect are:

- `mean_step_time_ms`
- `mean_tokens_per_second`
- `mean_padding_ratio`
- `mean_cache_reuse_effectiveness`
- `mean_scheduler_kv_pressure`
- `mean_scheduler_deferred_sequences`
- `mean_scheduler_max_concurrent_sequences`
- `mean_reward`
- `peak_vram_mb`
- `min_rollout_runtime_headroom_mb`
- `dominant_runtime_bottleneck`
- `efficiency_diagnosis`
- `comparison_verdict`

The benchmark is useful even when task reward is not the interesting part: it isolates rollout behavior and makes clear whether a multi-turn run was decode-limited, padding-limited, prefill-limited, or KV-budget-limited.

## Task Integration

AgentRL supports three task integration paths.

### High-Level Multi-Turn Agent Path

Use `AgentTaskRecord`, `ToolSpec`, and `make_tool_agent_task(...)` when your task is a deterministic multi-turn tool-agent workload.

```python
from agentrl import AgentTaskRecord, ToolSpec, make_tool_agent_task

records = [
    AgentTaskRecord(
        task_id="lookup-alpha",
        goal="Look up alpha, then submit it.",
        metadata={"answer": "4"},
        supervised_trace=("TOOL: lookup[alpha]", "FINAL: 4"),
    )
]

task = make_tool_agent_task(
    records=records,
    tools=[
        ToolSpec(
            name="lookup",
            description="Return a value from the task lookup table.",
            handler=lambda argument, state: {"alpha": "4"}[argument],
        )
    ],
    final_answer_fn=lambda record, state: str(record.metadata["answer"]),
)
```

This path adapts to the same `BaseEnvironment` / `BaseVerifier` contracts used by the trainer, while adding a standard action grammar, tool results, trajectory state, shaped reward defaults, and supervised trace export. Start with [docs/multi_turn_agents.md](docs/multi_turn_agents.md) for the multi-turn scaffold guide.

### High-Level BYOD Path

Use `BYODRecord` and `make_single_turn_task(...)` when your task is a common single-turn `prompt -> response -> verifier` setup.

```python
from agentrl import BYODRecord, make_single_turn_task

records = [
    BYODRecord(
        input="Return exactly: ok",
        reference_answer="ok",
        supervised_target="ok",
    )
]

task = make_single_turn_task(
    records=records,
    prompt_formatter=lambda record, tokenizer: f"User:\n{record.input}\n\nAssistant:\n",
    reward_fn=lambda response, state: 1.0 if response.strip() == state["reference_answer"] else 0.0,
    supervised_target_fn=lambda record: record.supervised_target,
)
```

This path keeps the simple-case public API small while still supporting custom prompts, deterministic reward logic, and supervised bootstrap targets. Start with [docs/bring_your_own_task.md](docs/bring_your_own_task.md) for the single-turn onboarding guide.

### Low-Level Custom Task Path

Use `BaseEnvironment` and `BaseVerifier` directly when you need custom state, task-specific transitions, multi-turn rollouts, or richer verifier logic.

Custom tasks implement:

- `BaseEnvironment.reset()`
- `BaseEnvironment.step(action)`
- `BaseEnvironment.state()`
- `BaseVerifier.verify(response, env_state)`

## Public Demo Paths

The strongest public story is:

1. **Systems proof**: compare standard, continuous, and paged-KV continuous rollouts on the same model, workload, and decode budget.
2. **Post-training proof**: run `bootstrap -> diagnostic eval -> GRPO -> final eval` on one primary task.
3. **Task abstraction proof**: show the same lifecycle on a BYOD task with a custom verifier.

Useful entry points:

- [docs/open_source_demo.md](docs/open_source_demo.md)
- [docs/multi_turn_agents.md](docs/multi_turn_agents.md)
- [docs/bring_your_own_task.md](docs/bring_your_own_task.md)
- [notebooks/codeDemo.ipynb](notebooks/codeDemo.ipynb)

For the single-turn MBPP comparison harness:

```bash
python -m examples.compare_single_turn_baselines \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --limit 32 \
  --steps 1 \
  --batch-size 1 \
  --group-size 4 \
  --output-dir ./outputs/single_turn_compare
```

## Validation Snapshot

Recent Colab T4 results on the MBPP BYOD code-task demo show the intended comparison shape rather than a broad benchmark claim.

Fair-track AgentRL vs TRL setup:

- same MBPP task construction
- same prompt and supervised-target shape
- same shaped reward function
- same strict held-out functional evaluation
- SFT bootstrap before GRPO

Observed on a small 16-example evaluation subset:

| Framework | Strict pass rate | Any pass rate | Mean test-pass fraction | Mean eval reward |
| --- | ---: | ---: | ---: | ---: |
| AgentRL | 0.5000 | 0.7500 | 0.6458 | 0.6604 |
| TRL | 0.5625 | 0.6875 | 0.6458 | 0.6635 |

Interpretation: eval quality was effectively tied on this tiny run, while the AgentRL path exposes the rollout/runtime metrics that make the systems comparison inspectable.

On a small MBPP BYOD code-task systems run, continuous batching versus standard rollout showed:

- 1.54x faster step time
- 1.56x faster generation
- 1.55x higher throughput
- 2842 MB lower rollout VRAM
- 2842 MB more runtime headroom
- +0.158 mean reward in that run

These are project validation results from small demo runs, not broad benchmark or SOTA claims.

## Interview Talking Points

- I built AgentRL to make verifier-based RL systems behavior measurable on one GPU, not to claim distributed serving scale.
- The runtime story is about controlling rollout cost: batching mode, padding waste, KV pressure, cache reuse, scheduler deferrals, and VRAM headroom are all surfaced in benchmark artifacts.
- The training story is deliberately bootstrap-first. GRPO is run from a supervised LoRA adapter because sparse verifier reward is only useful once the policy can already produce some correct trajectories.
- The task story is portability. Multi-turn tool agents, single-turn BYOD records, and fully custom environments all adapt to the same trainer-facing contracts.
- The validation snapshot is intentionally modest and honest: small demo runs show comparable task quality to TRL while exposing additional runtime telemetry.

## Repository Layout

- `agentrl/` - runtime, rollout, training, task APIs, memory, and observability
- `examples/` - runnable training, evaluation, benchmark, and comparison scripts
- `notebooks/` - demo notebooks
- `docs/` - BYOD, demo, benchmark, and planning notes
- `tests/` - automated validation of runtime, trainer, task, and example behavior

## What AgentRL Is Not

AgentRL is not, in its current open-source shape:

- a production serving engine
- a multi-node training stack
- a hosted task platform
- a hardened sandbox for executing untrusted code
- a broad benchmark or SOTA claim

The validated story is a research-grade single-GPU rollout/runtime and post-training scaffold for multi-turn agent workloads. More advanced runtime-engine work, including paged-KV-style improvements, belongs to the v2 track rather than a production serving claim.

## Security

- Replay trajectories use `torch.load(..., weights_only=False)`. Do not load untrusted trajectory files.
- Execution-based code-task demos may run model-generated Python in a subprocess for evaluation. This is demo-grade isolation, not a security boundary.
