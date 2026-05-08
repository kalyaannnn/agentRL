# Multi-Turn Agent Tasks

AgentRL's systems-first path is a deterministic multi-turn agent task running through the same single-GPU rollout runtime used for training. The scaffold is intentionally small: a task record defines the goal, tools provide deterministic observations, the model emits one text action per turn, and the verifier scores the terminal episode state.

Use this path when the workload should stress transcript growth, uneven episode lengths, continuous batching, KV pressure, and runtime headroom.

## Public API

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

The returned object matches the existing task shape:

- `task.environment` implements `BaseEnvironment`
- `task.verifier` implements `BaseVerifier`
- `task.supervised_samples()` exports prompt/target traces when records include `supervised_trace`

## Action Grammar

The v1 grammar is deliberately text-only:

- `TOOL: name[arg]`
- `FINAL: answer`

Invalid actions stay inside the episode as observations and are tracked in verifier-facing state. This keeps rollout collection simple and makes malformed action behavior visible to the reward function.

## Verifier State

`make_tool_agent_task(...)` exposes deterministic state for reward, replay, and debugging:

- `task_id`, `goal`, and `metadata`
- `tool_trace`
- `invalid_action_count`
- `completed_tool_steps`
- `total_tool_steps`
- `final_submitted`
- `submitted_answer`
- `expected_final_answer`
- `success`
- `trajectory`

The default verifier returns `1.0` for terminal success. Failed episodes receive a shaped reward from completed expected tool steps, a small final-submission bonus, and an invalid-action penalty, clamped below `1.0`.

Pass `reward_fn` to `make_tool_agent_task(...)` when the task needs custom reward logic.

## Benchmark Workload

The bundled tool-use benchmark is built on this API:

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

Inspect:

- `mean_step_time_ms`
- `mean_tokens_per_second`
- `mean_padding_ratio`
- `mean_cache_reuse_effectiveness`
- `mean_scheduler_kv_pressure`
- `mean_scheduler_deferred_sequences`
- `mean_reward`
- `dominant_runtime_bottleneck`
- `comparison_verdict`

The goal is not to claim production serving. The goal is to make single-GPU multi-turn rollout behavior inspectable enough to tell whether the workload is decode-limited, padding-limited, prefill-limited, or KV-budget-limited.

## When To Drop Lower

Use `BaseEnvironment` and `BaseVerifier` directly when the task does not fit deterministic text tools, needs custom state transitions, or relies on an external backend. The trainer does not require the high-level scaffold; the scaffold is the recommended first path for in-process multi-turn agent workloads.
