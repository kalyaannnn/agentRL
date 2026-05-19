# AgentRL

AgentRL is a readable **single-GPU inference runtime** for shared-prefix rollouts, with **SFT bootstrap** and **GRPO** for multi-turn, verifier-trained agents. Inspired by vLLM-style techniques (continuous batching, prefix caching, chunked prefill, CUDA graph decode), scoped for one GPU and a codebase you can read in an afternoon—not a production serving engine.

**Hardware:** one CUDA GPU for the full systems path, or CPU for smoke tests. No multi-node orchestration.

## Install

```bash
git clone https://github.com/kalyaannnn/agentRL.git
cd agentRL
pip install -e .
```

Colab:

```python
!git clone https://github.com/kalyaannnn/agentRL.git
%cd agentRL
!pip install -q -e .
```

## What you get

| Inference runtime | Post-training |
| --- | --- |
| Standard and **continuous batching** for uneven multi-turn episodes | **SFT bootstrap** on LoRA adapters |
| **Prefix cache** (block-aligned shared-prefix reuse) | **GRPO** with clipped sampled-token objective |
| **Chunked prefill** and persistent-KV decode | Policy / reference **shared-base LoRA** layout |
| **CUDA graph decode** on CUDA (optional, eager fallback) | JSONL metrics, checkpoints, shaped verifier rewards |

The runtime helps any workload with **shared prompt prefixes** on one GPU: grouped GRPO sampling, multi-turn agent transcripts, or few-shot prompts with a common system block. **You define the task** (environment + verifier, or `make_tool_agent_task`); the repo does not ship bundled datasets or example environments.

## Recommended workflow

Sparse verifier rewards rarely work from a cold start:

1. Define task records with `supervised_trace` when possible.
2. Run **`SFTBootstrapTrainer`** to teach action format.
3. Run **`GRPOTrainer`** with `init_adapter_path` pointing at the bootstrap adapter.
4. Compare runtime modes (standard → continuous → prefix cache → CUDA graphs) while checking reward stays stable.

## Minimal example

```python
from agentrl import (
    AgentTaskRecord,
    GRPOConfig,
    GRPOTrainer,
    SFTBootstrapTrainer,
    ToolSpec,
    make_tool_agent_task,
)
from agentrl.memory.layout import SharedWeightLayout
from peft import LoraConfig
from transformers import AutoTokenizer

LOOKUP = {"alpha": "4", "beta": "9"}

records = [
    AgentTaskRecord(
        task_id="sum",
        goal="Look up alpha and beta, add them, submit the result.",
        metadata={"final_answer": "13"},
        supervised_trace=(
            "TOOL: lookup[alpha]",
            "TOOL: lookup[beta]",
            "TOOL: add[4,9]",
            "FINAL: 13",
        ),
    ),
]

def split_args(raw: str) -> tuple[str, str]:
    left, right = raw.split(",", 1)
    return left.strip(), right.strip()

task = make_tool_agent_task(
    records=records,
    tools=[
        ToolSpec(
            name="lookup",
            description="Lookup a key.",
            handler=lambda arg, _state: LOOKUP[arg.strip()],
        ),
        ToolSpec(
            name="add",
            description="Add two integers.",
            handler=lambda arg, _state: str(
                int(split_args(arg)[0]) + int(split_args(arg)[1])
            ),
        ),
    ],
    final_answer_fn=lambda record, _state: str(record.metadata["final_answer"]),
    reward_fn=None,  # default shaped reward (not sparse 0/1)
)

# SFT bootstrap, then GRPO — see your Colab notebook for full ladder metrics.
config = GRPOConfig(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    batch_size=1,
    group_size=4,
    max_new_tokens=64,
    max_episode_steps=4,
    steps=5,
    use_continuous_batching=True,
    use_prefix_cache=True,
)

trainer = GRPOTrainer(
    config=config,
    environment=task.environment,
    verifier=task.verifier,
)
# trainer.train()
```

Pass `reward_fn=None` to use the built-in shaped tool-agent verifier (partial credit for tool steps, penalty for invalid actions, `1.0` on success).

## Runtime flags (`GRPOConfig`)

| Flag | Effect |
| --- | --- |
| `use_continuous_batching` | Uneven multi-turn batching (default `True`) |
| `use_prefix_cache` | Reuse shared prefixes across GRPO siblings |
| `use_cuda_graph_decode` | CUDA graph replay for decode steps (CUDA; default on when unset) |
| `init_adapter_path` | Load a bootstrap LoRA adapter before GRPO |

**Metrics to watch** during training: `total_step_time_ms`, `tokens_per_second`, `cache_hit_ratio`, `prefill_token_savings_pct`, `padding_ratio`, `peak_vram_mb`, `mean_reward`.

## Public API

```python
from agentrl import (
    AgentTaskRecord,
    BaseEnvironment,
    BaseVerifier,
    GRPOConfig,
    GRPOTrainer,
    SFTBootstrapTrainer,
    ToolSpec,
    make_tool_agent_task,
)
```

- **Custom agents:** implement `BaseEnvironment` + `BaseVerifier`.
- **Tool agents:** `make_tool_agent_task` with `TOOL: name[arg]` and `FINAL: answer` grammar.

## Architecture

```text
Your task (env + verifier)
    -> RolloutSource (standard or continuous orchestrator)
        -> PrefixCache + chunked prefill + CUDA graph decode (when enabled)
    -> RolloutBatch
    -> GRPOTrainer.step (PyTorch clipped GRPO objective)
```

`RolloutSource` is a protocol in `agentrl/core/rollout.py` for future async or remote collectors.

## Repository layout

```text
agentrl/
  agents.py          # make_tool_agent_task, shaped default verifier
  core/              # config, trainer, rollout, SFT
  generation/        # continuous batching, prefix cache, CUDA graphs
  memory/            # shared LoRA layout, trajectory buffer
  observability/     # metrics logger, systems profiler
  runtime/           # execution controller (OOM / headroom)
```

## Positioning

| Project | Role |
| --- | --- |
| **vLLM** | Production multi-GPU serving |
| **TRL** | HuggingFace post-training at scale |
| **AgentRL** | Readable single-GPU reference: rollout systems + GRPO on agent workloads |

## What AgentRL is not

- A production serving engine or vLLM replacement
- Multi-node training
- Bundled benchmarks, datasets, or example environments (use Colab / your own task code)
- Custom CUDA/Triton kernels (GRPO uses PyTorch)

## Security

- Trajectory buffer save/load uses `torch.load(..., weights_only=False)`—do not load untrusted checkpoint files.
- Register only trusted tool handlers; model-generated tool arguments run in your process.

## License

See [LICENSE](LICENSE).
