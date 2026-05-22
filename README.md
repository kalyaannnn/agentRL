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

The runtime helps any workload with **shared prompt prefixes** on one GPU: grouped GRPO sampling, multi-turn agent transcripts, or few-shot prompts with a common system block. **You define the task** (`BaseEnvironment` + `BaseVerifier`); the repo does not ship bundled datasets or example environments.

## Systems demo (GSM8K, single A100)

End-to-end run from [`systems_demo_abcd.ipynb`](systems_demo_abcd.ipynb): SFT bootstrap on GSM8K, then GRPO with three runtime modes from the same SFT adapter, same seed, 80 steps.

| Mode | Step time (ms) | Tokens/sec | Cache hit ratio | Prefill savings | Peak VRAM (MB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| A — standard | 19315.6 | 13.2 | 0.00 | 0.0% | 10841.9 |
| B — continuous batching | 7761.7 | 32.7 | 0.00 | 0.0% | 11008.4 |
| C — continuous + prefix cache | 7473.7 | 33.4 | 0.92 | 92.0% | 10877.3 |

Headline:

- B alone gives **~2.5× faster step time** and **~2.5× tokens/sec** vs A on the same GRPO workload.
- Adding prefix cache in C contributes **~92% prefill-token savings** and a **~0.92 cache hit ratio** with no extra VRAM cost.
- VRAM stays flat across modes (~10.6–10.7 GB), so throughput is gained without expanding the memory budget.

Model: `Qwen/Qwen2.5-1.5B-Instruct`, bf16, group_size=4, batch_size=2, max_new_tokens=48, single A100.

## Recommended workflow

Sparse verifier rewards rarely work from a cold start:

1. Define your `BaseEnvironment` + `BaseVerifier` and prepare prompt/target pairs.
2. Run **`SFTBootstrapTrainer`** to teach the output format.
3. Run **`GRPOTrainer`** with `init_adapter_path` pointing at the bootstrap adapter.
4. Compare runtime modes (standard → continuous → prefix cache → CUDA graphs) while checking reward stays stable.

## Minimal example

```python
from agentrl import (
    BaseEnvironment,
    BaseVerifier,
    GRPOConfig,
    GRPOTrainer,
)

class MyEnvironment(BaseEnvironment):
    def reset(self) -> str:
        ...
    def step(self, action: str) -> tuple[str, bool]:
        ...
    def state(self) -> dict:
        ...

class MyVerifier(BaseVerifier):
    def verify(self, response: str, env_state: dict) -> float:
        ...

config = GRPOConfig(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    batch_size=2,
    group_size=4,
    max_new_tokens=48,
    steps=80,
    use_continuous_batching=True,
    use_prefix_cache=True,
    init_adapter_path="./checkpoints/sft_bootstrap",  # optional warm start
)

trainer = GRPOTrainer(
    config=config,
    environment=MyEnvironment(),
    verifier=MyVerifier(),
)
# trainer.train()
```

For a working end-to-end example (SFT bootstrap on GSM8K, format-repair pass, GRPO A/B/C ladder), see [`systems_demo_abcd.ipynb`](systems_demo_abcd.ipynb).

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
    BaseEnvironment,
    BaseVerifier,
    GRPOConfig,
    GRPOTrainer,
    SFTBootstrapTrainer,
)
```

Implement `BaseEnvironment` (`reset`, `step`, `state`) and `BaseVerifier` (`verify`) for your task. See `systems_demo_abcd.ipynb` for a complete GSM8K example.

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

## License

See [LICENSE](LICENSE).
