# AgentRL TRL-Compatible GRPO Design

## Summary

Re-spec AgentRL's RL core so the GRPO objective is as close as practical to TRL's GRPO while preserving AgentRL's single-GPU systems moat.

This pass changes AgentRL from a GRPO-like policy-vs-reference delta objective to a TRL-compatible old-policy PPO-style clipped surrogate objective with optional reference KL. The runtime moat remains independent: standard, continuous batching, paged-KV, and speculative are preserved as internal rollout implementations rather than becoming part of the external parity claim.

The redesign is explicitly adapter-only. Full-model finetuning is out of scope.

## Goals

- Make AgentRL's GRPO semantics externally defensible as "TRL-compatible in the important ways."
- Preserve the single-GPU shared-weight memory story.
- Keep the runtime/systems benchmark track separate from the external TRL parity track.
- Avoid redesign work driven by future full-model finetuning support.

## Non-Goals

- Add full-model finetuning support.
- Add multi-GPU, multi-node, FSDP, DeepSpeed, or vLLM training abstractions.
- Implement `num_iterations > 1` reuse of the same rollout batch in this pass.
- Add TRL's broader loss surface beyond the default PPO-style GRPO path.
- Change AgentRL's task abstraction beyond what is required for correct completion masking.

## Final Decisions

- **Target algorithm:** TRL-compatible GRPO with rollout-time old-policy logprobs, PPO-style ratio/clipping, optional KL against reference, and completion-token-only averaging.
- **Reference handling:** keep a separate frozen reference adapter, not a full reference model copy.
- **Training layout:** one frozen base model plus two LoRA adapters:
  - `policy`: trainable
  - `reference`: frozen snapshot of initial policy adapter
- **Default parity regime:** `num_iterations = 1`
- **Parity path KL default:** `beta = 0.0`
- **Advantage normalization:** per-prompt-group mean/std normalization, zero advantage when group std is zero, no advantage clipping in the parity path.
- **Benchmark split:** TRL parity uses AgentRL standard rollout only; runtime benchmarking stays fully internal to AgentRL.

## Exact Target Behavior

### What "TRL-compatible GRPO" means in AgentRL

AgentRL must match TRL semantically in the parts that define the GRPO update:

- rollout stores **old-policy sampled-token logprobs**
- training computes **current vs old** importance ratios
- policy loss uses **PPO-style clipping**
- reference model is used only for **optional KL regularization**
- loss averages over **completion tokens only**
- rewards are converted to **group-relative normalized advantages**

This is based on the default GRPO behavior documented by TRL, especially the `num_iterations=1` path and the clipped surrogate objective used when old-policy reuse matters.

References:

- https://huggingface.co/docs/trl/grpo_trainer
- https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_config.py

### Semantics that must match

#### 1. Old-policy logprobs

At rollout collection time, AgentRL must store sampled-token logprobs under the exact policy that generated the completion.

These are the canonical "old policy" values for the later update step. They are not recomputed at train time.

#### 2. Current vs old ratio

During update, AgentRL must compute:

`ratio = exp(current_logprob - old_policy_logprob)`

This ratio is token-level and only defined on sampled completion tokens.

#### 3. PPO-style clipping

The policy term must be:

`min(ratio * advantage, clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage)`

with:

- `epsilon = 0.2` default
- symmetric clipping only in this pass

The current AgentRL objective based on policy-vs-reference logprob difference must be removed from the parity path.

#### 4. Optional KL against reference

If `beta == 0.0`, AgentRL must skip reference scoring entirely.

If `beta > 0.0`, AgentRL must score sampled completion tokens under the frozen reference policy and add a KL penalty term. The KL term in the parity path should follow the sampled-token approximator used in TRL-style GRPO:

`kl_token = exp(ref_logprob - current_logprob) - (ref_logprob - current_logprob) - 1`

The current full-vocab analytical KL may remain available for diagnostics, but it must not be the training KL in the parity path.

#### 5. Completion-token masking

Only assistant-generated completion tokens contribute to:

- policy objective
- KL objective
- clip statistics
- token-count normalization

Prompt tokens, observation tokens, tool outputs, and padding tokens must be masked out.

For multi-turn environments, "completion tokens" means all assistant spans across the transcript.

#### 6. Reward and advantage normalization

AgentRL must keep group-relative normalization:

- compute rewards for each prompt group
- subtract group mean
- divide by group std
- if std is zero, set the whole group's advantages to zero

Each completion gets one scalar advantage that is broadcast across that completion's masked tokens during loss computation.

### Semantics allowed to remain intentionally different

These are acceptable and not part of the external parity claim:

- shared-base dual-adapter layout instead of TRL's generic trainer internals
- AgentRL's environment/verifier API and transcript packing
- standard vs continuous vs paged-KV vs speculative rollout implementations
- runtime controller, scheduler metrics, KV accounting, and replay tooling
- adapter-only training and single-GPU memory optimizations

## Current Gap

Today AgentRL collects:

- `policy_logprobs`
- `ref_logprobs`

and the trainer computes a sequence-level delta between current policy and reference logprobs weighted by normalized advantages.

That is not close enough to TRL's GRPO to support a strong external parity claim because:

- it does not use rollout-time old-policy semantics as the core optimization target
- it does not use PPO-style current-vs-old clipping
- it makes the reference model central to the policy term instead of optional via KL

## Proposed Architecture

### Shared-weight dual-adapter layout

Replace the current "policy adapter enabled / reference = adapters disabled" behavior with a shared-base dual-adapter design.

The layout owns:

- one frozen base model loaded once
- one trainable `policy` LoRA adapter
- one frozen `reference` LoRA adapter

The `reference` adapter is initialized as a snapshot of the initial `policy` adapter state before training updates begin.

This keeps the single-GPU memory story:

- no duplicate base model
- only adapter weights are duplicated
- reference remains frozen

### Rollout batch contract

`RolloutBatch` becomes:

- `input_ids`
- `attention_mask`
- `completion_mask`
- `old_policy_logprobs`
- `rewards`
- `advantages`
- `metadata`

Changes:

- rename `action_mask` to `completion_mask`
- rename `policy_logprobs` to `old_policy_logprobs`
- remove `ref_logprobs` from the required batch contract

The rename is not cosmetic. It corrects the algorithmic meaning of the batch.

### Update-time objective

For one optimizer step on one rollout batch:

1. run current policy forward on packed sequences
2. gather sampled-token `current_logprobs`
3. read `old_policy_logprobs` from batch
4. compute token-level `ratio = exp(current - old)`
5. broadcast scalar completion advantages over masked completion tokens
6. compute PPO-style clipped surrogate token losses
7. if `beta > 0`, run reference forward and gather sampled-token `ref_logprobs`
8. compute sampled-token KL approximator
9. average over active completion tokens only

Total loss:

`loss = -mean_over_completion_tokens( surrogate - beta * kl_token )`

### Runtime backends

All rollout implementations must emit the same batch semantics:

- standard rollout
- continuous batching
- paged-KV continuous batching
- speculative decoding

They may differ internally in how they obtain rollout-time logprobs, but they must all export:

- `completion_mask`
- `old_policy_logprobs`

with identical meaning.

## Changes By File

### `agentrl/core/config.py`

Add:

- `epsilon: float = 0.2`
- `num_iterations: int = 1`
- `grpo_mode: str = "trl"`

Change:

- `beta` default from `0.01` to `0.0`

Deprecate for the TRL-compatible path:

- `clip_range`
- `use_adaptive_kl`
- `kl_target`
- `kl_beta_multiplier`
- `min_beta`
- `max_beta`

Validation rules:

- `use_lora` must stay `True`
- `num_iterations` must equal `1` in this pass
- `grpo_mode` must default to `"trl"` once this redesign lands
- if deprecated KL adaptation fields are set in `"trl"` mode, raise a configuration error rather than silently mixing semantics

### `agentrl/core/rollout.py`

Change `RolloutBatch` fields:

- `action_mask` -> `completion_mask`
- `policy_logprobs` -> `old_policy_logprobs`
- remove `ref_logprobs`

Collection behavior:

- keep transcript packing and assistant-span alignment
- store rollout-time old-policy sampled-token logprobs only
- stop computing reference logprobs during rollout

Advantage behavior:

- keep per-group normalization
- remove parity-path advantage clipping

### `agentrl/core/trainer.py`

Replace the current update rule with:

- gather current sampled-token logprobs
- compute current-vs-old ratio
- apply PPO-style clipping
- optionally compute reference KL
- average over active completion tokens

Additional metrics to log:

- `clip_ratio/region_mean`
- `clip_ratio/low_mean`
- `clip_ratio/high_mean`
- `mean_token_kl`
- `mean_ratio`

The existing sequence-delta policy loss and adaptive-KL beta update logic must not remain active in the TRL-compatible path.

### `agentrl/memory/layout.py`

Upgrade `SharedWeightLayout` to manage two named adapters on one base model.

Required behavior:

- `policy_forward(...)` activates adapter `"policy"`
- `reference_forward(...)` activates adapter `"reference"`
- only policy adapter params are trainable
- reference adapter params are frozen

The layout must expose adapter creation and snapshot logic cleanly enough that the trainer does not own adapter internals.

### `agentrl/memory/buffer.py`

Persist the renamed rollout artifacts:

- `completion_mask`
- `old_policy_logprobs`

Compatibility requirement:

- support loading older replay files with `action_mask` / `policy_logprobs` for one migration window
- when loading legacy payloads, map them onto the new field names

### `agentrl/generation/continuous.py`

Update continuous batching collection to emit:

- `completion_mask`
- `old_policy_logprobs`

Do not emit reference rollout logprobs in the parity path.

### `agentrl/generation/speculative.py`

Update speculative collection similarly.

If speculative verification already exposes policy logits on accepted tokens, those may be reused internally, but the exported batch field must still mean rollout-time old-policy sampled-token logprobs.

## Data and Loss Details

### Completion mask

The canonical mask is `completion_mask`, not `action_mask`.

Rationale:

- "completion" matches TRL terminology and benchmark framing
- the mask is not about generic actions in the RL sense
- it avoids ambiguity in multi-turn transcripts

### Old-policy logprobs

The canonical rollout tensor is `old_policy_logprobs`.

Rationale:

- it encodes update semantics, not just provenance
- it makes later `num_iterations > 1` support possible without renaming again

### Reference use

Reference scoring is update-time only and optional:

- no reference scoring during rollout collection
- no reference term inside the PPO surrogate
- no dependence on reference logprobs when `beta = 0.0`

### Reward and advantage assumptions

This pass keeps AgentRL's current group-relative reward normalization because it already matches the core GRPO assumption better than the current loss does.

The only required change is to stop clipping normalized advantages in the parity path.

## Scope Boundaries

### In scope

- TRL-compatible GRPO objective semantics
- batch schema rename and migration
- shared-base dual-adapter policy/reference layout
- rollout backend updates for old-policy logprob export
- trainer rewrite to ratio/clipping/KL semantics
- replay/buffer migration
- parity and runtime benchmark split

### Out of scope

- full-model finetuning
- `num_iterations > 1`
- sync/ref refresh during training
- dynamic/adaptive KL beta control
- extended TRL loss variants
- distributed training support

### Hard constraint

No design choice in this pass should be justified primarily by future full-model finetuning. The redesign serves the adapter-only single-GPU path first.

## Test Plan

### Unit tests

Add or update tests to verify:

- `RolloutBatch` emits `completion_mask` and `old_policy_logprobs`
- `completion_mask` excludes prompt, observation, tool, and padding tokens
- rollout-time old-policy logprobs are only populated on masked completion tokens
- dual-adapter switching selects policy vs reference correctly
- only policy adapter params receive gradients
- `beta == 0.0` skips reference forward
- `num_iterations != 1` raises configuration error

### Analytical objective tests

Add exact numerical tests with tiny handcrafted logits covering:

- ratio computation from `current` and `old`
- PPO clipping for positive and negative advantages
- unclipped vs clipped branch selection
- sampled-token KL approximator correctness
- token-averaged final loss correctness
- zero-std reward groups yielding zero policy contribution

These tests are required before claiming parity.

### Integration tests

Run end-to-end tests for:

- standard rollout path
- continuous batching rollout path
- speculative rollout path
- trainer step updating only policy adapter parameters
- replay/buffer roundtrip with renamed fields
- legacy replay payload compatibility

### Sufficient evidence for algorithmic parity

AgentRL is "close enough to TRL" only if all of the following are true:

- a frozen-batch parity harness matches a reference TRL-style loss implementation within floating-point tolerance
- external single-turn training runs show materially similar reward, clip, and KL behavior across at least 3 seeds under matched configs
- remaining differences are attributable to rollout/runtime implementation rather than objective definition

## Benchmark Plan

### Track 1: External parity

Purpose: establish the algorithmic claim.

Rules:

- use a clean single-turn task only
- use AgentRL **standard** rollout mode only
- disable continuous batching, paged-KV, and speculative
- match model, tokenizer, prompt format, SFT init, group size, batch size, generation params, `epsilon`, `beta`, LR, and seeds against TRL

Report:

- mean reward
- reward std
- advantage std
- clip ratio
- mean KL
- completion length
- loss curve

Success condition:

- objective-level parity on frozen batches
- training behavior close enough that "AgentRL uses TRL-style GRPO" is technically defensible

### Track 2: Internal systems

Purpose: measure AgentRL's actual moat.

Compare only inside AgentRL:

- standard
- continuous batching
- paged-KV
- speculative

Hold fixed:

- objective config
- task
- model
- seeds
- optimizer config

Report:

- reward / task success
- tokens per second
- prefill tokens per second
- decode tokens per second
- padding ratio
- peak VRAM
- runtime headroom
- scheduler metrics
- paged-KV allocator metrics
- speculative acceptance metrics where relevant

Hard separation rule:

- this track does not compare AgentRL to TRL
- this track does not support the external parity claim

## Implementation Order

1. change config fields and validation
2. change batch schema and replay compatibility layer
3. upgrade shared layout to dual adapters
4. update rollout backends to emit `old_policy_logprobs` and `completion_mask`
5. rewrite trainer loss to current-vs-old clipped surrogate with optional reference KL
6. add analytical objective tests
7. run external parity benchmark
8. run internal systems benchmark

## Risks

### Adapter-state confusion

The current code assumes "reference = adapters disabled." That is no longer sufficient once policy and reference must both be adapter states.

Mitigation:

- require explicit named adapter activation
- keep layout as the only owner of adapter switching

### Speculative logprob mismatch

Speculative decoding may tempt the implementation to mix draft-time and policy-time scores.

Mitigation:

- enforce that exported `old_policy_logprobs` always mean rollout-policy sampled-token logprobs
- add backend-specific tests

### Silent semantics drift from legacy config knobs

Old knobs like `clip_range` and adaptive KL fields can silently preserve non-parity behavior.

Mitigation:

- reject mixed semantics in config validation
- remove those knobs from parity examples and docs

## Decision

Proceed with a TRL-compatible GRPO redesign built around:

- rollout-time old-policy logprobs
- PPO-style current-vs-old clipping
- optional update-time reference KL
- completion-token masking
- shared-base dual-adapter LoRA layout

The external claim becomes algorithmic parity with TRL's default GRPO semantics. AgentRL's differentiation remains the single-GPU runtime layer.
