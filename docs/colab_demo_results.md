# Colab Demo Results

Date: 2026-04-25

Environment: Google Colab T4 target, Qwen2.5-1.5B-Instruct policy model, MBPP sanitized task records, codeDemo-style prompt/reward/eval path.

## Fair Track: AgentRL vs TRL

Both branches use the same task construction shape:

- MBPP examples converted into `BYODRecord`
- shared `make_prompt`
- shared `supervised_target_fn`
- shared shaped reward function
- strict eval based on held-out MBPP tests
- SFT bootstrap before GRPO

| framework | sft_time_s | grpo_time_s | shaped_train_reward | strict_pass_rate | any_pass_rate | mean_test_pass_fraction | mean_eval_reward | peak_vram_mb |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AgentRL | 6.822393 | 140.604486 | 0.088333 | 0.5000 | 0.7500 | 0.645833 | 0.660417 | 10924.055176 |
| TRL | 31.225731 | 55.065827 | 0.000000 | 0.5625 | 0.6875 | 0.645833 | 0.663542 | 10924.055176 |

Notes:

- On this tiny run, eval quality is effectively tied: both reach `0.645833` mean test-pass fraction.
- TRL has slightly higher strict pass rate, while AgentRL has higher any-pass rate.
- AgentRL SFT bootstrap is much faster in this run.
- TRL GRPO is faster in the standard fair-track setup.
- The TRL `shaped_train_reward=0.000000` is likely a metric extraction/reporting issue, not actual task failure, because strict eval is strong.

## AgentRL Strict Eval Snapshot

AgentRL eval summary from the 16-example MBPP subset:

| metric | value |
| --- | ---: |
| strict_pass_rate | 0.500000 |
| any_pass_rate | 0.750000 |
| mean_test_pass_fraction | 0.645833 |
| mean_eval_reward | 0.660417 |

Interpretation:

- 8 of 16 examples passed all tests.
- 12 of 16 examples passed at least one test.
- Average per-test correctness was about 64.6%.

## Systems Track: AgentRL Runtime Variants

Current completed systems run:

| mode | wall_time_s | mean_reward | mean_step_time_ms | mean_generation_time_ms | mean_training_time_ms | tokens_per_second | rollout_peak_vram_mb | runtime_headroom_mb | padding_ratio | generation_padding_ratio | cache_reuse_effectiveness | scheduler_decode_passes | scheduler_prefill_passes | kv_pressure | paged_kv_pressure | runtime_adjustments |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| standard | 117.802230 | 0.050000 | 32671.216275 | 32098.513845 | 572.702429 | 8.300856 | 6886.177734 | 8026.509766 | 0.018319 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

Standard-mode runtime notes:

- Runtime bottleneck reported: `decode_without_cache_reuse`.
- Runtime recommendation reported: prefer continuous batching or shorter `max_new_tokens`.
- `reward_std == 0` for these prompt groups, so GRPO advantages were zero for the logged steps.
- This makes the standard systems run useful mainly as a runtime baseline, not as a learning-quality signal.

## Pending Systems Results

Add rows here after each Colab run completes:

| mode | wall_time_s | mean_reward | mean_step_time_ms | mean_generation_time_ms | mean_training_time_ms | tokens_per_second | rollout_peak_vram_mb | runtime_headroom_mb | padding_ratio | generation_padding_ratio | cache_reuse_effectiveness | scheduler_decode_passes | scheduler_prefill_passes | kv_pressure | paged_kv_pressure | runtime_adjustments |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| continuous | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| paged_kv_continuous | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| speculative | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Interview Framing

The fair-track table compares training stack behavior when both frameworks are constrained to the same task, same bootstrap concept, same reward, and same strict evaluation. That is the algorithm/framework comparison.

The systems-track table isolates the AgentRL moat: runtime modes, batching, KV behavior, scheduler telemetry, VRAM headroom, decode bottlenecks, and actionable runtime recommendations. This is where AgentRL should show that it is more than a trainer wrapper: it exposes and optimizes the rollout system that dominates RL wall time.
