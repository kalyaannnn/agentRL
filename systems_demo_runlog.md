# Systems Demo Run Log (Public Demo) - GSM8K

Date: 2026-05-21  
Model: `Qwen/Qwen2.5-1.5B-Instruct`  
Setup: SFT bootstrap adapter loaded, GRPO systems ladder A/B/C

## Status
- Completed `A_standard` (`./runs/public_demo_A_standard_seed42`)
- Failed `B_continuous` with CUDA OOM during generation (after one runtime retry)
- Deferred `C_continuous_prefix` until memory is stabilized

## A_standard observations
- Step time around 32-34s/step
- Dominant bottleneck: `decode_without_cache_reuse`
- `cache_hit_ratio` around 0.0 (expected for standard mode without continuous/prefix features)
- Repeated runtime recommendation:
  - "Decode dominates with weak cache reuse; prefer continuous batching or shorter max_new_tokens."
- Reward was noisy but non-degenerate (several zero-std batches mixed with higher-reward batches)

## OOM details on B_continuous
- Runtime warning before failure:
  - `OOM during generation. Retrying with chunk_size=2, prefill_chunk_size=256 (1/1 retries used).`
- Final error:
  - `OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB`
  - GPU capacity: `14.56 GiB`
  - Free at failure: `19.81 MiB`
  - In use by process: `14.54 GiB`

## Next run plan (after cleanup/restart)
1. Restart runtime (recommended).
2. Re-run only B and C with safer knobs:
   - `max_new_tokens=64` (or keep 96 only if stable),
   - `batch_size=1`, `group_size=4`,
   - `prefill_chunk_size=128` or `256`.
3. Run conditions separately (not in one loop) with CUDA cleanup between runs.
4. Keep same seed and subset for fair A/B/C systems comparison.

