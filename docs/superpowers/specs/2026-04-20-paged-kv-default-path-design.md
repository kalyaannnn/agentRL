# Paged-KV Default Continuous Path Design

## Summary

Make paged-KV the default-worthy continuous runtime path by removing the per-step cache materialization bridge while keeping compatibility with the current Transformers `past_key_values` API.

The current paged-KV path is correct and stable, but it is slower than legacy continuous batching because each decode bucket performs a full cache round-trip:

1. read block-backed legacy cache
2. rebuild a cache object such as `DynamicCache`
3. run model forward
4. convert outputs back to legacy cache
5. split and clone cache tensors back into block storage

This design replaces that hot path with resident decode-time cache objects while preserving paged block ownership, allocation growth, and runtime accounting.

## Goals

- Keep paged-KV as the ownership and accounting model for continuous decode.
- Remove per-step legacy cache reconstruction from the active decode path.
- Preserve compatibility with the current Transformers `forward(..., past_key_values=..., use_cache=True)` interface.
- Keep paged-KV metrics and scheduler behavior visible in the benchmark harness.
- Make paged-KV competitive enough to become the default continuous path on single-GPU workloads.

## Non-Goals

- Introduce a true block-native attention kernel.
- Add a new public API for tasks, benchmarks, or models.
- Require custom model interfaces outside the current `past_key_values` contract.
- Remove the existing block materialization helpers entirely.

## Current Problem

Today the paged path uses `PagedKVCacheStore` as both:

- the allocator and block ownership model
- the decode-time source of truth for KV state

That forces the decode loop to reconstruct full cache tensors every step. This is the main reason the Colab tool-use benchmark showed:

- legacy continuous batching faster than standard rollout
- paged-KV continuous batching slower than legacy continuous batching

The regression is architectural rather than a scheduler bug. The paged implementation currently pays:

- the normal cached decode cost
- plus repeated block readback, concatenation, reconstruction, splitting, and cloning

## Proposed Architecture

`PagedKVCacheStore` becomes a hybrid runtime object with two internal layers.

### 1. Paged ownership layer

This layer remains the source of truth for:

- allocator state
- block tables
- token counts
- block growth
- reuse and pressure metrics
- resident sequence counts

### 2. Resident execution layer

This layer becomes the active decode-time source of truth for:

- one live cache object per active sequence
- one cache template per sequence

Examples of resident cache objects:

- legacy tuple cache
- `DynamicCache`
- other cache-like objects already supported by `_cache_to_legacy(...)` / `_cache_from_legacy(...)`

Block-backed legacy materialization remains available, but only for:

- tests
- debug inspection
- fallback paths when needed

It is no longer used on every decode step.

## Decode Flow

### Prefill

For each sequence:

1. run prompt prefill as today
2. reserve paged blocks based on prompt token count
3. store the resulting cache object as that sequence's resident cache
4. store the sequence cache template for reconstruction compatibility if needed later

### Decode

For each decode bucket:

1. gather resident cache objects for the admitted sequence bucket
2. stack them with `_stack_past_key_values(...)`
3. run model forward with the stacked cache
4. split the updated cache object with `_split_past_key_values(...)`
5. write those split cache objects back as resident caches
6. grow paged allocation metadata with `append_tokens(...)`
7. update runtime stats and scheduler metrics as today

The decode loop must not call these on the hot path:

- `read_batched_legacy_cache(...)`
- `_cache_from_legacy(...)`
- `_cache_to_legacy(...)`
- `write_batched_legacy_cache(...)`

### Release

When a sequence finishes:

1. clear its resident cache object
2. clear its template entry
3. release paged blocks

## Changes By File

### `agentrl/generation/paged_kv.py`

Add resident cache support to `PagedKVCacheStore`.

New responsibilities:

- hold `resident_cache` per sequence id
- expose getters/setters for resident caches
- clear resident caches on sequence release

Existing responsibilities retained:

- allocator ownership
- block-backed legacy cache storage
- cache template tracking
- materialization helpers

The block-backed read/write methods remain implemented, but become secondary.

### `agentrl/generation/continuous.py`

Refactor `_generate_active_batch_with_cache(...)` to use resident caches directly during decode.

Expected changes:

- prefill stores resident sequence caches instead of requiring later materialization
- decode buckets stack resident caches directly
- updated per-sequence caches are stored back into the paged store as resident caches
- paged block growth and scheduler accounting remain unchanged

No public benchmark CLI or task interface changes are required.

## Performance Hypothesis

This design should remove the largest avoidable overhead in the current paged path:

- repeated `torch.cat(...)` over block fragments
- repeated `DynamicCache.from_legacy_cache(...)`
- repeated `to_legacy_cache()` conversion
- repeated per-step split-and-clone back into `_storage`

Expected outcome:

- paged-KV should move materially closer to legacy continuous performance on small workloads
- paged-KV should have a better chance of winning on more memory-pressured or longer-lived workloads
- the runtime remains aligned with a vLLM-like direction instead of a pure bridge implementation

## Validation Plan

### Unit tests

Add or update tests to cover:

- setting and retrieving resident cache objects
- resident cache cleanup on release
- resident cache support for both tuple caches and `DynamicCache`
- no regression in block-backed legacy roundtrip helpers

### Integration tests

Keep and extend:

- paged-KV continuous collection with `DynamicCache`
- paged-KV block growth regression coverage
- scheduler block metrics and allocator stats

Add focused assertions that the paged continuous path can decode successfully using resident caches without requiring legacy materialization on each decode step.

### Benchmark validation

Rerun the same tool-use benchmark in Colab across:

- standard rollout
- legacy continuous batching
- paged-KV continuous batching

Success condition for this stage:

- paged-KV is stable
- paged-KV is materially closer to legacy continuous than the current regression
- if competitive enough, it becomes the default continuous path

## Risks

### Resident cache consistency

Resident caches and allocator token counts must stay synchronized. A mismatch would create silent correctness bugs or stale block accounting.

Mitigation:

- update token counts only when cache growth is committed
- keep targeted tests for block-boundary growth

### Mixed cache types

The runtime currently supports tuple caches and `DynamicCache` through generic conversion helpers. Resident storage must preserve the original type per sequence.

Mitigation:

- store cache objects per sequence directly
- keep templates for fallback reconstruction only

### Drift between resident and block-backed views

If block-backed storage is not updated on the hot path, debug materialization must not be treated as the primary source of truth during active decode.

Mitigation:

- document resident caches as the decode-time source of truth
- restrict block-backed materialization to debug/test/fallback usage

## Rollout Strategy

### Stage 1

Implement resident cache execution while preserving current public behavior and tests.

### Stage 2

Benchmark paged-KV again on the tool-use workload and compare against legacy continuous.

### Stage 3

If paged-KV becomes competitive, switch the default continuous path to paged-KV.

### Stage 4

After the default path is viable, evaluate a deeper block-native decode path as a follow-on systems project.

## Decision

Proceed with a vLLM-like transitional design:

- paged blocks remain the ownership model
- resident cache objects become the active decode model
- the current Transformers cache API remains the execution interface
- block materialization leaves the hot path
