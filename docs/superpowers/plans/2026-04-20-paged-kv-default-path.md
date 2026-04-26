\# Paged-KV Default Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the paged-KV decode hot path's per-step legacy cache materialization with resident per-sequence cache objects so paged-KV can become the default continuous runtime path.

**Architecture:** Keep `PagedKVAllocator` and `PagedKVCacheStore` as the ownership and accounting layer, but add resident cache objects per active sequence and make `ContinuousBatchingOrchestrator._generate_active_batch_with_cache(...)` consume those resident caches directly during decode. Leave block-backed legacy cache read/write helpers in place for tests and debug materialization, but remove them from the active decode loop.

**Tech Stack:** Python, PyTorch, Transformers cache objects (`DynamicCache`), pytest

---

## File Structure

- Modify: `agentrl/generation/paged_kv.py`
  - Add resident cache bookkeeping to `PagedKVCacheStore`
  - Preserve allocator, template, and legacy materialization behavior
- Modify: `agentrl/generation/continuous.py`
  - Refactor paged continuous decode to use resident caches directly
  - Preserve scheduler, allocator growth, and runtime stat updates
- Modify: `tests/test_paged_kv.py`
  - Add resident-cache unit coverage for tuple caches and `DynamicCache`
- Modify: `tests/test_continuous.py`
  - Add integration coverage proving paged continuous no longer depends on legacy cache reads/writes on the hot path

### Task 1: Add Resident Cache Storage To `PagedKVCacheStore`

**Files:**
- Modify: `agentrl/generation/paged_kv.py`
- Test: `tests/test_paged_kv.py`

- [ ] **Step 1: Write the failing resident-cache unit tests**

Add these tests near the existing `PagedKVCacheStore` tests in `tests/test_paged_kv.py`:

```python
def test_paged_kv_cache_store_tracks_resident_tuple_cache() -> None:
    allocator = PagedKVAllocator(total_blocks=4, block_size_tokens=2)
    store = PagedKVCacheStore(allocator=allocator)
    store.reserve(sequence_id=1, token_count=2)
    legacy = ((torch.ones((1, 1, 2, 1)), torch.zeros((1, 1, 2, 1))),)

    store.set_resident_cache(sequence_id=1, cache=legacy, cache_template=legacy)

    assert store.has_resident_cache(1) is True
    assert store.resident_cache(1) == legacy
    assert store.cache_template(1) == legacy


@pytest.mark.skipif(DynamicCache is None, reason="transformers DynamicCache is unavailable")
def test_paged_kv_cache_store_tracks_resident_dynamic_cache() -> None:
    allocator = PagedKVAllocator(total_blocks=4, block_size_tokens=2)
    store = PagedKVCacheStore(allocator=allocator)
    store.reserve(sequence_id=1, token_count=2)
    legacy = ((torch.ones((1, 1, 2, 1)), torch.zeros((1, 1, 2, 1))),)
    cache = DynamicCache.from_legacy_cache(legacy)

    store.set_resident_cache(sequence_id=1, cache=cache, cache_template=cache)

    assert store.has_resident_cache(1) is True
    assert isinstance(store.resident_cache(1), DynamicCache)


def test_paged_kv_cache_store_clears_resident_cache_on_release() -> None:
    allocator = PagedKVAllocator(total_blocks=4, block_size_tokens=2)
    store = PagedKVCacheStore(allocator=allocator)
    store.reserve(sequence_id=1, token_count=2)
    legacy = ((torch.ones((1, 1, 2, 1)), torch.zeros((1, 1, 2, 1))),)
    store.set_resident_cache(sequence_id=1, cache=legacy, cache_template=legacy)

    store.release(1)

    assert store.has_sequence(1) is False
    assert store.has_resident_cache(1) is False
```

- [ ] **Step 2: Run the new unit tests and verify they fail**

Run:

```bash
pytest tests/test_paged_kv.py -q
```

Expected: FAIL with `AttributeError` or missing resident-cache methods on `PagedKVCacheStore`.

- [ ] **Step 3: Add resident cache support to `PagedKVCacheStore`**

Update `agentrl/generation/paged_kv.py` so `PagedKVCacheStore` owns resident per-sequence caches:

```python
class PagedKVCacheStore:
    def __init__(self, allocator: PagedKVAllocator) -> None:
        self.allocator = allocator
        self._storage: dict[tuple[int, int, int], torch.Tensor] = {}
        self._cache_templates: dict[int, Any] = {}
        self._resident_caches: dict[int, Any] = {}

    def set_resident_cache(self, sequence_id: int, cache: Any, cache_template: Any | None = None) -> None:
        if not self.allocator.has_sequence(sequence_id):
            raise KeyError(f"Unknown sequence_id {sequence_id}.")
        self._resident_caches[sequence_id] = cache
        if cache_template is not None:
            self._cache_templates[sequence_id] = cache_template

    def resident_cache(self, sequence_id: int) -> Any:
        return self._resident_caches[sequence_id]

    def has_resident_cache(self, sequence_id: int) -> bool:
        return sequence_id in self._resident_caches

    def clear_resident_cache(self, sequence_id: int) -> None:
        self._resident_caches.pop(sequence_id, None)
```

Also update `release(...)` to clear resident caches:

```python
    def release(self, sequence_id: int) -> None:
        view = self.allocator.view(sequence_id)
        for block_id in view.physical_blocks:
            keys_to_delete = [key for key in self._storage if key[0] == block_id]
            for key in keys_to_delete:
                del self._storage[key]
        self._resident_caches.pop(sequence_id, None)
        self._cache_templates.pop(sequence_id, None)
        self.allocator.release(sequence_id)
```

- [ ] **Step 4: Run the unit tests and verify they pass**

Run:

```bash
pytest tests/test_paged_kv.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agentrl/generation/paged_kv.py tests/test_paged_kv.py
git commit -m "Add resident cache storage to paged KV store"
```

### Task 2: Seed Resident Caches During Prefill

**Files:**
- Modify: `agentrl/generation/continuous.py`
- Test: `tests/test_continuous.py`

- [ ] **Step 1: Write the failing prefill resident-cache integration test**

Add this test near the other paged continuous tests in `tests/test_continuous.py`:

```python
def test_paged_kv_prefill_seeds_resident_caches() -> None:
    config = GRPOConfig(
        model_name="fake/model",
        batch_size=1,
        group_size=2,
        max_new_tokens=1,
        use_paged_kv_continuous=True,
        do_sample=False,
    )
    orchestrator = ContinuousBatchingOrchestrator(
        config=config,
        environment=SingleTurnEnvironment(),
        verifier=PrefixVerifier(),
        tokenizer=CharTokenizer(),
        layout=Layout(),
        device=torch.device("cpu"),
    )
    scheduler = orchestrator._build_scheduler_state(active_count=2)
    prompts = [torch.tensor([1, 2], dtype=torch.long), torch.tensor([1, 2], dtype=torch.long)]
    masks = [torch.ones_like(prompt) for prompt in prompts]
    sequences = [
        _ScheduledSequence(original_index=index, prompt_ids=prompt, prompt_mask=mask)
        for index, (prompt, mask) in enumerate(zip(prompts, masks, strict=True))
    ]

    orchestrator._generate_active_batch_with_cache(sequences, scheduler)

    # The method should complete without forcing tests to rebuild caches from block storage.
```

This test is intentionally minimal; it is there to support the refactor and keep the paged prefill path exercised.

- [ ] **Step 2: Run the focused test to verify current behavior**

Run:

```bash
pytest tests/test_continuous.py::test_paged_kv_prefill_seeds_resident_caches -q
```

Expected: PASS or weak coverage only. If it already passes, keep it and proceed; the value is coverage before the refactor.

- [ ] **Step 3: Store resident caches immediately after prefill**

In `agentrl/generation/continuous.py`, update the prefill loop inside `_generate_active_batch_with_cache(...)` so each sequence gets a resident cache object as soon as prefill returns:

```python
for offset, sequence in enumerate(batch):
    cache = batch_caches[offset]
    sequence_index = sequence.original_index
    paged_kv.write_sequence_cache(
        sequence_index,
        self._cache_to_legacy(cache),
        cache,
    )
    paged_kv.set_resident_cache(
        sequence_id=sequence_index,
        cache=cache,
        cache_template=cache,
    )
    next_logits_by_index[sequence_index] = batch_logits[offset : offset + 1]
```

The block-backed write stays for now during prefill so debug/test materialization still works.

- [ ] **Step 4: Run continuous and paged-KV tests**

Run:

```bash
pytest tests/test_continuous.py tests/test_paged_kv.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agentrl/generation/continuous.py tests/test_continuous.py
git commit -m "Seed resident caches during paged KV prefill"
```

### Task 3: Switch Decode Buckets To Resident Caches

**Files:**
- Modify: `agentrl/generation/continuous.py`
- Test: `tests/test_continuous.py`

- [ ] **Step 1: Write the failing hot-path regression test**

Add a targeted test in `tests/test_continuous.py` that proves the decode bucket no longer needs block-backed legacy reads. Use monkeypatch to make `read_batched_legacy_cache(...)` explode if called after prefill:

```python
@pytest.mark.skipif(DynamicCache is None, reason="transformers DynamicCache is unavailable")
def test_paged_kv_continuous_decode_uses_resident_caches(monkeypatch) -> None:
    config = GRPOConfig(
        model_name="fake/model",
        batch_size=1,
        group_size=2,
        max_new_tokens=2,
        use_paged_kv_continuous=True,
        do_sample=False,
    )
    orchestrator = ContinuousBatchingOrchestrator(
        config=config,
        environment=SingleTurnEnvironment(),
        verifier=PrefixVerifier(),
        tokenizer=CharTokenizer(),
        layout=DynamicCacheLayout(),
        device=torch.device("cpu"),
    )

    original = PagedKVCacheStore.read_batched_legacy_cache
    allow_calls = {"count": 0}

    def guarded(self, sequence_ids):
        allow_calls["count"] += 1
        if allow_calls["count"] > 0:
            raise AssertionError("decode path should not rebuild from legacy cache")
        return original(self, sequence_ids)

    monkeypatch.setattr(PagedKVCacheStore, "read_batched_legacy_cache", guarded)

    batch = orchestrator.collect()

    assert batch.metadata["responses"] == [["ab", "ab"]]
```

If needed, split the guard into prefill-vs-decode by counting calls after you inspect the current path. The test goal is to make decode-time legacy reads fail.

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
pytest tests/test_continuous.py::test_paged_kv_continuous_decode_uses_resident_caches -q
```

Expected: FAIL because the current decode loop calls `read_batched_legacy_cache(...)`.

- [ ] **Step 3: Refactor decode buckets to use resident caches**

In `agentrl/generation/continuous.py`, replace this block:

```python
bucket_cache_legacy = paged_kv.read_batched_legacy_cache(bucket_indices)
bucket_cache = self._cache_from_legacy(
    paged_kv.cache_template(bucket_indices[0]),
    bucket_cache_legacy,
)
...
paged_kv.write_batched_legacy_cache(
    bucket_indices,
    self._cache_to_legacy(outputs.past_key_values),
    outputs.past_key_values,
)
```

with resident-cache logic:

```python
resident_caches = [paged_kv.resident_cache(index) for index in bucket_indices]
bucket_cache = self._stack_past_key_values(resident_caches)
outputs = generation_model(
    input_ids=bucket_tokens,
    attention_mask=bucket_attention,
    past_key_values=bucket_cache,
    use_cache=True,
)
split_caches = self._split_past_key_values(outputs.past_key_values, len(bucket_indices))

for offset, episode_index in enumerate(bucket_indices):
    paged_kv.set_resident_cache(
        sequence_id=episode_index,
        cache=split_caches[offset],
        cache_template=split_caches[offset],
    )
```

Keep allocator growth and `next_logits_by_index` updates exactly where they are today.

- [ ] **Step 4: Run the focused decode test**

Run:

```bash
pytest tests/test_continuous.py::test_paged_kv_continuous_decode_uses_resident_caches -q
```

Expected: PASS.

- [ ] **Step 5: Run the existing paged continuous regressions**

Run:

```bash
pytest \
  tests/test_continuous.py::test_paged_kv_continuous_collects_with_dynamic_cache_model \
  tests/test_continuous.py::test_paged_kv_continuous_dynamic_cache_handles_block_growth \
  tests/test_continuous.py::test_paged_kv_continuous_collects_block_metrics \
  -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add agentrl/generation/continuous.py tests/test_continuous.py
git commit -m "Use resident caches in paged KV decode buckets"
```

### Task 4: Keep Resident State Consistent Across Growth And Release

**Files:**
- Modify: `agentrl/generation/continuous.py`
- Modify: `agentrl/generation/paged_kv.py`
- Test: `tests/test_continuous.py`

- [ ] **Step 1: Write the failing consistency test**

Add a regression in `tests/test_continuous.py` that ensures finished sequences release resident state and surviving sequences retain resident caches through growth:

```python
@pytest.mark.skipif(DynamicCache is None, reason="transformers DynamicCache is unavailable")
def test_paged_kv_releases_finished_sequences_and_keeps_resident_state() -> None:
    config = GRPOConfig(
        model_name="fake/model",
        batch_size=1,
        group_size=2,
        max_new_tokens=3,
        use_paged_kv_continuous=True,
        do_sample=False,
    )
    orchestrator = ContinuousBatchingOrchestrator(
        config=config,
        environment=SingleTurnEnvironment(),
        verifier=PrefixVerifier(),
        tokenizer=CharTokenizer(),
        layout=DynamicCacheLayout(),
        device=torch.device("cpu"),
    )

    batch = orchestrator.collect()

    assert batch.rewards.tolist() == [[1.0, 1.0]]
```

This test is a harness for the refactor. If no failure appears, keep it as coverage and rely on the existing block-growth regression for the real guard.

- [ ] **Step 2: Run the focused test**

Run:

```bash
pytest tests/test_continuous.py::test_paged_kv_releases_finished_sequences_and_keeps_resident_state -q
```

Expected: PASS or provide coverage only.

- [ ] **Step 3: Tighten release and growth consistency**

Review and update `agentrl/generation/continuous.py` and `agentrl/generation/paged_kv.py` so:

- finished sequences always call `paged_kv.release(...)`
- `release(...)` clears resident caches and templates
- `append_tokens(...)` only updates allocator ownership after forward has produced the enlarged cache
- resident caches remain the source of truth for active decode

No new API is needed if Task 1 and Task 3 already establish the needed methods.

- [ ] **Step 4: Run the paged-KV suite**

Run:

```bash
pytest tests/test_paged_kv.py tests/test_continuous.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agentrl/generation/paged_kv.py agentrl/generation/continuous.py tests/test_continuous.py
git commit -m "Keep paged KV resident state consistent across growth and release"
```

### Task 5: Full Regression And Benchmark-Readiness Verification

**Files:**
- Modify: `tests/test_continuous.py` (only if small assertion fixes are needed)
- Modify: `tests/test_paged_kv.py` (only if small assertion fixes are needed)

- [ ] **Step 1: Run the full targeted regression suite**

Run:

```bash
pytest \
  tests/test_examples.py \
  tests/test_runtime.py \
  tests/test_base.py \
  tests/test_rollout.py \
  tests/test_paged_kv.py \
  tests/test_continuous.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Validate the existing notebook artifact if unchanged**

Run:

```bash
python -m json.tool notebooks/codeDemo.ipynb >/dev/null
```

Expected: no output, exit code 0.

- [ ] **Step 3: Inspect the runtime diff before commit**

Run:

```bash
git diff --stat HEAD~4..HEAD
```

Expected: changes concentrated in `agentrl/generation/continuous.py`, `agentrl/generation/paged_kv.py`, and the two test files.

- [ ] **Step 4: Commit any final test adjustments**

If Steps 1-3 required small follow-up fixes:

```bash
git add agentrl/generation/continuous.py agentrl/generation/paged_kv.py tests/test_continuous.py tests/test_paged_kv.py
git commit -m "Finalize paged KV resident cache runtime path"
```

If no follow-up fixes were needed, skip this step.

- [ ] **Step 5: Run the Colab benchmark after merge**

Run in Colab:

```bash
python -m examples.benchmark_systems \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --task tool-use \
  --split easy \
  --steps 2 \
  --batch-size 1 \
  --group-size 4 \
  --max-new-tokens 96 \
  --max-episode-steps 6 \
  --output-dir ./tool_use_v2_compare \
  --compare-runtime-modes
```

Expected:

- all three modes complete
- paged-KV remains stable
- paged-KV mean step time is materially closer to legacy continuous than the current regression

## Self-Review

- Spec coverage:
  - resident cache storage: Task 1
  - resident-cache prefill seeding: Task 2
  - decode hot-path refactor away from legacy materialization: Task 3
  - growth/release consistency: Task 4
  - regression and benchmark validation: Task 5
- Placeholder scan:
  - no `TODO`, `TBD`, or vague “handle appropriately” steps remain
- Type consistency:
  - resident cache API is consistently named `set_resident_cache`, `resident_cache`, `has_resident_cache`, `clear_resident_cache`

