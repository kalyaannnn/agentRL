# Phase 1 AgentRL vs TRL Single-Turn Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fair single-GPU comparison between AgentRL and TRL on the existing BYOD code task, then summarize both task quality and systems behavior.

**Architecture:** Reuse the MBPP-style single-turn task shape already validated in `notebooks/codeDemo.ipynb` and convert it into two parallel runners: one AgentRL path and one TRL path. Keep the task, model, bootstrap set, and evaluation metric aligned, then add a small comparison script or notebook section that aggregates quality and systems metrics into one table.

**Tech Stack:** AgentRL, TRL, Transformers, PEFT/LoRA, PyTorch, notebook/script comparison harness, pytest.

---

## File Structure

- `notebooks/codeDemo.ipynb`
  - Existing reference workflow for the validated single-turn BYOD code task.
- `examples/byod_task.py`
  - Existing task helpers and reference BYOD task structure.
- `docs/open_source_demo.md`
  - Existing demo documentation to align wording and scope.
- `examples/compare_single_turn_baselines.py`
  - New comparison entrypoint that runs AgentRL and TRL on the same task and saves metrics.
- `examples/trl_single_turn_baseline.py`
  - New TRL-specific baseline runner for bootstrap + RL or best-effort equivalent online post-training path.
- `examples/agentrl_single_turn_baseline.py`
  - New AgentRL-specific runner extracted from the notebook flow into a reproducible script.
- `tests/test_examples.py`
  - Coverage for new example entrypoints and summary formatting.
- `docs/superpowers/specs/`
  - Existing design/spec area; no new spec planned in this task.
- `docs/`
  - Optional final documentation touch-up if the comparison becomes part of the public story.

### Task 1: Freeze the comparison workload

**Files:**
- Modify: `notebooks/codeDemo.ipynb`
- Modify: `examples/byod_task.py`
- Test: `tests/test_byod_task.py`

- [ ] **Step 1: Inspect the validated single-turn workload and extract the minimum reusable pieces**

Read:

```bash
python - <<'PY'
import json
from pathlib import Path

nb = json.loads(Path("notebooks/codeDemo.ipynb").read_text())
for cell in nb["cells"]:
    src = "".join(cell.get("source", []))
    if "MBPPExample" in src or "make_prompt(" in src or "strict_pass_fail(" in src:
        print("=" * 80)
        print(src[:3000])
PY
```

Expected: The notebook prints the MBPP example shape, prompt formatter, and strict execution-based verifier helpers.

- [ ] **Step 2: Write down the frozen workload contract in comments or docstrings before extracting code**

Target contract:

```python
# Workload contract for the comparison:
# - same prompt text for AgentRL and TRL
# - same supervised target code for bootstrap
# - same strict pass/fail execution-based verifier
# - same eval subset and seed
```

Expected: The workload contract is explicit before any runner code is added.

- [ ] **Step 3: Add or confirm a reusable task builder for the notebook workload**

If `examples/byod_task.py` does not already expose a compatible builder, add a focused helper with a shape like:

```python
def build_mbpp_comparison_records(limit: int, seed: int = 0) -> list[BYODRecord]:
    ...
```

The helper must produce records with:

```python
BYODRecord(
    input=prompt_text,
    reference_answer=f"task::{task_id}",
    supervised_target=reference_code,
    metadata={"task_id": task_id},
)
```

- [ ] **Step 4: Add a regression test for the extracted workload builder**

Add a focused test in `tests/test_byod_task.py`:

```python
def test_build_mbpp_comparison_records_returns_prompt_target_metadata():
    records = build_mbpp_comparison_records(limit=2, seed=0)
    assert len(records) == 2
    assert "Write Python code" in records[0].input
    assert records[0].reference_answer.startswith("task::")
    assert records[0].supervised_target
    assert "task_id" in records[0].metadata
```

- [ ] **Step 5: Run the targeted workload-builder test**

Run:

```bash
pytest tests/test_byod_task.py::test_build_mbpp_comparison_records_returns_prompt_target_metadata -v
```

Expected: PASS

- [ ] **Step 6: Commit the workload freeze**

```bash
git add examples/byod_task.py tests/test_byod_task.py notebooks/codeDemo.ipynb
git commit -m "Add reusable MBPP comparison workload builder"
```

### Task 2: Add a reproducible AgentRL baseline runner

**Files:**
- Create: `examples/agentrl_single_turn_baseline.py`
- Test: `tests/test_examples.py`

- [ ] **Step 1: Write a failing example smoke test for the AgentRL baseline runner**

Add a test in `tests/test_examples.py` with monkeypatched trainer/task internals:

```python
def test_agentrl_single_turn_baseline_main_runs(monkeypatch, tmp_path, capsys):
    captured = {}

    class StubTrainer:
        def __init__(self, config, environment, verifier):
            captured["config"] = config
            captured["environment"] = environment
            captured["verifier"] = verifier
        def train(self):
            return [{"step": 0, "mean_reward": 0.5}]

    monkeypatch.setattr("examples.agentrl_single_turn_baseline.GRPOTrainer", StubTrainer)
    monkeypatch.setattr("examples.agentrl_single_turn_baseline.run_bootstrap", lambda *a, **k: "adapter-path")

    from examples.agentrl_single_turn_baseline import main
    main(["--limit", "4", "--output-dir", str(tmp_path)])

    out = capsys.readouterr().out
    assert "mean_reward" in out
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
pytest tests/test_examples.py::test_agentrl_single_turn_baseline_main_runs -v
```

Expected: FAIL with import or missing module error.

- [ ] **Step 3: Implement the AgentRL baseline runner**

Create `examples/agentrl_single_turn_baseline.py` with:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentrl import GRPOConfig, GRPOTrainer, SFTBootstrapTrainer
from examples.byod_task import build_mbpp_comparison_task


def run_bootstrap(task, config):
    samples = task.bootstrap_samples()
    trainer = SFTBootstrapTrainer(config=config)
    trainer.train(samples, epochs=1)
    return str(trainer.save_adapter(config.output_dir))


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="./outputs/agentrl_single_turn")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    task = build_mbpp_comparison_task(limit=args.limit, seed=args.seed)

    bootstrap_config = GRPOConfig(
        model_name=args.model,
        steps=1,
        batch_size=4,
        output_dir=str(Path(args.output_dir) / "bootstrap"),
        use_continuous_batching=False,
        seed=args.seed,
    )
    adapter_path = run_bootstrap(task, bootstrap_config)

    rl_config = GRPOConfig(
        model_name=args.model,
        steps=1,
        batch_size=1,
        group_size=4,
        output_dir=str(Path(args.output_dir) / "rl"),
        init_adapter_path=adapter_path,
        seed=args.seed,
    )
    trainer = GRPOTrainer(
        config=rl_config,
        environment=task.environment,
        verifier=task.verifier,
    )
    history = trainer.train()
    print(json.dumps(history[-1] if history else {}, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the smoke test again**

Run:

```bash
pytest tests/test_examples.py::test_agentrl_single_turn_baseline_main_runs -v
```

Expected: PASS

- [ ] **Step 5: Commit the AgentRL runner**

```bash
git add examples/agentrl_single_turn_baseline.py tests/test_examples.py
git commit -m "Add AgentRL single-turn comparison runner"
```

### Task 3: Add a reproducible TRL baseline runner

**Files:**
- Create: `examples/trl_single_turn_baseline.py`
- Modify: `tests/test_examples.py`

- [ ] **Step 1: Write a failing smoke test for the TRL runner**

Add a test like:

```python
def test_trl_single_turn_baseline_main_runs(monkeypatch, tmp_path, capsys):
    class StubTrainer:
        def __init__(self, *args, **kwargs):
            pass
        def train(self):
            return None

    monkeypatch.setattr("examples.trl_single_turn_baseline.SFTTrainer", StubTrainer)
    monkeypatch.setattr("examples.trl_single_turn_baseline.GRPOTrainer", StubTrainer)

    from examples.trl_single_turn_baseline import main
    main(["--limit", "4", "--output-dir", str(tmp_path)])

    out = capsys.readouterr().out
    assert "trl" in out.lower()
```

- [ ] **Step 2: Run the TRL smoke test and confirm failure**

Run:

```bash
pytest tests/test_examples.py::test_trl_single_turn_baseline_main_runs -v
```

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement the TRL runner with the closest feasible parity**

Create `examples/trl_single_turn_baseline.py` with:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from examples.byod_task import build_mbpp_comparison_dataset

try:
    from trl import GRPOTrainer, GRPOConfig, SFTTrainer, SFTConfig
except ImportError as exc:
    raise SystemExit("TRL is required for this baseline. Install with `pip install trl`.") from exc


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="./outputs/trl_single_turn")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    dataset = build_mbpp_comparison_dataset(limit=args.limit, seed=args.seed)

    sft_args = SFTConfig(
        output_dir=str(Path(args.output_dir) / "bootstrap"),
        per_device_train_batch_size=4,
        num_train_epochs=1,
    )
    sft_trainer = SFTTrainer(model=args.model, args=sft_args, train_dataset=dataset["sft"])
    sft_trainer.train()

    grpo_args = GRPOConfig(
        output_dir=str(Path(args.output_dir) / "rl"),
        per_device_train_batch_size=1,
        num_generations=4,
        max_completion_length=128,
    )
    grpo_trainer = GRPOTrainer(
        model=sft_trainer.model,
        args=grpo_args,
        train_dataset=dataset["rl"],
        reward_funcs=[dataset["reward_fn"]],
    )
    grpo_trainer.train()
    print(json.dumps({"framework": "trl", "status": "ok"}, indent=2))


if __name__ == "__main__":
    main()
```

Note: If exact TRL GRPO parity is impossible for this task shape, document the mismatch inline and keep the runner focused on the closest workable baseline rather than forcing fake symmetry.

- [ ] **Step 4: Run the TRL smoke test**

Run:

```bash
pytest tests/test_examples.py::test_trl_single_turn_baseline_main_runs -v
```

Expected: PASS

- [ ] **Step 5: Commit the TRL runner**

```bash
git add examples/trl_single_turn_baseline.py tests/test_examples.py
git commit -m "Add TRL single-turn comparison runner"
```

### Task 4: Build the comparison harness and summary table

**Files:**
- Create: `examples/compare_single_turn_baselines.py`
- Modify: `tests/test_examples.py`

- [ ] **Step 1: Write a failing smoke test for the comparison harness**

Add a test:

```python
def test_compare_single_turn_baselines_writes_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "examples.compare_single_turn_baselines.run_agentrl",
        lambda args: {"framework": "agentrl", "reward": 0.6, "peak_vram_mb": 4200},
    )
    monkeypatch.setattr(
        "examples.compare_single_turn_baselines.run_trl",
        lambda args: {"framework": "trl", "reward": 0.55, "peak_vram_mb": 5100},
    )

    from examples.compare_single_turn_baselines import main
    main(["--output-dir", str(tmp_path)])

    summary = (tmp_path / "comparison.json").read_text()
    assert '"framework": "agentrl"' in summary
    assert '"framework": "trl"' in summary
```

- [ ] **Step 2: Run the failing comparison test**

Run:

```bash
pytest tests/test_examples.py::test_compare_single_turn_baselines_writes_summary -v
```

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement the comparison harness**

Create `examples/compare_single_turn_baselines.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from examples.agentrl_single_turn_baseline import main as run_agentrl_main
from examples.trl_single_turn_baseline import main as run_trl_main


def run_agentrl(args):
    return run_agentrl_main(args, return_metrics=True)


def run_trl(args):
    return run_trl_main(args, return_metrics=True)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="./outputs/single_turn_compare")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shared_args = ["--model", args.model, "--limit", str(args.limit), "--seed", str(args.seed)]
    agentrl_metrics = run_agentrl(shared_args + ["--output-dir", str(out_dir / "agentrl")])
    trl_metrics = run_trl(shared_args + ["--output-dir", str(out_dir / "trl")])

    comparison = {
        "agentrl": agentrl_metrics,
        "trl": trl_metrics,
    }
    (out_dir / "comparison.json").write_text(json.dumps(comparison, indent=2))
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the comparison smoke test**

Run:

```bash
pytest tests/test_examples.py::test_compare_single_turn_baselines_writes_summary -v
```

Expected: PASS

- [ ] **Step 5: Commit the comparison harness**

```bash
git add examples/compare_single_turn_baselines.py tests/test_examples.py
git commit -m "Add single-turn AgentRL vs TRL comparison harness"
```

### Task 5: Define the metric schema and quality/system claims

**Files:**
- Modify: `examples/agentrl_single_turn_baseline.py`
- Modify: `examples/trl_single_turn_baseline.py`
- Modify: `examples/compare_single_turn_baselines.py`
- Modify: `tests/test_examples.py`

- [ ] **Step 1: Add a failing test that both runners emit a compatible metrics schema**

Add a test:

```python
def test_single_turn_runner_metrics_schema():
    required = {
        "framework",
        "task_name",
        "model_name",
        "seed",
        "quality_metric",
        "peak_vram_mb",
        "wall_time_s",
    }
    agentrl = build_agentrl_result_stub()
    trl = build_trl_result_stub()
    assert required.issubset(agentrl)
    assert required.issubset(trl)
```

- [ ] **Step 2: Run the schema test and verify failure**

Run:

```bash
pytest tests/test_examples.py::test_single_turn_runner_metrics_schema -v
```

Expected: FAIL until the runners return aligned dicts.

- [ ] **Step 3: Update both runners to return the same keys**

Use this schema:

```python
{
    "framework": "...",
    "task_name": "mbpp_single_turn_byod",
    "model_name": args.model,
    "seed": args.seed,
    "quality_metric": float(...),
    "bootstrap_loss": float(...) or None,
    "rl_mean_reward": float(...) or None,
    "peak_vram_mb": float(...) or None,
    "wall_time_s": float(...),
    "notes": "...",
}
```

- [ ] **Step 4: Run the schema test again**

Run:

```bash
pytest tests/test_examples.py::test_single_turn_runner_metrics_schema -v
```

Expected: PASS

- [ ] **Step 5: Commit the metrics schema alignment**

```bash
git add examples/agentrl_single_turn_baseline.py examples/trl_single_turn_baseline.py examples/compare_single_turn_baselines.py tests/test_examples.py
git commit -m "Align single-turn comparison metrics schema"
```

### Task 6: Verification and documentation handoff

**Files:**
- Modify: `README.md`
- Modify: `docs/open_source_demo.md`
- Test: `tests/test_examples.py`

- [ ] **Step 1: Add a short documentation note for the comparison entrypoint**

Add a short section in `README.md` or `docs/open_source_demo.md` like:

```md
## Phase 1 Single-Turn Comparison

Use `examples/compare_single_turn_baselines.py` to compare AgentRL and TRL
on the validated MBPP-style BYOD code task under a matched single-GPU setup.
```

- [ ] **Step 2: Add a documentation regression test if the README is updated**

Example:

```python
def test_readme_mentions_single_turn_comparison():
    readme = Path("README.md").read_text()
    assert "compare_single_turn_baselines.py" in readme
```

- [ ] **Step 3: Run the focused example/documentation test slice**

Run:

```bash
pytest tests/test_examples.py -q
```

Expected: PASS

- [ ] **Step 4: Run a real smoke invocation for each runner**

Run:

```bash
python -m examples.agentrl_single_turn_baseline --limit 4 --output-dir /tmp/agentrl_single_turn_smoke
python -m examples.trl_single_turn_baseline --limit 4 --output-dir /tmp/trl_single_turn_smoke
python -m examples.compare_single_turn_baselines --limit 4 --output-dir /tmp/single_turn_compare_smoke
```

Expected:
- all commands finish without import/config errors
- `comparison.json` is written for the final command

- [ ] **Step 5: Commit the docs and verification pass**

```bash
git add README.md docs/open_source_demo.md tests/test_examples.py
git commit -m "Document single-turn AgentRL vs TRL comparison"
```

## Self-Review

- Spec coverage:
  - Comparison target chosen: AgentRL vs TRL on the validated single-turn BYOD code task.
  - Fairness constraints covered: same workload, model, seed, approximate budget, shared metric schema.
  - Deliverable covered: runnable baselines, comparison harness, summary output, docs mention.
- Placeholder scan:
  - No `TODO` or `TBD` markers remain in task steps.
  - Where exact TRL parity may be imperfect, the plan explicitly requires documenting the mismatch rather than hiding it.
- Type consistency:
  - Shared metric schema is defined once and reused across tasks.
  - Runner names and file paths are consistent across tests and scripts.
