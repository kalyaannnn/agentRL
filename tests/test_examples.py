from __future__ import annotations

import sys

from examples.benchmark_gsm8k_subset import main as benchmark_gsm8k_main
from examples import benchmark_gsm8k_subset, train_math
from examples.gsm8k_subset import (
    GSM8KStyleProblem,
    GSM8KSubsetEnvironment,
    GSM8KSubsetVerifier,
)
from examples.math_env import MathEnvironment, MathProblem, MathVerifier


def test_math_environment_and_verifier_roundtrip() -> None:
    env = MathEnvironment(
        split="train",
        problems=[MathProblem("2 + 2", 4)],
        seed=123,
    )
    verifier = MathVerifier()

    prompt = env.reset()
    _, done = env.step("Final answer: 4")
    reward = verifier.verify("Reasoning...\nFinal answer: 4", env.state())

    assert "2 + 2" in prompt
    assert done is True
    assert reward == 1.0


def test_train_math_uses_public_api_shape(monkeypatch) -> None:
    captured = {}

    class StubTrainer:
        def __init__(self, config, environment, verifier):
            captured["config"] = config
            captured["environment"] = environment
            captured["verifier"] = verifier

        def train(self) -> None:
            captured["trained"] = True

    monkeypatch.setattr(train_math, "GRPOTrainer", StubTrainer)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_math.py",
            "--model",
            "fake/model",
            "--steps",
            "3",
            "--batch-size",
            "2",
            "--group-size",
            "4",
            "--max-new-tokens",
            "32",
            "--output-dir",
            "./artifacts",
            "--split",
            "eval",
        ],
    )

    train_math.main()

    assert captured["config"].model_name == "fake/model"
    assert captured["config"].steps == 3
    assert captured["config"].batch_size == 2
    assert captured["config"].group_size == 4
    assert captured["config"].max_new_tokens == 32
    assert captured["config"].output_dir == "./artifacts"
    assert captured["environment"].split == "eval"
    assert captured["verifier"].__class__.__name__ == "MathVerifier"
    assert captured["trained"] is True


def test_smoke_split_uses_easy_builtin_problems() -> None:
    env = MathEnvironment(split="smoke", seed=0)

    prompt = env.reset()
    answer = int(env.state()["answer"])

    assert "Reply with exactly one line and nothing else" in prompt
    assert answer in {2, 3, 4, 5, 6}


def test_smoke_verifier_requires_exact_one_line_format() -> None:
    verifier = MathVerifier()

    strict_state = {"answer": 6, "split": "smoke"}

    assert verifier.verify("Final answer: 6", strict_state) == 1.0
    assert verifier.verify("6\nThe answer is 6.", strict_state) == 0.0
    assert verifier.verify("6Human: Solve the following equation", strict_state) == 0.0


def test_easy_split_uses_harder_synthetic_problems_and_strict_format() -> None:
    env = MathEnvironment(split="easy", seed=0)
    verifier = MathVerifier()

    prompt = env.reset()
    strict_state = {"answer": 5, "split": "easy"}

    assert "Reply with exactly one line and nothing else" in prompt
    assert verifier.verify("Final answer: 5", strict_state) == 1.0
    assert verifier.verify("5\nextra text", strict_state) == 0.0


def test_gsm8k_subset_environment_and_verifier_roundtrip() -> None:
    env = GSM8KSubsetEnvironment(
        split="train",
        problems=[
            GSM8KStyleProblem(
                "A jar has 10 candies and 3 more are added. How many candies are in the jar?",
                13,
            )
        ],
        seed=123,
    )
    verifier = GSM8KSubsetVerifier()

    prompt = env.reset()
    _, done = env.step("Final answer: 13")
    reward = verifier.verify("Final answer: 13", env.state())

    assert "grade-school math word problem" in prompt
    assert "Final answer: <integer>" in prompt
    assert done is True
    assert reward == 1.0


def test_gsm8k_subset_verifier_requires_exact_one_line_format() -> None:
    verifier = GSM8KSubsetVerifier()
    state = {"answer": 18, "split": "train", "dataset": "gsm8k_style_subset"}

    assert verifier.verify("Final answer: 18", state) == 1.0
    assert verifier.verify("18", state) == 0.0
    assert verifier.verify("Final answer: 18\nextra", state) == 0.0


def test_benchmark_gsm8k_subset_uses_public_api_shape(monkeypatch) -> None:
    captured = {}

    class StubTrainer:
        def __init__(self, config, environment, verifier):
            captured["config"] = config
            captured["environment"] = environment
            captured["verifier"] = verifier

        def train(self) -> None:
            captured["trained"] = True

    monkeypatch.setattr(benchmark_gsm8k_subset, "GRPOTrainer", StubTrainer)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_gsm8k_subset.py",
            "--model",
            "fake/model",
            "--steps",
            "7",
            "--batch-size",
            "1",
            "--group-size",
            "4",
            "--max-new-tokens",
            "48",
            "--output-dir",
            "./bench",
            "--split",
            "eval",
            "--replay-every",
            "1",
        ],
    )

    benchmark_gsm8k_main()

    assert captured["config"].model_name == "fake/model"
    assert captured["config"].steps == 7
    assert captured["config"].batch_size == 1
    assert captured["config"].group_size == 4
    assert captured["config"].max_new_tokens == 48
    assert captured["config"].output_dir == "./bench"
    assert captured["config"].replay_every == 1
    assert captured["environment"].split == "eval"
    assert captured["verifier"].__class__.__name__ == "GSM8KSubsetVerifier"
    assert captured["trained"] is True
