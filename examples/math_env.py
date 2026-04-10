"""Reference math environment and verifier for AgentRL demos."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from agentrl import BaseEnvironment, BaseVerifier


_FINAL_ANSWER_RE = re.compile(r"final answer:\s*(-?\d+)", re.IGNORECASE)
_INTEGER_RE = re.compile(r"-?\d+")


@dataclass(frozen=True, slots=True)
class MathProblem:
    """One arithmetic training example."""

    prompt: str
    answer: int


class MathEnvironment(BaseEnvironment):
    """Single-turn arithmetic environment for smoke tests and demos.

    The environment samples from a small built-in dataset by default so the demo
    runs without external dependencies.
    """

    def __init__(
        self,
        split: str = "train",
        problems: list[MathProblem] | None = None,
        seed: int = 0,
    ) -> None:
        self.split = split
        self._rng = random.Random(seed)
        self._problems = problems or self._default_problems(split)
        self._current_problem: MathProblem | None = None

    def reset(self) -> str:
        """Return the next arithmetic prompt."""

        self._current_problem = self._rng.choice(self._problems)
        return (
            "Solve the following problem. Show your reasoning briefly, then put the final line as "
            "`Final answer: <integer>`.\n\n"
            f"Problem: {self._current_problem.prompt}"
        )

    def step(self, action: str) -> tuple[str, bool]:
        """Mark the single-turn episode as complete."""

        del action
        return ("done", True)

    def state(self) -> dict[str, int | str]:
        """Expose the ground-truth answer to the verifier."""

        if self._current_problem is None:
            raise RuntimeError("reset() must be called before state().")
        return {
            "prompt": self._current_problem.prompt,
            "answer": self._current_problem.answer,
            "split": self.split,
        }

    def _default_problems(self, split: str) -> list[MathProblem]:
        train = [
            MathProblem("7 + 5", 12),
            MathProblem("18 - 9", 9),
            MathProblem("6 * 4", 24),
            MathProblem("27 / 3", 9),
            MathProblem("3 * 8 + 2", 26),
            MathProblem("14 + 11 - 3", 22),
        ]
        eval_set = [
            MathProblem("9 + 9", 18),
            MathProblem("15 - 7", 8),
            MathProblem("5 * 5", 25),
        ]
        return train if split == "train" else eval_set


class MathVerifier(BaseVerifier):
    """Deterministic verifier that checks integer final answers."""

    def verify(self, response: str, env_state: dict[str, int | str]) -> float:
        """Return `1.0` on exact integer match and `0.0` otherwise."""

        answer = int(env_state["answer"])
        extracted = self._extract_answer(response)
        return 1.0 if extracted == answer else 0.0

    def _extract_answer(self, response: str) -> int | None:
        match = _FINAL_ANSWER_RE.search(response)
        if match is not None:
            return int(match.group(1))

        integers = _INTEGER_RE.findall(response)
        if not integers:
            return None
        return int(integers[-1])
