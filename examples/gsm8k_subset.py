"""Filtered GSM8K-style benchmark environment and verifier for AgentRL."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from agentrl import BaseEnvironment, BaseVerifier


_STRICT_FINAL_ANSWER_RE = re.compile(r"^\s*Final answer:\s*(-?\d+)\s*$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class GSM8KStyleProblem:
    """One single-step arithmetic word problem."""

    prompt: str
    answer: int


class GSM8KSubsetEnvironment(BaseEnvironment):
    """Single-turn environment backed by a small GSM8K-style word-problem subset.

    The bundled examples are intentionally filtered toward short, deterministic
    single-step or near-single-step arithmetic so small instruction-tuned models
    can produce a measurable reward distribution on a single GPU.
    """

    def __init__(
        self,
        split: str = "train",
        problems: list[GSM8KStyleProblem] | None = None,
        seed: int = 0,
    ) -> None:
        self.split = split
        self._rng = random.Random(seed)
        self._problems = problems or self._default_problems(split)
        self._current_problem: GSM8KStyleProblem | None = None

    def reset(self) -> str:
        """Return the next filtered word problem prompt."""

        self._current_problem = self._rng.choice(self._problems)
        return (
            "Solve the following grade-school math word problem.\n"
            "Reply with exactly one line and nothing else:\n"
            "Final answer: <integer>\n\n"
            f"Problem: {self._current_problem.prompt}"
        )

    def step(self, action: str) -> tuple[str, bool]:
        """Mark the single-turn episode as complete."""

        del action
        return ("done", True)

    def state(self) -> dict[str, int | str]:
        """Expose the answer key and split metadata to the verifier."""

        if self._current_problem is None:
            raise RuntimeError("reset() must be called before state().")
        return {
            "prompt": self._current_problem.prompt,
            "answer": self._current_problem.answer,
            "split": self.split,
            "dataset": "gsm8k_style_subset",
        }

    def _default_problems(self, split: str) -> list[GSM8KStyleProblem]:
        train = [
            GSM8KStyleProblem(
                "A bakery made 12 muffins in the morning and 7 more in the afternoon. "
                "How many muffins did the bakery make in total?",
                19,
            ),
            GSM8KStyleProblem(
                "Noah has 15 toy cars and gives 4 of them to his friend. "
                "How many toy cars does Noah have left?",
                11,
            ),
            GSM8KStyleProblem(
                "A bus has 18 passengers. At the next stop, 5 passengers get off and "
                "3 passengers get on. How many passengers are on the bus now?",
                16,
            ),
            GSM8KStyleProblem(
                "Lena reads 6 pages on Monday, 8 pages on Tuesday, and 5 pages on Wednesday. "
                "How many pages does Lena read altogether?",
                19,
            ),
            GSM8KStyleProblem(
                "There are 24 oranges packed equally into 6 boxes. "
                "How many oranges are in each box?",
                4,
            ),
            GSM8KStyleProblem(
                "Each notebook costs 3 dollars. Amir buys 7 notebooks. "
                "How many dollars does Amir spend?",
                21,
            ),
            GSM8KStyleProblem(
                "Maya had 30 stickers and used 12 of them. Then she bought 5 more stickers. "
                "How many stickers does Maya have now?",
                23,
            ),
            GSM8KStyleProblem(
                "A gardener plants 4 rows of flowers with 5 flowers in each row. "
                "How many flowers are planted in total?",
                20,
            ),
            GSM8KStyleProblem(
                "A library has 40 chairs. 9 chairs are moved out for cleaning. "
                "How many chairs remain in the library?",
                31,
            ),
            GSM8KStyleProblem(
                "Two friends save money. One saves 14 dollars and the other saves 9 dollars. "
                "How many dollars did they save together?",
                23,
            ),
            GSM8KStyleProblem(
                "A coach brings 28 soccer balls to practice and splits them equally among 4 teams. "
                "How many balls does each team get?",
                7,
            ),
            GSM8KStyleProblem(
                "Sofia buys 3 packs of pencils. Each pack has 6 pencils. "
                "How many pencils does Sofia buy?",
                18,
            ),
        ]
        eval_set = [
            GSM8KStyleProblem(
                "A farmer picks 11 apples in the morning and 13 apples in the afternoon. "
                "How many apples does the farmer pick that day?",
                24,
            ),
            GSM8KStyleProblem(
                "Jordan has 26 marbles and loses 8 marbles. "
                "How many marbles does Jordan have left?",
                18,
            ),
            GSM8KStyleProblem(
                "A store puts 32 cans equally on 4 shelves. "
                "How many cans are on each shelf?",
                8,
            ),
            GSM8KStyleProblem(
                "A class has 5 tables with 4 students at each table. "
                "How many students are in the class?",
                20,
            ),
            GSM8KStyleProblem(
                "Priya had 17 dollars, spent 6 dollars, and then earned 4 dollars. "
                "How many dollars does Priya have now?",
                15,
            ),
            GSM8KStyleProblem(
                "A train travels 9 miles in the first hour and 12 miles in the second hour. "
                "How many miles does the train travel in two hours?",
                21,
            ),
        ]
        if split not in {"train", "eval"}:
            raise ValueError(f"Unsupported split '{split}'. Expected 'train' or 'eval'.")
        return train if split == "train" else eval_set


class GSM8KSubsetVerifier(BaseVerifier):
    """Strict exact-match verifier for the bundled benchmark subset."""

    def verify(self, response: str, env_state: dict[str, int | str]) -> float:
        """Return 1.0 for an exact `Final answer: <integer>` match."""

        answer = int(env_state["answer"])
        match = _STRICT_FINAL_ANSWER_RE.fullmatch(response)
        if match is None:
            return 0.0
        return 1.0 if int(match.group(1)) == answer else 0.0
