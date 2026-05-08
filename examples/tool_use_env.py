"""Canonical deterministic multi-turn tool-use task for AgentRL benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

from agentrl import (
    AgentTaskRecord,
    BaseEnvironment,
    BaseVerifier,
    ToolSpec,
    make_tool_agent_task,
)


@dataclass(frozen=True, slots=True)
class ToolUseTask:
    """One deterministic task instance for the tool-use benchmark."""

    task_id: str
    goal: str
    optimal_actions: tuple[str, ...]
    final_answer: str


class ToolUseEnvironment(BaseEnvironment):
    """Small deterministic multi-turn environment built on AgentRL's public scaffold."""

    LOOKUP_TABLE = {
        "alpha": "4",
        "beta": "9",
        "gamma": "12",
        "capital_france": "Paris",
        "planet_red": "Mars",
        "metal_au": "gold",
        "word_prime": "prime",
    }

    def __init__(
        self,
        split: str = "train",
        tasks: list[ToolUseTask] | None = None,
        seed: int = 0,
    ) -> None:
        self.split = split
        self._tasks = tasks or self._default_tasks(split)
        self._task_adapter = make_tool_agent_task(
            records=[self._to_agent_record(task, split=split) for task in self._tasks],
            tools=self._tool_specs(),
            final_answer_fn=lambda record, state: str(record.metadata["final_answer"]),
            seed=seed,
        )
        self._environment = self._task_adapter.environment

    def reset(self) -> str:
        return self._environment.reset()

    def step(self, action: str) -> tuple[str, bool]:
        return self._environment.step(action)

    def state(self) -> dict[str, object]:
        state = self._environment.state()
        metadata = dict(state.get("metadata", {}))
        state["split"] = self.split
        state["optimal_actions"] = list(metadata.get("optimal_actions", []))
        return state

    def render_generation_prompt(
        self,
        tokenizer: object,
        observations: list[str],
        actions: list[str],
    ) -> str:
        renderer = getattr(self._environment, "render_generation_prompt")
        return renderer(tokenizer, observations, actions)

    def render_transcript(
        self,
        tokenizer: object,
        observations: list[str],
        actions: list[str],
    ) -> tuple[str, list[tuple[int, int]]]:
        renderer = getattr(self._environment, "render_transcript")
        return renderer(tokenizer, observations, actions)

    def _to_agent_record(self, task: ToolUseTask, *, split: str) -> AgentTaskRecord:
        return AgentTaskRecord(
            task_id=task.task_id,
            goal=task.goal,
            metadata={
                "split": split,
                "optimal_actions": list(task.optimal_actions),
                "final_answer": task.final_answer,
            },
            supervised_trace=(*task.optimal_actions, f"FINAL: {task.final_answer}"),
        )

    def _tool_specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="lookup",
                description="Return a value from the deterministic lookup table.",
                handler=lambda argument, state: self.LOOKUP_TABLE.get(argument, f"unknown:{argument}"),
            ),
            ToolSpec(
                name="add",
                description="Add two comma-separated integers.",
                handler=lambda argument, state: str(
                    int(self._split_two_args(argument)[0].strip())
                    + int(self._split_two_args(argument)[1].strip())
                ),
            ),
            ToolSpec(
                name="concat",
                description="Concatenate two comma-separated strings.",
                handler=lambda argument, state: "".join(part.strip() for part in self._split_two_args(argument)),
            ),
        ]

    def _split_two_args(self, raw_argument: str) -> tuple[str, str]:
        if "," not in raw_argument:
            raise ValueError(f"Expected two comma-separated arguments, got: {raw_argument!r}")
        left, right = raw_argument.split(",", 1)
        return left, right

    def _default_tasks(self, split: str) -> list[ToolUseTask]:
        smoke = [
            ToolUseTask(
                task_id="smoke-alpha",
                goal="Use the lookup tool to find the value for alpha, then submit it.",
                optimal_actions=("TOOL: lookup[alpha]",),
                final_answer="4",
            ),
            ToolUseTask(
                task_id="smoke-france",
                goal="Use the lookup tool to find the capital of France, then submit it.",
                optimal_actions=("TOOL: lookup[capital_france]",),
                final_answer="Paris",
            ),
        ]
        easy = [
            ToolUseTask(
                task_id="easy-add",
                goal="Look up alpha, add 3 to it, then submit the result.",
                optimal_actions=("TOOL: lookup[alpha]", "TOOL: add[4,3]"),
                final_answer="7",
            ),
            ToolUseTask(
                task_id="easy-concat",
                goal="Look up word_prime, concatenate it with metal_au, then submit the result.",
                optimal_actions=(
                    "TOOL: lookup[word_prime]",
                    "TOOL: lookup[metal_au]",
                    "TOOL: concat[prime,gold]",
                ),
                final_answer="primegold",
            ),
        ]
        train = [
            ToolUseTask(
                task_id="train-sum-two-lookups",
                goal="Look up alpha and beta, add them, then submit the result.",
                optimal_actions=(
                    "TOOL: lookup[alpha]",
                    "TOOL: lookup[beta]",
                    "TOOL: add[4,9]",
                ),
                final_answer="13",
            ),
            ToolUseTask(
                task_id="train-planet-gold",
                goal="Look up planet_red and metal_au, concatenate them, then submit the result.",
                optimal_actions=(
                    "TOOL: lookup[planet_red]",
                    "TOOL: lookup[metal_au]",
                    "TOOL: concat[Mars,gold]",
                ),
                final_answer="Marsgold",
            ),
        ]
        eval_set = [
            ToolUseTask(
                task_id="eval-gamma",
                goal="Look up gamma, add 5 to it, then submit the result.",
                optimal_actions=("TOOL: lookup[gamma]", "TOOL: add[12,5]"),
                final_answer="17",
            ),
        ]
        if split == "smoke":
            return smoke
        if split == "easy":
            return easy
        return train if split == "train" else eval_set


class ToolUseVerifier(BaseVerifier):
    """Episode-level verifier for the canonical tool-use task."""

    def verify(self, response: str, env_state: dict[str, object]) -> float:
        del response
        if bool(env_state["success"]):
            return 1.0

        total_tool_steps = max(1, int(env_state["total_tool_steps"]))
        completed_fraction = min(1.0, float(env_state["completed_tool_steps"]) / total_tool_steps)
        invalid_penalty = 0.2 * float(env_state["invalid_action_count"])
        final_bonus = 0.1 if bool(env_state["final_submitted"]) else 0.0
        reward = (0.45 * completed_fraction) + final_bonus - invalid_penalty
        return max(0.0, min(0.95, reward))
