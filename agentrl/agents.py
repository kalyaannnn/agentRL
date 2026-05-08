"""Public multi-turn agent task helpers for AgentRL."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence

from agentrl.byod import BYODTask
from agentrl.core.base import BaseEnvironment, BaseVerifier


ActionKind = Literal["tool", "final", "invalid"]
ToolHandler = Callable[[str, dict[str, Any]], str]
FinalAnswerFn = Callable[["AgentTaskRecord", dict[str, Any]], str]
RewardFn = Callable[[str, dict[str, Any]], float]

_TOOL_ACTION_RE = re.compile(r"^\s*TOOL:\s*([a-z_][a-z0-9_]*)\[(.*)\]\s*$", re.IGNORECASE)
_FINAL_ACTION_RE = re.compile(r"^\s*FINAL:\s*(.+?)\s*$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class AgentTaskRecord:
    """One multi-turn agent task instance."""

    task_id: str
    goal: str
    metadata: dict[str, Any] = field(default_factory=dict)
    supervised_trace: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class AgentAction:
    """Parsed text action emitted by a model during an agent episode."""

    kind: ActionKind
    raw_text: str
    name: str | None = None
    argument: str | None = None

    @classmethod
    def parse(cls, text: str) -> "AgentAction":
        """Parse the v1 text action grammar."""

        raw_text = text.strip()
        tool_match = _TOOL_ACTION_RE.fullmatch(raw_text)
        if tool_match is not None:
            return cls(
                kind="tool",
                raw_text=raw_text,
                name=tool_match.group(1).lower(),
                argument=tool_match.group(2).strip(),
            )

        final_match = _FINAL_ACTION_RE.fullmatch(raw_text)
        if final_match is not None:
            return cls(
                kind="final",
                raw_text=raw_text,
                argument=final_match.group(1).strip(),
            )

        return cls(kind="invalid", raw_text=raw_text)


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Deterministic in-process tool exposed to a multi-turn agent."""

    name: str
    description: str
    handler: ToolHandler


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Normalized result of executing one tool action."""

    tool_name: str
    argument: str
    result: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True, slots=True)
class AgentTurn:
    """One environment turn in a multi-turn agent trajectory."""

    observation: str
    action: AgentAction
    tool_result: ToolResult | None = None
    reward_diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AgentTrajectory:
    """Complete verifier-facing trajectory summary for one episode."""

    task_id: str
    goal: str
    turns: tuple[AgentTurn, ...]
    final_response: str
    done: bool
    success: bool


class _ToolAgentEnvironment(BaseEnvironment):
    def __init__(
        self,
        *,
        records: Sequence[AgentTaskRecord],
        tools: Sequence[ToolSpec],
        final_answer_fn: FinalAnswerFn,
        seed: int,
    ) -> None:
        if not records:
            raise ValueError("Tool agent tasks require at least one record.")
        if not tools:
            raise ValueError("Tool agent tasks require at least one tool.")
        self._records = list(records)
        self._tools = {tool.name.lower(): tool for tool in tools}
        self._final_answer_fn = final_answer_fn
        self._rng = random.Random(seed)
        self._current_record: AgentTaskRecord | None = None
        self._current_observation = ""
        self._turns: list[AgentTurn] = []
        self._tool_trace: list[dict[str, str]] = []
        self._invalid_action_count = 0
        self._completed_tool_steps = 0
        self._final_submitted = False
        self._submitted_answer = ""
        self._success = False
        self._last_action = ""

    def reset(self) -> str:
        self._current_record = self._rng.choice(self._records)
        self._current_observation = self._render_initial_observation(self._current_record)
        self._turns = []
        self._tool_trace = []
        self._invalid_action_count = 0
        self._completed_tool_steps = 0
        self._final_submitted = False
        self._submitted_answer = ""
        self._success = False
        self._last_action = ""
        return self._current_observation

    def step(self, action: str) -> tuple[str, bool]:
        record = self._require_record()
        parsed = AgentAction.parse(action)
        self._last_action = parsed.raw_text
        observation_before = self._current_observation

        if parsed.kind == "final":
            self._final_submitted = True
            self._submitted_answer = parsed.argument or ""
            expected = self._final_answer_fn(record, self._state_payload(include_trajectory=False))
            self._success = self._submitted_answer == expected
            self._current_observation = "episode complete"
            self._turns.append(
                AgentTurn(
                    observation=observation_before,
                    action=parsed,
                    reward_diagnostics={"success": self._success, "expected_final_answer": expected},
                )
            )
            return self._current_observation, True

        if parsed.kind == "invalid":
            return self._record_invalid_turn(
                observation_before=observation_before,
                action=parsed,
                message=(
                    "Invalid action. Use exactly one action of the form "
                    "`TOOL: name[arg]` or `FINAL: answer`."
                ),
            )

        assert parsed.name is not None
        assert parsed.argument is not None
        tool = self._tools.get(parsed.name)
        if tool is None:
            return self._record_invalid_turn(
                observation_before=observation_before,
                action=parsed,
                message=f"Invalid action. Unknown tool {parsed.name!r}.",
                tool_result=ToolResult(
                    tool_name=parsed.name,
                    argument=parsed.argument,
                    error=f"Unknown tool {parsed.name!r}.",
                ),
            )

        try:
            result = str(tool.handler(parsed.argument, self._state_payload(include_trajectory=False)))
        except Exception as exc:
            return self._record_invalid_turn(
                observation_before=observation_before,
                action=parsed,
                message=f"Invalid action. {exc}",
                tool_result=ToolResult(tool_name=parsed.name, argument=parsed.argument, error=str(exc)),
            )

        self._tool_trace.append({"tool": parsed.name, "argument": parsed.argument, "result": result})
        if parsed.raw_text == self._next_expected_tool_action(record):
            self._completed_tool_steps += 1

        tool_result = ToolResult(tool_name=parsed.name, argument=parsed.argument, result=result)
        self._current_observation = f"Tool result: {result}"
        self._turns.append(
            AgentTurn(
                observation=observation_before,
                action=parsed,
                tool_result=tool_result,
                reward_diagnostics={"completed_tool_steps": self._completed_tool_steps},
            )
        )
        return self._current_observation, False

    def state(self) -> dict[str, Any]:
        self._require_record()
        return self._state_payload(include_trajectory=True)

    def render_generation_prompt(
        self,
        tokenizer: object,
        observations: list[str],
        actions: list[str],
    ) -> str:
        del tokenizer
        parts = [self._system_prompt()]
        for index, observation in enumerate(observations):
            parts.append(f"Observation:\n{observation}\n\n")
            if index < len(actions):
                parts.append(f"Assistant:\n{actions[index]}\n\n")
        parts.append("Assistant:\n")
        return "".join(parts)

    def render_transcript(
        self,
        tokenizer: object,
        observations: list[str],
        actions: list[str],
    ) -> tuple[str, list[tuple[int, int]]]:
        del tokenizer
        parts: list[str] = []
        assistant_spans: list[tuple[int, int]] = []
        cursor = 0

        for index, observation in enumerate(observations):
            prefix = "Observation:\n"
            parts.extend((prefix, observation, "\n\n"))
            cursor += len(prefix) + len(observation) + 2
            if index < len(actions):
                assistant_prefix = "Assistant:\n"
                parts.append(assistant_prefix)
                cursor += len(assistant_prefix)
                start = cursor
                parts.append(actions[index])
                cursor += len(actions[index])
                assistant_spans.append((start, cursor))
                parts.append("\n\n")
                cursor += 2

        return "".join(parts), assistant_spans

    def _record_invalid_turn(
        self,
        *,
        observation_before: str,
        action: AgentAction,
        message: str,
        tool_result: ToolResult | None = None,
    ) -> tuple[str, bool]:
        self._invalid_action_count += 1
        self._current_observation = message
        self._turns.append(
            AgentTurn(
                observation=observation_before,
                action=action,
                tool_result=tool_result,
                reward_diagnostics={"invalid_action_count": self._invalid_action_count},
            )
        )
        return self._current_observation, False

    def _state_payload(self, *, include_trajectory: bool) -> dict[str, Any]:
        record = self._require_record()
        total_tool_steps = len(self._expected_tool_actions(record))
        payload: dict[str, Any] = {
            "task_id": record.task_id,
            "goal": record.goal,
            "metadata": dict(record.metadata),
            "tool_trace": list(self._tool_trace),
            "invalid_action_count": self._invalid_action_count,
            "completed_tool_steps": self._completed_tool_steps,
            "total_tool_steps": total_tool_steps,
            "final_submitted": self._final_submitted,
            "submitted_answer": self._submitted_answer,
            "success": self._success,
            "last_action": self._last_action,
            "turn_count": len(self._turns),
        }
        payload["expected_final_answer"] = self._final_answer_fn(record, payload)
        if include_trajectory:
            payload["trajectory"] = AgentTrajectory(
                task_id=record.task_id,
                goal=record.goal,
                turns=tuple(self._turns),
                final_response=self._last_action,
                done=self._final_submitted,
                success=self._success,
            )
        return payload

    def _render_initial_observation(self, record: AgentTaskRecord) -> str:
        return (
            f"Goal: {record.goal}\n"
            "Allowed actions:\n"
            f"{self._allowed_actions_text()}\n"
            "Return exactly one action."
        )

    def _system_prompt(self) -> str:
        return (
            "You are a tool-using agent.\n"
            "Return exactly one action per turn.\n"
            "Allowed actions:\n"
            f"{self._allowed_actions_text()}\n\n"
        )

    def _allowed_actions_text(self) -> str:
        tool_lines = [f"- TOOL: {name}[arg]" for name in sorted(self._tools)]
        return "\n".join([*tool_lines, "- FINAL: answer"])

    def _next_expected_tool_action(self, record: AgentTaskRecord) -> str | None:
        expected_actions = self._expected_tool_actions(record)
        if self._completed_tool_steps >= len(expected_actions):
            return None
        return expected_actions[self._completed_tool_steps]

    def _expected_tool_actions(self, record: AgentTaskRecord) -> tuple[str, ...]:
        if record.supervised_trace is None:
            return ()
        return tuple(action for action in record.supervised_trace if AgentAction.parse(action).kind == "tool")

    def _require_record(self) -> AgentTaskRecord:
        if self._current_record is None:
            raise RuntimeError("reset() must be called before using a tool agent environment.")
        return self._current_record


class _ToolAgentVerifier(BaseVerifier):
    def __init__(self, reward_fn: RewardFn | None) -> None:
        self._reward_fn = reward_fn

    def verify(self, response: str, env_state: dict[str, Any]) -> float:
        if self._reward_fn is not None:
            return _clamp_reward(self._reward_fn(response, env_state))

        if bool(env_state["success"]):
            return 1.0

        total_tool_steps = max(1, int(env_state["total_tool_steps"]))
        completed_fraction = min(1.0, float(env_state["completed_tool_steps"]) / total_tool_steps)
        invalid_penalty = 0.2 * float(env_state["invalid_action_count"])
        final_bonus = 0.1 if bool(env_state["final_submitted"]) else 0.0
        reward = (0.45 * completed_fraction) + final_bonus - invalid_penalty
        return max(0.0, min(0.95, reward))


def make_tool_agent_task(
    *,
    records: Sequence[AgentTaskRecord],
    tools: Sequence[ToolSpec],
    final_answer_fn: FinalAnswerFn,
    reward_fn: RewardFn | None = None,
    seed: int = 0,
) -> BYODTask:
    """Build a multi-turn tool-agent task from records, tools, and a verifier hook."""

    environment = _ToolAgentEnvironment(
        records=records,
        tools=tools,
        final_answer_fn=final_answer_fn,
        seed=seed,
    )
    verifier = _ToolAgentVerifier(reward_fn=reward_fn)

    def build_samples(tokenizer: Any | None) -> list[tuple[str, str]]:
        del tokenizer
        samples: list[tuple[str, str]] = []
        for record in records:
            if record.supervised_trace is None:
                continue
            samples.append((environment._render_initial_observation(record), "\n".join(record.supervised_trace)))
        if not samples:
            raise ValueError("No supervised traces found for this tool agent task.")
        return samples

    return BYODTask(
        environment=environment,
        verifier=verifier,
        _supervised_samples_fn=build_samples,
    )


def _clamp_reward(value: float) -> float:
    return min(1.0, max(0.0, float(value)))
