from __future__ import annotations

import pytest

import agentrl
from agentrl.agents import (
    AgentAction,
    AgentTaskRecord,
    AgentTrajectory,
    ToolSpec,
    make_tool_agent_task,
)
from agentrl.byod import BYODTask


def test_agent_action_parses_tool_final_and_invalid_actions() -> None:
    tool_action = AgentAction.parse(" TOOL: lookup[alpha] ")
    final_action = AgentAction.parse("FINAL: 4")
    invalid_action = AgentAction.parse("lookup alpha")

    assert tool_action.kind == "tool"
    assert tool_action.name == "lookup"
    assert tool_action.argument == "alpha"
    assert tool_action.raw_text == "TOOL: lookup[alpha]"
    assert final_action.kind == "final"
    assert final_action.argument == "4"
    assert invalid_action.kind == "invalid"
    assert invalid_action.raw_text == "lookup alpha"


def test_make_tool_agent_task_steps_tools_final_answer_and_state() -> None:
    task = make_tool_agent_task(
        records=[
            AgentTaskRecord(
                task_id="alpha",
                goal="Look up alpha, then submit it.",
                metadata={"answer": "4"},
                supervised_trace=("TOOL: lookup[alpha]", "FINAL: 4"),
            )
        ],
        tools=[
            ToolSpec(
                name="lookup",
                description="Return a lookup value.",
                handler=lambda argument, state: {"alpha": "4"}[argument],
            )
        ],
        final_answer_fn=lambda record, state: str(record.metadata["answer"]),
    )

    assert isinstance(task, BYODTask)

    initial_observation = task.environment.reset()
    tool_observation, tool_done = task.environment.step("TOOL: lookup[alpha]")
    final_observation, final_done = task.environment.step("FINAL: 4")
    state = task.environment.state()

    assert "Goal: Look up alpha" in initial_observation
    assert "TOOL: lookup[arg]" in initial_observation
    assert tool_observation == "Tool result: 4"
    assert tool_done is False
    assert final_observation == "episode complete"
    assert final_done is True
    assert state["success"] is True
    assert state["submitted_answer"] == "4"
    assert state["completed_tool_steps"] == 1
    assert state["total_tool_steps"] == 1
    assert state["tool_trace"] == [{"tool": "lookup", "argument": "alpha", "result": "4"}]
    assert isinstance(state["trajectory"], AgentTrajectory)
    assert task.verifier.verify("FINAL: 4", state) == 1.0


def test_tool_agent_task_tracks_invalid_actions_and_shaped_reward() -> None:
    task = make_tool_agent_task(
        records=[
            AgentTaskRecord(
                task_id="alpha",
                goal="Look up alpha, then submit it.",
                metadata={"answer": "4"},
                supervised_trace=("TOOL: lookup[alpha]", "FINAL: 4"),
            )
        ],
        tools=[
            ToolSpec(
                name="lookup",
                description="Return a lookup value.",
                handler=lambda argument, state: {"alpha": "4"}[argument],
            )
        ],
        final_answer_fn=lambda record, state: str(record.metadata["answer"]),
    )

    task.environment.reset()
    observation, done = task.environment.step("TOOL: unknown[alpha]")
    _, final_done = task.environment.step("FINAL: wrong")
    state = task.environment.state()
    reward = task.verifier.verify("FINAL: wrong", state)

    assert observation == "Invalid action. Unknown tool 'unknown'."
    assert done is False
    assert final_done is True
    assert state["invalid_action_count"] == 1
    assert state["success"] is False
    assert 0.0 <= reward < 1.0


def test_tool_agent_task_exports_supervised_trace_samples() -> None:
    task = make_tool_agent_task(
        records=[
            AgentTaskRecord(
                task_id="alpha",
                goal="Look up alpha, then submit it.",
                metadata={"answer": "4"},
                supervised_trace=("TOOL: lookup[alpha]", "FINAL: 4"),
            )
        ],
        tools=[
            ToolSpec(
                name="lookup",
                description="Return a lookup value.",
                handler=lambda argument, state: {"alpha": "4"}[argument],
            )
        ],
        final_answer_fn=lambda record, state: str(record.metadata["answer"]),
    )

    samples = task.supervised_samples(tokenizer="unused")

    assert len(samples) == 1
    prompt, target = samples[0]
    assert "Goal: Look up alpha" in prompt
    assert target == "TOOL: lookup[alpha]\nFINAL: 4"


def test_tool_agent_task_requires_supervised_traces_for_samples() -> None:
    task = make_tool_agent_task(
        records=[AgentTaskRecord(task_id="alpha", goal="Look up alpha.")],
        tools=[
            ToolSpec(
                name="lookup",
                description="Return a lookup value.",
                handler=lambda argument, state: "4",
            )
        ],
        final_answer_fn=lambda record, state: "4",
    )

    with pytest.raises(ValueError, match="No supervised traces found"):
        task.supervised_samples()


def test_package_root_exports_agent_api() -> None:
    assert agentrl.AgentAction is AgentAction
    assert agentrl.AgentTaskRecord is AgentTaskRecord
    assert agentrl.ToolSpec is ToolSpec
    assert agentrl.make_tool_agent_task is make_tool_agent_task
