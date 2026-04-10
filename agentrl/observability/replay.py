"""Trajectory replay and comparison helpers for AgentRL."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from agentrl.core.rollout import RolloutBatch
from agentrl.memory.buffer import TrajectoryBuffer


@dataclass(slots=True)
class TrajectoryStore:
    """Thin wrapper around serialized trajectory files."""

    output_dir: str
    buffer: TrajectoryBuffer = field(init=False)
    root: Path = field(init=False)

    def __post_init__(self) -> None:
        self.buffer = TrajectoryBuffer(output_dir=self.output_dir)
        self.root = Path(self.output_dir).expanduser()

    def load(self, step: int) -> RolloutBatch:
        """Load one saved rollout batch from disk."""

        return self.buffer.load(step, device="cpu")

    def list_steps(self) -> list[int]:
        """List saved step ids discovered under the output directory."""

        return sorted(int(path.stem.split("_")[-1]) for path in self.root.glob("trajectory_*.pt"))


class ReplayBuffer:
    """Human-readable replay interface over serialized rollout batches."""

    def __init__(self, output_dir: str = "./checkpoints") -> None:
        self.store = TrajectoryStore(output_dir=output_dir)

    def show(self, step: int) -> str:
        """Render all grouped responses for one saved training step."""

        batch = self.store.load(step)
        prompts = batch.metadata.get("prompts", [])
        responses = batch.metadata.get("responses", [])
        lines: list[str] = []

        for batch_index, prompt in enumerate(prompts):
            lines.append(f'=== Step {step} | Prompt: "{prompt}" ===')
            group_rewards = batch.rewards[batch_index].tolist()
            group_advantages = batch.advantages[batch_index].tolist()
            group_responses = responses[batch_index]
            for response_index, response_text in enumerate(group_responses):
                lines.append(
                    f"[Response {response_index + 1}] reward={group_rewards[response_index]:.2f} "
                    f"| advantage={group_advantages[response_index]:+.2f}"
                )
                lines.append(f'  "{response_text}"')
        return "\n".join(lines)

    def filter(self, min_reward: float = 0.8) -> list[RolloutBatch]:
        """Load saved batches whose max reward meets the threshold."""

        return self.store.buffer.filter(min_reward=min_reward)

    def compare(self, step_a: int, step_b: int) -> str:
        """Render a prompt-aligned comparison between two saved steps."""

        batch_a = self.store.load(step_a)
        batch_b = self.store.load(step_b)

        prompts_a = batch_a.metadata.get("prompts", [])
        prompts_b = batch_b.metadata.get("prompts", [])
        responses_a = batch_a.metadata.get("responses", [])
        responses_b = batch_b.metadata.get("responses", [])

        lines: list[str] = []
        max_prompts = max(len(prompts_a), len(prompts_b))
        for index in range(max_prompts):
            prompt_a = prompts_a[index] if index < len(prompts_a) else "<missing>"
            prompt_b = prompts_b[index] if index < len(prompts_b) else "<missing>"
            lines.append(f"=== Compare step {step_a} vs {step_b} | Prompt A: {prompt_a!r} | Prompt B: {prompt_b!r} ===")
            group_a = responses_a[index] if index < len(responses_a) else []
            group_b = responses_b[index] if index < len(responses_b) else []
            max_group = max(len(group_a), len(group_b))
            for response_index in range(max_group):
                left = group_a[response_index] if response_index < len(group_a) else "<missing>"
                right = group_b[response_index] if response_index < len(group_b) else "<missing>"
                lines.append(f"[Response {response_index + 1}] A={left!r} | B={right!r}")
        return "\n".join(lines)
