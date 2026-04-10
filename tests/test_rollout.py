from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from agentrl.core.base import BaseEnvironment, BaseVerifier
from agentrl.core.config import GRPOConfig
from agentrl.core.rollout import RolloutOrchestrator


class CharTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 3

    def __call__(
        self,
        text: str,
        return_tensors: str = "pt",
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict[str, torch.Tensor]:
        del return_tensors, add_special_tokens
        input_ids = torch.tensor([[ord(character) for character in text]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        encoded: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if return_offsets_mapping:
            encoded["offset_mapping"] = torch.tensor(
                [[(index, index + 1) for index, _ in enumerate(text)]],
                dtype=torch.long,
            )
        return encoded

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        values = token_ids.tolist()
        return "".join(chr(value) for value in values if value != self.pad_token_id)


class TwoTurnEnvironment(BaseEnvironment):
    def __init__(self) -> None:
        self.turn = 0

    def reset(self) -> str:
        self.turn = 0
        return "start"

    def step(self, action: str) -> tuple[str, bool]:
        if self.turn == 0:
            self.turn += 1
            return ("finish", False)
        self.turn += 1
        return ("done", True)

    def state(self) -> dict[str, str]:
        return {"expected": "done"}


class FinalAnswerVerifier(BaseVerifier):
    def verify(self, response: str, env_state: dict[str, str]) -> float:
        return 1.0 if response == env_state["expected"] else 0.0


class FakeGenerationModel(torch.nn.Module):
    def __init__(self, outputs: list[str]) -> None:
        super().__init__()
        self.outputs = list(outputs)
        self.config = SimpleNamespace(use_cache=False)
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
        pad_token_id: int,
        eos_token_id: int | None,
    ) -> torch.Tensor:
        del attention_mask, max_new_tokens, temperature, do_sample, pad_token_id, eos_token_id
        text = self.outputs.pop(0)
        response = torch.tensor([[ord(character) for character in text]], dtype=torch.long, device=input_ids.device)
        return torch.cat((input_ids, response), dim=-1)


class FakeLayout:
    def __init__(self, outputs: list[str]) -> None:
        self.model = FakeGenerationModel(outputs)

    def policy_forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        del attention_mask
        batch, seq = input_ids.shape
        vocab = 256
        logits = torch.zeros((batch, seq, vocab), dtype=torch.float32)
        logits[:, :, 65] = 1.0
        return logits

    def reference_forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        del attention_mask
        batch, seq = input_ids.shape
        vocab = 256
        logits = torch.zeros((batch, seq, vocab), dtype=torch.float32)
        logits[:, :, 66] = 1.0
        return logits


def test_rollout_collects_multi_turn_grouped_batch() -> None:
    config = GRPOConfig(
        model_name="fake/model",
        batch_size=1,
        group_size=2,
        max_new_tokens=8,
        max_episode_steps=3,
    )
    layout = FakeLayout(outputs=["plan", "done", "plan", "bad"])
    orchestrator = RolloutOrchestrator(
        config=config,
        environment=TwoTurnEnvironment(),
        verifier=FinalAnswerVerifier(),
        tokenizer=CharTokenizer(),
        layout=layout,
        device=torch.device("cpu"),
    )

    batch = orchestrator.collect()

    assert batch.input_ids.shape[0:2] == (1, 2)
    assert batch.input_ids.shape[-1] % config.pad_to_multiple_of == 0
    assert batch.attention_mask.shape == batch.input_ids.shape
    assert batch.action_mask.shape == batch.input_ids.shape
    assert batch.rewards.tolist() == [[1.0, 0.0]]
    assert batch.advantages[0, 0].item() == pytest.approx(1.0, rel=1e-5)
    assert batch.advantages[0, 1].item() == pytest.approx(-1.0, rel=1e-5)
    assert batch.action_mask.sum().item() > 0
    assert batch.metadata["responses"] == [["done", "bad"]]


def test_rollout_warns_when_episode_hits_turn_cap(caplog: pytest.LogCaptureFixture) -> None:
    class NeverDoneEnvironment(BaseEnvironment):
        def __init__(self) -> None:
            self.counter = 0

        def reset(self) -> str:
            self.counter = 0
            return "start"

        def step(self, action: str) -> tuple[str, bool]:
            del action
            self.counter += 1
            return (f"obs-{self.counter}", False)

        def state(self) -> dict[str, str]:
            return {"expected": "never"}

    caplog.set_level("WARNING")
    config = GRPOConfig(
        model_name="fake/model",
        batch_size=1,
        group_size=2,
        max_episode_steps=1,
    )
    layout = FakeLayout(outputs=["x", "y"])
    orchestrator = RolloutOrchestrator(
        config=config,
        environment=NeverDoneEnvironment(),
        verifier=FinalAnswerVerifier(),
        tokenizer=CharTokenizer(),
        layout=layout,
        device=torch.device("cpu"),
    )

    batch = orchestrator.collect()

    assert "max_episode_steps=1" in caplog.text
    assert batch.metadata["responses"] == [["x", "y"]]
