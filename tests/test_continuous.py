from __future__ import annotations

from types import SimpleNamespace

import torch

from agentrl.core.base import BaseEnvironment, BaseVerifier
from agentrl.core.config import GRPOConfig
from agentrl.generation.continuous import ContinuousBatchingOrchestrator


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
        return "".join(chr(int(value)) for value in token_ids.tolist() if value not in {0, 3})


class SingleTurnEnvironment(BaseEnvironment):
    def __init__(self, label: str = "task") -> None:
        self.label = label

    def reset(self) -> str:
        return self.label

    def step(self, action: str) -> tuple[str, bool]:
        del action
        return ("done", True)

    def state(self) -> dict[str, str]:
        return {"expected": "ab"}


class PrefixVerifier(BaseVerifier):
    def verify(self, response: str, env_state: dict[str, str]) -> float:
        return 1.0 if response.startswith(env_state["expected"]) else 0.0


class StepModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(use_cache=False)
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def generate_step(self, active_sequences: list[torch.Tensor], active_indices: list[int]) -> list[int]:
        del active_indices
        outputs = []
        for sequence in active_sequences:
            text = "".join(chr(int(value)) for value in sequence.tolist() if value != 0)
            generated = text.split("Assistant:\n")[-1]
            if generated == "":
                outputs.append(ord("a"))
            elif generated == "a":
                outputs.append(ord("b"))
            else:
                outputs.append(3)
        return outputs


class Layout:
    def __init__(self) -> None:
        self.model = StepModel()

    def policy_forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        del attention_mask
        batch, seq = input_ids.shape
        return torch.zeros((batch, seq, 256), dtype=torch.float32)

    def reference_forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        del attention_mask
        batch, seq = input_ids.shape
        return torch.zeros((batch, seq, 256), dtype=torch.float32)


def test_continuous_batching_collects_and_reports_padding_ratio() -> None:
    config = GRPOConfig(
        model_name="fake/model",
        batch_size=1,
        group_size=2,
        max_new_tokens=4,
    )
    orchestrator = ContinuousBatchingOrchestrator(
        config=config,
        environment=SingleTurnEnvironment(),
        verifier=PrefixVerifier(),
        tokenizer=CharTokenizer(),
        layout=Layout(),
        device=torch.device("cpu"),
    )

    batch = orchestrator.collect()

    assert batch.rewards.tolist() == [[1.0, 1.0]]
    assert "padding_ratio" in batch.metadata
    assert 0.0 <= batch.metadata["padding_ratio"] <= 1.0
