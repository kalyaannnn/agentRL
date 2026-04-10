from __future__ import annotations

from types import SimpleNamespace

import torch

from agentrl.core.base import BaseEnvironment, BaseVerifier
from agentrl.core.config import GRPOConfig
from agentrl.core.rollout import RolloutBatch
from agentrl.core.trainer import GRPOTrainer


class MinimalEnvironment(BaseEnvironment):
    def reset(self) -> str:
        return "start"

    def step(self, action: str) -> tuple[str, bool]:
        del action
        return ("done", True)

    def state(self) -> dict[str, str]:
        return {"expected": "x"}


class MinimalVerifier(BaseVerifier):
    def verify(self, response: str, env_state: dict[str, str]) -> float:
        return 1.0 if response == env_state["expected"] else 0.0


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 0
    eos_token = "<eos>"
    pad_token = "<pad>"


class TrainableLayout:
    def __init__(self) -> None:
        self.model = torch.nn.Linear(1, 1, bias=False)
        self.model.config = SimpleNamespace(use_cache=False)
        self.logit_scale = torch.nn.Parameter(torch.tensor(0.0))
        self.ref_bias = 0.25

    def trainable_parameters(self):
        yield self.logit_scale

    def policy_forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        del attention_mask
        batch, seq = input_ids.shape
        vocab = 4
        logits = torch.zeros((batch, seq, vocab), dtype=torch.float32)
        logits[:, :, 1] = self.logit_scale
        return logits

    def reference_forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        del attention_mask
        batch, seq = input_ids.shape
        vocab = 4
        logits = torch.zeros((batch, seq, vocab), dtype=torch.float32)
        logits[:, :, 1] = self.ref_bias
        return logits


class StaticRollout:
    def __init__(self, batch: RolloutBatch) -> None:
        self.batch = batch

    def collect(self) -> RolloutBatch:
        return self.batch


class ClosingLogger:
    def __init__(self) -> None:
        self.closed = False
        self.rows = []

    def log(self, step: int, metrics: dict[str, float]) -> str:
        self.rows.append((step, metrics))
        return "logged"

    def close(self) -> None:
        self.closed = True


def test_trainer_step_updates_trainable_parameter() -> None:
    config = GRPOConfig(
        model_name="fake/model",
        batch_size=1,
        group_size=2,
        steps=1,
        max_new_tokens=4,
    )
    batch = RolloutBatch(
        input_ids=torch.tensor([[[0, 1, 1], [0, 1, 2]]], dtype=torch.long),
        attention_mask=torch.tensor([[[1, 1, 1], [1, 1, 1]]], dtype=torch.long),
        action_mask=torch.tensor([[[0, 1, 1], [0, 1, 1]]], dtype=torch.bool),
        policy_logprobs=torch.zeros((1, 2, 3), dtype=torch.float32),
        ref_logprobs=torch.zeros((1, 2, 3), dtype=torch.float32),
        rewards=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        advantages=torch.tensor([[1.0, -1.0]], dtype=torch.float32),
        metadata={"unique_response_ratio": 1.0, "responses": [["x", "y"]]},
    )
    layout = TrainableLayout()
    trainer = GRPOTrainer(
        config=config,
        environment=MinimalEnvironment(),
        verifier=MinimalVerifier(),
        tokenizer=DummyTokenizer(),
        layout=layout,
        rollout_orchestrator=StaticRollout(batch),
    )

    before = layout.logit_scale.detach().clone()
    loss, metrics = trainer.step(batch)
    after = layout.logit_scale.detach().clone()

    assert loss.item() != 0.0
    assert after.item() != before.item()
    assert metrics["mean_reward"] == 0.5
    assert metrics["unique_response_ratio"] == 1.0


def test_trainer_closes_metrics_logger_after_train() -> None:
    config = GRPOConfig(
        model_name="fake/model",
        batch_size=1,
        group_size=2,
        steps=1,
        use_continuous_batching=False,
    )
    batch = RolloutBatch(
        input_ids=torch.tensor([[[0, 1, 1], [0, 1, 2]]], dtype=torch.long),
        attention_mask=torch.tensor([[[1, 1, 1], [1, 1, 1]]], dtype=torch.long),
        action_mask=torch.tensor([[[0, 1, 1], [0, 1, 1]]], dtype=torch.bool),
        policy_logprobs=torch.zeros((1, 2, 3), dtype=torch.float32),
        ref_logprobs=torch.zeros((1, 2, 3), dtype=torch.float32),
        rewards=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        advantages=torch.tensor([[1.0, -1.0]], dtype=torch.float32),
        metadata={"unique_response_ratio": 1.0, "responses": [["x", "y"]]},
    )
    logger = ClosingLogger()
    trainer = GRPOTrainer(
        config=config,
        environment=MinimalEnvironment(),
        verifier=MinimalVerifier(),
        tokenizer=DummyTokenizer(),
        layout=TrainableLayout(),
        rollout_orchestrator=StaticRollout(batch),
        metrics_logger=logger,
    )

    trainer.train()

    assert logger.closed is True
    assert len(logger.rows) == 1
