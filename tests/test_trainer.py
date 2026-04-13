from __future__ import annotations

from pathlib import Path
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
        self.saved_adapters = []

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

    def save_adapter(self, path):
        output = Path(path)
        output.mkdir(parents=True, exist_ok=True)
        self.saved_adapters.append(output)
        return output

    def vram_report(self) -> dict[str, float]:
        return {"base_mb": 10.0, "adapter_mb": 1.5, "total_mb": 11.5}


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
    assert "mean_token_kl" in metrics
    assert metrics["learning_rate"] == config.lr
    assert metrics["beta"] == config.beta
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
    logged_metrics = logger.rows[0][1]
    assert "prefill_time_ms" in logged_metrics
    assert "decode_time_ms" in logged_metrics
    assert "cache_reuse_effectiveness" in logged_metrics
    assert "rollout_peak_vram_mb" in logged_metrics


def test_trainer_saves_periodic_and_final_adapters(tmp_path) -> None:
    config = GRPOConfig(
        model_name="fake/model",
        batch_size=1,
        group_size=2,
        steps=2,
        save_every=1,
        output_dir=str(tmp_path),
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
    layout = TrainableLayout()
    trainer = GRPOTrainer(
        config=config,
        environment=MinimalEnvironment(),
        verifier=MinimalVerifier(),
        tokenizer=DummyTokenizer(),
        layout=layout,
        rollout_orchestrator=StaticRollout(batch),
        metrics_logger=ClosingLogger(),
    )

    trainer.train()

    saved_names = [path.name for path in layout.saved_adapters]
    assert saved_names == [
        "checkpoint_000001",
        "checkpoint_000002",
        "checkpoint_final",
    ]


def test_trainer_exposes_startup_vram_report() -> None:
    config = GRPOConfig(
        model_name="fake/model",
        batch_size=1,
        group_size=2,
        steps=1,
        device="cpu",
    )
    trainer = GRPOTrainer(
        config=config,
        environment=MinimalEnvironment(),
        verifier=MinimalVerifier(),
        tokenizer=DummyTokenizer(),
        layout=TrainableLayout(),
        rollout_orchestrator=StaticRollout(
            RolloutBatch(
                input_ids=torch.tensor([[[0, 1, 1], [0, 1, 2]]], dtype=torch.long),
                attention_mask=torch.tensor([[[1, 1, 1], [1, 1, 1]]], dtype=torch.long),
                action_mask=torch.tensor([[[0, 1, 1], [0, 1, 1]]], dtype=torch.bool),
                policy_logprobs=torch.zeros((1, 2, 3), dtype=torch.float32),
                ref_logprobs=torch.zeros((1, 2, 3), dtype=torch.float32),
                rewards=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
                advantages=torch.tensor([[1.0, -1.0]], dtype=torch.float32),
                metadata={"unique_response_ratio": 1.0, "responses": [["x", "y"]]},
            )
        ),
    )

    assert trainer.startup_report["device"] == "cpu"
    assert trainer.startup_report["parameter_total_mb"] == 11.5
    assert trainer.startup_report["runtime_headroom_mb"] is None


def test_trainer_creates_profile_trace(tmp_path) -> None:
    config = GRPOConfig(
        model_name="fake/model",
        batch_size=1,
        group_size=2,
        steps=1,
        use_continuous_batching=False,
        profile_steps=1,
        profile_dir=str(tmp_path / "profiles"),
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

    trace_path = Path(logger.rows[0][1]["profile_trace_path"])
    assert trace_path.exists()
    assert trace_path.stat().st_size > 0


def test_trainer_logs_scheduler_and_adaptive_beta(tmp_path) -> None:
    config = GRPOConfig(
        model_name="fake/model",
        batch_size=1,
        group_size=2,
        steps=2,
        use_continuous_batching=False,
        lr=1e-3,
        lr_scheduler="cosine",
        min_lr_ratio=0.5,
        use_adaptive_kl=True,
        kl_target=1e-6,
        kl_beta_multiplier=2.0,
        max_beta=1.0,
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

    first_metrics = logger.rows[0][1]
    second_metrics = logger.rows[1][1]
    assert first_metrics["learning_rate"] == config.lr
    assert second_metrics["learning_rate"] < config.lr
    assert first_metrics["beta"] > config.beta
    assert second_metrics["beta"] >= first_metrics["beta"]
    assert "mean_token_kl" in second_metrics


def test_trainer_rejects_unimplemented_async_and_vllm_flags() -> None:
    base_kwargs = {
        "environment": MinimalEnvironment(),
        "verifier": MinimalVerifier(),
        "tokenizer": DummyTokenizer(),
        "layout": TrainableLayout(),
        "rollout_orchestrator": StaticRollout(
            RolloutBatch(
                input_ids=torch.tensor([[[0, 1, 1], [0, 1, 2]]], dtype=torch.long),
                attention_mask=torch.tensor([[[1, 1, 1], [1, 1, 1]]], dtype=torch.long),
                action_mask=torch.tensor([[[0, 1, 1], [0, 1, 1]]], dtype=torch.bool),
                policy_logprobs=torch.zeros((1, 2, 3), dtype=torch.float32),
                ref_logprobs=torch.zeros((1, 2, 3), dtype=torch.float32),
                rewards=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
                advantages=torch.tensor([[1.0, -1.0]], dtype=torch.float32),
                metadata={"unique_response_ratio": 1.0, "responses": [["x", "y"]]},
            )
        ),
    }

    for flag_name in ("use_async_rollout_workers", "use_async_trajectory_copy", "experimental_vllm_rollout"):
        config = GRPOConfig(model_name="fake/model", **{flag_name: True})
        try:
            GRPOTrainer(config=config, **base_kwargs)
        except NotImplementedError:
            continue
        raise AssertionError(f"{flag_name} should raise NotImplementedError when enabled.")
