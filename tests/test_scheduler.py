from __future__ import annotations

from types import SimpleNamespace

import pytest

from agentrl.core.config import GRPOConfig
from agentrl.generation.scheduler import (
    available_vram_bytes,
    compute_safe_chunk_size,
    estimate_kv_cache_bytes,
)


def test_estimate_kv_cache_bytes_matches_formula() -> None:
    estimate = estimate_kv_cache_bytes(
        batch_size=4,
        group_size=8,
        max_new_tokens=256,
        num_layers=24,
        num_heads=16,
        head_dim=128,
        dtype_bytes=2,
    )

    assert estimate == 4 * 8 * 256 * 24 * 16 * 128 * 2 * 2


def test_available_vram_bytes_rejects_invalid_safety_factor() -> None:
    with pytest.raises(ValueError, match="safety_factor"):
        available_vram_bytes(safety_factor=0.0)


def test_compute_safe_chunk_size_uses_group_dimension_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = GRPOConfig(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        batch_size=4,
        group_size=8,
        max_new_tokens=256,
    )
    model_config = SimpleNamespace(
        num_hidden_layers=24,
        num_attention_heads=16,
        hidden_size=2048,
    )

    monkeypatch.setattr(
        "agentrl.generation.scheduler.available_vram_bytes",
        lambda safety_factor=0.85: 4 * 4 * 256 * 24 * 16 * 128 * 2 * 2,
    )

    assert compute_safe_chunk_size(config, model_config) == 4


def test_compute_safe_chunk_size_falls_back_to_one_without_cuda_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = GRPOConfig(model_name="Qwen/Qwen2.5-1.5B-Instruct", group_size=8)
    model_config = SimpleNamespace(
        num_hidden_layers=24,
        num_attention_heads=16,
        hidden_size=2048,
    )

    monkeypatch.setattr(
        "agentrl.generation.scheduler.available_vram_bytes",
        lambda safety_factor=0.85: 0,
    )

    assert compute_safe_chunk_size(config, model_config) == 1
