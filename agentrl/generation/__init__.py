"""Generation-time scheduling and rollout utilities for AgentRL."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "ChunkedPrefillMixin",
    "ContinuousBatchingOrchestrator",
    "CUDAGraphDecodeRunner",
    "PrefixCache",
    "available_vram_bytes",
    "compute_safe_chunk_size",
    "estimate_kv_cache_bytes",
]


def __getattr__(name: str) -> Any:
    if name in {"available_vram_bytes", "compute_safe_chunk_size", "estimate_kv_cache_bytes"}:
        module = import_module("agentrl.generation.scheduler")
        return getattr(module, name)
    if name == "ChunkedPrefillMixin":
        module = import_module("agentrl.generation.prefill")
        return getattr(module, name)
    if name == "ContinuousBatchingOrchestrator":
        module = import_module("agentrl.generation.continuous")
        return getattr(module, name)
    if name == "CUDAGraphDecodeRunner":
        module = import_module("agentrl.generation.cuda_graph_decode")
        return getattr(module, name)
    if name == "PrefixCache":
        module = import_module("agentrl.generation.prefix_cache")
        return getattr(module, name)
    raise AttributeError(f"module 'agentrl.generation' has no attribute {name!r}")
