"""Opportunistic CUDA graph decode replay for cache-aware generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch


DecodeCallable = Callable[[torch.Tensor, torch.Tensor, Any], Any]
CaptureFactory = Callable[[Any, torch.Tensor, torch.Tensor, Any], DecodeCallable]


@dataclass(frozen=True, slots=True)
class DecodeGraphKey:
    """Shape bucket key for a captured decode graph."""

    batch_size_bucket: int
    seq_len_bucket: int
    input_shape: tuple[int, ...]
    attention_shape: tuple[int, ...]
    cache_signature: tuple[tuple[tuple[int, ...], ...], ...]


class CUDAGraphDecodeRunner:
    """Route decode forwards through captured CUDA graphs when safe.

    The runner is intentionally conservative. Unsupported devices, torch.compile,
    bucket misses, and cache shapes that cannot be represented safely fall back to
    eager execution with the same model call.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        device: torch.device,
        torch_compile: bool = False,
        batch_size_buckets: tuple[int, ...] = (1, 2, 4, 8),
        seq_len_bucket_size: int = 64,
        max_seq_len_bucket: int = 4096,
        capture_factory: CaptureFactory | None = None,
        allow_non_cuda_graphs_for_tests: bool = False,
    ) -> None:
        self.enabled = bool(enabled)
        self.device = device
        self.torch_compile = bool(torch_compile)
        self.batch_size_buckets = tuple(sorted(batch_size_buckets))
        self.seq_len_bucket_size = int(seq_len_bucket_size)
        self.max_seq_len_bucket = int(max_seq_len_bucket)
        self.capture_factory = capture_factory
        self.allow_non_cuda_graphs_for_tests = allow_non_cuda_graphs_for_tests
        self._graphs: dict[DecodeGraphKey, DecodeCallable] = {}
        self.graph_replays = 0
        self.eager_fallbacks = 0
        self.capture_count = 0
        self.last_used_graph = False
        self.last_fallback_reason = "none"

    @property
    def requested(self) -> bool:
        """Return whether graph decode was requested by config."""

        return self.enabled

    def forward(
        self,
        model: Any,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Any,
    ) -> Any:
        """Run one decode forward through graph replay or eager fallback."""

        self.last_used_graph = False
        self.last_fallback_reason = "none"

        reason = self._ineligible_reason(input_ids=input_ids)
        if reason is not None:
            return self._eager(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                reason=reason,
            )

        key = self._graph_key(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
        if key is None:
            return self._eager(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                reason="bucket_miss",
            )

        graph = self._graphs.get(key)
        if graph is None:
            graph = self._capture(model, input_ids, attention_mask, past_key_values)
            if graph is None:
                return self._eager(
                    model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    reason="capture_unavailable",
                )
            self._graphs[key] = graph
            self.capture_count += 1

        self.graph_replays += 1
        self.last_used_graph = True
        return graph(input_ids, attention_mask, past_key_values)

    def _ineligible_reason(self, *, input_ids: torch.Tensor) -> str | None:
        if not self.enabled:
            return "disabled"
        if self.torch_compile:
            return "torch_compile"
        if self.device.type != "cuda" and not self.allow_non_cuda_graphs_for_tests:
            return "non_cuda"
        if self.device.type == "cuda" and not input_ids.is_cuda and not self.allow_non_cuda_graphs_for_tests:
            return "non_cuda_tensor"
        return None

    def _graph_key(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Any,
    ) -> DecodeGraphKey | None:
        batch_bucket = self._bucket(input_ids.shape[0], self.batch_size_buckets)
        seq_bucket = self._seq_bucket(attention_mask.shape[-1])
        cache_signature = self._cache_signature(past_key_values)
        if batch_bucket is None or seq_bucket is None or cache_signature is None:
            return None
        return DecodeGraphKey(
            batch_size_bucket=batch_bucket,
            seq_len_bucket=seq_bucket,
            input_shape=tuple(input_ids.shape),
            attention_shape=tuple(attention_mask.shape),
            cache_signature=cache_signature,
        )

    @staticmethod
    def _bucket(value: int, buckets: tuple[int, ...]) -> int | None:
        for bucket in buckets:
            if value <= bucket:
                return bucket
        return None

    def _seq_bucket(self, seq_len: int) -> int | None:
        if seq_len <= 0 or self.seq_len_bucket_size <= 0:
            return None
        bucket = ((int(seq_len) + self.seq_len_bucket_size - 1) // self.seq_len_bucket_size) * self.seq_len_bucket_size
        if bucket > self.max_seq_len_bucket:
            return None
        return bucket

    @staticmethod
    def _cache_signature(past_key_values: Any) -> tuple[tuple[tuple[int, ...], ...], ...] | None:
        if not isinstance(past_key_values, tuple):
            return None
        signature = []
        for layer in past_key_values:
            if not isinstance(layer, tuple):
                return None
            layer_shapes = []
            for tensor in layer:
                if not isinstance(tensor, torch.Tensor):
                    return None
                layer_shapes.append(tuple(tensor.shape))
            signature.append(tuple(layer_shapes))
        return tuple(signature)

    def _capture(
        self,
        model: Any,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Any,
    ) -> DecodeCallable | None:
        if self.capture_factory is not None:
            return self.capture_factory(model, input_ids, attention_mask, past_key_values)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return None
        # Real CUDA graph capture is deliberately opt-in until cache padding and
        # static output ownership are implemented for all supported cache types.
        return None

    def _eager(
        self,
        model: Any,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Any,
        reason: str,
    ) -> Any:
        self.eager_fallbacks += 1
        self.last_fallback_reason = reason
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
