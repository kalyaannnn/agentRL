"""Chunked prefill utilities for long-prompt generation."""

from __future__ import annotations

import logging
from typing import Any

import torch


LOGGER = logging.getLogger(__name__)


class ChunkedPrefillMixin:
    """Provide chunked prompt prefill for models that support cached decoding."""

    def chunked_prefill(
        self,
        model: Any,
        prompt_ids: torch.Tensor,
        chunk_size: int = 512,
        attention_mask: torch.Tensor | None = None,
    ) -> Any:
        """Run prompt prefill in chunks and return accumulated past KV state.

        Args:
            model: Causal LM module supporting `past_key_values` and `use_cache`.
            prompt_ids: Tensor of shape `[batch, seq]`.
            chunk_size: Tokens per prefill chunk.
            attention_mask: Optional attention mask aligned with `prompt_ids`.

        Returns:
            The model's `past_key_values` after the final chunk.
        """

        if prompt_ids.shape[-1] <= chunk_size:
            outputs = model(
                input_ids=prompt_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
            return outputs.past_key_values

        chunk_count = (prompt_ids.shape[-1] + chunk_size - 1) // chunk_size
        LOGGER.info(
            "Chunked prefill: prompt_len=%s, chunks=%s",
            prompt_ids.shape[-1],
            chunk_count,
        )

        past_key_values = None
        for start in range(0, prompt_ids.shape[-1], chunk_size):
            end = start + chunk_size
            chunk_ids = prompt_ids[:, start:end]
            chunk_mask = attention_mask[:, start:end] if attention_mask is not None else None
            outputs = model(
                input_ids=chunk_ids,
                attention_mask=chunk_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
        return past_key_values
