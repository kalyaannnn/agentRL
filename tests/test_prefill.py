from __future__ import annotations

from types import SimpleNamespace

import torch

from agentrl.generation.prefill import ChunkedPrefillMixin


class RecorderModel:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int | None]] = []

    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: object | None = None,
        use_cache: bool = True,
    ) -> SimpleNamespace:
        del attention_mask, use_cache
        marker = None if past_key_values is None else int(past_key_values)
        self.calls.append((input_ids.shape[-1], marker))
        next_marker = len(self.calls)
        return SimpleNamespace(past_key_values=next_marker)


class PrefillHarness(ChunkedPrefillMixin):
    pass


def test_chunked_prefill_splits_long_prompts_and_threads_past_kv() -> None:
    harness = PrefillHarness()
    model = RecorderModel()
    prompt_ids = torch.arange(0, 10, dtype=torch.long).unsqueeze(0)

    past = harness.chunked_prefill(model=model, prompt_ids=prompt_ids, chunk_size=4)

    assert model.calls == [(4, None), (4, 1), (2, 2)]
    assert past == 3
