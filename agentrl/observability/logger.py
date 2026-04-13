"""Metrics logging backends for AgentRL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class MetricsLogger:
    """Log scalar metrics to stdout, JSONL, and optionally Weights & Biases."""

    def __init__(
        self,
        output_dir: str,
        jsonl_name: str = "metrics.jsonl",
        log_to_wandb: bool = False,
        wandb_project: str | None = None,
        wandb_run_name: str | None = None,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_dir / jsonl_name
        self._wandb_run = None

        if log_to_wandb:
            try:
                import wandb
            except ImportError as exc:
                raise ImportError("log_to_wandb=True requires the `wandb` package.") from exc
            self._wandb_run = wandb.init(project=wandb_project, name=wandb_run_name, reinit=True)

    def log(self, step: int, metrics: dict[str, Any]) -> str:
        """Persist one metrics row and return the stdout rendering.

        Args:
            step: Training step.
            metrics: Scalar metrics or histogram-like values.

        Returns:
            Human-readable one-line rendering written to stdout.
        """

        serializable = {"step": step, **self._normalize(metrics)}
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(serializable, sort_keys=True) + "\n")

        if self._wandb_run is not None:
            self._wandb_run.log(serializable, step=step)

        rendered = self._format_stdout(serializable)
        print(rendered)
        return rendered

    def close(self) -> None:
        """Close any active logging backends."""

        if self._wandb_run is not None:
            self._wandb_run.finish()
            self._wandb_run = None

    def _normalize(self, metrics: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float, str, bool)):
                normalized[key] = value
            elif isinstance(value, list):
                normalized[key] = value
            else:
                normalized[key] = str(value)
        return normalized

    def _format_stdout(self, row: dict[str, Any]) -> str:
        ordered_keys = [
            "step",
            "mean_reward",
            "reward_std",
            "policy_loss",
            "kl_loss",
            "total_loss",
            "generation_time_ms",
            "prefill_time_ms",
            "decode_time_ms",
            "logprob_time_ms",
            "training_time_ms",
            "generation_peak_vram_mb",
            "rollout_peak_vram_mb",
            "generation_runtime_headroom_mb",
            "rollout_runtime_headroom_mb",
            "peak_vram_mb",
            "tokens_per_second",
            "prefill_tokens_per_second",
            "decode_tokens_per_second",
            "padding_ratio",
            "padding_waste_tokens",
            "cache_reuse_effectiveness",
            "unique_response_ratio",
        ]
        segments: list[str] = []
        for key in ordered_keys:
            if key not in row:
                continue
            value = row[key]
            if isinstance(value, float):
                segments.append(f"{key}={value:.4f}")
            else:
                segments.append(f"{key}={value}")
        for key, value in row.items():
            if key in ordered_keys:
                continue
            segments.append(f"{key}={value}")
        return " | ".join(segments)
