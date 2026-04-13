"""Per-phase timing and VRAM profiling for AgentRL."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch


@dataclass(slots=True)
class _PhaseStats:
    name: str
    time_ms: float
    peak_vram_mb: float
    runtime_headroom_mb: float


class SystemsProfiler:
    """Capture per-phase wall-clock and VRAM usage."""

    def __init__(self) -> None:
        self._phases: list[_PhaseStats] = []

    def __enter__(self) -> "SystemsProfiler":
        self._phases.clear()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        return None

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Record one named phase.

        Args:
            name: Phase label.
        """

        start = time.perf_counter()
        base_peak_mb = self._current_peak_vram_mb()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            peak_vram_mb = self._current_peak_vram_mb()
            self._phases.append(
                _PhaseStats(
                    name=name,
                    time_ms=elapsed_ms,
                    peak_vram_mb=max(peak_vram_mb, base_peak_mb),
                    runtime_headroom_mb=self._runtime_headroom_mb(max(peak_vram_mb, base_peak_mb)),
                )
            )

    def metrics(self) -> dict[str, float]:
        """Return scalar metrics for the recorded phases."""

        total_ms = sum(phase.time_ms for phase in self._phases)
        metrics: dict[str, float] = {}
        for phase in self._phases:
            metrics[f"{phase.name}_time_ms"] = phase.time_ms
            metrics[f"{phase.name}_peak_vram_mb"] = phase.peak_vram_mb
            metrics[f"{phase.name}_runtime_headroom_mb"] = phase.runtime_headroom_mb
        metrics["peak_vram_mb"] = max((phase.peak_vram_mb for phase in self._phases), default=0.0)
        metrics["total_step_time_ms"] = total_ms
        generation_ms = metrics.get("generation_time_ms", 0.0)
        metrics["phase_a_fraction"] = (generation_ms / total_ms) if total_ms > 0 else 0.0
        return metrics

    def report(self) -> str:
        """Render a simple per-phase timing table."""

        total_ms = sum(phase.time_ms for phase in self._phases)
        lines = [
            "Phase          Time (ms)    VRAM Peak (MB)    % of step",
            "--------------------------------------------------------",
        ]
        for phase in self._phases:
            fraction = (phase.time_ms / total_ms * 100.0) if total_ms > 0 else 0.0
            lines.append(
                f"{phase.name:<12} {phase.time_ms:>9.0f} {phase.peak_vram_mb:>17.0f} {fraction:>11.0f}%"
            )
        lines.append("--------------------------------------------------------")
        peak_vram = max((phase.peak_vram_mb for phase in self._phases), default=0.0)
        lines.append(f"{'total':<12} {total_ms:>9.0f} {peak_vram:>17.0f} {100 if total_ms > 0 else 0:>11}%")
        return "\n".join(lines)

    def _current_peak_vram_mb(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated() / (1024 * 1024)

    def _runtime_headroom_mb(self, peak_vram_mb: float) -> float:
        if not torch.cuda.is_available():
            return 0.0
        total_mb = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024 * 1024)
        return max(total_mb - peak_vram_mb, 0.0)
