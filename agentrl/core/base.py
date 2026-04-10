"""Public task-side contracts for AgentRL.

These two abstract base classes are the only interfaces users implement
directly. Everything else in the framework is internal runtime machinery.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEnvironment(ABC):
    """Minimal text environment contract used by AgentRL rollouts."""

    @abstractmethod
    def reset(self) -> str:
        """Return the initial prompt or observation for a new episode.

        Returns:
            The initial text observation presented to the model.
        """

    @abstractmethod
    def step(self, action: str) -> tuple[str, bool]:
        """Advance the environment using the model's text response.

        Args:
            action: The model-generated text for the current step.

        Returns:
            A tuple of `(next_observation, done)`. For single-turn tasks, the
            environment should typically return `done=True` immediately.
        """

    @abstractmethod
    def state(self) -> dict[str, Any]:
        """Return deterministic verifier-facing environment state.

        Returns:
            Arbitrary structured state needed by the verifier, such as ground
            truth values or intermediate task state.
        """


class BaseVerifier(ABC):
    """Deterministic scalar reward contract for AgentRL."""

    @abstractmethod
    def verify(self, response: str, env_state: dict[str, Any]) -> float:
        """Score a model response against environment state.

        Args:
            response: The final text response produced by the model.
            env_state: Structured environment state from
                :meth:`BaseEnvironment.state`.

        Returns:
            A scalar reward in the closed interval `[0.0, 1.0]`.
        """
