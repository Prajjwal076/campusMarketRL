"""Compatibility layer for optional OpenEnv base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

ActionT = TypeVar("ActionT")
ObservationT = TypeVar("ObservationT")
ResultT = TypeVar("ResultT")
StateT = TypeVar("StateT")

try:
    from openenv import EnvClient, Environment
except ImportError:
    class Environment(ABC, Generic[ActionT, ObservationT, StateT]):
        """Minimal fallback base class when OpenEnv is unavailable."""

        @abstractmethod
        def reset(self, seed: int | None = None) -> ObservationT:
            """Reset the environment and return the initial observation."""

        @abstractmethod
        def step(
            self,
            action: ActionT,
        ) -> tuple[ObservationT, float, bool, dict[str, str | int | float | bool]]:
            """Advance the environment by one step."""

        @property
        @abstractmethod
        def state(self) -> StateT:
            """Return the current environment state."""

    class EnvClient(ABC, Generic[ActionT, ResultT]):
        """Minimal fallback client base class when OpenEnv is unavailable."""

        @abstractmethod
        def reset(self, seed: int | None = None) -> ResultT:
            """Reset the remote environment and return the initial result."""

        @abstractmethod
        def step(self, action: ActionT) -> ResultT:
            """Send an action to the remote environment."""

__all__ = ["EnvClient", "Environment"]
