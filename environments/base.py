"""Base environment abstraction for robotics tasks, gym-compatible."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

try:  # Prefer gymnasium if available, else fallback to gym
    import gymnasium as gym
except ImportError:  # pragma: no cover - dependency optional
    import gym  # type: ignore


class BaseEnv(gym.Env, abc.ABC):
    """A minimal Gym/Gymnasium-compatible interface for robotics environments."""

    observation_space: Any
    action_space: Any

    @abc.abstractmethod
    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment and return (observation, info)."""
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advance the environment by one step given an action."""
        raise NotImplementedError

    def close(self):
        """Hook for cleaning up simulators/resources."""
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
