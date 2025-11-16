"""Wrapper for the PushT environment from gym_pusht.

This module provides a wrapper for the PushT environment from the gym_pusht
library, making it compatible with the BaseEnv interface.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

try:
    import gymnasium as gym
    import gym_pusht  # noqa: F401
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "gymnasium and gym_pusht are required for PushT. Install with `pip install gymnasium gym-pusht`."
    ) from e

from environments.base import BaseEnv


class PushTEnv(BaseEnv):
    """A thin wrapper around the PushT-v0 environment from gym_pusht.

    This class provides a thin wrapper around the PushT-v0 environment from
    the gym_pusht library, making it compatible with the BaseEnv interface.

    Attributes:
        observation_space: The observation space of the environment.
        action_space: The action space of the environment.
    """

    def __init__(
        self, env_name: str = "gym_pusht/PushT-v0", render_mode: str | None = None
    ):
        """Initializes a new PushTEnv environment.

        Args:
            env_name: The name of the PushT environment to use.
            render_mode: The rendering mode to use.
        """
        self._env = gym.make(env_name, render_mode=render_mode)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def reset(self) -> Tuple[Any, Dict[str, Any]]:
        """Resets the environment to its initial state.

        Returns:
            A tuple containing the initial observation and an info dict.
        """
        obs, info = self._env.reset()
        return obs, info

    def step(self, action) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Takes a step in the environment.

        Args:
            action: The action to take.

        Returns:
            A tuple containing the next observation, the reward, a boolean
            indicating whether the episode has terminated, a boolean indicating
            whether the episode has been truncated, and an info dict.
        """
        return self._env.step(action)

    def render(self):
        """Renders the environment.

        Returns:
            The rendered environment.
        """
        return self._env.render()

    def close(self):
        """Cleans up the environment's resources."""
        return self._env.close()
