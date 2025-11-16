"""AntMaze wrapper using Gymnasium robotics tasks.

This module provides a wrapper for the AntMaze environments from the
Gymnasium-Robotics library.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

try:  # Prefer gymnasium for robotics
    import gymnasium as gym
    import gymnasium_robotics

    gym.register_envs(gymnasium_robotics)
except ImportError as e:  # pragma: no cover - fail fast
    raise ImportError(
        'gymnasium[robotics] is required for AntMaze. Install with `pip install "gymnasium[robotics]"`.'
    ) from e

from environments.base import BaseEnv


class AntMazeEnv(BaseEnv):
    """A thin wrapper around the AntMaze tasks from gymnasium-robotics.

    This class provides a thin wrapper around the AntMaze tasks from the
    gymnasium-robotics library, making them compatible with the BaseEnv
    interface.

    Attributes:
        observation_space: The observation space of the environment.
        action_space: The action space of the environment.
    """

    def __init__(
        self, env_name: str = "AntMaze_UMaze-v5", render_mode: str | None = None
    ):
        """Initializes a new AntMazeEnv environment.

        Args:
            env_name: The name of the AntMaze environment to use. Can be one of
                the AntMaze variants, e.g., "AntMaze_UMaze-v4",
                "AntMaze_Medium-v4", "AntMaze_Large-v4".
            render_mode: The rendering mode to use. Can be "rgb_array" for
                offscreen rendering or "human" for onscreen rendering.
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

    def close(self):
        """Cleans up the environment's resources."""
        return self._env.close()

    def render(self):
        """Renders the environment.

        Returns:
            The rendered environment.
        """
        return self._env.render()
