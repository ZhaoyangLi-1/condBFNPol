"""Wrapper for D4RL environments (Deep Data-Driven Reinforcement Learning).

This module provides a wrapper for D4RL environments, making them compatible
with the BaseEnv (Gymnasium) interface. D4RL is built on the legacy `gym`
library, so this wrapper handles the API translation (e.g., converting
old-style step returns to new-style terminated/truncated).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

# D4RL requires the legacy 'gym' package.
try:
    import gym as old_gym
    import d4rl  # noqa: F401
except ImportError as e:
    raise ImportError(
        "D4RL is not installed. Please install it via "
        "`pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl` "
        "or follow the official D4RL instructions. Note that D4RL requires `gym<0.26`."
    ) from e

from environments.base import BaseEnv


class D4RLEnv(BaseEnv):
    """Wrapper for D4RL environments to adapt them to the Gymnasium API.

    Attributes:
        observation_space: The observation space.
        action_space: The action space.
    """

    def __init__(
        self, env_name: str, render_mode: str | None = "rgb_array", **kwargs: Any
    ):
        """Initializes the D4RL environment.

        Args:
            env_name: The ID of the D4RL environment (e.g., 'maze2d-umaze-v1').
            render_mode: Rendering mode ('rgb_array', 'human').
            **kwargs: Additional arguments passed to gym.make.
        """
        # D4RL uses the legacy gym.make
        self._env = old_gym.make(env_name, **kwargs)
        self.render_mode = render_mode

        # Expose spaces (Duck typing works between gym/gymnasium spaces mostly)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment.

        Adapts legacy `reset()` (returns obs) to Gymnasium (returns obs, info).
        """
        # D4RL envs often don't accept seed/options in reset(), so we handle seed separately
        # if we were to strictly follow the API, but D4RL usually relies on global seeding
        # or self._env.seed(). We'll just call reset().
        obs = self._env.reset()

        # Return empty info dict as legacy envs don't provide it on reset
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Takes a step in the environment.

        Adapts legacy `step()` (obs, reward, done, info) to Gymnasium
        (obs, reward, terminated, truncated, info).
        """
        # Legacy step
        obs, reward, done, info = self._env.step(action)

        # D4RL/Mujoco logic for distinguishing termination vs truncation
        # 'TimeLimit.truncated' is standard in gym wrappers
        truncated = info.get("TimeLimit.truncated", False)
        terminated = done and not truncated

        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        """Renders the environment."""
        if self.render_mode == "rgb_array":
            return self._env.render(mode="rgb_array")
        elif self.render_mode == "human":
            self._env.render(mode="human")
            return None
        return None

    def close(self):
        """Closes the environment."""
        return self._env.close()

    # --- D4RL Specific Methods ---

    def get_dataset(self, **kwargs) -> Dict[str, np.ndarray]:
        """Retrieves the offline dataset associated with the environment.

        Returns:
            Dictionary containing 'observations', 'actions', 'rewards', etc.
        """
        return self._env.get_dataset(**kwargs)

    def get_normalized_score(self, score: float) -> float:
        """Computes the normalized score (0.0 to 100.0 scale typically)."""
        if hasattr(self._env, "get_normalized_score"):
            return self._env.get_normalized_score(score)
        return score
