"""Wrapper for PushT environment from gym_pusht."""

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
    """Thin wrapper around gym_pusht.PushT-v0."""

    def __init__(self, env_name: str = "gym_pusht/PushT-v0", render_mode: str | None = None):
        self._env = gym.make(env_name, render_mode=render_mode)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def reset(self) -> Tuple[Any, Dict[str, Any]]:
        obs, info = self._env.reset()
        return obs, info

    def step(self, action) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        return self._env.step(action)

    def render(self):
        return self._env.render()

    def close(self):
        return self._env.close()
