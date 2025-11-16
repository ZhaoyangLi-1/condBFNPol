"""AntMaze wrapper using Gymnasium robotics tasks."""

from __future__ import annotations

from typing import Any, Dict, Tuple

try:  # Prefer gymnasium for robotics
    import gymnasium as gym
    import gymnasium_robotics
    gym.register_envs(gymnasium_robotics)
except ImportError as e:  # pragma: no cover - fail fast
    raise ImportError(
        "gymnasium[robotics] is required for AntMaze. Install with `pip install \"gymnasium[robotics]\"`."
    ) from e

from environments.base import BaseEnv


class AntMazeEnv(BaseEnv):
    """Thin wrapper around the AntMaze tasks from gymnasium-robotics."""

    def __init__(self, env_name: str = "AntMaze_UMaze-v5", render_mode: str | None = None):
        """
        Args:
            env_name: one of the AntMaze variants, e.g. AntMaze_UMaze-v4, AntMaze_Medium-v4, AntMaze_Large-v4.
            render_mode: gym render mode; use "rgb_array" for offscreen frames or "human" if available.
        """
        # Normalize some common aliases
        candidate_names = [env_name]
        if "_" in env_name and "-" not in env_name:
            candidate_names.append(env_name.replace("_", "-"))
        if not env_name.endswith("-v4") and not env_name.endswith("-v5"):
            candidate_names.append(f"{env_name}-v5")
            candidate_names.append(f"{env_name}-v4")
        if not env_name.replace("_", "-").endswith("-v4") and not env_name.replace("_", "-").endswith("-v5"):
            candidate_names.append(env_name.replace("_", "-") + "-v5")
            candidate_names.append(env_name.replace("_", "-") + "-v4")

        # Log what we have registered, to aid debugging
        registry = gym.envs.registry
        available = [k for k in registry.keys() if "AntMaze" in k]
        if not available:
            raise RuntimeError(
                "No AntMaze environments registered. Ensure gymnasium[robotics] is installed "
                "and gym.register_envs(gymnasium_robotics) has been called."
            )

        last_err = None
        self._env = None
        for name in candidate_names:
            try:
                self._env = gym.make(name, render_mode=render_mode)
                env_name = name
                break
            except Exception as e:  # pragma: no cover - fallback attempts
                last_err = e
                continue
        if self._env is None:
            raise last_err or RuntimeError(
                f"Could not create AntMaze env from {candidate_names}. Available: {available}"
            )

        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def reset(self) -> Tuple[Any, Dict[str, Any]]:
        obs, info = self._env.reset()
        return obs, info

    def step(self, action) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        return self._env.step(action)

    def close(self):
        return self._env.close()

    def render(self):
        return self._env.render()
