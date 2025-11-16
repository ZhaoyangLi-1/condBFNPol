"""A tiny grid-world maze environment for policy learning (Gym-friendly)."""

from __future__ import annotations

import numpy as np
from typing import Iterable, Tuple

from environments.base import BaseEnv

try:  # Prefer gymnasium if available, else fallback to gym
    import gymnasium as gym
except ImportError:  # pragma: no cover - dependency optional
    import gym  # type: ignore

spaces = gym.spaces


class MazeEnv(BaseEnv):
    metadata = {"render_modes": ["human", "ansi", "rgb_array"], "render_fps": 4}

    """A classic grid-world maze with walls, goal and per-step penalty."""

    def __init__(
        self,
        grid: Iterable[str] | None = None,
        *,
        max_steps: int = 100,
        step_cost: float = -0.01,
        goal_reward: float = 1.0,
        seed: int | None = None,
        render_mode: str | None = None,
    ):
        """
        Args:
            grid: an iterable of strings, each cell: "." free, "#" wall, "S" start, "G" goal.
                  If None, a default 5x5 maze is used.
            max_steps: episode length limit.
            step_cost: reward per time step (usually negative).
            goal_reward: reward when reaching the goal.
            seed: optional RNG seed for reproducibility.
        """
        self.rng = np.random.default_rng(seed)
        grid_rows = (
            list(grid)
            if grid is not None
            else [
                "S....",
                ".##..",
                "..#..",
                ".##..",
                "...G.",
            ]
        )
        # Convert to 2D array of single-character strings
        self.grid = np.array([list(row) for row in grid_rows], dtype="<U1")
        self.height, self.width = self.grid.shape
        self.step_cost = step_cost
        self.goal_reward = goal_reward
        self.max_steps = max_steps
        self.render_mode = render_mode

        if self.render_mode is not None and self.render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"Invalid render_mode {self.render_mode}. Supported: {self.metadata['render_modes']}"
            )

        self.start_pos = self._find("S")
        self.goal_pos = self._find("G")
        if self.start_pos is None or self.goal_pos is None:
            raise ValueError("Grid must contain one start 'S' and one goal 'G'.")

        # Observation is agent (y, x) position normalised to [0, 1]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(2,), dtype=np.float32
        )
        # Actions: 0=up, 1=down, 2=left, 3=right
        self.action_space = spaces.Discrete(4)

        self._pos = tuple(self.start_pos)
        self._steps = 0

    def _find(self, char: str) -> Tuple[int, int] | None:
        coords = np.argwhere(self.grid == char)
        return tuple(coords[0]) if coords.size > 0 else None

    def _valid(self, pos: Tuple[int, int]) -> bool:
        y, x = pos
        if y < 0 or y >= self.height or x < 0 or x >= self.width:
            return False
        return self.grid[y, x] != "#"

    def _obs(self) -> np.ndarray:
        y, x = self._pos
        return np.array([y / (self.height - 1), x / (self.width - 1)], dtype=np.float32)

    def reset(self) -> Tuple[np.ndarray, dict]:
        self._pos = tuple(self.start_pos)
        self._steps = 0
        return self._obs(), {}

    def step(self, action: np.ndarray | int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self._steps += 1
        act = int(np.squeeze(action))
        dy, dx = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1),
        }.get(act, (0, 0))

        new_pos = (self._pos[0] + dy, self._pos[1] + dx)
        if self._valid(new_pos):
            self._pos = new_pos

        terminated = self._pos == tuple(self.goal_pos)
        truncated = self._steps >= self.max_steps
        reward = self.goal_reward if terminated else self.step_cost

        info = {"pos": self._pos, "steps": self._steps}
        return self._obs(), reward, terminated, truncated, info

    def _grid_with_agent(self) -> np.ndarray:
        grid = self.grid.copy()
        y, x = self._pos
        grid[y, x] = "A"
        return grid

    def render(self):
        grid = self._grid_with_agent()

        if self.render_mode in (None, "human", "ansi"):
            text = "\n".join("".join(row) for row in grid)
            if self.render_mode == "human":
                print(text)
                return None
            return text

        if self.render_mode == "rgb_array":
            color_map = {
                ".": np.array([255, 255, 255], dtype=np.uint8),  # free
                "#": np.array([0, 0, 0], dtype=np.uint8),  # wall
                "S": np.array([0, 200, 0], dtype=np.uint8),  # start
                "G": np.array([255, 215, 0], dtype=np.uint8),  # goal
                "A": np.array([50, 100, 255], dtype=np.uint8),  # agent
            }
            img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            for y in range(self.height):
                for x in range(self.width):
                    img[y, x] = color_map.get(grid[y, x], np.array([255, 0, 0], dtype=np.uint8))
            return img

        raise ValueError(f"Unsupported render_mode {self.render_mode}")
