"""A tiny grid-world maze environment for policy learning.

This module provides a simple grid-world maze environment that is compatible
with the Gymnasium API. The environment is defined by a grid of characters,
where 'S' is the start, 'G' is the goal, '#' is a wall, and '.' is a free
space.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
from environments.base import BaseEnv

try:  # Prefer gymnasium if available, else fallback to gym
    import gymnasium as gym
except ImportError:  # pragma: no cover - dependency optional
    import gym  # type: ignore

spaces = gym.spaces


class MazeEnv(BaseEnv):
    """A classic grid-world maze with walls, a goal, and a per-step penalty.

    Attributes:
        metadata (dict): Metadata for the environment, including render modes and FPS.
        action_space (gym.spaces.Discrete): The action space for the environment.
        observation_space (gym.spaces.Box): The observation space for the environment.
    """

    metadata = {"render_modes": ["human", "ansi", "rgb_array"], "render_fps": 4}

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
        """Initializes a new MazeEnv environment.

        Args:
            grid: An iterable of strings representing the maze layout. Each
                character in the string can be one of the following:
                - ".": A free space.
                - "#": A wall.
                - "S": The starting position.
                - "G": The goal position.
                If None, a default 5x5 maze is used.
            max_steps: The maximum number of steps per episode.
            step_cost: The cost incurred at each step.
            goal_reward: The reward for reaching the goal.
            seed: The seed for the random number generator.
            render_mode: The rendering mode to use. Can be one of "human",
                "ansi", or "rgb_array".
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

        if (
            self.render_mode is not None
            and self.render_mode not in self.metadata["render_modes"]
        ):
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
        """Finds the coordinates of a character in the grid.

        Args:
            char: The character to find.

        Returns:
            A tuple of (y, x) coordinates, or None if the character is not found.
        """
        coords = np.argwhere(self.grid == char)
        return tuple(coords[0]) if coords.size > 0 else None

    def _valid(self, pos: Tuple[int, int]) -> bool:
        """Checks if a position is valid.

        Args:
            pos: The position to check.

        Returns:
            True if the position is valid, False otherwise.
        """
        y, x = pos
        if y < 0 or y >= self.height or x < 0 or x >= self.width:
            return False
        return self.grid[y, x] != "#"

    def _obs(self) -> np.ndarray:
        """Gets the current observation.

        Returns:
            The current observation.
        """
        y, x = self._pos
        return np.array([y / (self.height - 1), x / (self.width - 1)], dtype=np.float32)

    def reset(self) -> Tuple[np.ndarray, dict]:
        """Resets the environment to its initial state.

        Returns:
            A tuple containing the initial observation and an empty info dict.
        """
        self._pos = tuple(self.start_pos)
        self._steps = 0
        return self._obs(), {}

    def step(
        self, action: np.ndarray | int
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Takes a step in the environment.

        Args:
            action: The action to take.

        Returns:
            A tuple containing the next observation, the reward, a boolean
            indicating whether the episode has terminated, a boolean indicating
            whether the episode has been truncated, and an info dict.
        """
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
        """Returns a copy of the grid with the agent's position marked.

        Returns:
            A copy of the grid with the agent's position marked.
        """
        grid = self.grid.copy()
        y, x = self._pos
        grid[y, x] = "A"
        return grid

    def render(self):
        """Renders the environment.

        Returns:
            The rendered environment, or None if the render mode is "human".
        """
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
                    img[y, x] = color_map.get(
                        grid[y, x], np.array([255, 0, 0], dtype=np.uint8)
                    )
            return img

        raise ValueError(f"Unsupported render_mode {self.render_mode}")
