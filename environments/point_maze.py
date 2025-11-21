"""Continuous Point Maze environment with Expert Oracle.

This environment implements a continuous 2D maze where an agent (point mass)
must navigate to a goal.

Key Features:
1. Continuous Action Space: Actions are (vx, vy) velocity vectors in [-1, 1].
2. Continuous Observation Space: (x, y) coordinates in [0, 1].
3. Built-in Expert: `get_expert_action` uses A* search to provide optimal actions
   for data collection/Behavior Cloning.
"""

from __future__ import annotations

import heapq
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from environments.base import BaseEnv


class PointMazeEnv(BaseEnv):
    """Continuous maze where the agent controls velocity (vx, vy)."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        maze_layout: Optional[List[str]] = None,
        render_mode: Optional[str] = None,
        max_steps: int = 200,
        step_size: float = 0.15,  # Increased from 0.1 for faster travel
        goal_threshold: float = 0.1,  # Distance to consider goal reached
        seed: int | None = None,
    ):
        self.rng = np.random.default_rng(seed)
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.step_size = step_size
        self.goal_threshold = goal_threshold

        # 1. Define Maze Layout
        self.layout = maze_layout or [
            "S.....",
            ".###..",
            ".#....",
            ".###..",
            "....G.",
        ]
        self.height = len(self.layout)
        self.width = len(self.layout[0])

        # 2. Spaces
        # Obs: (x, y) normalized to [0, 1]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(2,), dtype=np.float32
        )
        # Action: (vx, vy) clipped to [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # 3. State
        self.start_cell = self._find_cell("S")
        self.goal_cell = self._find_cell("G")
        self.pos = np.array(self.start_cell, dtype=np.float32) + 0.5  # Center of cell
        self.goal_pos = np.array(self.goal_cell, dtype=np.float32) + 0.5
        self.steps = 0

        # Cache walls for collision detection
        self.walls = self._get_walls()

    def _find_cell(self, char: str) -> Tuple[int, int]:
        for r, row in enumerate(self.layout):
            for c, val in enumerate(row):
                if val == char:
                    return (r, c)
        raise ValueError(f"Maze must contain '{char}'")

    def _get_walls(self) -> Set[Tuple[int, int]]:
        walls = set()
        for r, row in enumerate(self.layout):
            for c, val in enumerate(row):
                if val == "#":
                    walls.add((r, c))
        return walls

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Start at center of S cell + small noise
        self.pos = np.array(self.start_cell, dtype=np.float32) + 0.5
        self.pos += self.rng.uniform(-0.1, 0.1, size=2)
        self.steps = 0

        return self._get_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.steps += 1

        # Clip action
        vel = np.clip(action, -1.0, 1.0)

        # Proposed new position
        new_pos = self.pos + vel * self.step_size

        # Simple collision check (treat agent as point)
        # If new cell is a wall, stay in current cell
        r, c = int(new_pos[0]), int(new_pos[1])

        # Boundary check
        if 0 <= r < self.height and 0 <= c < self.width:
            if (r, c) not in self.walls:
                self.pos = new_pos

        # Rewards / Termination
        dist = np.linalg.norm(self.pos - self.goal_pos)
        terminated = dist < self.goal_threshold
        truncated = self.steps >= self.max_steps
        reward = 1.0 if terminated else -0.01

        return self._get_obs(), reward, terminated, truncated, {"dist": dist}

    def _get_obs(self) -> np.ndarray:
        # Normalize to [0, 1]
        return np.array(
            [self.pos[0] / self.height, self.pos[1] / self.width], dtype=np.float32
        )

    # --- EXPERT / ORACLE ---

    def get_expert_action(self, obs: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculates the optimal action (velocity) towards the goal.

        Uses BFS/A* on the grid to find the next cell, then creates a vector
        pointing to that cell's center.
        """
        curr_r, curr_c = int(self.pos[0]), int(self.pos[1])
        target_r, target_c = self.goal_cell

        # A* Search
        queue = [(0, curr_r, curr_c)]
        came_from = {}
        cost_so_far = {(curr_r, curr_c): 0}

        found = False
        while queue:
            _, r, c = heapq.heappop(queue)

            if (r, c) == (target_r, target_c):
                found = True
                break

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.height and 0 <= nc < self.width:
                    if (nr, nc) not in self.walls:
                        new_cost = cost_so_far[(r, c)] + 1
                        if (nr, nc) not in cost_so_far or new_cost < cost_so_far[
                            (nr, nc)
                        ]:
                            cost_so_far[(nr, nc)] = new_cost
                            priority = (
                                new_cost + abs(target_r - nr) + abs(target_c - nc)
                            )
                            heapq.heappush(queue, (priority, nr, nc))
                            came_from[(nr, nc)] = (r, c)

        if not found:
            return np.zeros(2, dtype=np.float32)  # No path

        # Backtrack to find immediate next step
        curr = (target_r, target_c)
        path = [curr]
        while curr != (curr_r, curr_c):
            curr = came_from.get(curr)
            if curr is None:
                break  # Should not happen if found
            path.append(curr)

        # path[-1] is current, path[-2] is next step
        if len(path) >= 2:
            next_r, next_c = path[-2]
            # Target is center of next cell
            target_pos = np.array([next_r + 0.5, next_c + 0.5])

            # Vector to target
            direction = target_pos - self.pos

            # Normalize to unit length (max speed)
            norm = np.linalg.norm(direction)
            if norm > 1e-5:
                action = direction / norm
            else:
                action = np.zeros(2)
            return action.astype(np.float32)

        # If we are in the goal cell, move exactly to center
        direction = self.goal_pos - self.pos
        return np.clip(direction / self.step_size, -1, 1).astype(np.float32)

    def render(self):
        if self.render_mode == "rgb_array":
            img = np.zeros((self.height * 10, self.width * 10, 3), dtype=np.uint8)
            # Draw Walls (White)
            for r, c in self.walls:
                img[r * 10 : (r + 1) * 10, c * 10 : (c + 1) * 10] = 255
            # Draw Goal (Green)
            gr, gc = self.goal_cell
            img[gr * 10 : (gr + 1) * 10, gc * 10 : (gc + 1) * 10] = [0, 255, 0]
            # Draw Agent (Red)
            ar, ac = int(self.pos[0] * 10), int(self.pos[1] * 10)
            # Simple 2x2 pixel agent
            img[ar - 1 : ar + 1, ac - 1 : ac + 1] = [255, 0, 0]
            return img
        elif self.render_mode == "human":
            print(f"Pos: {self.pos}")
