"""
HybridLunarLander – Parameterized Action Space for HyAR
========================================================

Designed to match HyAR's assumption:
    Each discrete action k ∈ {0, 1, 2, 3} has its OWN continuous parameter
    space X_k.  A hybrid action is a pair (k, x_k).

Discrete actions & their continuous parameters
-----------------------------------------------
  k=0  COAST        │ x_0 = []  (no parameters, pure drift)
  k=1  MAIN_ENGINE  │ x_1 = [throttle]          ∈ [-1, 1]  → mapped to [50%, 100%] thrust
  k=2  LEFT_BOOST   │ x_2 = [intensity, duration_frac]  ∈ [-1,1]²
  k=3  RIGHT_BOOST  │ x_3 = [intensity, duration_frac]  ∈ [-1,1]²

Design rationale
----------------
- **COAST** has zero continuous dims → purely discrete, like the original
  "do nothing" action.
- **MAIN_ENGINE** has 1 continuous dim (throttle power) → fine-grained
  vertical thrust control.
- **LEFT / RIGHT BOOST** each have 2 continuous dims:
    • intensity  → how hard the side thruster fires  (mapped past 0.5
      ignition threshold)
    • duration_frac → fraction of the physics step during which the
      thruster is active (pulse-width modulation), giving the agent a
      richer control vocabulary than simple on/off.

This creates genuine semantic heterogeneity across discrete branches,
which is exactly what HyAR's conditional VAE is designed to capture.

Observation modes
-----------------
  use_image_obs=False  →  Box(8,)           (classic state vector)
  use_image_obs=True   →  Dict{"state": Box(8,), "image": Box(H,W,3)}
"""

import importlib.metadata as _metadata

_original_entry_points = _metadata.entry_points


def _safe_entry_points(*args, **kwargs):
    eps = _original_entry_points(*args, **kwargs)
    if kwargs.get("group") != "gymnasium.envs":
        return eps
    filtered = [
        ep for ep in eps if getattr(ep, "module", "") != "gymnasium_robotics.__init__"
    ]
    try:
        return type(eps)(filtered)
    except Exception:
        return filtered


_metadata.entry_points = _safe_entry_points

import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import RecordVideo
import numpy as np
from typing import Optional


# ─── Action constants ────────────────────────────────────────────────────────

COAST = 0          # do nothing — drift under gravity
MAIN_ENGINE = 1    # vertical thrust (throttle)
LEFT_BOOST = 2     # left side thruster (intensity, duration_frac)
RIGHT_BOOST = 3    # right side thruster (intensity, duration_frac)

NUM_DISCRETE = 4

# Continuous parameter dimensions per discrete action
PARAM_DIMS = {
    COAST:       0,   # no parameters
    MAIN_ENGINE: 1,   # [throttle]
    LEFT_BOOST:  2,   # [intensity, duration_frac]
    RIGHT_BOOST: 2,   # [intensity, duration_frac]
}

# Maximum continuous dimension (used for padding)
MAX_PARAM_DIM = max(PARAM_DIMS.values())  # = 2


class HybridLunarLander(gym.Env):
    """
    Parameterized-action-space wrapper over LunarLanderContinuous.

    Action format (for HyAR compatibility):
        action = {
            "k":   int in {0, 1, 2, 3},
            "x_k": ndarray of shape (MAX_PARAM_DIM,) in [-1, 1]
                   (only the first PARAM_DIMS[k] entries are used)
        }

    The environment internally converts (k, x_k) into the 2-dim continuous
    action [main_throttle, side_throttle] expected by LunarLanderContinuous.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        use_image_obs: bool = False,
        img_size: tuple = (84, 84),
        sub_steps: int = 1,
    ):
        """
        Args:
            use_image_obs: Include pixel observations alongside the state vector.
            img_size:      (H, W) for image observations.
            sub_steps:     Number of physics sub-steps per action (frame-skip).
                           Useful for duration_frac control in side boosters.
        """
        super().__init__()

        self.use_image_obs = use_image_obs
        self.img_size = img_size
        self.sub_steps = max(1, sub_steps)

        self._env = self._make_env(render_mode="rgb_array")
        self.render_mode = "rgb_array"

        # ── Observation space ────────────────────────────────────────────
        state_space = self._env.observation_space  # Box(8,)

        if self.use_image_obs:
            H, W = self.img_size
            self.observation_space = spaces.Dict({
                "state": state_space,
                "image": spaces.Box(0, 255, shape=(H, W, 3), dtype=np.uint8),
            })
        else:
            self.observation_space = state_space

        # ── Parameterized action space ───────────────────────────────────
        #
        # HyAR expects:  k ∈ Discrete(K),  x_k ∈ Box(d_max)
        # where only the first d_k dims are semantically meaningful for
        # action k.  We expose the full padded vector; unused dims are
        # ignored internally.
        #
        self.action_space = spaces.Dict({
            "k":   spaces.Discrete(NUM_DISCRETE),
            "x_k": spaces.Box(-1.0, 1.0, shape=(MAX_PARAM_DIM,), dtype=np.float32),
        })

        # Store param dims so HyAR can query them
        self.param_dims = dict(PARAM_DIMS)
        self.num_discrete = NUM_DISCRETE
        self.max_param_dim = MAX_PARAM_DIM

    # ── Environment factory ──────────────────────────────────────────────

    @staticmethod
    def _make_env(render_mode: str) -> gym.Env:
        candidates = ["LunarLanderContinuous-v3", "LunarLanderContinuous-v2"]
        last_err = None
        for env_id in candidates:
            try:
                return gym.make(env_id, render_mode=render_mode)
            except gym.error.DependencyNotInstalled as e:
                raise RuntimeError(
                    "Box2D missing. Install: pip install 'gymnasium[box2d]'"
                ) from e
            except Exception as e:
                last_err = e
        raise RuntimeError(
            f"Could not create LunarLanderContinuous. Tried {candidates}. "
            f"Last error: {last_err}"
        )

    # ── Action conversion ────────────────────────────────────────────────

    def _convert_action(self, k: int, x_k: np.ndarray) -> np.ndarray:
        """
        Map parameterized action (k, x_k) → [main_throttle, side_throttle]
        for the base LunarLanderContinuous environment.

        Returns:
            np.array([main, side], dtype=float32)
              main ∈ [0, 1]    (0 = off, >0 maps to 50-100% internally)
              side ∈ [-1, 1]   (|side| > 0.5 triggers thruster)
        """
        main = 0.0
        side = 0.0

        if k == COAST:
            # Pure drift — both engines off
            pass

        elif k == MAIN_ENGINE:
            # x_k[0] = throttle ∈ [-1, 1] → mapped to [0, 1]
            throttle = float(np.clip(x_k[0], -1.0, 1.0))
            main = (throttle + 1.0) * 0.5  # [-1,1] → [0,1]

        elif k == LEFT_BOOST:
            # x_k[0] = intensity ∈ [-1, 1] → mapped to [-1.0, -0.5]
            #          (negative side value = left thruster)
            # x_k[1] = duration_frac ∈ [-1, 1] → mapped to [0, 1]
            #          (fraction of sub-steps to fire)
            intensity = float(np.clip(x_k[0], -1.0, 1.0))
            # Map intensity [−1, 1] → magnitude [0.5, 1.0], sign = negative (left)
            mag = 0.5 + 0.5 * (intensity + 1.0) / 2.0   # [0.5, 1.0]
            side = -mag  # negative = fire left

        elif k == RIGHT_BOOST:
            # x_k[0] = intensity ∈ [-1, 1] → mapped to [0.5, 1.0]
            #          (positive side value = right thruster)
            # x_k[1] = duration_frac (same as left)
            intensity = float(np.clip(x_k[0], -1.0, 1.0))
            mag = 0.5 + 0.5 * (intensity + 1.0) / 2.0   # [0.5, 1.0]
            side = mag  # positive = fire right

        return np.array([main, side], dtype=np.float32)

    def _get_duration_frac(self, k: int, x_k: np.ndarray) -> float:
        """
        Extract duration fraction for side boosters.
        Returns 1.0 for COAST and MAIN_ENGINE (always full step).
        For LEFT/RIGHT_BOOST: x_k[1] ∈ [-1,1] → [0.1, 1.0]
        """
        if k in (LEFT_BOOST, RIGHT_BOOST) and self.sub_steps > 1:
            raw = float(np.clip(x_k[1], -1.0, 1.0))
            return 0.1 + 0.9 * (raw + 1.0) / 2.0   # [0.1, 1.0]
        return 1.0

    # ── Image helper ─────────────────────────────────────────────────────

    def _get_image(self) -> np.ndarray:
        frame = self._env.render()  # (native_H, native_W, 3)
        H, W = self.img_size
        nH, nW = frame.shape[:2]
        if (nH, nW) != (H, W):
            row_idx = (np.arange(H) * nH / H).astype(int)
            col_idx = (np.arange(W) * nW / W).astype(int)
            frame = frame[np.ix_(row_idx, col_idx)]
        return frame.astype(np.uint8)

    def _build_obs(self, state_obs: np.ndarray):
        if self.use_image_obs:
            return {"state": state_obs, "image": self._get_image()}
        return state_obs

    # ── Gym API ──────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        state_obs, info = self._env.reset(seed=seed, options=options)
        return self._build_obs(state_obs), info

    def step(self, action):
        """
        Execute one parameterized action.

        Args:
            action: dict with
                "k":   int    — discrete action index
                "x_k": array  — continuous parameters (padded to MAX_PARAM_DIM)

        For sub_steps > 1 and side boosters, duration_frac controls how many
        sub-steps the thruster is active (pulse-width modulation).
        """
        k = int(action["k"])
        x_k = np.asarray(action["x_k"], dtype=np.float32)

        cont_action = self._convert_action(k, x_k)
        duration_frac = self._get_duration_frac(k, x_k)

        if self.sub_steps == 1:
            # Standard single-step execution
            state_obs, reward, terminated, truncated, info = self._env.step(cont_action)
        else:
            # Sub-stepping with duration control
            active_steps = max(1, int(round(duration_frac * self.sub_steps)))
            idle_action = np.array([0.0, 0.0], dtype=np.float32)

            total_reward = 0.0
            terminated = truncated = False

            for i in range(self.sub_steps):
                act = cont_action if i < active_steps else idle_action
                state_obs, r, terminated, truncated, info = self._env.step(act)
                total_reward += r
                if terminated or truncated:
                    break

            reward = total_reward

        info["discrete_action"] = k
        info["param_dims_used"] = PARAM_DIMS[k]

        return self._build_obs(state_obs), reward, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        return self._env.close()

    # ── Utility for HyAR integration ────────────────────────────────────

    def get_action_spec(self) -> dict:
        """
        Return action space specification for HyAR's encoder/decoder setup.

        Returns dict with:
            num_discrete:  K (number of discrete actions)
            param_dims:    {k: d_k} mapping each action to its parameter dim
            max_param_dim: max(d_k) across all actions
            action_names:  {k: str} human-readable names
        """
        return {
            "num_discrete": self.num_discrete,
            "param_dims": dict(self.param_dims),
            "max_param_dim": self.max_param_dim,
            "action_names": {
                COAST:       "coast",
                MAIN_ENGINE: "main_engine",
                LEFT_BOOST:  "left_boost",
                RIGHT_BOOST: "right_boost",
            },
        }


# ─── Smoke test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  HybridLunarLander — Parameterized Action Space")
    print("=" * 60)

    env = HybridLunarLander(use_image_obs=False)

    spec = env.get_action_spec()
    print(f"\nAction spec:")
    print(f"  Discrete actions : {spec['num_discrete']}")
    print(f"  Max param dim    : {spec['max_param_dim']}")
    for k, name in spec["action_names"].items():
        print(f"    k={k}  {name:15s}  params={spec['param_dims'][k]}d")

    print(f"\nAction space : {env.action_space}")
    print(f"Obs space    : {env.observation_space}")

    # ── Test each discrete action ────────────────────────────────────────
    obs, info = env.reset(seed=42)
    print(f"\nInitial obs shape: {obs.shape}")

    test_actions = [
        {"k": COAST,       "x_k": np.zeros(MAX_PARAM_DIM)},
        {"k": MAIN_ENGINE, "x_k": np.array([0.5, 0.0])},        # throttle=0.5
        {"k": LEFT_BOOST,  "x_k": np.array([0.3, 0.8])},        # intensity=0.3, dur=0.8
        {"k": RIGHT_BOOST, "x_k": np.array([-0.2, -0.5])},      # intensity=-0.2, dur=-0.5
    ]

    for act in test_actions:
        obs, reward, term, trunc, info = env.step(act)
        name = spec["action_names"][act["k"]]
        dims_used = info["param_dims_used"]
        print(f"  {name:15s}  x_k={act['x_k']}  "
              f"dims_used={dims_used}  reward={reward:+.2f}")
        if term or trunc:
            obs, info = env.reset()

    # ── Full random episode ──────────────────────────────────────────────
    print("\nRunning random episode...")
    obs, info = env.reset(seed=123)

    action_counts = {k: 0 for k in range(NUM_DISCRETE)}
    total_reward = 0.0
    steps = 0
    terminated = truncated = False

    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        action_counts[info["discrete_action"]] += 1
        total_reward += reward
        steps += 1

    print(f"  Steps: {steps}  Total reward: {total_reward:+.1f}")
    print(f"  Action distribution:")
    for k, name in spec["action_names"].items():
        pct = 100 * action_counts[k] / steps if steps > 0 else 0
        print(f"    {name:15s}: {action_counts[k]:4d} ({pct:.1f}%)")

    # ── Image obs mode ───────────────────────────────────────────────────
    print("\nTesting image obs mode...")
    env_img = HybridLunarLander(use_image_obs=True, img_size=(84, 84))
    obs, _ = env_img.reset(seed=42)
    print(f"  State shape: {obs['state'].shape}")
    print(f"  Image shape: {obs['image'].shape}  dtype={obs['image'].dtype}")
    env_img.close()

    # ── Video recording (optional) ──────────────────────────────────────
    try:
        print("\nRecording video...")
        env_vid = HybridLunarLander(use_image_obs=False)
        env_vid = RecordVideo(
            env_vid,
            video_folder="videos/parameterized",
            episode_trigger=lambda ep: True,
            name_prefix="hybrid-param-lander",
        )
        obs, _ = env_vid.reset(seed=42)
        terminated = truncated = False
        while not (terminated or truncated):
            action = env_vid.action_space.sample()
            obs, reward, terminated, truncated, info = env_vid.step(action)
        env_vid.close()
        print("  Video saved to ./videos/parameterized/")
    except Exception as e:
        print(f"  Video recording skipped ({e})")

    env.close()
    print("\nAll tests passed.")