"""
Reproducible PPO training for HybridLunarLanderV2 with fully hybrid action space:
  - discrete mode: Categorical(3)
  - continuous u: tanh-squashed Gaussian Box(2,)

Key features:
1. Reproducibility:
   - global seeding (python/numpy/torch/cuda)
   - per-env reproducible RNG for reset seed sampling
   - fixed evaluation seed set

2. Robustness-oriented training:
   - reset seeds are sampled from a Gaussian distribution
   - different envs see different but reproducible seed streams
   - deterministic evaluation on a fixed seed set

3. Correct PPO for vectorized envs:
   - rollout buffer stores tensors with shape [n_steps, n_envs, ...]
   - GAE / returns computed per-environment
   - hybrid log-prob = log p(mode) + log p(u)

Usage:
    python scripts/train_ppo_lunar_lander.py
    python scripts/train_ppo_lunar_lander.py ppo.total_timesteps=2000000 ppo.lr=1e-4
    python scripts/train_ppo_lunar_lander.py env.n_envs=32 ppo.n_steps=512
"""

import os
import sys
import time
import random
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
import gymnasium as gym
from gymnasium import spaces
from tqdm.auto import tqdm
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from environments.lunar_lander import HybridLunarLanderV2


# ═══════════════════════════════════════════════════════════════════════════════
# Reproducibility utils
# ═══════════════════════════════════════════════════════════════════════════════

def set_global_seeds(base_seed: int, torch_deterministic: bool = False):
    random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(base_seed)
        torch.cuda.manual_seed_all(base_seed)

    if torch_deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as e:
            print(f"[Warning] Could not enable full deterministic algorithms: {e}")
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def sample_gaussian_seed(rng, mean, std, seed_min, seed_max):
    raw = int(round(rng.normal(mean, std)))
    raw = abs(raw)
    return int(np.clip(raw, seed_min, seed_max))


# ═══════════════════════════════════════════════════════════════════════════════
# Env wrapper
# ═══════════════════════════════════════════════════════════════════════════════

class PPOLunarLanderWrapper(gym.Env):
    """
    HybridLunarLanderV2 for PPO:
      - observation: Box(8,)
      - action: Dict(mode=Discrete(3), u=Box(2,))
    """
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        seed_mean=0.0,
        seed_std=10000.0,
        seed_min=0,
        seed_max=2147483647,
        trigger_side=True,
        rng_seed=0,
    ):
        super().__init__()
        self._env = HybridLunarLanderV2(use_image_obs=False, trigger_side=trigger_side)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

        self.seed_mean = seed_mean
        self.seed_std = seed_std
        self.seed_min = seed_min
        self.seed_max = seed_max
        self._rng = np.random.default_rng(rng_seed)

    def reset(self, *, seed=None, options=None):
        if seed is None:
            seed = sample_gaussian_seed(
                self._rng,
                self.seed_mean,
                self.seed_std,
                self.seed_min,
                self.seed_max,
            )
        obs, info = self._env.reset(seed=seed, options=options)
        info = dict(info) if info is not None else {}
        info["reset_seed"] = int(seed)
        return obs.astype(np.float32), info

    def step(self, action):
        mode = int(action["mode"])
        u = np.asarray(action["u"], dtype=np.float32)
        u = np.clip(u, -1.0, 1.0)

        hybrid_action = {"mode": mode, "u": u}
        obs, reward, terminated, truncated, info = self._env.step(hybrid_action)
        return obs.astype(np.float32), float(reward), terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        return self._env.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Minimal synchronous vectorized env
# ═══════════════════════════════════════════════════════════════════════════════

class VecEnv:
    """Minimal synchronous vectorized environment."""
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.n = len(self.envs)
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(self):
        obs_list, infos = [], []
        for env in self.envs:
            obs, info = env.reset()
            obs_list.append(obs)
            infos.append(info)
        return np.stack(obs_list), infos

    def step(self, actions):
        obs_list, rew_list, term_list, trunc_list, done_list, info_list = [], [], [], [], [], []

        if isinstance(actions, dict):
            modes = actions["mode"]
            us = actions["u"]
            per_env_actions = [{"mode": int(modes[i]), "u": us[i]} for i in range(self.n)]
        else:
            per_env_actions = actions

        for env, act in zip(self.envs, per_env_actions):
            obs, reward, term, trunc, info = env.step(act)
            done = term or trunc
            if done:
                next_obs, reset_info = env.reset()
                info = dict(info) if info is not None else {}
                info["terminal_observation"] = obs
                info["reset_info"] = reset_info
                obs = next_obs

            obs_list.append(obs)
            rew_list.append(reward)
            term_list.append(term)
            trunc_list.append(trunc)
            done_list.append(done)
            info_list.append(info)

        return (
            np.stack(obs_list),
            np.asarray(rew_list, dtype=np.float32),
            np.asarray(term_list, dtype=np.float32),
            np.asarray(trunc_list, dtype=np.float32),
            np.asarray(done_list, dtype=np.float32),
            info_list,
        )

    def close(self):
        for env in self.envs:
            env.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Hybrid Actor-Critic network
# ═══════════════════════════════════════════════════════════════════════════════

class HybridActorCritic(nn.Module):
    def __init__(self, obs_dim, cont_act_dim, n_modes=3, hidden=256):
        super().__init__()
        self.n_modes = n_modes
        self.cont_act_dim = cont_act_dim

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        self.mode_logits = nn.Linear(hidden, n_modes)
        self.policy_mean = nn.Linear(hidden, cont_act_dim)
        self.policy_log_std = nn.Parameter(torch.zeros(cont_act_dim))
        self.value_head = nn.Linear(hidden, 1)

        for layer in [*self.shared, self.mode_logits, self.policy_mean, self.value_head]:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)

        nn.init.orthogonal_(self.mode_logits.weight, gain=0.01)
        nn.init.orthogonal_(self.policy_mean.weight, gain=0.01)
        nn.init.zeros_(self.mode_logits.bias)
        nn.init.zeros_(self.policy_mean.bias)

    def forward(self, obs):
        h = self.shared(obs)
        mode_logits = self.mode_logits(h)
        mean = self.policy_mean(h)
        std = self.policy_log_std.exp().expand_as(mean)
        value = self.value_head(h).squeeze(-1)
        return mode_logits, mean, std, value

    def get_action(self, obs, deterministic=False):
        mode_logits, mean, std, value = self(obs)

        mode_dist = Categorical(logits=mode_logits)
        cont_dist = Normal(mean, std)

        if deterministic:
            mode = torch.argmax(mode_logits, dim=-1)
            pre_tanh_u = mean
        else:
            mode = mode_dist.sample()
            pre_tanh_u = cont_dist.rsample()

        u = torch.tanh(pre_tanh_u)

        mode_log_prob = mode_dist.log_prob(mode)
        cont_log_prob = cont_dist.log_prob(pre_tanh_u).sum(-1)
        cont_log_prob -= torch.log(1 - u.pow(2) + 1e-6).sum(-1)

        log_prob = mode_log_prob + cont_log_prob

        return mode, u, log_prob, value

    def evaluate(self, obs, modes, cont_actions):
        mode_logits, mean, std, value = self(obs)

        mode_dist = Categorical(logits=mode_logits)
        cont_dist = Normal(mean, std)

        mode_log_prob = mode_dist.log_prob(modes)
        mode_entropy = mode_dist.entropy()

        cont_actions = cont_actions.clamp(-0.999999, 0.999999)
        pre_tanh = torch.atanh(cont_actions)

        cont_log_prob = cont_dist.log_prob(pre_tanh).sum(-1)
        cont_log_prob -= torch.log(1 - cont_actions.pow(2) + 1e-6).sum(-1)
        cont_entropy = cont_dist.entropy().sum(-1)

        log_prob = mode_log_prob + cont_log_prob
        entropy = mode_entropy + cont_entropy

        return log_prob, value, entropy, mode_entropy, cont_entropy


# ═══════════════════════════════════════════════════════════════════════════════
# Rollout buffer for vectorized hybrid PPO
# ═══════════════════════════════════════════════════════════════════════════════

class RolloutBuffer:
    def __init__(self, n_steps, n_envs, obs_dim, cont_act_dim, device):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.obs_dim = obs_dim
        self.cont_act_dim = cont_act_dim
        self.device = device

        self.obs = np.zeros((n_steps, n_envs, obs_dim), dtype=np.float32)
        self.modes = np.zeros((n_steps, n_envs), dtype=np.int64)
        self.cont_actions = np.zeros((n_steps, n_envs, cont_act_dim), dtype=np.float32)
        self.log_probs = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.dones = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values = np.zeros((n_steps, n_envs), dtype=np.float32)

        self.advantages = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.returns = np.zeros((n_steps, n_envs), dtype=np.float32)

        self.ptr = 0

    def add(self, obs, modes, cont_actions, log_probs, rewards, dones, values):
        self.obs[self.ptr] = obs
        self.modes[self.ptr] = modes
        self.cont_actions[self.ptr] = cont_actions
        self.log_probs[self.ptr] = log_probs
        self.rewards[self.ptr] = rewards
        self.dones[self.ptr] = dones
        self.values[self.ptr] = values
        self.ptr += 1

    def compute_returns_and_advantages(self, last_values, gamma=0.99, lam=0.95):
        gae = np.zeros((self.n_envs,), dtype=np.float32)

        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_values = last_values
            else:
                next_values = self.values[t + 1]

            not_done = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_values * not_done - self.values[t]
            gae = delta + gamma * lam * not_done * gae
            self.advantages[t] = gae

        self.returns = self.advantages + self.values

    def get_tensors(self):
        obs = torch.tensor(
            self.obs.reshape(self.n_steps * self.n_envs, self.obs_dim),
            dtype=torch.float32,
            device=self.device,
        )
        modes = torch.tensor(
            self.modes.reshape(self.n_steps * self.n_envs),
            dtype=torch.long,
            device=self.device,
        )
        cont_actions = torch.tensor(
            self.cont_actions.reshape(self.n_steps * self.n_envs, self.cont_act_dim),
            dtype=torch.float32,
            device=self.device,
        )
        old_log_probs = torch.tensor(
            self.log_probs.reshape(self.n_steps * self.n_envs),
            dtype=torch.float32,
            device=self.device,
        )
        advantages = torch.tensor(
            self.advantages.reshape(self.n_steps * self.n_envs),
            dtype=torch.float32,
            device=self.device,
        )
        returns = torch.tensor(
            self.returns.reshape(self.n_steps * self.n_envs),
            dtype=torch.float32,
            device=self.device,
        )
        values = torch.tensor(
            self.values.reshape(self.n_steps * self.n_envs),
            dtype=torch.float32,
            device=self.device,
        )
        return obs, modes, cont_actions, old_log_probs, advantages, returns, values


# ═══════════════════════════════════════════════════════════════════════════════
# Wandb utils
# ═══════════════════════════════════════════════════════════════════════════════

def maybe_init_wandb(cfg: DictConfig):
    if cfg.logging.mode == "disabled":
        return None

    run = wandb.init(
        project=cfg.logging.project,
        entity=cfg.logging.entity,
        mode=cfg.logging.mode,
        name=cfg.logging.name,
        group=cfg.logging.group,
        tags=list(cfg.logging.tags) if cfg.logging.tags is not None else None,
        notes=cfg.logging.notes,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
    )
    return run


def wandb_log(data: dict, step: int):
    if wandb.run is not None:
        wandb.log(data, step=step)


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def build_eval_seeds(base_seed: int, n_episodes: int):
    rng = np.random.default_rng(base_seed + 10_000)
    seeds = []
    for _ in range(n_episodes):
        s = int(rng.integers(low=0, high=2_147_483_647, endpoint=False))
        seeds.append(s)
    return seeds


def evaluate_policy(ac, eval_env, eval_seeds, device):
    ac.eval()
    episode_rewards = []
    mode_hist = []

    for seed in eval_seeds:
        obs, _ = eval_env.reset(seed=int(seed))
        done = False
        ep_reward = 0.0

        while not done:
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                mode_t, u_t, _, _ = ac.get_action(obs_t, deterministic=True)
                mode = int(mode_t.item())
                u = u_t.cpu().numpy()[0]

            obs, reward, term, trunc, _ = eval_env.step({"mode": mode, "u": u})
            ep_reward += reward
            mode_hist.append(mode)
            done = term or trunc

        episode_rewards.append(ep_reward)

    return np.asarray(episode_rewards, dtype=np.float32), np.asarray(mode_hist, dtype=np.int64)


# ═══════════════════════════════════════════════════════════════════════════════
# PPO training loop
# ═══════════════════════════════════════════════════════════════════════════════

def train_ppo(cfg: DictConfig):
    set_global_seeds(
        base_seed=cfg.seed.base_seed,
        torch_deterministic=getattr(cfg.seed, "torch_deterministic", False),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = cfg.training.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)
    print(f"Using device: {device}")

    run = maybe_init_wandb(cfg)

    env_cfg = cfg.env
    ppo_cfg = cfg.ppo

    env_fns = []
    for i in range(env_cfg.n_envs):
        env_seed = cfg.seed.base_seed + i
        env_fns.append(
            lambda i=i, env_seed=env_seed: PPOLunarLanderWrapper(
                seed_mean=env_cfg.seed_mean,
                seed_std=env_cfg.seed_std,
                seed_min=env_cfg.seed_min,
                seed_max=env_cfg.seed_max,
                trigger_side=env_cfg.trigger_side,
                rng_seed=env_seed,
            )
        )
    vec_env = VecEnv(env_fns)

    eval_env = PPOLunarLanderWrapper(
        seed_mean=env_cfg.seed_mean,
        seed_std=env_cfg.seed_std,
        seed_min=env_cfg.seed_min,
        seed_max=env_cfg.seed_max,
        trigger_side=env_cfg.trigger_side,
        rng_seed=cfg.seed.base_seed + 9999,
    )
    eval_seeds = build_eval_seeds(cfg.seed.base_seed, cfg.training.eval_episodes)

    obs_dim = vec_env.observation_space.shape[0]
    cont_act_dim = vec_env.action_space["u"].shape[0]
    n_modes = vec_env.action_space["mode"].n
    hidden_dim = cfg.network.hidden_dim

    ac = HybridActorCritic(
        obs_dim=obs_dim,
        cont_act_dim=cont_act_dim,
        n_modes=n_modes,
        hidden=hidden_dim,
    ).to(device)

    optimizer = torch.optim.Adam(ac.parameters(), lr=ppo_cfg.lr, eps=1e-5)

    if run is not None and cfg.logging.watch_model:
        wandb.watch(ac, log="all", log_freq=cfg.logging.watch_log_freq)

    total_updates = ppo_cfg.total_timesteps // (ppo_cfg.n_steps * env_cfg.n_envs)

    def lr_fn(update_idx):
        frac = 1.0 - update_idx / max(total_updates, 1)
        return max(frac, 0.0)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_fn)

    obs, _ = vec_env.reset()
    global_step = 0
    episode_count = 0
    best_eval_reward = -float("inf")
    recent_episode_rewards = deque(maxlen=100)
    per_env_episode_rewards = np.zeros((env_cfg.n_envs,), dtype=np.float32)

    print(
        f"Training Hybrid PPO | device={device} | envs={env_cfg.n_envs} | "
        f"steps/update={ppo_cfg.n_steps * env_cfg.n_envs} | total_updates={total_updates}"
    )

    t0 = time.time()
    progress = tqdm(
        range(1, total_updates + 1),
        disable=not cfg.tqdm.enable,
        leave=cfg.tqdm.leave,
        desc="PPO Training",
    )

    for update in progress:
        rollout = RolloutBuffer(
            n_steps=ppo_cfg.n_steps,
            n_envs=env_cfg.n_envs,
            obs_dim=obs_dim,
            cont_act_dim=cont_act_dim,
            device=device,
        )

        rollout_reward_values = []
        done_values = []
        sampled_reset_seeds = []
        sampled_modes = []

        ac.eval()

        for _ in range(ppo_cfg.n_steps):
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
                modes_t, cont_actions_t, log_probs_t, values_t = ac.get_action(
                    obs_t, deterministic=False
                )

            modes = modes_t.cpu().numpy()
            cont_actions = cont_actions_t.cpu().numpy()
            log_probs = log_probs_t.cpu().numpy()
            values = values_t.cpu().numpy()

            next_obs, rewards, terms, truncs, dones, infos = vec_env.step(
                {"mode": modes, "u": cont_actions}
            )

            rollout.add(
                obs=obs,
                modes=modes,
                cont_actions=cont_actions,
                log_probs=log_probs,
                rewards=rewards,
                dones=dones,
                values=values,
            )

            rollout_reward_values.extend(rewards.tolist())
            done_values.extend(dones.tolist())
            sampled_modes.extend(modes.tolist())

            per_env_episode_rewards += rewards
            for i in range(env_cfg.n_envs):
                if dones[i]:
                    recent_episode_rewards.append(float(per_env_episode_rewards[i]))
                    per_env_episode_rewards[i] = 0.0
                    episode_count += 1

                    reset_info = infos[i].get("reset_info", {})
                    if "reset_seed" in reset_info:
                        sampled_reset_seeds.append(int(reset_info["reset_seed"]))

            obs = next_obs
            global_step += env_cfg.n_envs

        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            _, _, _, last_values_t = ac(obs_t)
            last_values = last_values_t.cpu().numpy()

        rollout.compute_returns_and_advantages(
            last_values=last_values,
            gamma=ppo_cfg.gamma,
            lam=ppo_cfg.gae_lambda,
        )

        obs_b, mode_b, cont_act_b, old_logp_b, adv_b, ret_b, old_val_b = rollout.get_tensors()
        adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

        ac.train()
        batch_size = ppo_cfg.batch_size
        n_samples = obs_b.shape[0]

        policy_losses = []
        value_losses = []
        entropies = []
        mode_entropies = []
        cont_entropies = []
        total_losses = []
        approx_kls = []
        clipfracs = []
        grad_norms = []

        early_stop = False

        for _epoch in range(ppo_cfg.n_epochs):
            indices = np.random.permutation(n_samples)

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                idx = indices[start:end]

                new_logp, new_value, entropy, mode_entropy, cont_entropy = ac.evaluate(
                    obs_b[idx], mode_b[idx], cont_act_b[idx]
                )

                log_ratio = new_logp - old_logp_b[idx]
                ratio = torch.exp(log_ratio)

                surr1 = ratio * adv_b[idx]
                surr2 = torch.clamp(
                    ratio,
                    1.0 - ppo_cfg.clip_range,
                    1.0 + ppo_cfg.clip_range,
                ) * adv_b[idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * (new_value - ret_b[idx]).pow(2).mean()
                entropy_bonus = entropy.mean()

                loss = (
                    policy_loss
                    + ppo_cfg.vf_coef * value_loss
                    - ppo_cfg.ent_coef * entropy_bonus
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(ac.parameters(), ppo_cfg.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    approx_kl = (old_logp_b[idx] - new_logp).mean().item()
                    clipfrac = ((ratio - 1.0).abs() > ppo_cfg.clip_range).float().mean().item()

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy_bonus.item()))
                mode_entropies.append(float(mode_entropy.mean().item()))
                cont_entropies.append(float(cont_entropy.mean().item()))
                total_losses.append(float(loss.item()))
                approx_kls.append(float(approx_kl))
                clipfracs.append(float(clipfrac))
                grad_norms.append(float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm))

                if ppo_cfg.target_kl is not None and approx_kl > float(ppo_cfg.target_kl):
                    early_stop = True
                    break

            if early_stop:
                break

        scheduler.step()

        elapsed = time.time() - t0
        fps = global_step / max(elapsed, 1e-6)

        mean_recent_ep_reward = (
            float(np.mean(recent_episode_rewards))
            if len(recent_episode_rewards) > 0 else 0.0
        )
        mean_rollout_reward = float(np.mean(rollout_reward_values))
        done_rate = float(np.mean(done_values))
        sampled_seed_mean = float(np.mean(sampled_reset_seeds)) if sampled_reset_seeds else 0.0
        sampled_seed_std = float(np.std(sampled_reset_seeds)) if sampled_reset_seeds else 0.0

        mode_freq = np.bincount(np.asarray(sampled_modes, dtype=np.int64), minlength=n_modes)
        mode_freq = mode_freq / max(mode_freq.sum(), 1)

        train_metrics = {
            "train/update": update,
            "train/global_step": global_step,
            "train/episodes": episode_count,
            "train/fps": fps,
            "train/lr": scheduler.get_last_lr()[0],
            "train/mean_recent_ep_reward_100": mean_recent_ep_reward,
            "train/mean_rollout_reward": mean_rollout_reward,
            "train/done_rate": done_rate,
            "train/sample_reset_seed_mean": sampled_seed_mean,
            "train/sample_reset_seed_std": sampled_seed_std,
            "loss/policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "loss/value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "loss/entropy": float(np.mean(entropies)) if entropies else 0.0,
            "loss/mode_entropy": float(np.mean(mode_entropies)) if mode_entropies else 0.0,
            "loss/cont_entropy": float(np.mean(cont_entropies)) if cont_entropies else 0.0,
            "loss/total_loss": float(np.mean(total_losses)) if total_losses else 0.0,
            "diagnostics/approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
            "diagnostics/clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
            "diagnostics/grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
            "diagnostics/adv_mean": float(rollout.advantages.mean()),
            "diagnostics/adv_std": float(rollout.advantages.std()),
            "diagnostics/return_mean": float(rollout.returns.mean()),
            "diagnostics/return_std": float(rollout.returns.std()),
            "diagnostics/value_mean": float(rollout.values.mean()),
            "diagnostics/cont_action_mean": float(rollout.cont_actions.mean()),
            "diagnostics/cont_action_std": float(rollout.cont_actions.std()),
        }

        for m in range(n_modes):
            train_metrics[f"train/mode_{m}_freq"] = float(mode_freq[m])

        wandb_log(train_metrics, step=global_step)

        progress.set_postfix(
            step=global_step,
            reward=f"{mean_recent_ep_reward:.1f}",
            eval_best=f"{best_eval_reward:.1f}" if np.isfinite(best_eval_reward) else "None",
            ploss=f"{train_metrics['loss/policy_loss']:.3f}",
            vloss=f"{train_metrics['loss/value_loss']:.3f}",
            kl=f"{train_metrics['diagnostics/approx_kl']:.4f}",
            lr=f"{train_metrics['train/lr']:.2e}",
        )

        if update % cfg.training.log_interval == 0:
            mode_msg = " | ".join([f"m{m}={mode_freq[m]:.2f}" for m in range(n_modes)])
            print(
                f"[update {update:>5d} | step {global_step:>8d}] "
                f"episodes={episode_count:>5d} | "
                f"mean_reward_100={mean_recent_ep_reward:>8.2f} | "
                f"rollout_reward={mean_rollout_reward:>8.3f} | "
                f"policy_loss={train_metrics['loss/policy_loss']:.4f} | "
                f"value_loss={train_metrics['loss/value_loss']:.4f} | "
                f"entropy={train_metrics['loss/entropy']:.4f} | "
                f"kl={train_metrics['diagnostics/approx_kl']:.5f} | "
                f"clipfrac={train_metrics['diagnostics/clipfrac']:.4f} | "
                f"{mode_msg} | "
                f"fps={fps:.0f}"
            )

        if update % cfg.training.eval_interval == 0:
            eval_rewards, eval_modes = evaluate_policy(ac, eval_env, eval_seeds, device)
            mean_eval = float(eval_rewards.mean())
            std_eval = float(eval_rewards.std())

            eval_mode_freq = np.bincount(eval_modes, minlength=n_modes)
            eval_mode_freq = eval_mode_freq / max(eval_mode_freq.sum(), 1)

            print(f"  [EVAL] mean_reward={mean_eval:.2f} +/- {std_eval:.2f}")

            eval_metrics = {
                "eval/mean_reward": mean_eval,
                "eval/std_reward": std_eval,
                "eval/min_reward": float(eval_rewards.min()),
                "eval/max_reward": float(eval_rewards.max()),
                "eval/best_mean_reward": max(best_eval_reward, mean_eval),
            }
            for m in range(n_modes):
                eval_metrics[f"eval/mode_{m}_freq"] = float(eval_mode_freq[m])

            wandb_log(eval_metrics, step=global_step)

            if mean_eval > best_eval_reward:
                best_eval_reward = mean_eval
                if cfg.training.save_best:
                    save_path = os.path.join(output_dir, "best_model.pt")
                    torch.save(
                        {
                            "model": ac.state_dict(),
                            "obs_dim": obs_dim,
                            "cont_act_dim": cont_act_dim,
                            "n_modes": n_modes,
                            "hidden_dim": hidden_dim,
                            "cfg": OmegaConf.to_container(cfg, resolve=True),
                            "eval_best_mean_reward": best_eval_reward,
                            "global_step": global_step,
                        },
                        save_path,
                    )
                    print(f"  New best! Saved to {save_path}")
                    if run is not None and cfg.logging.log_model:
                        wandb.save(save_path)

    if cfg.training.save_final:
        final_path = os.path.join(output_dir, "final_model.pt")
        torch.save(
            {
                "model": ac.state_dict(),
                "obs_dim": obs_dim,
                "cont_act_dim": cont_act_dim,
                "n_modes": n_modes,
                "hidden_dim": hidden_dim,
                "cfg": OmegaConf.to_container(cfg, resolve=True),
                "eval_best_mean_reward": best_eval_reward,
                "global_step": global_step,
            },
            final_path,
        )
        print(f"\nTraining complete. Final model: {final_path}")
        if run is not None and cfg.logging.log_model:
            wandb.save(final_path)
    else:
        print("\nTraining complete.")

    vec_env.close()
    eval_env.close()

    if run is not None:
        run.finish()


# ═══════════════════════════════════════════════════════════════════════════════
# Hydra entry
# ═══════════════════════════════════════════════════════════════════════════════

OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="train_ppo_lunar_lander",
)
def main(cfg: DictConfig):
    train_ppo(cfg)


if __name__ == "__main__":
    main()