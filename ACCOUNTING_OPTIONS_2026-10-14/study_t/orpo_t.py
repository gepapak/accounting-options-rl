"""Study T baseline: occupancy-measure regularized PPO with the sqrt-chi^2 divergence (ORPO), after Laidlaw, Singhal and Dragan,
"Correlated Proxies: A New Definition and Improved Mitigation for Reward Hacking" (ICLR 2025).

Implemented from the paper's main-text formulation and the semantics of the authors' public code (occupancy_measures/agents/orpo.py):
  * a discriminator d(s, a) is trained every PPO iteration, on that iteration's fresh rollout of the current policy (positive class)
    against fresh rollouts of the reference ("safe") policy (negative class), with the logistic loss
        E_pi[softplus(-d)] + E_ref[softplus(d)],
    so that d approximates log(mu_pi / mu_ref);
  * the divergence of each sample is chi2 = exp(d) - 1; for "sqrt_chi2" it is divided by sqrt(m), m the 1%-trimmed mean of chi2 over
    the iteration's samples (chi2 itself if m <= 0), and the reward penalty is  - lambda * distance  with lambda = c * sigma_ref,
    where sigma_ref is the standard deviation of the per-step reward under the reference policy (the repository's "scale-independent
    coefficient" c);
  * the discriminator is trained BEFORE the penalty is computed, the penalty is added to the iteration's rewards, and returns and
    advantages are recomputed (as the authors' code does: discriminator trained first, then rewards augmented).
Our implementation differs from the authors' in the framework (Stable-Baselines3 PPO instead of RLlib), the discriminator (a 13-64-64-1
tanh MLP on observation and one-hot action, Adam 3e-4, minibatch 256, 10 epochs per iteration) and in clipping d to [-20, 20] before
exp. The reward penalty is added on the scale of the learner's normalized reward (VecNormalize), by dividing by the running return std.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import DummyVecEnv

IDX_LONG1, IDX_SHORT1 = 6, 2          # LEV[6] = +1, LEV[2] = -1
N_ACTIONS = 9


def ref_actions(obs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Reference ("safe") policy: leverage 1 in the direction of the signal with probability 0.7, the other eight actions 0.0375 each."""
    target = np.where(obs[:, 1] > 0, IDX_LONG1, IDX_SHORT1)
    other = rng.integers(0, N_ACTIONS - 1, size=len(obs))
    other = other + (other >= target)                       # uniform over the eight non-target actions
    return np.where(rng.random(len(obs)) < 0.7, target, other)


class Discriminator(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim + N_ACTIONS, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, F.one_hot(act.long(), N_ACTIONS).float()], dim=-1)).squeeze(-1)


def trimmed_mean(x: np.ndarray, frac: float = 0.01) -> float:
    xs = np.sort(x)
    k = int(frac * len(xs))
    return float(xs[k:len(xs) - k].mean()) if len(xs) > 2 * k else float(xs.mean())


class ORPOPPO(PPO):
    """PPO with an occupancy-measure (sqrt-chi^2) penalty. Call `configure_orpo` after construction."""

    def configure_orpo(self, ref_env_fns, coeff_c: float, seed: int, disc_epochs: int = 10, disc_batch: int = 256):
        self.om_c = float(coeff_c)
        self.om_rng = np.random.default_rng(seed + 777)
        self.om_ref_env = DummyVecEnv(ref_env_fns)
        self.om_ref_obs = self.om_ref_env.reset()
        obs_dim = self.observation_space.shape[0]
        torch.manual_seed(seed + 555)
        self.om_disc = Discriminator(obs_dim)
        self.om_opt = torch.optim.Adam(self.om_disc.parameters(), lr=3e-4)
        self.om_epochs, self.om_batch = disc_epochs, disc_batch
        self.om_sigma = None
        self.om_log = []

    def _reference_rollout(self, n_steps: int):
        obs_l, act_l, rew_l = [], [], []
        obs = self.om_ref_obs
        for _ in range(n_steps):
            a = ref_actions(obs, self.om_rng)
            obs_l.append(obs.copy())
            act_l.append(a)
            obs, r, _, _ = self.om_ref_env.step(a)           # DummyVecEnv resets finished episodes automatically
            rew_l.append(r)
        self.om_ref_obs = obs
        return (np.concatenate(obs_l).astype(np.float32), np.concatenate(act_l).astype(np.int64), np.concatenate(rew_l))

    def _train_disc(self, cur_obs, cur_act, ref_obs, ref_act):
        n = min(len(cur_obs), len(ref_obs))
        co, ca = torch.as_tensor(cur_obs), torch.as_tensor(cur_act)
        ro, ra = torch.as_tensor(ref_obs), torch.as_tensor(ref_act)
        acc = 0.0
        for _ in range(self.om_epochs):
            perm_c, perm_r = torch.randperm(len(co))[:n], torch.randperm(len(ro))[:n]
            for i in range(0, n, self.om_batch):
                ic, ir = perm_c[i:i + self.om_batch], perm_r[i:i + self.om_batch]
                dc, dr = self.om_disc(co[ic], ca[ic]), self.om_disc(ro[ir], ra[ir])
                loss = F.softplus(-dc).mean() + F.softplus(dr).mean()
                self.om_opt.zero_grad()
                loss.backward()
                self.om_opt.step()
        with torch.no_grad():
            acc = 0.5 * (float((self.om_disc(co, ca) > 0).float().mean()) + float((self.om_disc(ro, ra) <= 0).float().mean()))
        return acc

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        ok = super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)
        if not ok:
            return False
        n_steps, n_envs = rollout_buffer.buffer_size, rollout_buffer.n_envs
        cur_obs = rollout_buffer.observations.reshape(n_steps * n_envs, -1).astype(np.float32)
        cur_act = rollout_buffer.actions.reshape(n_steps * n_envs).astype(np.int64)
        ref_obs, ref_act, ref_rew = self._reference_rollout(n_steps)
        if self.om_sigma is None:
            self.om_sigma = float(np.std(ref_rew))               # fixed after the first iteration: the reference policy's reward std
        acc = self._train_disc(cur_obs, cur_act, ref_obs, ref_act)   # discriminator first, then the penalty
        with torch.no_grad():
            d = self.om_disc(torch.as_tensor(cur_obs), torch.as_tensor(cur_act)).clamp(-20.0, 20.0).numpy().astype(np.float64)
        chi2 = np.exp(d) - 1.0
        m = trimmed_mean(chi2)
        dist = chi2 / np.sqrt(m) if m > 0 else chi2
        penalty_raw = -self.om_c * self.om_sigma * dist                       # raw reward units (percent of K0 per step)
        scale = 1.0 / np.sqrt(env.ret_rms.var + env.epsilon)                  # learner works on VecNormalize'd rewards
        rollout_buffer.rewards += (penalty_raw * scale).reshape(n_steps, n_envs).astype(np.float32)
        with torch.no_grad():
            values = self.policy.predict_values(obs_as_tensor(self._last_obs, self.device))
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=self._last_episode_starts)
        self.om_log.append((float(m), float(np.mean(penalty_raw)), float(acc)))
        return True
