"""Study S training-reward wrappers (PREREGISTRATION_S.md).

SPR (sign-paired reward): before each step the simulator is cloned and the clone takes the same action with the sign of this
quarter's exogenous price change reversed; the training reward is (r - r_twin) / 2 - own trading cost. The trainer uses the reward
channel, a clone, a sign override of the exogenous input and its own cost schedule; it never reads the ledger or the accounting rule.

KL (baseline): reported reward minus beta * [log pi(a|s) - log pi_ref(a|s)], a per-step KL penalty toward a safe reference policy
(leverage 1 in the direction of the signal with probability 0.7; the other eight actions share 0.3).
"""
from __future__ import annotations

import copy

import gymnasium as gym
import numpy as np

IDX_LONG1, IDX_SHORT1 = 6, 2          # LEV[6] = +1, LEV[2] = -1


def make_spr_edge(env):
    """SPR for envs_s.EdgeEnv (the testbed exposes the one-step sign override `flip_next`)."""
    from envs_s import K0

    class SignPairedEdge(gym.Wrapper):
        def step(self, action):
            twin = copy.copy(self.env)
            twin.flip_next = True
            _, r_twin, _, _, _ = twin.step(action)
            obs, r, done, trunc, info = self.env.step(action)
            return obs, 0.5 * (r - r_twin) - info["cost"] / K0 * 100.0, done, trunc, info

    return SignPairedEdge(env)


class _FlipOne:
    """Array view with element t sign-reversed (the one-step counterfactual exogenous input)."""

    def __init__(self, a, t):
        self.a, self.t = a, t

    def __getitem__(self, i):
        return -self.a[i] if i == self.t else self.a[i]


class _FlippedMarket:
    def __init__(self, m, t):
        self._m, self.spread = m, _FlipOne(m.spread, t)

    def __getattr__(self, name):
        return getattr(self._m, name)


def make_spr_fair(env):
    """SPR for study_o/envs_o.PartialEnv: the clone reads the market through a view with this quarter's spread sign reversed;
    the own fee is the agent's fee schedule applied to its position."""
    from envs_b import FEE, K0, LEV, Q1

    class SignPairedFair(gym.Wrapper):
        def step(self, action):
            base = self.env.unwrapped
            t = base.t
            q = 0.0 if base.closed else float(np.clip(float(LEV[int(action)]) * Q1, -base.m.cap_mwh[t], base.m.cap_mwh[t]))
            fee = FEE * abs(q) / K0 * 100.0
            twin = copy.copy(base)
            twin.m = _FlippedMarket(base.m, t)
            _, r_twin, _, _, _ = twin.step(action)
            obs, r, done, trunc, info = self.env.step(action)
            return obs, 0.5 * (r - r_twin) - fee, done, trunc, info

    return SignPairedFair(env)


def ref_probs(obs, actions):
    target = np.where(obs[:, 1] > 0, IDX_LONG1, IDX_SHORT1)
    return np.where(actions == target, 0.7, 0.3 / 8)


def make_kl_wrapper():
    import torch
    from stable_baselines3.common.vec_env import VecEnvWrapper

    class KLPenalty(VecEnvWrapper):
        def __init__(self, venv, beta):
            super().__init__(venv)
            self.beta, self.model, self.last_obs, self.actions = beta, None, None, None

        def reset(self):
            self.last_obs = self.venv.reset()
            return self.last_obs

        def step_async(self, actions):
            self.actions = np.asarray(actions)
            self.venv.step_async(actions)

        def step_wait(self):
            obs, rew, done, info = self.venv.step_wait()
            if self.model is not None:
                with torch.no_grad():
                    dist = self.model.policy.get_distribution(torch.as_tensor(self.last_obs, dtype=torch.float32))
                    logp = dist.log_prob(torch.as_tensor(self.actions)).numpy()
                rew = rew - self.beta * (logp - np.log(ref_probs(self.last_obs, self.actions)))
            self.last_obs = obs
            return obs, rew, done, info

    return KLPenalty
