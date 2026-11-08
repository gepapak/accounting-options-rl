"""Study T training-reward wrappers for envs_t.EdgeEnvT.

SPR (sign-paired reward; identical in construction to Study S's spr_s.py): before each step the simulator is cloned and the clone takes
the same action with the sign of this quarter's exogenous price change reversed; the training reward is (r - r_twin) / 2 - own cost.

TMR (twin-maximum reward): the same twin query, a different estimator. Under every accounting rule of the ladder the reported reward is a
left-censored observation of the booked P&L, r = max(b, tau) with a hidden threshold tau. With the booked P&L b = sigma*g - c (g >= 0 the
hidden gross magnitude, sigma the realized gross sign, c the own cost), the twin pair gives g_hat = max(r, r_twin) + c, which equals g
whenever the favourable branch is uncensored, and the reward is  sigma*g_hat - c.  Neither wrapper reads the ledger or the rule.

TIR (twin-identified reward): TMR where the twin pair identifies g, the conservative value where it does not. Because r != r_twin holds
exactly when the favourable branch is uncensored (g > 0), g_hat = max(r, r_twin) + c is then exact; when r == r_twin both branches are
censored (or g = 0), g is only known to lie in [0, r + c], and TIR uses g_hat = 0, i.e. the reward -c (the booked expectation under sign
fairness, never an invented gain). Its error is never larger than SPR's, and it is exact on every identified step.
"""
from __future__ import annotations

import copy

import gymnasium as gym
import numpy as np


def make_spr(env):
    from envs_t import K0

    class SignPaired(gym.Wrapper):
        def step(self, action):
            twin = copy.copy(self.env)
            twin.flip_next = True
            _, r_twin, _, _, _ = twin.step(action)
            obs, r, done, trunc, info = self.env.step(action)
            return obs, 0.5 * (r - r_twin) - info["cost"] / K0 * 100.0, done, trunc, info

    return SignPaired(env)


def make_tmr(env):
    from envs_t import K0, LEV

    class TwinMax(gym.Wrapper):
        def step(self, action):
            base = self.env
            eps = base.eps_ep[base.t - base.t0]                    # realized exogenous sign of this quarter
            sigma = float(np.sign(LEV[int(action)])) * eps         # realized sign of the position's gross P&L
            twin = copy.copy(base)
            twin.flip_next = True
            _, r_twin, _, _, _ = twin.step(action)
            obs, r, done, trunc, info = base.step(action)
            c = info["cost"] / K0 * 100.0
            g_hat = max(r, r_twin) + c
            return obs, sigma * g_hat - c, done, trunc, info

    return TwinMax(env)


def make_tir(env):
    from envs_t import K0, LEV

    class TwinIdentified(gym.Wrapper):
        def step(self, action):
            base = self.env
            eps = base.eps_ep[base.t - base.t0]
            sigma = float(np.sign(LEV[int(action)])) * eps
            twin = copy.copy(base)
            twin.flip_next = True
            _, r_twin, _, _, _ = twin.step(action)
            obs, r, done, trunc, info = base.step(action)
            c = info["cost"] / K0 * 100.0
            g_hat = max(r, r_twin) + c if abs(r - r_twin) > 1e-12 else 0.0
            return obs, sigma * g_hat - c, done, trunc, info

    return TwinIdentified(env)


# ---- Part C: the same estimators for Study N's environments (envs_tn.LagEnvT) -----------------------------------------------------
def _twin_step_n(base, action):
    """Realized exogenous sign and gross sign before the step, the twin's reward and the real step."""
    from envs_tn import LEV
    t = base.t
    sigma = float(np.sign(LEV[int(action)])) * float(np.sign(base.m.spread[t]))
    twin = copy.copy(base)
    twin.flip_next = True
    _, r_twin, _, _, _ = twin.step(action)
    obs, r, term, trunc, info = base.step(action)
    return sigma, r, r_twin, obs, term, trunc, info


def make_spr_n(env):
    from envs_tn import K0

    class SignPairedN(gym.Wrapper):
        def step(self, action):
            _, r, r_twin, obs, term, trunc, info = _twin_step_n(self.env, action)
            return obs, 0.5 * (r - r_twin) - info["cost"] / K0 * 100.0, term, trunc, info

    return SignPairedN(env)


def make_tir_n(env):
    from envs_tn import K0

    class TwinIdentifiedN(gym.Wrapper):
        def step(self, action):
            sigma, r, r_twin, obs, term, trunc, info = _twin_step_n(self.env, action)
            c = info["cost"] / K0 * 100.0
            g_hat = max(r, r_twin) + c if abs(r - r_twin) > 1e-12 else 0.0
            return obs, sigma * g_hat - c, term, trunc, info

    return TwinIdentifiedN(env)
