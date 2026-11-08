"""Study Z training-reward wrappers for envs_z (PREREGISTRATION_Z.md).

RUINP(lam)  Study V's ruin penalty measured on the PORTFOLIO: the reported reward minus lam (percent of K0) on every step that takes the
            portfolio's reported equity from a positive value to zero. It reads the portfolio's reported equity only, as a trainer that sees
            the portfolio but not its account structure would.
RUINA(lam)  The same penalty measured per ACCOUNT: lam for every account whose reported equity goes from a positive value to zero (it needs
            the account structure).
SPR, TIR    The sign-twin rewards of Studies S to Y applied to the PORTFOLIO reward channel: the clone takes the same action with the sign of
            the quarter's exogenous price change reversed for both legs. SPR = (r - r_twin) / 2 - own cost. TIR uses the realized sign of the
            portfolio's gross P&L, sign(q_A + q_B) * eps, and the single-floor formula on the portfolio pair. Neither reads the ledger, the
            accounting rule or the account structure.
"""
from __future__ import annotations

import copy

import gymnasium as gym
import numpy as np

from envs_z import K0


def make_ruin_portfolio(env, lam: float):
    class RuinPortfolio(gym.Wrapper):
        def step(self, action):
            obs, r, done, trunc, info = self.env.step(action)
            return obs, r - lam * info["ruin_portfolio"], done, trunc, info

    return RuinPortfolio(env)


def make_ruin_accounts(env, lam: float):
    class RuinAccounts(gym.Wrapper):
        def step(self, action):
            obs, r, done, trunc, info = self.env.step(action)
            return obs, r - lam * info["ruin_accounts"], done, trunc, info

    return RuinAccounts(env)


def twin_pair(base, action):
    """The sign twin's reward (from a clone), then the real step. Returns (r_twin, sigma, step results)."""
    qa, qb, _, _ = base.legs(action)
    eps = float(base.eps_ep[base.t - base.t0])
    sigma = float(np.sign(qa + qb)) * eps                       # realized sign of the portfolio's gross P&L (0 for a flat or perfect hedge)
    twin = copy.copy(base)
    twin.flip_next = True
    r_twin = twin.step(action)[1]
    return r_twin, sigma, base.step(action)


def make_spr(env):
    class SignPaired(gym.Wrapper):
        def step(self, action):
            r_twin, _, (obs, r, done, trunc, info) = twin_pair(self.env, action)
            return obs, 0.5 * (r - r_twin) - info["cost"] / K0 * 100.0, done, trunc, info

    return SignPaired(env)


def make_tir(env):
    class TwinIdentified(gym.Wrapper):
        def step(self, action):
            r_twin, sigma, (obs, r, done, trunc, info) = twin_pair(self.env, action)
            c = info["cost"] / K0 * 100.0
            g_hat = max(r, r_twin) + c if abs(r - r_twin) > 1e-12 else 0.0
            return obs, sigma * g_hat - c, done, trunc, info

    return TwinIdentified(env)
