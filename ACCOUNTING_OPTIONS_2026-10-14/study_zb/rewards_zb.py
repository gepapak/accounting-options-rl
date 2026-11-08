"""Study Z, Part B training-reward wrappers for envs_zb.IsolatedEnv (PREREGISTRATION_ZB.md).

RUINP(lam)  Study V's ruin penalty on the account's (portfolio's) reported equity: minus lam (percent of K0) on every step that takes it from a
            positive value to zero. It reads the reported equity only.
LIQ(lam)    minus lam on every liquidation of an isolated position (a loss beyond its margin). It needs the venue's margin rule, which a trainer
            of a black-box simulator does not have; it is included as the structure-aware remedy.
SPR, TIR    the sign-twin rewards from the reward channel, a sign-reversed clone and the own cost (Studies S to Z); TIR uses the realized sign
            of the position's gross P&L. Neither reads the ledger, the margin mode's meaning or the margin rule.
FINDER      a ledger-free exploit finder: the reward is the width of the twin bracket, r + r_twin + 2 * own cost, which for any forgiving
            accounting is the credit paid on the pair (Proposition 10). It reads what SPR reads and nothing else; an agent trained on it
            searches the simulator for forgiveness.
"""
from __future__ import annotations

import copy

import gymnasium as gym
import numpy as np

from envs_zb import K0


def make_ruin_portfolio(env, lam: float):
    class RuinPortfolio(gym.Wrapper):
        def step(self, action):
            obs, r, done, trunc, info = self.env.step(action)
            return obs, r - lam * info["ruin_portfolio"], done, trunc, info

    return RuinPortfolio(env)


def make_liquidation(env, lam: float):
    class Liquidation(gym.Wrapper):
        def step(self, action):
            obs, r, done, trunc, info = self.env.step(action)
            return obs, r - lam * info["liquidation"], done, trunc, info

    return Liquidation(env)


def twin_pair(base, action):
    q, _, _ = base.position(action)
    sigma = float(np.sign(q)) * float(base.eps_ep[base.t - base.t0])
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


def make_finder(env):
    class Finder(gym.Wrapper):
        def step(self, action):
            r_twin, _, (obs, r, done, trunc, info) = twin_pair(self.env, action)
            return obs, r + r_twin + 2.0 * info["cost"] / K0 * 100.0, done, trunc, info

    return Finder(env)


def make_tir(env):
    class TwinIdentified(gym.Wrapper):
        def step(self, action):
            r_twin, sigma, (obs, r, done, trunc, info) = twin_pair(self.env, action)
            c = info["cost"] / K0 * 100.0
            g_hat = max(r, r_twin) + c if abs(r - r_twin) > 1e-12 else 0.0
            return obs, sigma * g_hat - c, done, trunc, info

    return TwinIdentified(env)
