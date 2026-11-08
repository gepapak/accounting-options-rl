"""Study FT training rewards (PREREGISTRATION_FT.md). None reads the ledger or the accounting rule.

SPR, TIR   The sign-twin rewards of Studies S to Z: one clone with the sign reversed; SPR = (r - r_twin)/2 - C; TIR = sigma*(max(r, r_twin) + C) - C if the pair differs, else -C.
TSR        Tail-slope reward. Four clones on the line through the realized move, at t = +T, +2T, -T, -2T. The tail slopes are a_plus = (phi(2T) - phi(T))/T and
           a_minus = (phi(-T) - phi(-2T))/T. Gross estimate: a_plus + a_minus if they have opposite signs (accounts of opposite exposure, each floored), else the larger in
           magnitude (one exposure, or an unfloored part left on the other tail). TSR = estimate - C. It needs control of the input's size as well as its sign, and the own cost.
"""
from __future__ import annotations

import copy

import gymnasium as gym

from envs_ft import K0

T_FAR = 1e9


def tail_gross(ap: float, am: float) -> float:
    if ap * am < 0.0:
        return ap + am
    return ap if abs(ap) >= abs(am) else am


def line_reward(base, action, t: float) -> float:
    c = copy.copy(base)
    c.move_scale = t
    return c.step(action)[1]


def tail_estimate(base, action, T: float = T_FAR) -> float:
    pp, p2, mm, m2 = (line_reward(base, action, x) for x in (T, 2.0 * T, -T, -2.0 * T))
    return tail_gross((p2 - pp) / T, (mm - m2) / T)


def twin_reward(base, action) -> float:
    c = copy.copy(base)
    c.flip_next = True
    return c.step(action)[1]


def make_spr(env):
    class SignPaired(gym.Wrapper):
        def step(self, action):
            r_twin = twin_reward(self.env, action)
            obs, r, done, trunc, info = self.env.step(action)
            return obs, 0.5 * (r - r_twin) - info["cost"] / K0 * 100.0, done, trunc, info

    return SignPaired(env)


def make_tir(env):
    class TwinIdentified(gym.Wrapper):
        def step(self, action):
            sigma = self.env.realized_sign(action)
            r_twin = twin_reward(self.env, action)
            obs, r, done, trunc, info = self.env.step(action)
            c = info["cost"] / K0 * 100.0
            g_hat = max(r, r_twin) + c if abs(r - r_twin) > 1e-12 else 0.0
            return obs, sigma * g_hat - c, done, trunc, info

    return TwinIdentified(env)


def make_tsr(env):
    class TailSlope(gym.Wrapper):
        def step(self, action):
            g_hat = tail_estimate(self.env, action)
            obs, r, done, trunc, info = self.env.step(action)
            return obs, g_hat - info["cost"] / K0 * 100.0, done, trunc, info

    return TailSlope(env)
