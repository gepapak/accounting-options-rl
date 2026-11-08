"""Study V training-reward wrappers for envs_t.EdgeEnvT: rewards that need NO clone of the simulator.

Study T's sign-twin rewards (SPR, TIR) query a clone of the simulator with the exogenous sign reversed. Study V asks what can be done with
the reward channel alone plus quantities every trainer has anyway: the agent's action (hence its position and own cost), the realized
exogenous sign and the public reported equity.

RUIN(lam)  Ruin-event penalty (the obvious accounting-style patch, observable from the reported equity): the reported reward minus
           lam (percent of K0) on every step that takes the reported equity from a positive value to zero.
MV(lam)    Mean-variance (risk-adjusted) reward, the usual trading-RL remedy: the reported step reward r minus lam * r**2.
PMR        Pooled-magnitude reward (clone-free). Under Assumption C of the paper the reported reward is r = max(sigma*g - c, tau) with
           g = |q| * m the gross magnitude (m: the exogenous per-MWh move). On a favourable step (sigma = +1) the booked profit is above
           every credit threshold unless g is tiny, so (r + c) / |q| is a draw of m. The trainer pools these draws over all favourable
           steps, whatever the equity (this needs m to be exogenous: independent of the agent's equity and of the sign), and rewards
           sigma * |q| * m_hat - c  with m_hat the running mean. In a fair market the expected reward is -c, so no credit can be
           gambled for; with an edge the expected reward is the booked one up to the error of m_hat.
None of the wrappers reads the ledger, the accounting rule or the hidden threshold.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from envs_t import EdgeEnvT


class MagnitudePool:
    """Running mean of the per-MWh gross move, estimated on favourable steps; shared by the environments of one training run."""

    def __init__(self):
        self.total = 0.0
        self.n = 0

    def add(self, u: float) -> None:
        self.total += float(u)
        self.n += 1

    @property
    def mean(self) -> float:
        return self.total / self.n if self.n else 0.0


class EdgeEnvZ(EdgeEnvT):
    """Study T's edge-market sleeve in which half of the training episodes start at zero reported equity (an account that has already hit the
    floor and keeps trading). Test episodes are those of EdgeEnvT; the start equity of a test episode is always given explicitly."""

    def reset(self, *, seed=None, options=None):
        opts = dict(options or {})
        if "e0" not in opts and self.rng.random() < 0.5:
            opts["e0"] = 0.0
        return super().reset(seed=seed, options=opts)


def _position(base, action) -> float:
    from envs_t import LEV, Q1
    t = base.t
    return float(np.clip(float(LEV[int(action)]) * Q1, -base.cap[t], base.cap[t]))


def make_ruin(env, lam: float):
    class Ruin(gym.Wrapper):
        def step(self, action):
            before = self.env.reported
            obs, r, done, trunc, info = self.env.step(action)
            hit = before > 0.0 and self.env.reported <= 0.0
            return obs, r - (lam if hit else 0.0), done, trunc, info

    return Ruin(env)


def make_mv(env, lam: float):
    class MeanVariance(gym.Wrapper):
        def step(self, action):
            obs, r, done, trunc, info = self.env.step(action)
            return obs, r - lam * r * r, done, trunc, info

    return MeanVariance(env)


def make_pmr(env, pool: MagnitudePool):
    from envs_t import K0

    class Pooled(gym.Wrapper):
        def step(self, action):
            base = self.env
            q = _position(base, action)
            eps = float(base.eps_ep[base.t - base.t0])
            sigma = float(np.sign(q)) * eps                      # realized sign of the position's gross P&L (0 when flat)
            obs, r, done, trunc, info = base.step(action)
            cost = info["cost"]                                  # EUR
            x_hat = sigma * abs(q) * pool.mean - cost            # the pooled mean before this step: the step's own reward does not enter it
            if abs(q) > 0.0 and sigma > 0.0:
                pool.add((r * K0 / 100.0 + cost) / abs(q))       # a draw of the per-MWh move, from a favourable step
            return obs, float(x_hat / K0 * 100.0), done, trunc, info

    return Pooled(env)
