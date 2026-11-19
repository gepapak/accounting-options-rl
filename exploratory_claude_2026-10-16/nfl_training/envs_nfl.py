"""No-free-lunch (NFL) pilot environment (EXPLORATORY, see PLAN.md).

The Study B/O leverage sleeve (fixed sizing from K0, liquidity cap, signed leverage grid LEV) on real Danish spread magnitudes
with an independent fair sign per quarter, plus a signal z observed before acting:
    edge market     P(z_t = eps_t) = P_EDGE  (a legitimate, learnable edge)
    placebo market  z_t independent of eps_t (no policy can profit; E[booked + cost] = 0 for every policy)
Costs are fee * |q| + KAPPA * q^2, so the legitimate optimum is a moderate position in the direction of z at every equity.

Accounting: the zero floor (ZF) of Studies B/O, or the other ladder rules (see ACCOUNTING). The ledger books every loss.
Reward modes:
    reported     change in reported equity (what a floored simulator rewards)
    booked       booked P&L (every loss charged; needs the ledger)
    nfl_placebo  -(change in reported equity + own trading cost); used only in placebo environments by NFL training
All rewards are in percent of K0.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "LIABILITY_STUDY_2026-09-29"))
from envs_b import E0_TRAIN, FEE, K0, LEV, Q1, load_market  # noqa: E402
from envs import EPISODE  # noqa: E402  (GENERALIZATION_STUDY envs, put on the path by envs_b)

KAPPA = 8.0      # EUR / MWh^2 convex trading cost
P_EDGE = 0.6     # signal accuracy in the edge market
MARKET_SEED = {("edge", "train"): 7001, ("placebo", "train"): 7002, ("edge", "test"): 7003, ("placebo", "test"): 7004}
ACCOUNTING = ("ZF", "LL")


@lru_cache(maxsize=None)
def market_arrays(region: str, split: str, kind: str):
    """Signed spread, signal and liquidity cap for one market draw (cached per process)."""
    m = load_market(region, split, "RM")
    abs_s = np.abs(np.asarray(m.spread, dtype=float))
    cap = np.asarray(m.cap_mwh, dtype=float)
    rng = np.random.default_rng(MARKET_SEED[(kind, split)] + (0 if region == "DK1" else 100))
    eps = rng.choice([-1.0, 1.0], size=len(abs_s))
    if kind == "edge":
        z = np.where(rng.random(len(abs_s)) < P_EDGE, eps, -eps)
    elif kind == "placebo":
        z = rng.choice([-1.0, 1.0], size=len(abs_s))
    else:
        raise ValueError(kind)
    return abs_s * eps, z, cap


class NFLEnv(gym.Env):
    metadata: dict = {}

    def __init__(self, region: str, split: str, kind: str, mode: str, accounting: str = "ZF", seed: int = 0):
        super().__init__()
        assert mode in ("reported", "booked", "nfl_placebo") and accounting in ACCOUNTING
        self.s, self.z, self.cap = market_arrays(region, split, kind)
        self.n = len(self.s)
        self.mode, self.accounting = mode, accounting
        self.action_space = gym.spaces.Discrete(len(LEV))
        self.observation_space = gym.spaces.Box(-5.0, 5.0, shape=(2,), dtype=np.float32)
        self.rng = np.random.default_rng(seed)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        opts = options or {}
        self.t0 = int(opts["t0"]) if "t0" in opts else int(self.rng.integers(200, self.n - EPISODE))
        self.t, self.t_end = self.t0, self.t0 + EPISODE
        e0 = opts.get("e0")
        if e0 is None:
            e0 = float(np.exp(self.rng.uniform(np.log(E0_TRAIN[0]), np.log(E0_TRAIN[1]))))
        self.e0 = K0 * float(e0)
        self.ledger = self.reported = self.e0
        self.closed = False
        self.forgiven = self.cost_sum = 0.0
        self.floor_hits = 0
        return self._obs(), {}

    def _obs(self):
        return np.array([min(self.reported / K0, 5.0), self.z[self.t]], dtype=np.float32)

    def step(self, action):
        t = self.t
        lev = 0.0 if self.closed else float(LEV[int(action)])
        q = float(np.clip(lev * Q1, -self.cap[t], self.cap[t]))
        gross = q * self.s[t]
        cost = FEE * abs(q) + KAPPA * q * q
        pnl = gross - cost
        self.ledger += pnl
        before = self.reported
        new = before + pnl
        forg = max(0.0, -new)
        if forg > 0.0:
            self.floor_hits += 1
            if self.accounting == "LL":
                self.closed = True
        self.reported = max(new, 0.0)
        self.forgiven += forg
        self.cost_sum += cost
        d_rep = self.reported - before
        if self.mode == "reported":
            r = d_rep
        elif self.mode == "booked":
            r = pnl
        else:
            r = -(d_rep + cost)
        self.t += 1
        done = self.t >= self.t_end
        obs = self._obs() if not done else np.zeros(2, np.float32)
        return obs, float(r / K0 * 100.0), done, False, {"pnl": pnl, "cost": cost, "forgiven": forg}
