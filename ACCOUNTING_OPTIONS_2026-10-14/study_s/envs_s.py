"""Study S environments (PREREGISTRATION_S.md).

Part A, the edge-market testbed (frozen copy of the exploratory v2 design, exploratory_claude_2026-10-16/nfl_training/envs_nfl2.py):
real Danish spread magnitudes (Study A panels) with an independent fair sign per quarter, a signal z observed before acting with
P(z = eps) = P_EDGE, the Study B sleeve (fixed sizing from K0, liquidity cap, signed leverage grid LEV), costs fee * |q| + KAPPA * q^2,
and the zero floor (reported equity max(e + pnl, 0), trading continues). The ledger books every loss.
    reward modes: reported (pnl + forgiven), booked (pnl)
    observation: (reported equity / K0 clipped to [-5, 5], z, share of the episode remaining, 0)   [the 4th slot is unused under ZF]
`flip_next = True` makes the next step use the opposite sign of its exogenous price change (the counterfactual used by SPR).
The edge-market draw (signs and signal) is the one fixed by MARKET_SEED, the same draw as the exploratory pilot.

Part B uses the registered Study O environment unchanged (study_o/envs_o.PartialEnv).
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import gymnasium as gym
import numpy as np

LIAB = Path(__file__).resolve().parents[2] / "LIABILITY_STUDY_2026-09-29"
sys.path.insert(0, str(LIAB))
from envs_b import E0_TRAIN, FEE, K0, LEV, Q1, load_market  # noqa: E402
from envs import EPISODE  # noqa: E402  (Study A envs, on the path through envs_b)

P_EDGE = 0.6
KAPPA = 4.0
MARKET_SEED = {("edge", "train"): 7101, ("edge", "test"): 7103, ("placebo", "train"): 7102, ("placebo", "test"): 7104}


@lru_cache(maxsize=None)
def market_arrays(region: str, split: str, kind: str):
    m = load_market(region, split, "RM")
    abs_s = np.abs(np.asarray(m.spread, dtype=float))
    cap = np.asarray(m.cap_mwh, dtype=float)
    rng = np.random.default_rng(MARKET_SEED[(kind, split)] + (0 if region == "DK1" else 100))
    eps = rng.choice([-1.0, 1.0], size=len(abs_s))
    z = np.where(rng.random(len(abs_s)) < P_EDGE, eps, -eps)
    return abs_s, eps, z, cap


class EdgeEnv(gym.Env):
    """Zero-floor sleeve in the edge market (kind 'edge') or the placebo market (kind 'placebo': fresh fair signs and an
    independent signal every episode)."""
    metadata: dict = {}

    def __init__(self, region: str, split: str, kind: str, mode: str, seed: int = 0):
        super().__init__()
        assert mode in ("reported", "booked") and kind in ("edge", "placebo")
        self.abs_s, self.eps, self.zz, self.cap = market_arrays(region, split, "edge" if kind == "edge" else "placebo")
        self.n = len(self.abs_s)
        self.kind, self.mode = kind, mode
        self.action_space = gym.spaces.Discrete(len(LEV))
        self.observation_space = gym.spaces.Box(-5.0, 5.0, shape=(4,), dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.flip_next = False

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        opts = options or {}
        self.t0 = int(opts["t0"]) if "t0" in opts else int(self.rng.integers(200, self.n - EPISODE))
        self.t, self.t_end = self.t0, self.t0 + EPISODE
        if self.kind == "placebo":
            self.eps_ep = self.rng.choice([-1.0, 1.0], size=EPISODE)
            self.z_ep = self.rng.choice([-1.0, 1.0], size=EPISODE)
        else:
            self.eps_ep = self.eps[self.t0:self.t_end]
            self.z_ep = self.zz[self.t0:self.t_end]
        e0 = opts.get("e0")
        if e0 is None:
            e0 = float(np.exp(self.rng.uniform(np.log(E0_TRAIN[0]), np.log(E0_TRAIN[1]))))
        self.e0 = K0 * float(e0)
        self.ledger = self.reported = self.e0
        self.forgiven = 0.0
        self.flip_next = False
        return self._obs(), {}

    def _obs(self):
        rem = (self.t_end - self.t) / EPISODE
        return np.array([np.clip(self.reported / K0, -5.0, 5.0), self.z_ep[self.t - self.t0], rem, 0.0], dtype=np.float32)

    def step(self, action):
        t = self.t
        q = float(np.clip(float(LEV[int(action)]) * Q1, -self.cap[t], self.cap[t]))
        cost = FEE * abs(q) + KAPPA * q * q
        sign = self.eps_ep[t - self.t0] * (-1.0 if self.flip_next else 1.0)
        self.flip_next = False
        pnl = q * self.abs_s[t] * sign - cost
        self.ledger += pnl
        new = self.reported + pnl
        f = max(0.0, -new)
        self.reported = max(new, 0.0)
        self.forgiven += f
        r = pnl + f if self.mode == "reported" else pnl
        self.t += 1
        done = self.t >= self.t_end
        obs = self._obs() if not done else np.zeros(4, np.float32)
        return obs, float(r / K0 * 100.0), done, False, {"pnl": pnl, "cost": cost, "forgiven": f}
