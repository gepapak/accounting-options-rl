"""NFL v2 environment (EXPLORATORY, PLAN.md "Pilot v2"): the pilot testbed under four accounting rules of the ladder.

Market: real Danish spread magnitudes, fair sign; signal z with P(z = eps) = P_EDGE in the edge market. In the placebo market the
signs and the signal are redrawn independently every episode. Costs: fee * |q| + KAPPA * q^2.

The ledger books every loss. The reported reward is booked P&L plus any forgiven deficit, r_t = pnl_t + f_t, where
    ZF    zero floor: f_t = max(0, -(e_t + pnl_t)) every quarter (reported equity floored at zero), trading continues
    LL    limited liability: the same f_t when equity would fall to zero or below, then the account closes
    TERM  terminal limited liability: f_t = 0 until the last quarter, where the remaining deficit max(0, -ledger_T) is forgiven
    LAG   settlement lag: insolvency is detected W quarters after it starts; the deficit at detection is forgiven and the
          account closes (if equity has recovered by the detection quarter, nothing is forgiven and the clock resets)
Observation: (reported equity / K0 clipped to [-5, 5], signal z, share of the episode remaining, insolvency clock / W).
Reward modes: reported (pnl + f), booked (pnl), nfl_placebo = -(pnl + f + own cost). Percent of K0.
An account that closes ends the episode (as in Study B's training environment).
`flip_next = True` makes the next step use the opposite sign of its exogenous price change (the one-step counterfactual used by
sign-paired reward); it resets after that step.
"""
from __future__ import annotations

from functools import lru_cache

import gymnasium as gym
import numpy as np

from envs_nfl import E0_TRAIN, EPISODE, FEE, K0, LEV, Q1  # noqa: F401
from envs_b import load_market

P_EDGE = 0.6
KAPPA = 4.0
ACCOUNTING = ("ZF", "LL", "TERM", "LAG")
W_LAG = 96   # one day of quarter-hours
SEED = {("edge", "train"): 7101, ("edge", "test"): 7103, ("placebo", "train"): 7102, ("placebo", "test"): 7104}


@lru_cache(maxsize=None)
def market_arrays(region: str, split: str, kind: str):
    m = load_market(region, split, "RM")
    abs_s = np.abs(np.asarray(m.spread, dtype=float))
    cap = np.asarray(m.cap_mwh, dtype=float)
    rng = np.random.default_rng(SEED[(kind, split)] + (0 if region == "DK1" else 100))
    eps = rng.choice([-1.0, 1.0], size=len(abs_s))
    z = np.where(rng.random(len(abs_s)) < P_EDGE, eps, -eps)
    return abs_s, eps, z, cap


class NFLEnv2(gym.Env):
    metadata: dict = {}

    def __init__(self, region: str, split: str, kind: str, mode: str, accounting: str, seed: int = 0, p_insolvent: float = 0.0):
        super().__init__()
        assert mode in ("reported", "booked", "nfl_placebo") and accounting in ACCOUNTING and kind in ("edge", "placebo")
        assert p_insolvent == 0.0 or accounting in ("TERM", "LAG"), "insolvent starts only where a deficit can persist"
        self.p_insolvent = float(p_insolvent)   # stage 3: share of episodes starting insolvent, equity U[-0.5, 0) K0
        self.abs_s, self.eps, self.zz, self.cap = market_arrays(region, split, kind)
        self.n = len(self.abs_s)
        self.kind, self.mode, self.acc = kind, mode, accounting
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
            if self.p_insolvent > 0.0 and self.rng.random() < self.p_insolvent:
                e0 = -float(self.rng.uniform(0.0, 0.5))
            else:
                e0 = float(np.exp(self.rng.uniform(np.log(E0_TRAIN[0]), np.log(E0_TRAIN[1]))))
        self.e0 = K0 * float(e0)
        self.ledger = self.reported = self.e0
        self.closed = False
        self.sigma = self.t0 if (self.acc == "LAG" and self.ledger < 0.0) else None
        self.forgiven = self.cost_sum = 0.0
        self.flip_next = False
        return self._obs(), {}

    def _obs(self):
        clock = 0.0 if self.sigma is None else (self.t - self.sigma) / W_LAG
        rem = (self.t_end - self.t) / EPISODE
        return np.array([np.clip(self.reported / K0, -5.0, 5.0), self.z_ep[self.t - self.t0], rem, clock], dtype=np.float32)

    def step(self, action):
        t = self.t
        lev = float(LEV[int(action)])
        q = float(np.clip(lev * Q1, -self.cap[t], self.cap[t]))
        cost = FEE * abs(q) + KAPPA * q * q
        sign = self.eps_ep[t - self.t0] * (-1.0 if self.flip_next else 1.0)
        self.flip_next = False
        pnl = q * self.abs_s[t] * sign - cost
        self.ledger += pnl
        last = t + 1 >= self.t_end
        f = 0.0
        if self.acc in ("ZF", "LL"):
            new = self.reported + pnl
            f = max(0.0, -new)
            self.reported = max(new, 0.0)
            if self.acc == "LL" and new <= 0.0:
                self.closed = True
        else:
            self.reported = self.ledger
            if self.acc == "LAG":
                if self.ledger < 0.0 and self.sigma is None:
                    self.sigma = t
                if self.sigma is not None and t + 1 - self.sigma >= W_LAG:
                    if self.ledger < 0.0:
                        f = -self.ledger
                        self.closed = True
                    self.sigma = None
            elif last and self.ledger < 0.0:          # TERM
                f = -self.ledger
        self.forgiven += f
        self.cost_sum += cost
        if self.mode == "reported":
            r = pnl + f
        elif self.mode == "booked":
            r = pnl
        else:
            r = -(pnl + f + cost)
        self.t += 1
        done = last or self.closed
        obs = self._obs() if not done else np.zeros(4, np.float32)
        return obs, float(r / K0 * 100.0), done, False, {"pnl": pnl, "cost": cost, "forgiven": f}
