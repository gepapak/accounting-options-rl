"""Study FT environments (PREREGISTRATION_FT.md): three accountings that a sign query cannot read, on the Study T sleeve's market, costs and sizing.

Every environment accepts a point on the line through the realized move: `move_scale = t` makes the step use t times its exogenous move, so that t = 1 is the real
step, t = -1 the sign twin, t = 0 no move and large |t| the far tails. `flip_next = True` is the sign twin of Studies S to Z. Own costs do not depend on the move.

BucketEnv    Unrealized bucket, as the engine's percent payoff. The gross goes to an unfloored bucket; the second action chooses the fraction realized at the end
             of the step, which moves to cash, floored at zero; the cost is written to cash next, floored at zero. Reported equity = cash + bucket.
DeficitEnv   Carried deficit, as an owner's claim. The ledger carries every loss and the reported equity is max(ledger, 0). A step that leaves the ledger below zero reports nothing.
PartialEnv   Partial forgiveness: of each new loss below zero equity the share lam is written off and the rest is carried. lam = 0 is the plain ledger with a negative balance;
             lam = 1 is the zero floor.

The ledger books every gross and every cost in all three. The reported reward is the change of reported equity. Rewards are percent of K0, as in Studies T to Z.
"""
from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "study_t"), str(HERE.parents[1] / "LIABILITY_STUDY_2026-09-29")]
from envs_t import E0_TRAIN, EPISODE, FEE, K0, LEV, Q1, SETTINGS, market_arrays  # noqa: E402

REALIZE = (0.0, 0.5, 1.0)
NLEV = len(LEV)
SETTING = "S_DK1"


class _Base(gym.Env):
    metadata: dict = {}

    def __init__(self, split: str, kind: str, mode: str, seed: int = 0, setting: str = SETTING):
        super().__init__()
        assert mode in ("reported", "booked") and kind in ("edge", "placebo") and setting in SETTINGS
        self.abs_s, self.eps, self.zz, self.cap = market_arrays(setting, split, "edge" if kind == "edge" else "placebo")
        self.kappa = SETTINGS[setting]["kappa"]
        self.n = len(self.abs_s)
        self.kind, self.mode = kind, mode
        self.action_space = gym.spaces.Discrete(NLEV)
        self.observation_space = gym.spaces.Box(-5.0, 5.0, shape=(4,), dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.flip_next = False
        self.move_scale = 1.0

    def _start_window(self, opts):
        self.t0 = int(opts["t0"]) if "t0" in opts else int(self.rng.integers(200, self.n - EPISODE))
        self.t, self.t_end = self.t0, self.t0 + EPISODE
        if self.kind == "placebo":
            self.eps_ep = self.rng.choice([-1.0, 1.0], size=EPISODE)
            self.z_ep = self.rng.choice([-1.0, 1.0], size=EPISODE)
        else:
            self.eps_ep = self.eps[self.t0:self.t_end]
            self.z_ep = self.zz[self.t0:self.t_end]
        self.flip_next = False
        self.move_scale = 1.0

    def _draw_e0(self):
        return float(np.exp(self.rng.uniform(np.log(E0_TRAIN[0]), np.log(E0_TRAIN[1]))))

    def leverage(self, action) -> float:
        a = np.asarray(action).ravel()
        return float(LEV[int(a[0])])

    def position(self, action) -> float:
        return float(np.clip(self.leverage(action) * Q1, -self.cap[self.t], self.cap[self.t]))

    def _move(self, q):
        t = self.t
        sign = self.eps_ep[t - self.t0] * (-1.0 if self.flip_next else 1.0)
        self.flip_next = False
        return q * self.abs_s[t] * sign * self.move_scale

    def realized_sign(self, action) -> float:
        """The realized sign of the position's gross P&L (used by TIR only)."""
        return float(np.sign(self.position(action))) * float(self.eps_ep[self.t - self.t0])

    def _advance(self):
        self.t += 1
        done = self.t >= self.t_end
        return (self._obs() if not done else np.zeros(4, np.float32)), done


class BucketEnv(_Base):
    def __init__(self, split, kind, mode, seed=0, setting=SETTING, zero_starts=False):
        super().__init__(split, kind, mode, seed, setting)
        self.action_space = gym.spaces.MultiDiscrete([NLEV, len(REALIZE)])
        self.zero_starts = zero_starts

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        opts = dict(options or {})
        self._start_window(opts)
        if "cash0" in opts:
            cash, bucket = float(opts["cash0"]), float(opts.get("bucket0", 0.0))
        elif "e0" in opts:
            cash, bucket = float(opts["e0"]), 0.0
        else:
            u = self.rng.random() if self.zero_starts else 1.0
            if u < 0.25:
                cash, bucket = 0.0, 0.0
            elif u < 0.5:
                cash, bucket = 0.0, -self._draw_e0()
            else:
                cash, bucket = self._draw_e0(), 0.0
        self.cash, self.bucket = K0 * cash, K0 * bucket
        self.e0 = self.cash + self.bucket
        self.ledger = self.e0
        self.forgiven = 0.0
        return self._obs(), {}

    @property
    def reported(self):
        return self.cash + self.bucket

    def _obs(self):
        rem = (self.t_end - self.t) / EPISODE
        return np.array([np.clip(self.reported / K0, -5.0, 5.0), self.z_ep[self.t - self.t0], rem, np.clip(self.bucket / K0, -5.0, 5.0)], dtype=np.float32)

    def step(self, action):
        a = np.asarray(action).ravel()
        q = self.position(a)
        f = REALIZE[int(a[1])]
        cost = FEE * abs(q) + self.kappa * q * q
        gross = self._move(q)
        pre = self.reported
        self.bucket += gross
        realized = f * self.bucket
        self.cash += realized
        self.bucket -= realized
        self.cash = max(self.cash, 0.0)
        self.cash = max(self.cash - cost, 0.0)
        pnl = gross - cost
        self.ledger += pnl
        rep = self.reported - pre
        forgiven = rep - pnl
        self.forgiven += forgiven
        r = rep if self.mode == "reported" else pnl
        obs, done = self._advance()
        return obs, float(r / K0 * 100.0), done, False, {"pnl": pnl, "cost": cost, "forgiven": forgiven, "gross": gross}


class DeficitEnv(_Base):
    def __init__(self, split, kind, mode, seed=0, setting=SETTING, deficit_starts=False):
        super().__init__(split, kind, mode, seed, setting)
        self.deficit_share = 0.5 if deficit_starts is True else float(deficit_starts)     # False: none; True: half of the training episodes; a number: that share

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        opts = dict(options or {})
        self._start_window(opts)
        if "e0" in opts:
            e0 = float(opts["e0"])
        elif self.deficit_share > 0.0 and self.rng.random() < self.deficit_share:
            e0 = -self._draw_e0()
        else:
            e0 = self._draw_e0()
        self.e0 = K0 * e0
        self.ledger = self.e0
        self.forgiven = 0.0
        return self._obs(), {}

    @property
    def reported(self):
        return max(self.ledger, 0.0)

    def _obs(self):
        rem = (self.t_end - self.t) / EPISODE
        return np.array([np.clip(self.reported / K0, -5.0, 5.0), self.z_ep[self.t - self.t0], rem, 0.0], dtype=np.float32)

    def step(self, action):
        q = self.position(action)
        cost = FEE * abs(q) + self.kappa * q * q
        gross = self._move(q)
        pre = self.reported
        pnl = gross - cost
        self.ledger += pnl
        rep = self.reported - pre
        forgiven = rep - pnl
        self.forgiven += forgiven
        r = rep if self.mode == "reported" else pnl
        obs, done = self._advance()
        return obs, float(r / K0 * 100.0), done, False, {"pnl": pnl, "cost": cost, "forgiven": forgiven, "gross": gross}


class PartialEnv(_Base):
    def __init__(self, split, kind, mode, seed=0, setting=SETTING, lam=1.0, zero_starts=False):
        super().__init__(split, kind, mode, seed, setting)
        self.lam = float(lam)
        self.zero_starts = zero_starts

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        opts = dict(options or {})
        self._start_window(opts)
        if "e0" in opts:
            e0 = float(opts["e0"])
        elif self.zero_starts and self.rng.random() < 0.5:
            e0 = 0.0
        else:
            e0 = self._draw_e0()
        self.e0 = K0 * e0
        self.ledger = self.reported = self.e0
        self.forgiven = 0.0
        return self._obs(), {}

    def _obs(self):
        rem = (self.t_end - self.t) / EPISODE
        return np.array([np.clip(self.reported / K0, -5.0, 5.0), self.z_ep[self.t - self.t0], rem, 0.0], dtype=np.float32)

    def step(self, action):
        q = self.position(action)
        cost = FEE * abs(q) + self.kappa * q * q
        gross = self._move(q)
        pnl = gross - cost
        self.ledger += pnl
        raw = self.reported + pnl
        new_deficit = max(max(-raw, 0.0) - max(-self.reported, 0.0), 0.0)
        credit = self.lam * new_deficit
        self.reported = raw + credit
        self.forgiven += credit
        r = pnl + credit if self.mode == "reported" else pnl
        obs, done = self._advance()
        return obs, float(r / K0 * 100.0), done, False, {"pnl": pnl, "cost": cost, "forgiven": credit, "gross": gross}
