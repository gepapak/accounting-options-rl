"""Study Z environments (PREREGISTRATION_Z.md): the Study T edge-market sleeve split across two separately floored accounts.

Two accounts A and B trade the same Danish imbalance spread (setting S_DK1, the Study T market draw). The agent chooses a signed leverage for
each account (MultiDiscrete([9, 9]) over the LEV grid), sized as in the sleeve (fixed sizing from Q1, the liquidity cap applied to each leg),
and each leg pays its own cost fee*|q| + kappa*q^2. The ledger books both legs in full.

    SplitEnv   two accounts with a zero floor each: E_k' = max(E_k + x_k, 0), trading continues (the split structure)
    PooledEnv  the same two legs written to one account with one zero floor: E' = max(E + x_A + x_B, 0) (the pooled control)

Observation (both): (E_A / K0, E_B / K0, z, share of the episode remaining), clipped to [-5, 5]. PooledEnv shows its single equity as two
equal halves, so a symmetric split state and the pooled state with the same total look alike to the policy.
Reward modes: reported (change of the portfolio's reported equity, percent of K0), booked (booked P&L of both legs, percent of K0).
`flip_next = True` reverses the sign of the next step's exogenous price change for both legs (the sign twin of the whole portfolio).
Zero starts: with `zero_starts=True` half of the episodes that are not given a start equity begin with every account at zero.
info: pnl and cost of the step (EUR, both legs), forgiven (EUR), ruin_portfolio (1 if the portfolio's reported equity went from a positive
value to zero), ruin_accounts (number of accounts whose reported equity went from a positive value to zero).
"""
from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "study_t"), str(HERE.parents[1] / "LIABILITY_STUDY_2026-09-29")]
from envs_t import E0_TRAIN, EPISODE, FEE, K0, LEV, Q1, SETTINGS, market_arrays  # noqa: E402

SETTING = "S_DK1"
NLEV = len(LEV)


class _TwoLegEnv(gym.Env):
    metadata: dict = {}
    split = True

    def __init__(self, split: str, kind: str, mode: str, seed: int = 0, zero_starts: bool = False, setting: str = SETTING):
        super().__init__()
        assert mode in ("reported", "booked") and kind in ("edge", "placebo")
        self.abs_s, self.eps, self.zz, self.cap = market_arrays(setting, split, "edge" if kind == "edge" else "placebo")
        self.kappa = SETTINGS[setting]["kappa"]
        self.n = len(self.abs_s)
        self.kind, self.mode, self.zero_starts = kind, mode, zero_starts
        self.action_space = gym.spaces.MultiDiscrete([NLEV, NLEV])
        self.observation_space = gym.spaces.Box(-5.0, 5.0, shape=(4,), dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.flip_next = False

    # ---- episode ------------------------------------------------------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        opts = dict(options or {})
        self.t0 = int(opts["t0"]) if "t0" in opts else int(self.rng.integers(200, self.n - EPISODE))
        self.t, self.t_end = self.t0, self.t0 + EPISODE
        if self.kind == "placebo":
            self.eps_ep = self.rng.choice([-1.0, 1.0], size=EPISODE)
            self.z_ep = self.rng.choice([-1.0, 1.0], size=EPISODE)
        else:
            self.eps_ep = self.eps[self.t0:self.t_end]
            self.z_ep = self.zz[self.t0:self.t_end]
        if "ea" in opts:                                     # explicit account equities (multiples of K0)
            ea, eb = float(opts["ea"]), float(opts["eb"])
        else:
            e0 = opts.get("e0")
            if e0 is None:
                e0 = 0.0 if (self.zero_starts and self.rng.random() < 0.5) else \
                    float(np.exp(self.rng.uniform(np.log(E0_TRAIN[0]), np.log(E0_TRAIN[1]))))
            ea = eb = 0.5 * float(e0)
        self.ea, self.eb = K0 * ea, K0 * eb
        self.e0 = self.ea + self.eb
        self.ledger = self.e0
        self.forgiven = 0.0
        self.n_ruin_portfolio = 0
        self.n_ruin_accounts = 0
        self.flip_next = False
        return self._obs(), {}

    @property
    def reported(self) -> float:
        return self.ea + self.eb

    def _obs(self):
        rem = (self.t_end - self.t) / EPISODE
        return np.array([np.clip(self.ea / K0, -5.0, 5.0), np.clip(self.eb / K0, -5.0, 5.0), self.z_ep[self.t - self.t0], rem],
                        dtype=np.float32)

    def legs(self, action):
        """Positions (MWh) and own costs (EUR) of the two legs at the current step."""
        t = self.t
        qa = float(np.clip(float(LEV[int(action[0])]) * Q1, -self.cap[t], self.cap[t]))
        qb = float(np.clip(float(LEV[int(action[1])]) * Q1, -self.cap[t], self.cap[t]))
        return qa, qb, FEE * abs(qa) + self.kappa * qa * qa, FEE * abs(qb) + self.kappa * qb * qb

    # ---- one quarter-hour ---------------------------------------------------------------------------------------------------------
    def step(self, action):
        t = self.t
        qa, qb, ca, cb = self.legs(action)
        sign = self.eps_ep[t - self.t0] * (-1.0 if self.flip_next else 1.0)
        self.flip_next = False
        xa = qa * self.abs_s[t] * sign - ca
        xb = qb * self.abs_s[t] * sign - cb
        pnl, cost = xa + xb, ca + cb
        before_a, before_b = self.ea, self.eb
        before = before_a + before_b
        self._write(xa, xb)
        self.ledger += pnl
        after = self.ea + self.eb
        change = after - before
        forgiven = change - pnl
        self.forgiven += forgiven
        ruin_p = int(before > 0.0 and after <= 0.0)
        ruin_a = self._account_ruins(before_a, before_b)
        self.n_ruin_portfolio += ruin_p
        self.n_ruin_accounts += ruin_a
        r = change if self.mode == "reported" else pnl
        self.t += 1
        done = self.t >= self.t_end
        obs = self._obs() if not done else np.zeros(4, np.float32)
        info = {"pnl": pnl, "cost": cost, "forgiven": forgiven, "ruin_portfolio": ruin_p, "ruin_accounts": ruin_a,
                "net": qa + qb, "gross": abs(qa) + abs(qb)}
        return obs, float(r / K0 * 100.0), done, False, info

    def _write(self, xa, xb):
        raise NotImplementedError

    def _account_ruins(self, before_a, before_b) -> int:
        raise NotImplementedError


class SplitEnv(_TwoLegEnv):
    """Two accounts, each with its own zero floor."""
    split = True

    def _write(self, xa, xb):
        self.ea = max(self.ea + xa, 0.0)
        self.eb = max(self.eb + xb, 0.0)

    def _account_ruins(self, before_a, before_b) -> int:
        return int(before_a > 0.0 and self.ea <= 0.0) + int(before_b > 0.0 and self.eb <= 0.0)


class PooledEnv(_TwoLegEnv):
    """The same two legs written to one account with one zero floor (equity shown as two equal halves)."""
    split = False

    def _write(self, xa, xb):
        e = max(self.ea + self.eb + xa + xb, 0.0)
        self.ea = self.eb = 0.5 * e

    def _account_ruins(self, before_a, before_b) -> int:
        return int(before_a + before_b > 0.0 and self.ea + self.eb <= 0.0)


def make_env(structure: str, split: str, kind: str, mode: str, seed: int = 0, zero_starts: bool = False):
    cls = {"split": SplitEnv, "pooled": PooledEnv}[structure]
    return cls(split, kind, mode, seed=seed, zero_starts=zero_starts)
