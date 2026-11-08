"""Study RE environment (PREREGISTRATION_RE.md): Study F's sleeve with the market's reach scaled.

The Study B/F sleeve is unchanged (fixed sizing from K0, liquidity cap, signed leverage grid LEV, fee per MWh, placebo market with
independent fair signs, training equity log-uniform in [0.05, 2] K0, episodes of 672 quarters, equity as the only input) except
that every spread and the fee are multiplied by a scale c. Scaling both leaves the sleeve's geometry (cost per unit of spread,
leverage grid, cap) unchanged and moves only the distance of the floor, in units of what an episode's P&L can travel.

Rules: FL (full liability; the account closes at non-positive equity with the overshoot booked) and ZF (zero floor; reported
equity max(e + pnl, 0), trading continues). The ledger books every loss. The counter `floor_steps` counts the training steps on
which the floored write differed from the booked one (ZF) or the account closed (FL): the steps on which the two rules' data differ.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

LIAB = Path(__file__).resolve().parents[2] / "LIABILITY_STUDY_2026-09-29"
sys.path.insert(0, str(LIAB))
from envs_b import FEE, K0, LEV, LiabilityEnv, load_market  # noqa: E402,F401

SCALES = (1.0, 0.3, 0.1, 0.03, 0.01)


class ScaledLiabilityEnv(LiabilityEnv):
    def __init__(self, market, liability: str, scale: float, train: bool, seed: int = 0):
        super().__init__(market, liability, train, seed)
        self.scale = float(scale)
        self.floor_steps = 0

    def step(self, action):
        t, m = self.t, self.m
        lev = 0.0 if self.closed else float(LEV[int(action)])
        if not self.closed:
            self.open_q += 1
            self.lev_abs_sum += abs(lev)
            self.max_lev_q += int(abs(lev) == LEV.max())
        q = float(np.clip(lev * 0.25, -m.cap_mwh[t], m.cap_mwh[t]))  # Q1 = 0.25 MWh at leverage 1, as in envs_b
        pnl = self.scale * (q * m.spread[t] - FEE * abs(q))
        self.ledger += pnl
        before = self.reported
        if self.liab == "FL":
            self.reported = self.ledger
            if self.reported <= 0.0:
                self.closed = True
                self.floor_steps += 1
        else:
            new = self.reported + pnl
            if new < 0.0:
                self.floor_bound = True
                self.floor_steps += 1
            self.reported = max(new, 0.0)
        reward = (self.reported - before) / K0 * 100.0
        self.max_gap_pp = max(self.max_gap_pp, 100.0 * abs(self.reported - self.ledger) / K0)
        self.t += 1
        done = self.t >= self.t_end or (self.train and self.closed)
        return (self._obs() if not done else np.zeros(11, np.float32)), float(reward), done, False, {"pnl": pnl}
