"""Study R: Study O's reflected account, with an explicit cost parameter.

No registered dependency is modified. At fee .10 this environment must match
PartialEnv pathwise. The market, state, resets, sizing and rewards are unchanged
apart from the declared fee. Training counters are observational only.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "LIABILITY_STUDY_2026-09-29"))
from envs_b import K0, LEV, Q1, LiabilityEnv

VISIT_EDGES = np.array([0, .02, .05, .1, .25, .5, 1, 2, np.inf])


class CostPartialEnv(LiabilityEnv):
    def __init__(self, market, alpha, fee, train, seed=0):
        super().__init__(market, "ZF", train=train, seed=seed)
        self.alpha, self.fee = float(alpha), float(fee)
        self.visit_counts = np.zeros((len(VISIT_EDGES) - 1, len(LEV)), dtype=np.int64)
        self.training_sums = np.zeros(4, dtype=float)  # pnl, cost, injections, reward (EUR)

    def step(self, action):
        t, m = self.t, self.m
        lev = float(LEV[int(action)])
        if self.train:
            b = int(np.clip(np.searchsorted(VISIT_EDGES, self.reported / K0, side="right") - 1,
                            0, len(VISIT_EDGES) - 2))
            self.visit_counts[b, int(action)] += 1
        self.open_q += 1
        self.lev_abs_sum += abs(lev)
        self.max_lev_q += int(abs(lev) == LEV.max())
        q = float(np.clip(lev * Q1, -m.cap_mwh[t], m.cap_mwh[t]))
        cost = self.fee * abs(q)
        pnl = q * m.spread[t] - cost
        self.ledger += pnl
        before = self.reported
        new = before + pnl
        if new < 0:
            self.floor_bound = True
        self.reported = max(new, 0.0)
        injection = max(-new, 0.0)
        # Preserve O's floating-point order: mix the already-percent rewards.
        reward = (1 - self.alpha) * ((self.reported - before) / K0 * 100) + self.alpha * pnl / K0 * 100
        reward_eur = reward * K0 / 100
        self.max_gap_pp = max(self.max_gap_pp, 100 * abs(self.reported - self.ledger) / K0)
        if self.train:
            self.training_sums += (pnl, cost, injection, reward_eur)
        self.t += 1
        done = self.t >= self.t_end
        return (self._obs() if not done else np.zeros(11, np.float32)), float(reward), done, False, {
            "pnl": pnl, "cost": cost, "injection": injection}
