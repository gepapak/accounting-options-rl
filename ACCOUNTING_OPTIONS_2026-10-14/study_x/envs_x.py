"""Study X environments (PREREGISTRATION_X.md): the Study T edge-market sleeve with a TWO-WRITE floor.

Study T's sleeve books a step as one floored write: equity E' = max(E + gross - fee, 0), so the reported reward is max(x, -E), a single monotone
censor of the booked P&L x = gross - fee (Assumption C). The public engine instead floors its cash writes one after the other (the opening fee
is written with its own floor, the settlement afterwards). This variant does the same:

    E1 = max(E - fee, 0)        fee write, floored: an account that cannot pay the fee is not charged for the unpaid part
    E2 = max(E1 + gross, 0)     gross write, floored
    reported reward = E2 - E;   the ledger books the whole fee (booked P&L = gross - fee, as in Study T)

It coincides with the single-floor sleeve whenever E >= fee and differs only for accounts that cannot pay the fee, where the twin pair no
longer identifies the booked profit (Proposition on testing Assumption C). EdgeEnvMFZ starts half of the training episodes at zero equity, as
Study V's Part C does, so that those accounts are visited.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "study_t"), str(HERE.parents[1] / "LIABILITY_STUDY_2026-09-29")]
from envs_t import FEE, K0, LEV, Q1, EdgeEnvT  # noqa: E402


class EdgeEnvMF(EdgeEnvT):
    def step(self, action):
        t = self.t
        q = float(np.clip(float(LEV[int(action)]) * Q1, -self.cap[t], self.cap[t]))
        cost = FEE * abs(q) + self.kappa * q * q
        sign = self.eps_ep[t - self.t0] * (-1.0 if self.flip_next else 1.0)
        self.flip_next = False
        gross = q * self.abs_s[t] * sign
        e0 = self.reported
        e1 = max(e0 - cost, 0.0)
        e2 = max(e1 + gross, 0.0)
        pnl = gross - cost
        self.ledger += pnl
        forgiven = (e2 - e0) - pnl
        self.forgiven += forgiven
        self.reported = e2
        r = (e2 - e0) if self.mode == "reported" else pnl
        self.t += 1
        done = self.t >= self.t_end
        obs = self._obs() if not done else np.zeros(4, np.float32)
        return obs, float(r / K0 * 100.0), done, False, {"pnl": pnl, "cost": cost, "forgiven": forgiven}


class EdgeEnvMFZ(EdgeEnvMF):
    def reset(self, *, seed=None, options=None):
        opts = dict(options or {})
        if "e0" not in opts and self.rng.random() < 0.5:
            opts["e0"] = 0.0
        return super().reset(seed=seed, options=opts)
