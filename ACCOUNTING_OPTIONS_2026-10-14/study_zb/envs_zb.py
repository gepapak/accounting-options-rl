"""Study Z, Part B environment (PREREGISTRATION_ZB.md): the one-account sleeve with per-position (isolated) margin, whose floors the trainer
cannot see.

The Study T edge-market sleeve (setting S_DK1, the Study T market draw) with one account whose reported equity P is all that the agent and
the trainer observe. Each quarter the agent chooses a signed leverage (the sleeve's grid) and a margin mode:
    mode 0, cross margin    the position's P&L is written to the whole account with the zero floor: P' = max(P + x, 0)  (the sleeve's rule)
    mode 1, isolated margin the position can lose at most its margin m = c + M_ISO * |q| (its own cost plus M_ISO EUR per MWh, never more
                            than P); any loss beyond it is absorbed by the venue: P' = P + max(x, -min(m, P))
as on derivatives exchanges that margin positions separately and cover a liquidated position's deficit from a pooled fund. Booked P&L
x = q s - c in both modes (the ledger books every loss). M_ISO = 100 EUR/MWh covers 84% of the quarter-hour moves of the DK1 training panel.

    action:       MultiDiscrete([9, 2]) = (leverage index, margin mode)
    observation:  (P / K0 clipped to [-5, 5], z, share of the episode remaining, 0), as in the one-account sleeve
    reward modes: reported (P' - P), booked (x), percent of K0
    info:         pnl, cost, forgiven (EUR), ruin_portfolio (P went from positive to zero), liquidation (an isolated position lost more than
                  its margin), isolated (1 if mode 1 with a nonzero position)
`flip_next = True` reverses the sign of the next step's exogenous price change (the sign twin).
"""
from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "study_t"), str(HERE.parents[1] / "LIABILITY_STUDY_2026-09-29")]
from envs_t import FEE, K0, LEV, Q1, EdgeEnvT  # noqa: E402

SETTING = "S_DK1"
NLEV = len(LEV)
M_ISO = 100.0


class IsolatedEnv(EdgeEnvT):
    def __init__(self, split: str, kind: str, mode: str, seed: int = 0, setting: str = SETTING):
        super().__init__(setting, split, kind, mode, seed=seed)
        self.action_space = gym.spaces.MultiDiscrete([NLEV, 2])

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.n_ruin_portfolio = 0
        self.n_liquidations = 0
        return obs, info

    def position(self, action):
        t = self.t
        q = float(np.clip(float(LEV[int(action[0])]) * Q1, -self.cap[t], self.cap[t]))
        return q, FEE * abs(q) + self.kappa * q * q, int(action[1])

    def step(self, action):
        t = self.t
        q, cost, iso = self.position(action)
        sign = self.eps_ep[t - self.t0] * (-1.0 if self.flip_next else 1.0)
        self.flip_next = False
        pnl = q * self.abs_s[t] * sign - cost
        before = self.reported
        if iso and q != 0.0:
            cap_loss = min(cost + M_ISO * abs(q), before)
            change = max(pnl, -cap_loss)
            liq = int(pnl < -cap_loss)
        else:
            change = max(before + pnl, 0.0) - before
            liq = 0
        self.reported = before + change
        self.ledger += pnl
        forgiven = change - pnl
        self.forgiven += forgiven
        ruin = int(before > 0.0 and self.reported <= 0.0)
        self.n_ruin_portfolio += ruin
        self.n_liquidations += liq
        r = change if self.mode == "reported" else pnl
        self.t += 1
        done = self.t >= self.t_end
        obs = self._obs() if not done else np.zeros(4, np.float32)
        info = {"pnl": pnl, "cost": cost, "forgiven": forgiven, "ruin_portfolio": ruin, "liquidation": liq,
                "isolated": int(iso and q != 0.0)}
        return obs, float(r / K0 * 100.0), done, False, info
