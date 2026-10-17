"""Study O environment (PREREGISTRATION_O.md): the Study B/F placebo sleeve under partial liability.

The account keeps trading under the zero floor (reported equity e' = max(e + pnl, 0), never closed), and a share alpha of
every forgiven loss is charged in the reward:
    reward = [(1 - alpha) * (e' - e) + alpha * pnl] / K0 * 100      (percent of the allocation)
alpha = 0 is Study B's zero floor (ZF); alpha = 1 charges every loss (booked reward with continued trading, as the full
charge of Study L). Everything else is envs_b.LiabilityEnv: fixed sizing, liquidity cap, fee 0.10 EUR/MWh, training
equity log-uniform in [0.05, 2] allocations, episodes of envs_b.EPISODE quarter-hours.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "LIABILITY_STUDY_2026-09-29"))
from envs_b import K0, LiabilityEnv  # noqa: E402


class PartialEnv(LiabilityEnv):
    def __init__(self, market, alpha: float, train: bool, seed: int = 0):
        super().__init__(market, "ZF", train=train, seed=seed)
        self.alpha = float(alpha)

    def step(self, action):
        obs, reward, done, trunc, info = super().step(action)           # zero-floor reward: (e' - e) / K0 * 100
        reward = (1.0 - self.alpha) * reward + self.alpha * info["pnl"] / K0 * 100.0
        return obs, float(reward), done, trunc, info
