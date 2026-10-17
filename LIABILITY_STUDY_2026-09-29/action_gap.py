"""EXPLORATORY (not pre-registered): the action gap of the zero-floor dynamic program in every condition in which
floor-trained agents were trained (Supplementary Section S11, Table tab:actiongap).

For each condition the DP of dp_full.py is solved on the full training distribution (2,000 equal-probability bins per
leverage level, equity grid to 6 allocations), and the action gap A(e) = Q(e, 16) - Q(e, 0), the advantage of maximal
leverage over staying flat in euros of discounted reward, is read at the probe equity levels. sigma16 is the standard
deviation of a single quarter-hour's P&L at leverage 16 on the binned distribution the DP uses (the reward noise).

Conditions: DK1 and DK2 at gamma 0.95, 0.98, 0.99, 0.995 (Studies F and K), DK1 and DK2 under the frictions of Study H
(gamma 0.99), FI and NO2 at gamma 0.99 (Study J; needs the Study J panels, see get_esett_j.py).

Also writes the optimal leverage on the equity grid at gamma 0.99 (results_k/dp_policy_b2000_g0.99.csv), which
false_profit_bound.py simulates.

    python action_gap.py        # writes results_k/action_gap.csv and results_k/dp_policy_b2000_g0.99.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import dp_full
import dp_theory
from dp_frictions import CAPF_H, FEE_H, KAPPA_H
from envs_b import K0, Q1, load_market
from envs_j import load_market_j
from theory_k import dp_threshold

HERE = Path(__file__).resolve().parent
BINS = 2000
PROBE = (0.0, 0.02, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0)
CONDITIONS = ([(z, f"gamma {g}", g, "base") for z in ("DK1", "DK2") for g in (0.95, 0.98, 0.99, 0.995)]
              + [(z, "Study H costs", 0.99, "H") for z in ("DK1", "DK2")]
              + [(z, "gamma 0.99", 0.99, "base") for z in ("FI", "NO2")])


def train_market(zone: str):
    return load_market(zone, "train", "RM") if zone in ("DK1", "DK2") else load_market_j(zone, "train", "RM")


def binned_pnl_h(zone: str, bins: int) -> np.ndarray:
    """dp_full.binned_pnl under the frictions of Study H (fee 1.0, cap 10% of balancing volume, quadratic impact)."""
    m = train_market(zone)
    s, cap = np.abs(m.spread[200:]), m.cap_mwh[200:] * CAPF_H / 0.25
    rows = []
    for L in dp_theory.LEVELS:
        q = np.minimum(L * Q1, cap)
        z, c = q * s, FEE_H * q + KAPPA_H * q ** 2 / np.maximum(cap, 0.05)
        parts = np.array_split(np.argsort(z), bins)
        zb, cb = np.array([z[p].mean() for p in parts]), np.array([c[p].mean() for p in parts])
        rows.append(np.concatenate([zb - cb, -zb - cb]))
    return np.array(rows)


def main() -> int:
    rows, policy = [], {}
    i16 = int(np.flatnonzero(dp_theory.LEVELS == 16)[0])
    idx = {e: int(np.argmin(np.abs(dp_theory.GRID - e * K0))) for e in PROBE}
    for zone, label, g, costs in CONDITIONS:
        X = binned_pnl_h(zone, BINS) if costs == "H" else dp_full.binned_pnl(zone, BINS)
        dp_theory.GAMMA = g
        V, best, it, Q = dp_theory.solve("ZF", X, iters=20000)
        gap = Q[:, i16] - Q[:, 0]
        rows.append(dict(zone=zone, condition=label, gamma=g, costs=costs, sigma16=float(np.std(X[i16])),
                         threshold_K=dp_threshold(best) / K0, iterations=it + 1,
                         **{f"A_{e:g}": float(gap[i]) for e, i in idx.items()}))
        if costs == "base" and g == 0.99:
            policy[zone] = dp_theory.LEVELS[best]
        print({k: round(v, 2) if isinstance(v, float) else v for k, v in rows[-1].items()}, flush=True)
    pd.DataFrame(rows).to_csv(HERE / "results_k" / "action_gap.csv", index=False)
    pd.DataFrame({"equity_K": dp_theory.GRID / K0, **policy}).to_csv(HERE / "results_k" / "dp_policy_b2000_g0.99.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
