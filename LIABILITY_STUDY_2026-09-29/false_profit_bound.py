"""EXPLORATORY (not pre-registered): the universal cap of Theorem 2 in the four zones of the learning studies
(Supplementary Section S11, Table tab:cap).

Placebo market over the test period (from index 200), liquidity cap and fee of the environment, N sign draws (the same
signs for every policy). The cap is B_T = sqrt(sum_t cbar_t^2 s_t^2) with cbar_t = min(16 q1, cap_t). Policies under
the zero floor:
    fixed maximal   leverage 16 throughout;
    threshold (DP)  the optimal policy of the DP on the full training distribution at gamma 0.99 (action_gap.py);
    closed form     maximal leverage below the closed-form barrier of Proposition 7 at gamma 0.99, flat above
                    (theory_k.py, Denmark only);
    learned agents  the shipped test-period reported returns of the equity-only floor-trained agents
                    (Studies F and K at gamma 0.99 in Denmark, Study J in Finland and southern Norway).
From zero equity: expected reported equity as a share of the cap. From one allocation: expected reported profit.

    python action_gap.py && python false_profit_bound.py     # writes results_k/false_profit_bound.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from envs_b import FEE, K0, Q1, load_market
from envs_j import load_market_j

HERE = Path(__file__).resolve().parent
R = HERE / "results_k"
N = 20000
SEED = 20261014
ZONES = ("DK1", "DK2", "FI", "NO2")


def simulate(mag_per_mwh: np.ndarray, cap: np.ndarray, lev_of_equity, e0: float) -> float:
    """Mean final reported equity (EUR) under the zero floor; lev_of_equity maps an equity array to |leverage|."""
    rng = np.random.default_rng(SEED)
    W = np.full(N, e0)
    for t in range(len(cap)):
        q = np.minimum(lev_of_equity(W) * Q1, cap[t])
        W = np.maximum(W + rng.choice([-1.0, 1.0], N) * q * mag_per_mwh[t] - FEE * q, 0.0)
    return float(W.mean())


def learned() -> dict[str, float]:
    c = "drawmean_reported_return_pct"
    F, J, K = (pd.read_csv(HERE / p) for p in ("results_f/runs_f.csv", "results_j/runs_j.csv", "results_k/runs_k.csv"))
    dk = pd.concat([F[(F.obsset == "EQ") & (F.rule == "ZF")], K[(K.rule == "ZF") & (K.gamma == 0.99)]])
    out = (dk.groupby("region")[c].mean() / 100).to_dict()
    out.update((J[(J.obsset == "EQ") & (J.rule == "ZF")].groupby("region")[c].mean() / 100).to_dict())
    return out


def main() -> int:
    pol = pd.read_csv(R / "dp_policy_b2000_g0.99.csv")
    grid = pol.equity_K.to_numpy() * K0
    barrier = pd.read_csv(R / "theory_k.csv").query("gamma == 0.99").set_index("region").closed_form_barrier_K.to_dict()
    lrn = learned()
    rows = []
    for zone in ZONES:
        m = load_market(zone, "test", "RM") if zone in ("DK1", "DK2") else load_market_j(zone, "test", "RM")
        s, cap = np.abs(m.spread[200:]), m.cap_mwh[200:]
        B = float(np.sqrt(np.sum((np.minimum(16 * Q1, cap) * s) ** 2)))
        lev = pol[zone].to_numpy()
        fixed = lambda W: np.full_like(W, 16.0)  # noqa: E731
        dp = lambda W, lev=lev: lev[np.clip(np.searchsorted(grid, W, side="right") - 1, 0, len(grid) - 1)]  # noqa: E731
        row = dict(zone=zone, cap_B_K=B / K0,
                   zero_fixed_share=simulate(s, cap, fixed, 0.0) / B, zero_dp_share=simulate(s, cap, dp, 0.0) / B,
                   one_fixed_K=simulate(s, cap, fixed, K0) / K0 - 1, one_dp_K=simulate(s, cap, dp, K0) / K0 - 1,
                   one_closed_form_K=np.nan, one_learned_K=lrn[zone])
        if zone in barrier:
            b = barrier[zone] * K0
            row["one_closed_form_K"] = simulate(s, cap, lambda W: np.where(W < b, 16.0, 0.0), K0) / K0 - 1
        rows.append(row)
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
    pd.DataFrame(rows).to_csv(R / "false_profit_bound.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
