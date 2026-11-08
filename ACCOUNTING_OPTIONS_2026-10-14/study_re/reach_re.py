"""Predicted reach per scale (PREREGISTRATION_RE.md): how often can an episode's P&L travel to the floor at all?

For each scale c, 20,000 training-like episodes (random start in the DK1 placebo training market, equity log-uniform in
[0.05, 2] K0, 672 quarters) are run under the most aggressive constant behaviour, maximal leverage every quarter, and the share
whose cumulative P&L reaches the floor is recorded, together with the same share from the lowest training equity 0.05 K0. The
sign of the leverage is irrelevant in a placebo market; the long side is used. The reach R_c is the share times the number of
episodes in a 10^6-step run (1,488): the expected number of floor-reaching episodes if the learner were always at maximal leverage.
This is the registered predictor, not a pathwise bound: a non-anticipating policy cannot choose its leverage by the coming sign, so
its cumulative P&L has at most this policy's variance, but on a given draw it can lose faster by luck. The rigorous bound per episode
is Kolmogorov's inequality for the P&L martingale, P(reach) <= c^2 Var(cumulative gross at maximal leverage) / (e0 - cost)^2, which
is also written out (it is loose: at c = 0.01 it allows a few per cent from the lowest equity, where the Monte Carlo finds none).

    python reach_re.py            # writes reach_re.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from envs_re import FEE, K0, SCALES, load_market

EPISODE = 672
N_EP = 20_000
STEPS_PER_RUN = 1_000_000
Q_MAX = 16.0 * 0.25


def main() -> int:
    m = load_market("DK1", "train", "NE")
    rng = np.random.default_rng(20261107)
    t0 = rng.integers(200, m.n - EPISODE, size=N_EP)
    e0 = K0 * np.exp(rng.uniform(np.log(0.05), np.log(2.0), size=N_EP))
    idx = t0[:, None] + np.arange(EPISODE)[None, :]
    q = np.minimum(Q_MAX, m.cap_mwh[idx])
    base = q * m.spread[idx] - FEE * q                      # unscaled P&L at maximal leverage
    cum = np.cumsum(base, axis=1)
    run_min = np.minimum.accumulate(cum, axis=1)[:, -1]     # most negative cumulative P&L in the episode
    out = {"n_episodes": N_EP, "episodes_per_run": STEPS_PER_RUN // EPISODE, "scales": {}}
    gross = q * m.spread[idx]
    var_gross = float((gross ** 2).sum(axis=1).mean())           # E[sum of squared gross] = Var of the martingale part
    cost_ep = float((FEE * q).sum(axis=1).mean())
    for c in SCALES:
        hit = (c * run_min <= -e0)
        hit_low = (c * run_min <= -0.05 * K0)
        kol_low = min(1.0, c * c * var_gross / (0.05 * K0 - c * cost_ep) ** 2)
        kol_all = float(np.minimum(1.0, c * c * var_gross / (e0 - c * cost_ep) ** 2).mean())
        out["scales"][str(c)] = {
            "share_episodes_reaching_floor": float(hit.mean()),
            "share_from_lowest_equity": float(hit_low.mean()),
            "reach_R": float(hit.mean() * (STEPS_PER_RUN // EPISODE)),
            "sd_cumulative_pnl_at_max_leverage": float(c * cum[:, -1].std()),
            "kolmogorov_bound_from_lowest_equity": kol_low,
            "kolmogorov_bound_training_equities": kol_all,
        }
        print(f"c={c:<5} share {hit.mean():.4f}  from 0.05K0 {hit_low.mean():.4f}  R={hit.mean() * (STEPS_PER_RUN // EPISODE):.1f}"
              f"  sd(cum P&L)={c * cum[:, -1].std():.0f} EUR  Kolmogorov bound (0.05K0) {kol_low:.3f}, (training e0) {kol_all:.3f}")
    Path(__file__).with_name("reach_re.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
