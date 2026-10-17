"""EXPLORATORY (not pre-registered): the best-moment theorem (Theorem 1) in all twelve Nordic bidding zones.

Prices alone, no liquidity cap: a fixed 4 MWh position (leverage 16 of the sleeve) in the eSett imbalance-minus-spot
spread is held from zero equity under the zero floor in the placebo market (independent random signs), January-August
2026, fee 0.10 EUR/MWh, N sign draws. Per zone: mean |spread|, excess kurtosis of the spread, Kolmogorov-Smirnov
distance between reported equity and the running maximum of the booked P&L, mean reported equity, the prediction of the
law of the maximum of a Brownian motion with the fee drift (no fitted parameter), CV and the power of a five-draw
placebo test. Supplementary Section S11, Table tab:zones.

Needs the eSett EXP14 files: python ../data/fetch_esett.py (from this folder) or python data/fetch_esett.py (root).

    python best_moment_zones.py        # writes results_k/best_moment_zones.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import integrate, stats

from envs_b import FEE, K0, Q1

HERE = Path(__file__).resolve().parent
ESETT = HERE.parent / "data" / "esett"
ZONES = ("DK1", "DK2", "FI", "NO1", "NO2", "NO3", "NO4", "NO5", "SE1", "SE2", "SE3", "SE4")
START, END = "2026-01-01", "2026-09-01"          # January-August 2026 (UTC)
N = 20000                                         # Monte Carlo s.e. of the mean about 0.6% of it
Q = 16 * Q1                                       # 4 MWh
rng = np.random.default_rng(20261014)


def power(R: np.ndarray, n: int, reps: int = 20000) -> float:
    """Share of one-sided 5% t-tests on n resampled placebo draws that flag a positive mean (as make_best_moment_figure)."""
    S = R[rng.integers(0, len(R), (reps, n))]
    sd = S.std(1, ddof=1)
    t = np.where(sd > 0, S.mean(1) / np.where(sd > 0, sd, 1) * np.sqrt(n), np.inf)
    return float(np.mean(t > stats.t.ppf(0.95, n - 1)))


def spread(zone: str) -> np.ndarray:
    d = pd.read_csv(ESETT / f"esett_EXP14_{zone}.csv.gz", usecols=["timestampUTC", "imblSpotDifferencePrice"])
    d = d[(d.timestampUTC >= START) & (d.timestampUTC < END)].dropna()
    return d.imblSpotDifferencePrice.to_numpy(float)


def max_drift_bm_mean(sigma: float, mu: float, T: int) -> float:
    """E[max_{t<=T} X_t] for X a Brownian motion with drift mu and volatility sigma per step (reflection principle)."""
    s, mT = sigma * np.sqrt(T), mu * T
    P = lambda a: 1 - stats.norm.cdf((a - mT) / s) + np.exp(2 * mu * a / sigma ** 2) * stats.norm.cdf((-a - mT) / s)  # noqa: E731
    return integrate.quad(P, 0, 20 * s, limit=200)[0]


def main() -> int:
    rows = []
    for zone in ZONES:
        s = spread(zone)
        mag, fee = Q * np.abs(s), FEE * Q
        T = len(mag)
        W, L, Lmax = np.zeros(N), np.zeros(N), np.zeros(N)
        for t in range(T):
            x = rng.choice([-1.0, 1.0], N) * mag[t] - fee
            W = np.maximum(W + x, 0.0)
            L = L + x
            Lmax = np.maximum(Lmax, L)
        sigma = float(np.sqrt(np.mean(mag ** 2)))
        pred = max_drift_bm_mean(sigma, -fee, T)
        ks = stats.ks_2samp(W, Lmax)
        rows.append(dict(zone=zone, T=T, mean_abs_spread=float(np.mean(np.abs(s))), excess_kurtosis=float(stats.kurtosis(s)),
                         ks_D=ks.statistic, mean_reported_K=W.mean() / K0, predicted_K=pred / K0,
                         error_pct=100 * (W.mean() / pred - 1), cv=W.std() / W.mean(), power_n5=power(W / K0, 5)))
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in rows[-1].items()}, flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(HERE / "results_k" / "best_moment_zones.csv", index=False)
    r = stats.spearmanr(D.excess_kurtosis, -D.error_pct)
    print(f"Spearman(kurtosis, over-prediction) = {r.statistic:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
