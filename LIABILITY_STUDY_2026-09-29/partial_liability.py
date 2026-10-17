"""EXPLORATORY (not pre-registered): partial liability on the full Danish training distribution (Corollary 3).

If a share alpha of every forgiven loss is booked, the zero-floor reward of a quarter-hour becomes
    (1 - alpha) * (max(e + pnl, 0) - e) + alpha * pnl,
with the same reflected equity transition. The DP of dp_full.py (2,000 bins, gamma 0.99) is solved for each alpha, and
the gambling region (the largest equity below which maximal leverage is optimal everywhere) is recorded. The paper
compares its width with the log law of Corollary 3: each tenfold reduction of the unbooked share 1 - alpha removes the
same width.

    python partial_liability.py     # writes results_k/partial_liability.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import dp_full
import dp_theory
from envs_b import K0

HERE = Path(__file__).resolve().parent
ALPHAS = (0.0, 0.5, 0.9, 0.99, 0.995, 0.996, 0.997, 0.998, 0.999)
GAMMA = 0.99


def solve_partial(X: np.ndarray, alpha: float, iters: int = 20000, tol: float = dp_theory.TOL) -> np.ndarray:
    """dp_theory.solve for the zero floor with a share alpha of the forgiven loss booked; returns the optimal level index."""
    G = dp_theory.GRID
    E = G[:, None, None]
    nxt = E + X[None, :, :]
    e2 = np.maximum(nxt, 0.0)
    r_mean = ((1 - alpha) * (e2 - E) + alpha * (nxt - E)).mean(axis=2)
    ec = np.clip(e2, G[0], G[-1])
    i1 = np.clip(np.searchsorted(G, ec, side="right"), 1, len(G) - 1)
    i0 = i1 - 1
    w1 = (ec - G[i0]) / (G[i1] - G[i0])
    del nxt, e2, ec
    tie = 1e-9 * K0 * (-np.arange(X.shape[0]))[None, :]
    rows = np.arange(len(G))

    def q_all(V):
        return r_mean + GAMMA * (V[i0] * (1 - w1) + V[i1] * w1).mean(axis=2)

    V = np.zeros_like(G)
    for _ in range(iters):
        best = np.argmax(q_all(V) + tie, axis=1)
        a0, a1, aw, ar = i0[rows, best], i1[rows, best], w1[rows, best], r_mean[rows, best]
        V_new = q_all(V)[rows, best]
        for _ in range(40):
            V_new = ar + GAMMA * (V_new[a0] * (1 - aw) + V_new[a1] * aw).mean(axis=1)
        delta = float(np.max(np.abs(V_new - V)))
        V = V_new
        if delta < tol * K0:
            break
    return np.argmax(q_all(V) + tie, axis=1)


def region16(best: np.ndarray) -> float:
    """Largest equity below which leverage 16 is optimal at every grid point; 0 if it is not optimal at zero equity.
    (theory_k.dp_threshold measures the largest leverage used, which is 0 once the region has closed.)"""
    ok = dp_theory.LEVELS[best] == 16
    if not ok[0]:
        return 0.0
    first_bad = int(np.argmin(ok)) if not ok.all() else len(ok)
    return float(dp_theory.GRID[first_bad - 1])


def main() -> int:
    rows = []
    for zone in ("DK1", "DK2"):
        X = dp_full.binned_pnl(zone, 2000)
        for a in ALPHAS:
            best = solve_partial(X, a)
            rows.append(dict(zone=zone, alpha=a, threshold_K=region16(best) / K0))
            print(rows[-1], flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(HERE / "results_k" / "partial_liability.csv", index=False)
    fit = D[D.alpha.isin((0.0, 0.5, 0.9, 0.99))]
    slope = np.polyfit(np.log10(1 - fit.alpha), fit.threshold_K, 1)[0]   # region shrinks as log10(1 - alpha) falls
    print(f"width removed per tenfold reduction of the unbooked share (alpha 0-0.99, both zones): {slope:.2f} allocations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
