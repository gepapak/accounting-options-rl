"""Study O design calculation (run before registration): the learnability index of the zero floor with a share alpha of
every forgiven loss booked, for candidate alphas, DK1 and DK2.

Reward per quarter-hour (EUR): (1 - alpha) * (max(e + pnl, 0) - e) + alpha * pnl; equity e' = max(e + pnl, 0) (the account
keeps trading). alpha = 0 is the zero floor, alpha = 1 books every loss (the matched control). The DP is that of
dp_full.py (full training distribution, 2,000 equal-probability bins per leverage, gamma 0.99, equity grid to 6
allocations).

Learnability index (the formula of the retrospective analysis, fixed here):
    A(e) = Q(e, 16) - Q(e, 0) read at e in PROBE, interpolated linearly in log e;
    index = mean over 400 log-spaced e in [0.05, 2] (the training-start distribution, log-uniform) of max(A(e), 0),
            divided by sigma16 (standard deviation of the binned one-quarter-hour P&L at leverage 16), in percent.

    python dp_o.py --zone DK1 ; python dp_o.py --zone DK2     # writes dp_o_<zone>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "LIABILITY_STUDY_2026-09-29"))
import dp_full  # noqa: E402  (sets dp_theory.GRID to 0..6 allocations)
import dp_theory  # noqa: E402
from envs_b import K0  # noqa: E402

GAMMA = 0.99
ALPHAS = (0.0, 0.25, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0)
PROBE = (0.02, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0)


def learnability_index(A_probe: np.ndarray, sigma16: float) -> float:
    e = np.exp(np.linspace(np.log(0.05), np.log(2.0), 400))
    A = np.interp(np.log(e), np.log(PROBE), A_probe)
    return float(100 * np.mean(np.maximum(A, 0.0)) / sigma16)


def solve(X: np.ndarray, alpha: float, iters: int = 20000) -> np.ndarray:
    """Value iteration (dp_theory.solve, zero floor) with a share alpha of the forgiven loss booked; returns Q."""
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
    q_all = lambda V: r_mean + GAMMA * (V[i0] * (1 - w1) + V[i1] * w1).mean(axis=2)  # noqa: E731
    V = np.zeros_like(G)
    for _ in range(iters):
        Q = q_all(V)
        best = np.argmax(Q + tie, axis=1)
        a0, a1, aw, ar = i0[rows, best], i1[rows, best], w1[rows, best], r_mean[rows, best]
        V_new = Q[rows, best]
        for _ in range(40):
            V_new = ar + GAMMA * (V_new[a0] * (1 - aw) + V_new[a1] * aw).mean(axis=1)
        delta = float(np.max(np.abs(V_new - V)))
        V = V_new
        if delta < dp_theory.TOL * K0:
            break
    return q_all(V)


def region16(Q: np.ndarray) -> float:
    """Largest equity below which leverage 16 is optimal at every grid point (0 if not optimal at zero equity)."""
    tie = 1e-9 * K0 * (-np.arange(Q.shape[1]))[None, :]
    ok = dp_theory.LEVELS[np.argmax(Q + tie, axis=1)] == 16
    if not ok[0]:
        return 0.0
    first_bad = int(np.argmin(ok)) if not ok.all() else len(ok)
    return float(dp_theory.GRID[first_bad - 1] / K0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", required=True)
    a = ap.parse_args()
    X = dp_full.binned_pnl(a.zone, 2000)
    i16 = int(np.flatnonzero(dp_theory.LEVELS == 16)[0])
    sigma16 = float(np.std(X[i16]))
    idx = [int(np.argmin(np.abs(dp_theory.GRID - e * K0))) for e in PROBE]
    out = dict(zone=a.zone, gamma=GAMMA, sigma16=sigma16, probe=PROBE, arms={})
    for alpha in ALPHAS:
        Q = solve(X, alpha)
        A = (Q[:, i16] - Q[:, 0])[idx]
        out["arms"][f"{alpha:g}"] = dict(A=[float(x) for x in A], region16_K=region16(Q),
                                         index_pct=learnability_index(A, sigma16))
        print(a.zone, alpha, {k: (round(v, 3) if isinstance(v, float) else [round(x, 2) for x in v])
                              for k, v in out["arms"][f"{alpha:g}"].items()}, flush=True)
    (HERE / f"dp_o_{a.zone}.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
