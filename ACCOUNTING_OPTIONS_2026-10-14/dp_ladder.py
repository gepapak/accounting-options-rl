"""EXPLORATORY DP: optimal leverage of the Study B/F placebo sleeve under the liability ladder.

Rules differ only in what happens when booked equity first falls to <= 0 (breach):
  FL      breach is terminal, the loss is booked (reward = pnl)
  LAG_W   breach detected W quarter-hours later: the sleeve keeps trading in a 'zombie window' with the ledger
          as reported equity; at detection the sleeve closes and reported equity becomes max(n, 0)
          (bonus (-n)^+). W = 0 is the capped-loss rule LL.
  ZF      reported equity is reflected at zero and trading continues
Sleeve, fee, levels and payoff distribution as dp_full.py (DK training distribution, quantile-binned).
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
LIAB = Path(__file__).resolve().parents[1] / "LIABILITY_STUDY_2026-09-29"
sys.path.insert(0, str(LIAB))
import dp_theory                              # noqa: E402
from envs_b import FEE, K0, Q1, load_market   # noqa: E402

LEVELS = dp_theory.LEVELS
def binned(zone, bins):
    m = load_market(zone, "train", "RM")
    s, cap = np.abs(m.spread[200:]), m.cap_mwh[200:]
    rows = []
    for L in LEVELS:
        q = np.minimum(L * Q1, cap); z, f = q * s, FEE * q
        parts = np.array_split(np.argsort(z), bins)
        zb = np.array([z[p].mean() for p in parts]); fb = np.array([f[p].mean() for p in parts])
        rows.append(np.concatenate([zb - fb, -zb - fb]))
    return np.array(rows)                     # (levels, 2*bins) equally likely outcomes

def interp(V, grid, x):
    xc = np.clip(x, grid[0], grid[-1])
    i1 = np.clip(np.searchsorted(grid, xc, side="right"), 1, len(grid) - 1); i0 = i1 - 1
    w = (xc - grid[i0]) / (grid[i1] - grid[i0])
    return V[i0] * (1 - w) + V[i1] * w

def zombie_tables(X, W, gamma, ngrid):
    """Z_k(n) on n in [-6K, 6K] for k = 0..W remaining quarters; Z_0(n) = (-n)^+ (closure bonus)."""
    G = np.linspace(-6 * K0, 6 * K0, ngrid)
    Z = np.maximum(-G, 0.0)
    lev = np.zeros(ngrid)
    for k in range(1, W + 1):
        nxt = G[:, None, None] + X[None, :, :]
        Q = (X[None, :, :] + gamma * interp(Z, G, nxt)).mean(axis=2)
        best = np.argmax(Q - 1e-9 * K0 * np.arange(len(LEVELS))[None, :], axis=1)
        Z, lev = Q[np.arange(ngrid), best], LEVELS[best]
    return G, Z, lev                          # Z = value of entering the window with W quarters left

def solve(rule, X, gamma, W=0, ngrid=1201, iters=20000, tol=1e-6):
    E = np.concatenate([np.linspace(0, 0.5, 201), np.linspace(0.505, 6.0, 1000)]) * K0
    if rule.startswith("LAG"):
        Gz, Zw, zlev = zombie_tables(X, W, gamma, ngrid)
    V = np.zeros_like(E)
    nxt = E[:, None, None] + X[None, :, :]
    for it in range(iters):
        if rule == "FL":
            cont = np.where(nxt > 0, interp(V, E, nxt), 0.0); Q = (X[None] + gamma * cont).mean(axis=2)
        elif rule == "ZF":
            fl = np.maximum(nxt, 0.0); Q = ((fl - E[:, None, None]) + gamma * interp(V, E, fl)).mean(axis=2)
        else:  # LAG_W, W=0 -> LL
            if W == 0:
                fl = np.maximum(nxt, 0.0)
                cont = np.where(nxt > 0, interp(V, E, nxt), 0.0)
                Q = ((fl - E[:, None, None]) + gamma * cont).mean(axis=2)
            else:
                cont = np.where(nxt > 0, interp(V, E, nxt), interp(Zw, Gz, nxt))
                Q = (X[None] + gamma * cont).mean(axis=2)
        best = np.argmax(Q - 1e-9 * K0 * np.arange(len(LEVELS))[None, :], axis=1)
        Vn = Q[np.arange(len(E)), best]
        d = np.max(np.abs(Vn - V)); V = Vn
        if d < tol * K0: break
    gain = Q.max(axis=1) - Q[:, 0]
    return E, LEVELS[best], gain, it

def region(E, lev):
    """largest equity (allocations) up to which the optimal |lev| is >= 8 (half the maximum)."""
    ok = lev >= 8
    if not ok[0]: return 0.0
    j = np.argmax(~ok) if (~ok).any() else len(ok)
    return float(E[max(j - 1, 0)] / K0)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--bins", type=int, default=400)
    ap.add_argument("--gamma", type=float, default=0.99); ap.add_argument("--zone", default="DK1")
    ap.add_argument("--lags", type=int, nargs="+", default=[0, 4, 16, 48, 96, 288, 672])
    a = ap.parse_args()
    X = binned(a.zone, a.bins); rows = []
    probes = (0.02, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
    for rule, W in [("FL", 0)] + [("LAG", w) for w in a.lags] + [("ZF", 0)]:
        E, lev, gain, it = solve(rule, X, a.gamma, W)
        idx = {e: int(np.argmin(np.abs(E - e * K0))) for e in probes}
        r = dict(zone=a.zone, gamma=a.gamma, rule=rule if rule != "LAG" else ("LL" if W == 0 else f"LAG{W}"), W=W,
                 region_alloc=region(E, lev), iters=it + 1,
                 **{f"lev_{e:g}": float(lev[i]) for e, i in idx.items()}, **{f"gain_{e:g}": float(gain[i]) for e, i in idx.items()})
        rows.append(r); print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    pd.DataFrame(rows).to_csv(f"dp_ladder_{a.zone}_g{a.gamma}.csv", index=False)
