"""DP benchmarks for Study N (written before any training): optimal |leverage| of the Study F placebo sleeve
in two matched settings, each with a FORGIVEN and a BOOKED variant that differ only in the accounting of the deficit.

Z (zombie window, settlement lag W quarter-hours): when the ledger first reaches <= 0 the breach is detected W quarters
  later; trading continues meanwhile (state: ledger n, clock k); at detection the sleeve closes.
  ZFG: reported equity at closure = max(n, 0) (deficit forgiven: bonus (-n)^+);  ZBK: deficit booked (no bonus).
T (terminal limited liability, horizon H): the ledger may go negative and trading continues; at the horizon
  TFG: bonus (-L_H)^+ (Merton's call);  TBK: no bonus.
Reward = change in reported equity (EUR); discount 0.99 per quarter-hour (the learners'); payoff distribution:
quantile-binned training-period P&L per leverage level (as dp_full.py), DK1 or DK2.
"""
from __future__ import annotations
import argparse, sys, json
from pathlib import Path
import numpy as np, pandas as pd
LIAB = Path(r"C:/Users/gepap/Desktop/Thesis/Paper1/rl-trading-reality-check-main/LIABILITY_STUDY_2026-09-29")
sys.path.insert(0, str(LIAB))
from envs_b import FEE, K0, Q1, LEV, load_market   # noqa: E402
LEVELS = np.unique(np.abs(LEV))
GAMMA = 0.99

def binned(zone, bins, fee=FEE):
    m = load_market(zone, "train", "RM")
    s, cap = np.abs(m.spread[200:]), m.cap_mwh[200:]
    rows = []
    for L in LEVELS:
        q = np.minimum(L * Q1, cap); z, f = q * s, fee * q
        parts = np.array_split(np.argsort(z), bins)
        zb = np.array([z[p].mean() for p in parts]); fb = np.array([f[p].mean() for p in parts])
        rows.append(np.concatenate([zb - fb, -zb - fb]))
    return np.array(rows)

class Interp:
    """precomputed linear interpolation of a value vector on GRID at points GRID[:,None,None] + X."""
    def __init__(self, grid, X):
        pts = np.clip(grid[:, None, None] + X[None], grid[0], grid[-1])
        self.i1 = np.clip(np.searchsorted(grid, pts, side="right"), 1, len(grid) - 1); self.i0 = self.i1 - 1
        self.w = (pts - grid[self.i0]) / (grid[self.i1] - grid[self.i0]); self.pts = grid[:, None, None] + X[None]
    def __call__(self, V):
        return V[self.i0] * (1 - self.w) + V[self.i1] * self.w

def argbest(Q):
    return np.argmax(Q - 1e-9 * K0 * np.arange(Q.shape[1])[None, :], axis=1)

def zombie(X, W, forgiven, grid):
    """Z[k] = value of the window with k quarters left at ledger n (k=0: closure). Returns tables and policy."""
    I = Interp(grid, X); xm = X.mean(axis=1)[None, :]
    Z = np.maximum(-grid, 0.0) if forgiven else np.zeros_like(grid)
    tabs, pols = [Z], [np.zeros(len(grid), int)]
    for k in range(1, W + 1):
        Q = xm + GAMMA * I(Z).mean(axis=2)
        b = argbest(Q); Z = Q[np.arange(len(grid)), b]
        tabs.append(Z); pols.append(b)
    return tabs, pols

def solvent(X, Zw, gridZ, grid):
    """infinite-horizon value for solvent states; breach -> enter window valued by Zw (interpolated on gridZ)."""
    nxt = grid[:, None, None] + X[None]
    Ipos = Interp(grid, X)
    zval = np.interp(np.clip(nxt, gridZ[0], gridZ[-1]), gridZ, Zw)
    xm = X.mean(axis=1)[None, :]
    V = np.zeros_like(grid)
    for it in range(20000):
        cont = np.where(nxt > 0, Ipos(V), zval)
        Q = xm + GAMMA * cont.mean(axis=2)
        b = argbest(Q); Vn = Q[np.arange(len(grid)), b]
        d = np.max(np.abs(Vn - V)); V = Vn
        if d < 1e-6 * K0: break
    return V, b, Q

def terminal(X, H, forgiven, grid):
    I = Interp(grid, X); xm = X.mean(axis=1)[None, :]
    V = np.maximum(-grid, 0.0) if forgiven else np.zeros_like(grid)
    pols, gains = [None], [None]
    for tau in range(1, H + 1):
        Q = xm + GAMMA * I(V).mean(axis=2)
        b = argbest(Q); V = Q[np.arange(len(grid)), b]
        pols.append(b); gains.append(Q.max(axis=1) - Q[:, 0])
    return pols, gains

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--zone", default="DK1"); ap.add_argument("--bins", type=int, default=300)
    ap.add_argument("--W", type=int, default=96); ap.add_argument("--H", type=int, default=192); ap.add_argument("--fee", type=float, default=FEE)
    a = ap.parse_args()
    X = binned(a.zone, a.bins, a.fee)
    gridZ = np.linspace(-3 * K0, 3 * K0, 1201)
    gridS = np.concatenate([np.linspace(0, 0.5, 201)[1:], np.linspace(0.505, 6.0, 900)]) * K0
    out = {"zone": a.zone, "W": a.W, "H": a.H}
    NPROBE = (-0.5, -0.25, -0.1, -0.02); KPROBE = (a.W, a.W // 2, a.W // 8, 1)
    EPROBE = (0.02, 0.1, 0.25, 0.5, 1.0, 2.0)
    for forgiven, tag in ((True, "ZFG"), (False, "ZBK")):
        tabs, pols = zombie(X, a.W, forgiven, gridZ)
        zl = {f"n{n:g}_k{k}": float(LEVELS[pols[k][int(np.argmin(np.abs(gridZ - n * K0)))]]) for n in NPROBE for k in KPROBE}
        zg = {}
        for n in NPROBE:
            for k in KPROBE:
                i = int(np.argmin(np.abs(gridZ - n * K0)))
                Ik = Interp(gridZ, X); Q = X.mean(axis=1)[None, :] + GAMMA * Ik(tabs[k - 1]).mean(axis=2)
                zg[f"n{n:g}_k{k}"] = float(Q[i].max() - Q[i, 0])
        V, b, Q = solvent(X, tabs[a.W], gridZ, gridS)
        sl = {f"e{e:g}": float(LEVELS[b[int(np.argmin(np.abs(gridS - e * K0)))]]) for e in EPROBE}
        sg = {f"e{e:g}": float((Q.max(axis=1) - Q[:, 0])[int(np.argmin(np.abs(gridS - e * K0)))]) for e in EPROBE}
        out[tag] = {"zombie_lev": zl, "zombie_gain_eur": zg, "solvent_lev": sl, "solvent_gain_eur": sg}
        print(tag, "zombie lev", zl); print(tag, "zombie gain", {k: round(v, 2) for k, v in zg.items()})
        print(tag, "solvent lev", sl, "gain", {k: round(v, 2) for k, v in sg.items()}, flush=True)
    gridT = np.linspace(-3 * K0, 6 * K0, 1801)
    LPROBE = (-0.25, -0.1, 0.02, 0.1, 0.25, 0.5, 1.0); TPROBE = (1, 4, 16, 48, 96, 192)
    for forgiven, tag in ((True, "TFG"), (False, "TBK")):
        pols, gains = terminal(X, a.H, forgiven, gridT)
        tl = {f"L{l:g}_tau{t}": float(LEVELS[pols[t][int(np.argmin(np.abs(gridT - l * K0)))]]) for l in LPROBE for t in TPROBE if t <= a.H}
        tg = {f"L{l:g}_tau{t}": float(gains[t][int(np.argmin(np.abs(gridT - l * K0)))]) for l in LPROBE for t in TPROBE if t <= a.H}
        out[tag] = {"lev": tl, "gain_eur": tg}
        print(tag, pd.Series(tl).values.reshape(len(LPROBE), -1)); print(tag, "gain", np.round(pd.Series(tg).values.reshape(len(LPROBE), -1), 2), flush=True)
    Path(f"dp_n_{a.zone}_fee{a.fee}.json").write_text(json.dumps(out, indent=1))
