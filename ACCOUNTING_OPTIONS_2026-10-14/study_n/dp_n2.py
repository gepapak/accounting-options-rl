"""DP benchmark for the registered Study N design (envs_n.py after pilot 1): fee FEE_N, zombie forgiven arm with the
owner's claim max(L, 0) reported once insolvent (reward = change in the claim), booked arms reward = P&L.
Otherwise as dp_n.py (quantile-binned training distribution, discount 0.99)."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dp_n_fee import binned, Interp, argbest, LEVELS, GAMMA, K0   # noqa: E402
from envs_n import FEE_N, W_LAG, H_TERM                         # noqa: E402

def zombie(X, W, forgiven, grid):
    I = Interp(grid, X)
    if forgiven:
        r = (np.maximum(I.pts, 0.0) - np.maximum(grid, 0.0)[:, None, None]).mean(axis=2)
    else:
        r = np.broadcast_to(X.mean(axis=1)[None, :], (len(grid), X.shape[0]))
    Z = np.zeros_like(grid); tabs, pols = [Z], [np.zeros(len(grid), int)]
    for k in range(1, W + 1):
        Q = r + GAMMA * I(Z).mean(axis=2); b = argbest(Q); Z = Q[np.arange(len(grid)), b]; tabs.append(Z); pols.append(b)
    return tabs, pols, r

def solvent(X, Zw, gridZ, grid, forgiven):
    nxt = grid[:, None, None] + X[None]; Ipos = Interp(grid, X)
    zval = np.interp(np.clip(nxt, gridZ[0], gridZ[-1]), gridZ, Zw)
    rew = np.where((nxt <= 0) & forgiven, -grid[:, None, None], X[None])   # forgiven breach: claim falls to 0
    rm = rew.mean(axis=2)
    V = np.zeros_like(grid)
    for it in range(20000):
        Q = rm + GAMMA * np.where(nxt > 0, Ipos(V), zval).mean(axis=2)
        b = argbest(Q); Vn = Q[np.arange(len(grid)), b]; d = np.max(np.abs(Vn - V)); V = Vn
        if d < 1e-6 * K0: break
    return b, Q

def terminal(X, H, forgiven, grid):
    I = Interp(grid, X); xm = X.mean(axis=1)[None, :]
    V = np.maximum(-grid, 0.0) if forgiven else np.zeros_like(grid); pols, gains = [None], [None]
    for tau in range(1, H + 1):
        Q = xm + GAMMA * I(V).mean(axis=2); b = argbest(Q); V = Q[np.arange(len(grid)), b]; pols.append(b); gains.append(Q.max(1) - Q[:, 0])
    return pols, gains

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--zone", default="DK1"); ap.add_argument("--bins", type=int, default=300)
    a = ap.parse_args()
    X = binned(a.zone, a.bins, FEE_N)
    gridZ = np.linspace(-3 * K0, 3 * K0, 1201)
    gridS = np.concatenate([np.linspace(0, 0.5, 201)[1:], np.linspace(0.505, 6.0, 900)]) * K0
    out = {"zone": a.zone, "fee": FEE_N, "W": W_LAG, "H": H_TERM}
    for forgiven, tag in ((True, "ZFG"), (False, "ZBK")):
        tabs, pols, r = zombie(X, W_LAG, forgiven, gridZ)
        zl, zg = {}, {}
        for n in (-0.5, -0.25, -0.1, -0.02):
            i = int(np.argmin(np.abs(gridZ - n * K0)))
            for k in (96, 48, 12):
                I = Interp(gridZ, X); Q = r + GAMMA * I(tabs[k - 1]).mean(axis=2)
                zl[f"n{n:g}_k{k}"] = float(LEVELS[pols[k][i]]); zg[f"n{n:g}_k{k}"] = float(Q[i].max() - Q[i, 0])
        b, Q = solvent(X, tabs[W_LAG], gridZ, gridS, forgiven)
        sl = {f"e{e:g}": float(LEVELS[b[int(np.argmin(np.abs(gridS - e * K0)))]]) for e in (0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 3)}
        sg = {f"e{e:g}": float((Q.max(1) - Q[:, 0])[int(np.argmin(np.abs(gridS - e * K0)))]) for e in (0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 3)}
        out[tag] = dict(zombie_lev=zl, zombie_gain_eur=zg, solvent_lev=sl, solvent_gain_eur=sg)
        print(tag, "zombie", zl, "\n   gain", {k: round(v, 1) for k, v in zg.items()}, "\n   solvent", sl, {k: round(v, 2) for k, v in sg.items()}, flush=True)
    gridT = np.linspace(-3 * K0, 6 * K0, 1801)
    LP, TP = (-0.25, -0.1, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0), (1, 4, 16, 48, 96, 192)
    for forgiven, tag in ((True, "TFG"), (False, "TBK")):
        pols, gains = terminal(X, H_TERM, forgiven, gridT)
        tl = {f"L{l:g}_tau{t}": float(LEVELS[pols[t][int(np.argmin(np.abs(gridT - l * K0)))]]) for l in LP for t in TP}
        tg = {f"L{l:g}_tau{t}": float(gains[t][int(np.argmin(np.abs(gridT - l * K0)))]) for l in LP for t in TP}
        out[tag] = dict(lev=tl, gain_eur=tg)
        print(tag, "\n", pd.Series(tl).values.reshape(len(LP), -1), "\n gain\n", np.round(pd.Series(tg).values.reshape(len(LP), -1), 1), flush=True)
    Path(f"dp_n2_{a.zone}.json").write_text(json.dumps(out, indent=1))
