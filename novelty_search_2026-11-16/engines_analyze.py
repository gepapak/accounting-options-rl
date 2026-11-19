"""Analysis of engines_pilot.csv (exploratory). For each leverage: mean reported and booked return per engine with 95% bootstrap
intervals over paths; how often an optimizer of the mean reported return (argmax over L) picks each L under bootstrap resampling of
paths; the share of paths on which the two engines' reported returns differ in sign; and backtesting.py's gap between its reported
equity and its own trade records (the forgiven loss)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
df = pd.read_csv(HERE / "engines_pilot.csv")
LEVS = sorted(df.L.unique())
P = df.path.nunique()
rng = np.random.default_rng(7)
B = 4000
idx = rng.integers(0, P, size=(B, P))


def mat(col):
    return np.stack([df[df.L == L].sort_values("path")[col].to_numpy() for L in LEVS])   # L x paths


def ci(m):
    bs = m[:, idx].mean(axis=2)                                                           # L x B
    return m.mean(axis=1), np.quantile(bs, 0.025, axis=1), np.quantile(bs, 0.975, axis=1), bs


print(f"{P} paths per leverage, returns in % of the starting cash, 95% bootstrap intervals over paths")
cols = {"backtesting.py reported": "btpy_rep", "backtesting.py booked": "btpy_book", "backtrader reported": "bt_rep", "ledger": "ledger"}
res = {k: ci(mat(c)) for k, c in cols.items()}
print(f"{'L':>4} {'reach0':>7} | " + " | ".join(f"{k:^28}" for k in cols))
zero = mat("ref_hit").mean(axis=1)
for i, L in enumerate(LEVS):
    cells = [f"{100 * m[i]:+8.2f} [{100 * lo[i]:+8.2f},{100 * hi[i]:+8.2f}]" for (m, lo, hi, _) in res.values()]
    print(f"{L:>4} {zero[i]:7.1%} | " + " | ".join(cells))
for k in ("backtesting.py reported", "backtrader reported"):
    bs = res[k][3]
    pick = np.array(LEVS)[bs.argmax(axis=0)]
    vals, cnt = np.unique(pick, return_counts=True)
    print(f"optimizer of the mean {k} return picks: " + ", ".join(f"L={v} in {c / B:.1%}" for v, c in zip(vals, cnt)))
a, b = mat("btpy_rep"), mat("bt_rep")
for i, L in enumerate(LEVS):
    if L >= 20:
        print(f"L={L}: paths with opposite signs {np.mean(np.sign(a[i]) != np.sign(b[i])):.1%}; mean sign: backtesting.py {np.sign(a[i].mean()):+.0f}, backtrader {np.sign(b[i].mean()):+.0f}")
gap = mat("btpy_rep") - mat("btpy_book")
print("backtesting.py: reported minus its own trade records (forgiven loss), mean over paths: " +
      ", ".join(f"L={L} {100 * gap[i].mean():+.2f}" for i, L in enumerate(LEVS)))
exact = (np.abs(mat("btpy_rep") - mat("floor_stop")).max(), np.abs(mat("bt_rep") - mat("ledger")).max())
print(f"largest deviation from the references: backtesting.py vs FLOOR+STOP {100 * exact[0]:.2e} points; backtrader vs LEDGER {100 * exact[1]:.2e} points")
agree = [L for L in LEVS if np.abs(a[LEVS.index(L)] - b[LEVS.index(L)]).max() < 1e-9]
print("leverages at which the two engines agree on every path: " + ", ".join(str(L) for L in agree))
