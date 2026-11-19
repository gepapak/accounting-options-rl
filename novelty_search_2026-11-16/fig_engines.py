"""Figure (exploratory): mean reported return against leverage for the same trade in two engines and the ledger, 2,000 fair placebo
paths, with 95% bootstrap bands; and the share of paths that reach zero. Writes fig_engines.pdf and fig_engines.png."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
df = pd.read_csv(HERE / "engines_pilot.csv")
LEVS = sorted(df.L.unique())
P = df.path.nunique()
rng = np.random.default_rng(3)
idx = rng.integers(0, P, size=(2000, P))


def band(col):
    m = np.stack([df[df.L == L].sort_values("path")[col].to_numpy() for L in LEVS]) * 100
    bs = m[:, idx].mean(axis=2)
    return m.mean(axis=1), np.quantile(bs, 0.025, axis=1), np.quantile(bs, 0.975, axis=1)


fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.4), gridspec_kw=dict(width_ratios=[2.2, 1]))
series = [("btpy_rep", "backtesting.py, reported", "#c0392b", "-"), ("btpy_book", "backtesting.py, its own trade records", "#c0392b", ":"),
          ("bt_rep", "backtrader, reported (= ledger)", "#2c3e50", "-")]
for col, lab, c, ls in series:
    m, lo, hi = band(col)
    ax.plot(LEVS, m, ls, color=c, marker="o", ms=3.5, lw=1.6, label=lab)
    ax.fill_between(LEVS, lo, hi, color=c, alpha=0.12 if ls == "-" else 0.06, lw=0)
ax.axhline(0, color="0.6", lw=0.8)
ax.set_xscale("log")
ax.set_xticks(LEVS)
ax.set_xticklabels([str(L) for L in LEVS])
ax.set_xlabel("leverage of the held position")
ax.set_ylabel("mean return over paths (% of cash)")
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_title("Same trade, same fair paths", fontsize=9)
zero = [df[df.L == L].ref_hit.mean() * 100 for L in LEVS]
ax2.bar(range(len(LEVS)), zero, color="0.55")
ax2.set_xticks(range(len(LEVS)))
ax2.set_xticklabels([str(L) for L in LEVS], fontsize=7)
ax2.set_xlabel("leverage")
ax2.set_ylabel("paths reaching zero equity (%)")
ax2.set_title("Where the engines part", fontsize=9)
fig.tight_layout()
fig.savefig(HERE / "fig_engines.pdf")
fig.savefig(HERE / "fig_engines.png", dpi=160)
print("written fig_engines.pdf / .png")
