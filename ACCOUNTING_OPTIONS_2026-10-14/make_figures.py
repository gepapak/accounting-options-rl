"""Figures for the accounting-options extension (palette: documented reference slots 1-3, validated all-pairs)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})

# Figure A: the ladder, subsidy per MW-year from zero equity, DK1 and DK2, first vs last regime
ml = pd.read_csv("market_ladder.csv")
rules = ["LL", "LAG_1d", "LAG_1w", "LAG_3w", "TERM", "ZF"]
labels = ["Capped\nloss", "Lag\n1 day", "Lag\n1 week", "Lag 3 weeks\n(Nordic)", "Terminal\nlimited liab.", "Zero floor\n(simulator)"]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for ax, zone in zip(axes, ["DK1", "DK2"]):
    d = ml[(ml.zone == zone) & (ml.start == "e=0")]
    r1 = d[d.regime.str.startswith("R1")][rules].iloc[0].values
    r4 = d[d.regime.str.startswith("R4")][rules].iloc[0].values
    x = np.arange(len(rules)); w = 0.38
    ax.bar(x - w / 2 - 0.01, r1, w, color=BLUE, label="2019-11 to 2021-10", zorder=3)
    ax.bar(x + w / 2 + 0.01, r4, w, color=ORANGE, label="2025-03 to 2026-08 (after the balancing reform)", zorder=3)
    ax.set_yscale("log"); ax.set_ylim(1, 2e4)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
    for xi, v in zip(x + w / 2, r4):
        ax.text(xi, v * 1.15, f"{v:,.0f}", ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_title(zone, color=INK, fontsize=10, loc="left")
axes[0].set_ylabel("Forgiven loss, EUR per MW-year (log)")
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.suptitle("Forgiven loss by accounting rule: Danish spreads, placebo signs, 1 MW, fee 0.10 EUR/MWh, from zero equity",
             fontsize=9.5, color=INK, x=0.01, ha="left")
fig.tight_layout(); fig.savefig("fig_ladder.pdf"); fig.savefig("fig_ladder.png", dpi=170); plt.close(fig)

# Figure B: monthly coverage of level-indexed collateral relative to the incentive-compatible buffer
cv = pd.read_csv("coverage_monthly.csv").replace(np.inf, np.nan)
cv["t"] = pd.PeriodIndex(cv.month, freq="M").to_timestamp()
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
for ax, zone in zip(axes, ["DK1", "DK2"]):
    d = cv[cv.zone == zone]
    ax.plot(d.t, d["cov_ESETT_V_0.1"], color=BLUE, lw=1.6, label="eSett volume term, m(V1+V2)P", zorder=3)
    ax.plot(d.t, d["cov_DK_C1_0.1"], color=ORANGE, lw=1.6, label="Energinet C1 consumption rule, 5% x month x forward", zorder=3)
    ax.axhline(1.0, color=INK, lw=0.9, ls="--", zorder=2)
    for dt_, lab in [("2021-11-01", "single price"), ("2025-03-04", "mFRR EAM go-live")]:
        ax.axvline(pd.Timestamp(dt_), color=INK2, lw=0.7, ls=":", zorder=1)
        ax.text(pd.Timestamp(dt_), 13.5, " " + lab, fontsize=7, color=INK2, va="top")
    ax.set_yscale("log"); ax.set_ylim(0.2, 15)
    ax.set_title(zone, color=INK, fontsize=10, loc="left")
axes[0].set_ylabel("Collateral / incentive-compatible buffer")
axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")
fig.suptitle("Collateral per MW / buffer that deters a 3-week limited-liability gamble (cost 0.10 EUR/MWh); below 1 = not deterred",
             fontsize=9, color=INK, x=0.01, ha="left")
fig.tight_layout(); fig.savefig("fig_coverage.pdf"); fig.savefig("fig_coverage.png", dpi=170); plt.close(fig)
print("ok")
