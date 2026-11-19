"""Figure: the monitoring law against its tests. Writes fig_monitoring_law.pdf and .png.
(a) capped loss in unmodified backtesting.py: forgiven amount against checks per day, engine points with 95% intervals and the law's lines
(b) zero floor over terminal limited liability against checks per day: simulation points and the law's lines (additive model)
(c) the paper's 24 Nordic zone-years: measured ratio against the law's ratio (post hoc form)"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
C = json.loads((HERE / "test_c_engine.json").read_text())["rows"]
A2 = json.loads((HERE / "test_a2_sim.json").read_text())["A2"]["rows"]
P1 = json.loads((HERE / "predictions.json").read_text())["A"]
N = pd.read_csv(HERE / "test_d4b_nordic.csv")

fig, ax = plt.subplots(1, 3, figsize=(11.2, 3.5))
col = {20: "#7f8c8d", 50: "#2980b9", 200: "#c0392b"}
for L in (20, 50, 200):
    r = [x for x in C if x["L"] == L and x["fee"] == 0.0]
    m = [x["m"] for x in r]
    ax[0].plot(m, [100 * x["pred"] for x in r], "-", color=col[L], lw=1.4)
    ax[0].errorbar(m, [100 * x["forgiven"] for x in r], yerr=[[100 * (x["forgiven"] - x["lo"]) for x in r], [100 * (x["hi"] - x["forgiven"]) for x in r]],
                   fmt="o", color=col[L], ms=4, capsize=2, label=f"leverage {L}")
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xticks([1, 4, 16]); ax[0].set_xticklabels(["1", "4", "16"])
ax[0].set_xlabel("ruin checks per day"); ax[0].set_ylabel("forgiven loss (% of equity)")
ax[0].set_title("(a) Capped loss, unmodified backtesting.py", fontsize=9)
ax[0].legend(frameon=False, fontsize=8)
for L in (20, 50, 200):
    r = [x for x in A2 if x["L"] == L]
    p = [x for x in P1 if x["L"] == L]
    m = [x["m"] for x in r]
    ax[1].plot(m, [x["ZF"] / x["TERM"] for x in p], "-", color=col[L], lw=1.4)
    ax[1].plot(m, [x["ZF"] / x["TERM"] for x in r], "o", color=col[L], ms=4)
ax[1].axhline(2.0, color="0.6", lw=0.8, ls="--")
ax[1].set_xscale("log"); ax[1].set_xticks([1, 4, 16, 64]); ax[1].set_xticklabels(["1", "4", "16", "64"])
ax[1].set_xlabel("ruin checks per day"); ax[1].set_ylabel("zero floor / terminal limited liability")
ax[1].set_title("(b) Zero floor: rises to 2", fontsize=9)
ax[2].errorbar(N.pred_b, N.measured, yerr=2 * N.se, fmt="o", color="#2c3e50", ms=3.5, capsize=1.5, lw=0.7)
lim = [1.6, 2.0]
ax[2].plot(lim, lim, "-", color="0.6", lw=0.8)
ax[2].set_xlim(lim); ax[2].set_ylim(lim)
ax[2].set_xlabel("law: 2 - sqrt(2 pi) rho / sigma_year"); ax[2].set_ylabel("measured (placebo signs, real spreads)")
ax[2].set_title("(c) 24 Nordic zone-years", fontsize=9)
fig.tight_layout()
fig.savefig(HERE / "fig_monitoring_law.pdf"); fig.savefig(HERE / "fig_monitoring_law.png", dpi=160)
print("written")
