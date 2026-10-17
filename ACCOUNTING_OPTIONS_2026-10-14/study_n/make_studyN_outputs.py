"""Figure for Study N: learned |leverage| (mean of runs) against the DP benchmark (dp_n2_*.json)."""
import json, sys
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
TAG = sys.argv[1] if len(sys.argv) > 1 else ""
BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})
runs = [json.loads(f.read_text()) for f in sorted((HERE / f"results_n{TAG}" / "jobs").glob("*.json"))]
dp = json.loads((HERE / "dp_n2_DK1.json").read_text())
def mean_probe(arm, key):
    v = [r["probes"][key] for r in runs if r["arm"] == arm]
    return np.mean(v) if v else np.nan
fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0), gridspec_kw=dict(wspace=0.35))
# (a) zombie window: insolvent probes (k = 48) and solvent probes
xs_n = [-0.5, -0.25, -0.1, -0.02]; xs_e = [0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0]
for arm, col, lab in [("ZFG", ORANGE, "deficit forgiven (ZFG)"), ("ZBK", BLUE, "deficit booked (ZBK)")]:
    yi = [mean_probe(arm, f"zombie_n{n:g}_k48") for n in xs_n]; ys = [mean_probe(arm, f"solvent_e{e:g}") for e in xs_e]
    ax[0].plot(xs_n, yi, "o-", color=col, lw=1.6, ms=4, label=lab); ax[0].plot(xs_e, ys, "o-", color=col, lw=1.6, ms=4)
dpi = [dp["ZFG"]["zombie_lev"][f"n{n:g}_k48"] for n in xs_n]; dps = [dp["ZFG"]["solvent_lev"][f"e{e:g}"] for e in xs_e]
ax[0].plot(xs_n, dpi, "--", color=INK2, lw=1); ax[0].plot(xs_e, dps, "--", color=INK2, lw=1, label="optimum, forgiven")
ax[0].axvline(0, color=INK, lw=0.8); ax[0].set_xscale("symlog", linthresh=0.02)
ax[0].set_xlabel("Ledger / allocation\n(left of 0: insolvent, undetected, 48 quarter-hours left)"); ax[0].set_ylabel("|Leverage| chosen")
ax[0].set_title("(a) Zombie window, 1-day detection lag", loc="left", fontsize=9); ax[0].legend(frameon=False, fontsize=7)
# (b, c) terminal limited liability: maps over (L, tau)
Ls = [-0.25, -0.1, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]; Ts = [1, 4, 16, 48, 96, 192]
for a, arm, title in [(ax[1], "TFG", "(b) Terminal liability, forgiven (TFG)"), (ax[2], "TBK", "(c) Terminal liability, booked (TBK)")]:
    M = np.array([[mean_probe(arm, f"L{l:g}_tau{t}") for t in Ts] for l in Ls])
    im = a.imshow(M, cmap="Oranges", vmin=0, vmax=16, aspect="auto", origin="lower")
    a.set_xticks(range(len(Ts))); a.set_xticklabels(Ts); a.set_yticks(range(len(Ls))); a.set_yticklabels([f"{l:g}" for l in Ls])
    a.set_xlabel("Quarter-hours to the horizon"); a.set_ylabel("Ledger / allocation" if arm == "TFG" else ""); a.set_title(title, loc="left", fontsize=9); a.grid(False)
    D = np.array([[dp[arm]["lev"][f"L{l:g}_tau{t}"] for t in Ts] for l in Ls])
    for i in range(len(Ls)):
        for j in range(len(Ts)):
            a.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=6.5, color=INK if M[i, j] < 10 else "white")
            if D[i, j] >= 8: a.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=0.8))
fig.colorbar(im, ax=ax[1:], shrink=0.85, label="mean |leverage| (boxed: optimum gambles)")
fig.savefig(HERE / f"fig_studyN{TAG}.png", dpi=170, bbox_inches="tight"); fig.savefig(HERE / f"fig_studyN{TAG}.pdf", bbox_inches="tight")
print("ok", len(runs))
