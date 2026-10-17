"""Twelve-zone heat strip: coverage of the eSett volume term relative to the deterrence buffer, monthly 2022-2026."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10.5, "axes.labelsize": 10.5, "xtick.labelsize": 10.5,
                     "ytick.labelsize": 10.5, "pdf.fonttype": 42})
cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
d = pd.read_csv("coverage12_fee0.1.csv")
d["cov"] = d.LEVEL / d.kappa.replace(0, np.nan)
order = ["NO1", "NO2", "NO5", "SE1", "SE2", "SE3", "SE4", "FI", "NO3", "NO4", "DK1", "DK2"]
P = d.pivot(index="zone", columns="month", values="cov").reindex(order)
months = list(P.columns)
fig, ax = plt.subplots(figsize=(7.4, 4.1))
im = ax.imshow(np.log10(P.values.astype(float)), cmap=cmap, norm=TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1.3), aspect="auto")
ax.set_yticks(range(len(order))); ax.set_yticklabels([z + (" *" if z.startswith("DK") else " †" if z == "FI" else "") for z in order])
for zone in ("DK1", "DK2"):
    ax.add_patch(Rectangle((-0.5, order.index(zone) - 0.5), len(months), 1, facecolor="none",
                           edgecolor="#777777", hatch="//", linewidth=0))
fi_start = months.index("2024-12")
ax.add_patch(Rectangle((fi_start - 0.5, order.index("FI") - 0.5), len(months) - fi_start, 1,
                       facecolor="none", edgecolor="#777777", hatch="//", linewidth=0))
ticks = [i for i, m in enumerate(months) if m.endswith("-01")]
ax.set_xticks(ticks); ax.set_xticklabels([months[i][:4] for i in ticks])
rf = months.index("2025-03")
ax.axvline(rf - 0.5, color=INK, lw=1.2)
cb = fig.colorbar(im, ax=ax, shrink=0.85, ticks=np.log10([0.1, 0.3, 1, 3, 10, 20]))
cb.ax.set_yticklabels(["0.1", "0.3", "1", "3", "10", "20"]); cb.set_label("Component / modeled buffer")
ax.set_title("Historical volume component / modeled buffer\nExternal owner risk cost: 0.10 EUR/MWh", fontsize=10.5, loc="left", color=INK)
fig.text(0.10, 0.045, "Hatched: * DK; † FI from Dec 2024 (change: 30 Nov).", fontsize=10.5, color=INK2)
fig.text(0.10, 0.005, "Counterfactual components. Black line: March 2025.", fontsize=10.5, color=INK2)
fig.tight_layout(rect=(0, 0.09, 1, 1)); fig.savefig("fig_coverage12.png", dpi=170); fig.savefig("fig_coverage12.pdf")
print("ok")
