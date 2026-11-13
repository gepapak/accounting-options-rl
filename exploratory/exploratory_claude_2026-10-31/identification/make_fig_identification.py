"""Figure for the identification section (descriptive): fig_identification.pdf / .png

(a) The reported reward is a censored observation of the booked P&L, r = max(b, tau); the sign twin gives two points of this curve,
    symmetric about -c (b = +-g - c).
(b) Mean absolute error of each reward against the booked P&L on credited steps, by accounting rule (ladder_identification_results.csv.gz).

    python make_fig_identification.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#a9a7a1"
plt.rcParams.update({"font.family": "serif", "font.size": 9.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "legend.frameon": False, "pdf.fonttype": 42})


def panel_a(ax):
    tau, c = 0.0, 0.5
    b = np.linspace(-3.4, 3.4, 400)
    ax.plot(b, b, color=GREY, lw=0.9, ls="--", zorder=1)
    ax.plot(b, np.maximum(b, tau), color=INK, lw=1.5, zorder=2)
    ax.text(2.35, 2.05, "booked $b$", color=MUTED, fontsize=8.5, rotation=40)
    ax.text(-3.3, 0.14, "reported $r=\\max(b,\\tau)$", color=INK, fontsize=8.5)
    # identified pair: g = 2.0  ->  b = +1.5 (uncensored) and -2.5 (censored): r = 1.5, r' = 0
    ax.scatter([1.5, -2.5], [1.5, 0.0], s=36, color=BLUE, zorder=4)
    ax.annotate("", xy=(-2.5, 0.05), xytext=(1.5, 1.5), arrowprops=dict(arrowstyle="<->", color=BLUE, lw=0.9, shrinkA=4, shrinkB=4))
    # censored pair: g = 0.3  ->  b = -0.2 and -0.8: r = r' = 0
    ax.scatter([-0.2, -0.8], [0.0, 0.0], s=36, color=ORANGE, zorder=4)
    ax.annotate("", xy=(-0.8, -0.03), xytext=(-0.2, -0.03), arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=0.9, shrinkA=3, shrinkB=3))
    ax.text(-3.3, 2.65, "twin pair with $r\\neq r'$:\n$g=\\max(r,r')+c$ exactly", color=BLUE, fontsize=8.5, va="top")
    ax.text(-3.3, 1.75, "twin pair with $r=r'$:\n$g\\in[0,\\,r+c]$ only", color=ORANGE, fontsize=8.5, va="top")
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-0.35, 2.9)
    ax.set_xlabel("booked P&L $b$ (pairs are symmetric about $-c$)")
    ax.set_ylabel("reported reward $r$")
    ax.set_title("(a) The credit censors the reward", loc="left", fontsize=9.5, color=INK)
    ax.grid(True, color=GRID, lw=0.5, zorder=0)


def panel_b(ax):
    df = pd.read_csv(HERE / "ladder_identification_results.csv.gz")
    cred = df[df.credited & (df.market == "edge")]
    rules = [("ZF", 0.0, "Zero\nfloor"), ("LL", 0.0, "Limited\nliability"), ("TERM", 0.5, "Terminal\n(carried)"), ("LAG", 0.5, "Settlement\nlag (carried)")]
    est = [("reported_abs", "Reported", ORANGE), ("spr_abs", "SPR", GREY), ("tmr_abs", "TMR", MUTED), ("tir_abs", "TIR", BLUE)]
    w = 0.2
    floor = 2e-4
    for j, (col, name, color) in enumerate(est):
        vals = [float(cred[(cred.rule == r) & (cred.p_insolvent == p)][col].mean()) for r, p, _ in rules]
        for i, v in enumerate(vals):
            x = i + (j - 1.5) * w
            if v < 1e-6:                                       # exactly zero: mark it instead of drawing a bar
                ax.text(x, floor * 1.5, "0", ha="center", va="bottom", fontsize=8, color=color)
            else:
                ax.bar(x, v, w * 0.92, color=color, zorder=2, label=name if i == 0 else None)
    ax.set_yscale("log")
    ax.set_ylim(floor, 4e3)
    ax.set_xticks(range(len(rules)), [r[2] for r in rules], fontsize=8.5)
    ax.set_ylabel("Mean abs. error vs booked P&L\n(% of allocation per quarter)")
    ax.grid(True, axis="y", color=GRID, lw=0.5, zorder=0)
    h, l = ax.get_legend_handles_labels()
    order = [l.index(n) for n in ("Reported", "SPR", "TMR", "TIR")]
    ax.legend([h[i] for i in order], [l[i] for i in order], ncol=4, fontsize=8.5, loc="upper center", handlelength=1.1, columnspacing=1.0)
    ax.set_title("(b) Error on credited steps", loc="left", fontsize=9.5, color=INK)


if __name__ == "__main__":
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.3), gridspec_kw=dict(width_ratios=[1.0, 1.15]))
    panel_a(a)
    panel_b(b)
    fig.tight_layout(w_pad=1.6)
    fig.savefig(HERE / "fig_identification.pdf")
    fig.savefig(HERE / "fig_identification.png", dpi=180)
    print("wrote fig_identification.pdf/.png")
