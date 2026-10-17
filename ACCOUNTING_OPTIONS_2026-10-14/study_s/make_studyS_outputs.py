"""Figure for Study S (written after the verdicts; descriptive only): fig_studyS.pdf / .png from runs_s_A.csv and runs_s_B.csv.

    python make_studyS_outputs.py
"""
from __future__ import annotations

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
rng = np.random.default_rng(7)

ARMS_A = [("PLAIN", "Reported", ORANGE), ("ORACLE", "Ledger", GREY), ("SPR", "SPR", BLUE),
          ("KL0.003", "KL .003", GREY), ("KL0.03", "KL .03", GREY), ("KL0.3", "KL .3", GREY)]


def boot_ci(x, n=5000):
    x = np.asarray(x, float)
    m = np.array([rng.choice(x, len(x)).mean() for _ in range(n)])
    return np.quantile(m, [0.025, 0.975])


def bars(ax, A, col, ylabel, title, pct=False):
    for i, (arm, label, color) in enumerate(ARMS_A):
        v = A.loc[A.arm == arm, col].to_numpy() * (100 if pct else 1)
        lo, hi = boot_ci(v)
        ax.bar(i, v.mean(), 0.62, color=color, alpha=0.85 if color != GREY else 0.6, zorder=2)
        ax.errorbar(i, v.mean(), yerr=[[v.mean() - lo], [hi - v.mean()]], fmt="none", ecolor=INK, lw=0.8, capsize=2, zorder=4)
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=7, color=INK, alpha=0.45, lw=0, zorder=3)
    ax.set_xticks(range(len(ARMS_A)), [a[1] for a in ARMS_A], rotation=35, ha="right")
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", color=GRID, lw=0.5, zorder=0)
    ax.set_title(title, loc="left", fontsize=9.5, color=INK)


def main():
    A = pd.read_csv(HERE / "runs_s_A.csv")
    B = pd.read_csv(HERE / "runs_s_B.csv")
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.2, 3.1), gridspec_kw=dict(width_ratios=[1.1, 1.1, 1.15]))
    bars(a, A, "E", "Booked edge (% capital / week)", "(a) Booked edge")
    bars(b, A, "X_A", "Probes at max. leverage (%)", "(b) Gambling at low equity", pct=True)
    for j, (ln, label) in enumerate((("PPO", "PPO"), ("A2C-n64", "A2C"))):
        for k, (arm, name, color) in enumerate((("ZF", "Reported", ORANGE), ("SPR", "SPR", BLUE), ("LEDGER", "Ledger", GREY))):
            v = B.loc[(B.learner == ln) & (B.arm == arm), "X_B"].to_numpy()
            x = j * 3.6 + k
            lo, hi = boot_ci(v)
            c.scatter(x + rng.uniform(-0.22, 0.22, len(v)), v, s=7, color=color, alpha=0.55, lw=0, zorder=3)
            c.errorbar(x, v.mean(), yerr=[[v.mean() - lo], [hi - v.mean()]], fmt="o", color=INK, ms=3.5, lw=0.8, capsize=2, zorder=4)
        c.text(j * 3.6 + 1, 17.4, label, ha="center", fontsize=9.5, color=INK)
    c.set_xticks([0, 1, 2, 3.6, 4.6, 5.6], ["Reported", "SPR", "Ledger"] * 2, rotation=35, ha="right")
    c.axhline(0, color=MUTED, lw=0.6)
    c.set_ylim(-17, 18.5)
    c.set_ylabel("Exploit shape L(0.02) - L(2)")
    c.grid(True, axis="y", color=GRID, lw=0.5, zorder=0)
    c.set_title("(c) Fair market", loc="left", fontsize=9.5, color=INK)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(HERE / "fig_studyS.pdf")
    fig.savefig(HERE / "fig_studyS.png", dpi=180)
    print("wrote fig_studyS.pdf/.png")


if __name__ == "__main__":
    main()
