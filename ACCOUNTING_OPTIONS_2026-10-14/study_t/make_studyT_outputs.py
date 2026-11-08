"""Figure for Study T (written after the verdicts; descriptive only): fig_studyT.pdf / .png from runs_t.csv.

    python make_studyT_outputs.py [--runs runs_t.csv] [--out DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREY, TEAL, PLUM = "#2a78d6", "#eb6834", "#a9a7a1", "#1f9d8a", "#8a6bbe"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "legend.frameon": False, "pdf.fonttype": 42})
rng = np.random.default_rng(7)

CORE = [("PLAIN", "Reported", ORANGE), ("ORACLE", "Ledger", GREY), ("SPR", "SPR", BLUE), ("TIR", "TIR", TEAL)]
ORPO = [("OM1", "ORPO 1", PLUM), ("OM0.1", "ORPO .1", PLUM), ("OM0.05", "ORPO .05", PLUM), ("OM0.02", "ORPO .02", PLUM), ("OM0.01", "ORPO .01", PLUM)]


def boot_ci(x, n=5000):
    x = np.asarray(x, float)
    m = np.array([rng.choice(x, len(x)).mean() for _ in range(n)])
    return np.quantile(m, [0.025, 0.975])


def draw(ax, d, col, arms, x0=0.0, pct=False):
    for i, (arm, _, color) in enumerate(arms):
        v = d.loc[d.arm == arm, col].to_numpy() * (100 if pct else 1)
        if not len(v):
            continue
        lo, hi = boot_ci(v)
        x = x0 + i
        ax.bar(x, v.mean(), 0.62, color=color, alpha=0.85 if color != GREY else 0.6, zorder=2)
        ax.errorbar(x, v.mean(), yerr=[[v.mean() - lo], [hi - v.mean()]], fmt="none", ecolor=INK, lw=0.8, capsize=2, zorder=4)
        ax.scatter(x + rng.uniform(-0.18, 0.18, len(v)), v, s=7, color=INK, alpha=0.45, lw=0, zorder=3)


def style(ax, ticks, labels, ylabel, title):
    ax.set_xticks(ticks, labels, rotation=35, ha="right")
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", color=GRID, lw=0.5, zorder=0)
    ax.set_title(title, loc="left", fontsize=9.5, color=INK)


def write_table(df, path):
    """LaTeX table of per-arm means for every setting (supplement): runs, X, E, placebo reported and forgiven."""
    names = {"PLAIN": "Reported", "ORACLE": "Ledger", "SPR": "SPR", "TIR": "TIR", "OM1": "ORPO 1", "OM0.1": "ORPO 0.1", "OM0.05": "ORPO 0.05",
             "OM0.02": "ORPO 0.02", "OM0.01": "ORPO 0.01"}
    label = {"S_DK1": "A: DK1", "T_DK2": "B: DK2", "T_DK1M": "B: DK1 weak"}
    lines = ["\\begin{tabularx}{\\linewidth}{@{}llrrrrrr@{}}", "\\toprule",
             "Setting & Reward & Runs & $X$ & Edge & $X>0$ & Plac.\\ rep. & Plac.\\ forg. \\\\", "\\midrule"]
    for s in ("S_DK1", "T_DK2", "T_DK1M"):
        d = df[df.setting == s]
        first = True
        for arm in names:
            x = d[d.arm == arm]
            if not len(x):
                continue
            lines.append(f"{label[s] if first else ''} & {names[arm]} & {len(x)} & {x.X.mean():.3f} & ${x.E.mean():+.2f}$ & {int((x.X > 0).sum())} & "
                         f"${x.placebo_reported.mean():+.2f}$ & {x.placebo_forgiven.mean():.2f} \\\\")
            first = False
        lines.append("\\midrule" if s != "T_DK1M" else "\\bottomrule")
    lines.append("\\end{tabularx}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(HERE / "runs_t.csv"))
    ap.add_argument("--out", default=str(HERE))
    a = ap.parse_args()
    df = pd.read_csv(a.runs)
    write_table(df, Path(a.out) / "tab_studyT_full.tex")
    print("wrote tab_studyT_full.tex")
    arms_a = CORE + ORPO
    A = df[df.setting == "S_DK1"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), gridspec_kw=dict(width_ratios=[1.25, 1.0]))
    for ax, col, ylabel, title, pct in ((axes[0, 0], "E", "Booked edge (% capital / week)", "(a) Part A, DK1: booked edge", False),
                                         (axes[0, 1], "X", "Probes at max. leverage (%)", "(b) Part A: gambling at low equity", True)):
        draw(ax, A, col, arms_a, pct=pct)
        style(ax, range(len(arms_a)), [x[1] for x in arms_a], ylabel, title)
    settings = [s for s in ("T_DK2", "T_DK1M") if s in set(df.setting)]
    gap = len(CORE) + 0.8
    for ax, col, ylabel, title, pct in ((axes[1, 0], "E", "Booked edge (% capital / week)", "(c) Part B: booked edge", False),
                                         (axes[1, 1], "X", "Probes at max. leverage (%)", "(d) Part B: gambling at low equity", True)):
        ticks, labels = [], []
        for j, s in enumerate(settings):
            draw(ax, df[df.setting == s], col, CORE, x0=j * gap, pct=pct)
            ticks += [j * gap + i for i in range(len(CORE))]
            labels += [x[1] for x in CORE]
            ax.text(j * gap + (len(CORE) - 1) / 2, 1.02, {"T_DK2": "DK2", "T_DK1M": "DK1, weaker edge"}.get(s, s), ha="center", fontsize=9,
                    color=INK, transform=ax.get_xaxis_transform())
        style(ax, ticks, labels, ylabel, title)
        ax.set_title(title, loc="left", fontsize=9.5, color=INK, pad=14)
    fig.tight_layout(h_pad=1.4, w_pad=1.2)
    out = Path(a.out)
    fig.savefig(out / "fig_studyT.pdf")
    fig.savefig(out / "fig_studyT.png", dpi=180)
    print("wrote fig_studyT.pdf/.png")


if __name__ == "__main__":
    main()
