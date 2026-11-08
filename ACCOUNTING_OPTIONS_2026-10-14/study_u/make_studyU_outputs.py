"""Figure for Study U (written after the verdicts; descriptive only): fig_studyU.pdf / .png from runs_u.csv.

    python make_studyU_outputs.py [--runs runs_u.csv] [--out DIR]
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
BLUE, ORANGE, GREY, TEAL = "#2a78d6", "#eb6834", "#a9a7a1", "#1f9d8a"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "legend.frameon": False, "pdf.fonttype": 42})
rng = np.random.default_rng(7)
PANELS = [("Z", "(a) Settlement lag (zombie window)", (("ZFG", "Reported", ORANGE), ("ZBK", "Ledger", GREY), ("ZSPR", "SPR", BLUE), ("ZTIR", "TIR", TEAL))),
          ("T", "(b) Terminal limited liability", (("TFG", "Reported", ORANGE), ("TBK", "Ledger", GREY), ("TSPR", "SPR", BLUE), ("TTIR", "TIR", TEAL)))]


def boot_ci(x, n=5000):
    x = np.asarray(x, float)
    m = np.array([rng.choice(x, len(x)).mean() for _ in range(n)])
    return np.quantile(m, [0.025, 0.975])


def write_table(df, path):
    """LaTeX table of per-arm results (supplement): runs, mean X, range, runs with X >= 8, region means."""
    names = {"ZFG": "Deficit forgiven (reported)", "ZBK": "Deficit booked (ledger)", "ZSPR": "SPR", "ZTIR": "TIR",
             "TFG": "Deficit forgiven (reported)", "TBK": "Deficit booked (ledger)", "TSPR": "SPR", "TTIR": "TIR"}
    env = {"Z": "Settlement lag", "T": "Terminal liability"}
    lines = ["\\begin{tabularx}{\\linewidth}{@{}llrrrrrr@{}}", "\\toprule",
             "Environment & Reward & Runs & Mean $X$ & Range & $X\\ge8$ & DK1 & DK2 \\\\", "\\midrule"]
    for s in ("Z", "T"):
        first = True
        for arm in [a for a in names if a[0] == s]:
            d = df[df.arm == arm]
            if not len(d):
                continue
            dk = d.groupby("region").X.mean()
            lines.append(f"{env[s] if first else ''} & {names[arm]} & {len(d)} & {d.X.mean():.2f} & {d.X.min():.2f}--{d.X.max():.2f} & "
                         f"{int((d.X >= 8).sum())} & {dk.get('DK1', float('nan')):.2f} & {dk.get('DK2', float('nan')):.2f} \\\\")
            first = False
        lines.append("\\midrule" if s == "Z" else "\\bottomrule")
    lines.append("\\end{tabularx}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(HERE / "runs_u.csv"))
    ap.add_argument("--out", default=str(HERE))
    a = ap.parse_args()
    df = pd.read_csv(a.runs)
    write_table(df, Path(a.out) / "tab_studyU_full.tex")
    print("wrote tab_studyU_full.tex")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for ax, (s, title, arms) in zip(axes, PANELS):
        for i, (arm, _, color) in enumerate(arms):
            d = df[df.arm == arm]
            if not len(d):
                continue
            v = d.X.to_numpy()
            lo, hi = boot_ci(v)
            ax.bar(i, v.mean(), 0.62, color=color, alpha=0.85 if color != GREY else 0.6, zorder=2)
            ax.errorbar(i, v.mean(), yerr=[[v.mean() - lo], [hi - v.mean()]], fmt="none", ecolor=INK, lw=0.8, capsize=2, zorder=4)
            for region, marker in (("DK1", "o"), ("DK2", "^")):
                w = d.loc[d.region == region, "X"].to_numpy()
                ax.scatter(i + rng.uniform(-0.2, 0.2, len(w)), w, s=9, marker=marker, color=INK, alpha=0.5, lw=0, zorder=3,
                           label=region if (i == 0) else None)
        ax.set_xticks(range(len(arms)), [x[1] for x in arms], rotation=35, ha="right")
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_ylabel("Mean |leverage| over the probes")
        ax.grid(True, axis="y", color=GRID, lw=0.5, zorder=0)
        ax.set_title(title, loc="left", fontsize=9.5, color=INK)
    axes[0].legend(loc="upper right", fontsize=8, handletextpad=0.2)
    fig.tight_layout(w_pad=1.4)
    out = Path(a.out)
    fig.savefig(out / "fig_studyU.pdf")
    fig.savefig(out / "fig_studyU.png", dpi=180)
    print("wrote fig_studyU.pdf/.png")


if __name__ == "__main__":
    main()
