"""Figure and tables for Study V (written after the verdicts; descriptive only): fig_studyV.pdf / .png, tab_studyV_full.tex and
tab_studyV_audit.tex from runs_v.csv (and Study T's runs_t.csv for Part B's reference arms).

    python make_studyV_outputs.py [--runs runs_v.csv] [--out DIR]
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
BLUE, ORANGE, GREY, TEAL, GOLD, VIOLET, BROWN = "#2a78d6", "#eb6834", "#a9a7a1", "#1f9d8a", "#c9a227", "#7a5195", "#8c564b"
plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "legend.frameon": False, "pdf.fonttype": 42})
rng = np.random.default_rng(11)

ARMS_A = [("PLAIN", "Reported", ORANGE), ("ORACLE", "Ledger", GREY), ("PMR", "PMR", GOLD), ("RUIN1", "Ruin 1", VIOLET), ("RUIN5", "Ruin 5", VIOLET),
          ("RUIN25", "Ruin 25", VIOLET), ("MV0.05", "MV .05", BROWN), ("MV0.2", "MV .2", BROWN), ("MV1", "MV 1", BROWN)]
ARMS_C = [("PLAIN", "Reported", ORANGE), ("ORACLE", "Ledger", GREY), ("TIR", "TIR", TEAL), ("PMR", "PMR", GOLD), ("RUIN1", "Ruin 1", VIOLET),
          ("RUIN5", "Ruin 5", VIOLET), ("RUIN25", "Ruin 25", VIOLET)]
NAMES = {a: n for a, n, _ in ARMS_A + ARMS_C}
EST = [("reported", "Reported", ORANGE), ("spr", "SPR", BLUE), ("pmr", "PMR", GOLD), ("tir", "TIR", TEAL)]


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
        ax.scatter(x + rng.uniform(-0.18, 0.18, len(v)), v, s=6, color=INK, alpha=0.45, lw=0, zorder=3)


def style(ax, ticks, labels, ylabel, title):
    ax.set_xticks(ticks, labels, rotation=40, ha="right")
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", color=GRID, lw=0.5, zorder=0)
    ax.set_title(title, loc="left", fontsize=9, color=INK)


def write_tables(df, t, out: Path):
    lines = ["\\begin{tabularx}{\\linewidth}{@{}llrrrrrr@{}}", "\\toprule",
             "Part & Reward & Runs & $X$ & Edge & Zero-start & $X>0$ & Plac.\\ forg. \\\\", "\\midrule"]
    blocks = [("A: DK1", df[df.setting == "S_DK1"], ARMS_A), ("C: DK1, zero starts", df[df.setting == "S_DK1+Z"], ARMS_C)]
    for label, d, arms in blocks:
        first = True
        for arm, name, _ in arms:
            x = d[d.arm == arm]
            if not len(x):
                continue
            ez = f"${x.Ez.mean():+.2f}$" if x.Ez.notna().any() else "--"
            lines.append(f"{label if first else ''} & {name} & {len(x)} & {x.X.mean():.3f} & ${x.E.mean():+.2f}$ & {ez} & {int((x.X > 0).sum())} & {x.placebo_forgiven.mean():.2f} \\\\")
            first = False
        lines.append("\\midrule")
    first = True
    for st, label in (("T_DK2", "B: DK2"), ("T_DK1M", "B: DK1 weak")):
        for name, x in (("Reported (Study T)", t[(t.setting == st) & (t.arm == "PLAIN")]), ("Ledger (Study T)", t[(t.setting == st) & (t.arm == "ORACLE")]),
                        ("PMR", df[(df.setting == st) & (df.arm == "PMR")])):
            if len(x):
                lines.append(f"{label} & {name} & {len(x)} & {x.X.mean():.3f} & ${x.E.mean():+.2f}$ & -- & {int((x.X > 0).sum())} & {x.placebo_forgiven.mean():.2f} \\\\")
                label = ""
        lines.append("\\midrule" if st == "T_DK2" else "\\bottomrule")
    lines.append("\\end{tabularx}")
    (out / "tab_studyV_full.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    d = df[df.setting.isin(["S_DK1", "S_DK1+Z"])]
    g = d[d.X >= 0.25]
    al = ["\\begin{tabularx}{\\linewidth}{@{}l*{4}{>{\\raggedleft\\arraybackslash}X}@{}}", "\\toprule",
          "Estimate & Max error, edge & Max error, placebo & Median error, placebo, gamblers & Median error, placebo, all \\\\", "\\midrule"]
    for est, name, _ in EST:
        al.append(f"{name} & {d[f'err_{est}_edge'].max():.2f} & {d[f'err_{est}_plac'].max():.2f} & "
                  f"{(g[f'err_{est}_plac'].median() if len(g) else float('nan')):.3f} & {d[f'err_{est}_plac'].median():.3f} \\\\")
    al += ["\\bottomrule", "\\end{tabularx}"]
    (out / "tab_studyV_audit.tex").write_text("\n".join(al) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(HERE / "runs_v.csv"))
    ap.add_argument("--runs-t", default=str(HERE.parent / "study_t" / "runs_t.csv"))
    ap.add_argument("--out", default=str(HERE))
    a = ap.parse_args()
    df, t, out = pd.read_csv(a.runs), pd.read_csv(a.runs_t), Path(a.out)
    write_tables(df, t, out)
    print("wrote tab_studyV_full.tex, tab_studyV_audit.tex")
    A, C = df[df.setting == "S_DK1"], df[df.setting == "S_DK1+Z"]
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.4), gridspec_kw=dict(width_ratios=[1.3, 1.3, 1.0]))
    for ax, d, arms, col, ylabel, title, pct in (
            (axes[0, 0], A, ARMS_A, "E", "Booked edge (% capital / week)", "(a) Part A: booked edge", False),
            (axes[0, 1], A, ARMS_A, "X", "Probes at max. leverage (%)", "(b) Part A: gambling", True),
            (axes[1, 0], C, ARMS_C, "E", "Booked edge (% capital / week)", "(c) Part C: booked edge", False),
            (axes[1, 1], C, ARMS_C, "X", "Probes at max. leverage (%)", "(d) Part C: gambling", True)):
        draw(ax, d, col, arms, pct=pct)
        style(ax, range(len(arms)), [x[1] for x in arms], ylabel, title)
    # (e) Part B: PMR against Study T's reported and ledger runs
    ax = axes[0, 2]
    ticks, labels = [], []
    for j, (st, lab) in enumerate((("T_DK2", "DK2"), ("T_DK1M", "DK1 weak"))):
        parts = [("PLAIN", "Reported", ORANGE, t[(t.setting == st) & (t.arm == "PLAIN")]), ("ORACLE", "Ledger", GREY, t[(t.setting == st) & (t.arm == "ORACLE")]),
                 ("PMR", "PMR", GOLD, df[(df.setting == st) & (df.arm == "PMR")])]
        for i, (arm, name, color, d) in enumerate(parts):
            if len(d):
                draw(ax, d.assign(arm=arm), "E", [(arm, name, color)], x0=j * 3.6 + i)
            ticks.append(j * 3.6 + i)
            labels.append(name)
        ax.text(j * 3.6 + 1, 1.02, lab, ha="center", fontsize=8.5, color=INK, transform=ax.get_xaxis_transform())
    style(ax, ticks, labels, "Booked edge (% capital / week)", "(e) Part B: PMR")
    ax.set_title("(e) Part B: PMR", loc="left", fontsize=9, color=INK, pad=14)
    # (f) audit: error of the estimate of the booked return, placebo market, every agent of Parts A and C
    ax = axes[1, 2]
    d = df[df.setting.isin(["S_DK1", "S_DK1+Z"])]
    for i, (est, name, color) in enumerate(EST):
        v = np.maximum(d[f"err_{est}_plac"].to_numpy(), 1e-3)
        ax.scatter(i + rng.uniform(-0.2, 0.2, len(v)), v, s=7, color=color, alpha=0.6, lw=0, zorder=3)
        ax.hlines(np.median(v), i - 0.3, i + 0.3, color=INK, lw=1.2, zorder=4)
    ax.set_yscale("log")
    ax.set_xticks(range(len(EST)), [e[1] for e in EST], rotation=40, ha="right")
    ax.set_ylabel("|estimate $-$ booked| (points of capital)")
    ax.grid(True, axis="y", color=GRID, lw=0.5, zorder=0)
    ax.set_title("(f) Audit error, placebo", loc="left", fontsize=9, color=INK)
    fig.tight_layout(h_pad=1.6, w_pad=1.0)
    fig.savefig(out / "fig_studyV.pdf")
    fig.savefig(out / "fig_studyV.png", dpi=180)
    print("wrote fig_studyV.pdf/.png")


if __name__ == "__main__":
    main()
