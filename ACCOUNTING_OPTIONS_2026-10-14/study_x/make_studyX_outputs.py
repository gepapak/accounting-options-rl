"""Figure and tables for Study X (written after the verdicts; descriptive only): fig_studyX.pdf / .png, tab_studyX_full.tex, tab_studyX_audit.tex
and tab_studyX_hyp.tex from runs_x.csv and VERDICTS_X.md (run aggregate_x.py first).

    python make_studyX_outputs.py [--runs runs_x.csv] [--verdicts VERDICTS_X.md] [--tables DIR] [--figures DIR]

By default the tables go to the sibling manuscript repository's sections/ folder and the figure to its figures/ folder.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MANUSCRIPT = HERE.parents[2] / "accounting-options-rl-latex-latex"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREY, TEAL = "#2a78d6", "#eb6834", "#a9a7a1", "#1f9d8a"
plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "legend.frameon": False, "pdf.fonttype": 42})
rng = np.random.default_rng(13)

ARMS = [("PLAIN", "Reported", ORANGE), ("ORACLE", "Ledger", GREY), ("SPR", "SPR", BLUE), ("TIR", "TIR", TEAL)]
EST = [("reported", "Reported", ORANGE), ("spr", "SPR", BLUE), ("tir", "TIR", TEAL)]
GAMBLER_X = 0.25


def boot_ci(x, n=5000):
    x = np.asarray(x, float)
    m = np.array([rng.choice(x, len(x)).mean() for _ in range(n)])
    return np.quantile(m, [0.025, 0.975])


def fmt(v: float) -> str:
    return f"${v:+.2f}$"


def write_tables(df: pd.DataFrame, verdict_file: Path, tables: Path) -> None:
    lines = ["\\begin{tabular}{@{}lrrrrrrr@{}}", "\\toprule",
             "Reward & Runs & $X$ & $X>0$ & Edge & Zero-start edge & Plac.\\ reported & Plac.\\ forgiven \\\\", "\\midrule"]
    for arm, name, _ in ARMS:
        x = df[df.arm == arm]
        if len(x):
            lines.append(f"{name} & {len(x)} & {x.X.mean():.3f} & {int((x.X > 0).sum())} & {fmt(x.E.mean())} & {fmt(x.Ez.mean())} & "
                         f"{fmt(x.placebo_reported.mean())} & {x.placebo_forgiven.mean():.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (tables / "tab_studyX_full.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    g = df[df.X >= GAMBLER_X]
    al = ["\\begin{tabularx}{\\linewidth}{@{}l*{4}{>{\\raggedleft\\arraybackslash}X}@{}}", "\\toprule",
          "Estimate & Max error, edge & Max error, placebo & Median error, placebo, gamblers & Median error, placebo, all \\\\", "\\midrule"]
    for est, name, _ in EST:
        gm = f"{g[f'err_{est}_plac'].median():.3f}" if len(g) else "--"
        al.append(f"{name} & {df[f'err_{est}_edge'].max():.2f} & {df[f'err_{est}_plac'].max():.2f} & {gm} & {df[f'err_{est}_plac'].median():.3f} \\\\")
    al += ["\\bottomrule", "\\end{tabularx}"]
    (tables / "tab_studyX_audit.tex").write_text("\n".join(al) + "\n", encoding="utf-8")

    verdicts = []
    for line in verdict_file.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\|\s*((?:XA|XB)\d[^|]*)\|\s*(.*)\|\s*(SUPPORTED|NOT SUPPORTED|NOT TESTABLE)\s*\|\s*$", line)
        if m:
            verdicts.append((m.group(1).strip(), m.group(2).strip(), m.group(3).strip()))
    body = []
    for vid, text, verdict in verdicts:
        text = text.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("<=", "$\\le$").replace(">=", "$\\ge$")
        text = text.replace("<", "$<$").replace(">", "$>$").replace("|", "$|$").replace(" - ", " $-$ ")
        verdict = {"SUPPORTED": "supported", "NOT SUPPORTED": "\\textbf{not supported}", "NOT TESTABLE": "not testable"}[verdict]
        body.append(f"{vid} & {text} & {verdict} \\\\")
    head = ("\\begin{tabularx}{\\linewidth}{@{}>{\\raggedright\\arraybackslash}p{0.08\\linewidth}X>{\\raggedright\\arraybackslash}p{0.17\\linewidth}@{}}\n"
            "\\toprule\nID & Result & Verdict \\\\\n\\midrule\n")
    (tables / "tab_studyX_hyp.tex").write_text(head + "\n".join(body) + "\n\\bottomrule\n\\end{tabularx}\n", encoding="utf-8")
    print(f"wrote tab_studyX_full.tex, tab_studyX_audit.tex and tab_studyX_hyp.tex ({len(verdicts)} verdicts)")


def draw(ax, d, col, pct=False):
    for i, (arm, _, color) in enumerate(ARMS):
        v = d.loc[d.arm == arm, col].to_numpy() * (100 if pct else 1)
        if not len(v):
            continue
        lo, hi = boot_ci(v)
        ax.bar(i, v.mean(), 0.62, color=color, alpha=0.85 if color != GREY else 0.6, zorder=2)
        ax.errorbar(i, v.mean(), yerr=[[v.mean() - lo], [hi - v.mean()]], fmt="none", ecolor=INK, lw=0.8, capsize=2, zorder=4)
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=7, color=INK, alpha=0.45, lw=0, zorder=3)


def style(ax, ylabel, title):
    ax.set_xticks(range(len(ARMS)), [a[1] for a in ARMS], rotation=40, ha="right")
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", color=GRID, lw=0.5, zorder=0)
    ax.set_title(title, loc="left", fontsize=9, color=INK)


def write_figure(df: pd.DataFrame, figures: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7), gridspec_kw=dict(width_ratios=[1.2, 1.2, 1.0]))
    draw(axes[0], df, "E", False)
    style(axes[0], "Booked edge (% capital / week)", "(a) Booked edge")
    draw(axes[1], df, "X", True)
    style(axes[1], "Probes at max. leverage (%)", "(b) Gambling")
    ax = axes[2]
    for i, (est, name, color) in enumerate(EST):
        v = np.maximum(df[f"err_{est}_plac"].to_numpy(), 1e-3)
        gam = df.X.to_numpy() >= GAMBLER_X
        ax.scatter(i + rng.uniform(-0.2, 0.2, (~gam).sum()), v[~gam], s=8, color=color, alpha=0.55, lw=0, zorder=3)
        ax.scatter(i + rng.uniform(-0.2, 0.2, gam.sum()), v[gam], s=12, facecolors="none", edgecolors=INK, lw=0.7, zorder=4)
        ax.hlines(np.median(v), i - 0.3, i + 0.3, color=INK, lw=1.2, zorder=5)
    ax.set_yscale("log")
    ax.set_xticks(range(len(EST)), [e[1] for e in EST], rotation=40, ha="right")
    ax.set_ylabel("|estimate $-$ booked| (points)")
    ax.grid(True, axis="y", color=GRID, lw=0.5, zorder=0)
    ax.set_title("(c) Audit error, placebo", loc="left", fontsize=9, color=INK)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(figures / "fig_studyX.pdf")
    fig.savefig(HERE / "fig_studyX.png", dpi=180)                  # a preview next to this script; the manuscript uses the PDF
    print("wrote fig_studyX.pdf (manuscript figures folder) and fig_studyX.png (this folder)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(HERE / "runs_x.csv"))
    ap.add_argument("--verdicts", default=str(HERE / "VERDICTS_X.md"))
    ap.add_argument("--tables", default=str(MANUSCRIPT / "sections"))
    ap.add_argument("--figures", default=str(MANUSCRIPT / "figures"))
    a = ap.parse_args()
    df = pd.read_csv(a.runs)
    write_tables(df, Path(a.verdicts), Path(a.tables))
    write_figure(df, Path(a.figures))


if __name__ == "__main__":
    main()
