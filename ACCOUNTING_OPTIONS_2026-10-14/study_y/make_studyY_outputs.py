"""Figure and tables for Study Y (written after the verdicts; descriptive only): fig_studyY.pdf / .png, tab_studyY_full.tex and tab_studyY_hyp.tex
from runs_y.csv and VERDICTS_Y.md (run aggregate_y.py first).

    python make_studyY_outputs.py [--runs runs_y.csv] [--verdicts VERDICTS_Y.md] [--tables DIR] [--figures DIR]

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
rng = np.random.default_rng(14)
ARMS = [("PLAIN", "Reported", ORANGE), ("ORACLE", "Ledger", GREY), ("SPR", "SPR", BLUE), ("TIR", "TIR", TEAL)]
SETTINGS = [("S_DK1+Z", "Single floor, zero starts"), ("X_DK1+Z", "Two writes, zero starts"), ("S_DK1", "Single floor, no zero starts")]


def boot_ci(x, n=5000):
    x = np.asarray(x, float)
    m = np.array([rng.choice(x, len(x)).mean() for _ in range(n)])
    return np.quantile(m, [0.025, 0.975])


def fmt(v: float) -> str:
    return f"${v:+.2f}$"


def write_tables(df: pd.DataFrame, verdict_file: Path, tables: Path) -> None:
    lines = ["\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{0.17\\linewidth}lrrrrrr@{}}", "\\toprule",
             "Setting & Reward & Runs & $X$ & $X>0$ & Edge & Zero-start edge & Plac.\\ forgiven \\\\", "\\midrule"]
    for k, (setting, sname) in enumerate(SETTINGS):
        d = df[df.setting == setting]
        first = True
        for arm, name, _ in ARMS:
            x = d[d.arm == arm]
            if not len(x):
                continue
            lines.append(f"{sname if first else ''} & {name} & {len(x)} & {x.X.mean():.3f} & {int((x.X > 0).sum())} & {fmt(x.E.mean())} & "
                         f"{fmt(x.Ez.mean())} & {x.placebo_forgiven.mean():.2f} \\\\")
            first = False
        if k < len(SETTINGS) - 1 and len(d):
            lines.append("\\addlinespace")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (tables / "tab_studyY_full.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    verdicts = []
    for line in verdict_file.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\|\s*((?:YA|YB|YC)\d[^|]*)\|\s*(.*)\|\s*(SUPPORTED|NOT SUPPORTED|NOT TESTABLE)\s*\|\s*$", line)
        if m:
            verdicts.append((m.group(1).strip(), m.group(2).strip(), m.group(3).strip()))
    body = []
    for vid, text, verdict in verdicts:
        text = text.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("<=", "$\\le$").replace(">=", "$\\ge$")
        text = text.replace("<", "$<$").replace(">", "$>$").replace("|", "$|$").replace(" - ", " $-$ ")
        verdict = {"SUPPORTED": "supported", "NOT SUPPORTED": "\\textbf{not supported}", "NOT TESTABLE": "not testable"}[verdict]
        body.append(f"{vid} & {text} & {verdict} \\\\")
    head = ("\\begin{tabularx}{\\linewidth}{@{}>{\\raggedright\\arraybackslash}p{0.09\\linewidth}X>{\\raggedright\\arraybackslash}p{0.15\\linewidth}@{}}\n"
            "\\toprule\nID & Result & Verdict \\\\\n\\midrule\n")
    (tables / "tab_studyY_hyp.tex").write_text(head + "\n".join(body) + "\n\\bottomrule\n\\end{tabularx}\n", encoding="utf-8")
    print(f"wrote tab_studyY_full.tex and tab_studyY_hyp.tex ({len(verdicts)} verdicts)")


def write_figure(df: pd.DataFrame, figures: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8))
    for ax, col, pct, ylabel, title in ((axes[0], "E", False, "Booked edge (% capital / week)", "(a) Booked edge"),
                                       (axes[1], "X", True, "Probes at max. leverage (%)", "(b) Gambling")):
        xt, xl = [], []
        pos = 0.0
        for setting, sname in SETTINGS:
            d = df[df.setting == setting]
            if not len(d):
                continue
            start = pos
            for arm, name, color in ARMS:
                v = d.loc[d.arm == arm, col].to_numpy() * (100 if pct else 1)
                if not len(v):
                    continue
                lo, hi = boot_ci(v)
                ax.bar(pos, v.mean(), 0.7, color=color, alpha=0.85 if color != GREY else 0.6, zorder=2, label=name if setting == SETTINGS[0][0] else None)
                ax.errorbar(pos, v.mean(), yerr=[[v.mean() - lo], [hi - v.mean()]], fmt="none", ecolor=INK, lw=0.8, capsize=2, zorder=4)
                ax.scatter(pos + rng.uniform(-0.2, 0.2, len(v)), v, s=6, color=INK, alpha=0.45, lw=0, zorder=3)
                pos += 1.0
            xt.append((start + pos - 1.0) / 2)
            xl.append(sname.replace(", ", ",\n"))
            pos += 0.8
        ax.set_xticks(xt, xl)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", color=GRID, lw=0.5, zorder=0)
        ax.set_title(title, loc="left", fontsize=9, color=INK)
    axes[0].legend(loc="lower right", fontsize=7, ncol=2)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(figures / "fig_studyY.pdf")
    fig.savefig(HERE / "fig_studyY.png", dpi=180)                  # a preview next to this script; the manuscript uses the PDF
    print("wrote fig_studyY.pdf (manuscript figures folder) and fig_studyY.png (this folder)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(HERE / "runs_y.csv"))
    ap.add_argument("--verdicts", default=str(HERE / "VERDICTS_Y.md"))
    ap.add_argument("--tables", default=str(MANUSCRIPT / "sections"))
    ap.add_argument("--figures", default=str(MANUSCRIPT / "figures"))
    a = ap.parse_args()
    df = pd.read_csv(a.runs)
    write_tables(df, Path(a.verdicts), Path(a.tables))
    write_figure(df, Path(a.figures))


if __name__ == "__main__":
    main()
