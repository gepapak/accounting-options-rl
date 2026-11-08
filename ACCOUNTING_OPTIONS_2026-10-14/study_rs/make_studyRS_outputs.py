"""Figure and tables for Study RS (written after the verdicts; descriptive only): fig_studyRS.pdf / .png, tab_studyRS_full.tex and tab_studyRS_hyp.tex
from runs_rs.csv and VERDICTS_RS.md.

    python make_studyRS_outputs.py [--runs runs_rs.csv] [--verdicts VERDICTS_RS.md] [--tables DIR] [--figures DIR]

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
BLUE, ORANGE, GREY, TEAL, VIOLET = "#2a78d6", "#eb6834", "#a9a7a1", "#1f9d8a", "#7a5cc0"
plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "legend.frameon": False, "pdf.fonttype": 42})
rng = np.random.default_rng(14)
ARMS = [("PLAIN", "Reported", ORANGE), ("ORACLE", "Ledger", GREY), ("SPR", "SPR", BLUE), ("TIR", "TIR", TEAL), ("RUIN5", "Ruin penalty 5", VIOLET)]


def boot_ci(x, n=5000):
    x = np.asarray(x, float)
    m = np.array([rng.choice(x, len(x)).mean() for _ in range(n)])
    return np.quantile(m, [0.025, 0.975])


def fmt(v: float) -> str:
    return f"${v:+.2f}$"


def write_tables(df: pd.DataFrame, verdict_file: Path, tables: Path) -> None:
    lines = ["\\begin{tabular}{@{}lrrrrrrrr@{}}", "\\toprule",
             "Reward & Runs & $X$ & $X>0$ & Edge & Edge $>0$ & Zero-start edge & Plac.\\ booked & Plac.\\ forgiven \\\\", "\\midrule"]
    for arm, name, _ in ARMS:
        x = df[df.arm == arm]
        lines.append(f"{name} & {len(x)} & {x.X.mean():.3f} & {int((x.X > 0).sum())} & {fmt(x.E.mean())} & {int((x.E > 0).sum())} & {fmt(x.E_zero.mean())} & "
                     f"{fmt(x.placebo_booked.mean())} & {x.placebo_forgiven.mean():.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (tables / "tab_studyRS_full.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    verdicts = []
    for line in verdict_file.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\|\s*(RS-?D?\d)\s*\|\s*(.*)\|\s*(SUPPORTED|NOT SUPPORTED|NOT TESTABLE)\s*\|\s*$", line)
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
    (tables / "tab_studyRS_hyp.tex").write_text(head + "\n".join(body) + "\n\\bottomrule\n\\end{tabularx}\n", encoding="utf-8")
    print(f"wrote tab_studyRS_full.tex and tab_studyRS_hyp.tex ({len(verdicts)} verdicts)")


def write_figure(df: pd.DataFrame, figures: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8))
    for ax, col, pct, ylabel, title in ((axes[0], "E", False, "Booked edge (% allocation / week, symlog)", "(a) Booked edge"),
                                       (axes[1], "X", True, "Probes at max. leverage (%)", "(b) Gambling")):
        for pos, (arm, name, color) in enumerate(ARMS):
            v = df.loc[df.arm == arm, col].to_numpy() * (100 if pct else 1)
            lo, hi = boot_ci(v)
            ax.bar(pos, v.mean(), 0.7, color=color, alpha=0.85 if color != GREY else 0.6, zorder=2)
            ax.errorbar(pos, v.mean(), yerr=[[v.mean() - lo], [hi - v.mean()]], fmt="none", ecolor=INK, lw=0.8, capsize=2, zorder=4)
            ax.scatter(pos + rng.uniform(-0.2, 0.2, len(v)), v, s=6, color=INK, alpha=0.45, lw=0, zorder=3)
        ax.set_xticks(range(len(ARMS)), [a[1].replace(" penalty ", "\npenalty ") for a in ARMS])
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", color=GRID, lw=0.5, zorder=0)
        ax.set_title(title, loc="left", fontsize=9, color=INK)
    axes[0].set_yscale("symlog", linthresh=20)
    axes[0].set_yticks([-200, -100, -20, 0, 20], ["$-200$", "$-100$", "$-20$", "0", "20"])
    fig.tight_layout(w_pad=1.2)
    fig.savefig(figures / "fig_studyRS.pdf")
    fig.savefig(HERE / "fig_studyRS.png", dpi=180)                  # a preview next to this script; the manuscript uses the PDF
    print("wrote fig_studyRS.pdf (manuscript figures folder) and fig_studyRS.png (this folder)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(HERE / "runs_rs.csv"))
    ap.add_argument("--verdicts", default=str(HERE / "VERDICTS_RS.md"))
    ap.add_argument("--tables", default=str(MANUSCRIPT / "sections"))
    ap.add_argument("--figures", default=str(MANUSCRIPT / "figures"))
    a = ap.parse_args()
    df = pd.read_csv(a.runs)
    write_tables(df, Path(a.verdicts), Path(a.tables))
    write_figure(df, Path(a.figures))


if __name__ == "__main__":
    main()
