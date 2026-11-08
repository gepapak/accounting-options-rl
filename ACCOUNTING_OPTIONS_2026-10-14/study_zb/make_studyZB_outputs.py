"""Tables and figure for Study Z, Part B (descriptive only): tab_studyZB_full.tex, tab_studyZB_hyp.tex and fig_studyZB.pdf / .png from
runs_zb.csv and VERDICTS_ZB.md (run aggregate_zb.py first). By default the tables go to the sibling manuscript repository's sections/ folder and
the figure to its figures/ folder.

    python make_studyZB_outputs.py [--runs runs_zb.csv] [--verdicts VERDICTS_ZB.md] [--tables DIR] [--figures DIR]
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
BLUE, ORANGE, GREY, TEAL, RED, PURPLE = "#2a78d6", "#eb6834", "#a9a7a1", "#1f9d8a", "#b3261e", "#7b5ea7"
plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "legend.frameon": False, "pdf.fonttype": 42})
rng = np.random.default_rng(16)
ARMS = [("REP", "Reported", ORANGE), ("LEDGER", "Ledger", GREY), ("RUINP5", "Ruin penalty 5", RED), ("RUINP25", "Ruin penalty 25", RED),
        ("LIQ5", "Liquidation penalty 5", PURPLE), ("SPR", "SPR", BLUE), ("TIR", "TIR", TEAL), ("FINDER", "Finder (bracket width)", INK)]


def boot_ci(x, n=5000):
    x = np.asarray(x, float)
    m = np.array([rng.choice(x, len(x)).mean() for _ in range(n)])
    return np.quantile(m, [0.025, 0.975])


def fmt(v: float) -> str:
    return f"${v:+.1f}$"


def write_tables(df: pd.DataFrame, verdict_file: Path, tables: Path) -> None:
    lines = ["\\begin{tabular}{@{}lrrrrrrrrr@{}}", "\\toprule",
             " & & \\multicolumn{2}{c}{Exploit share} & \\multicolumn{4}{c}{Edge market, per week} & \\multicolumn{2}{c}{Placebo market} \\\\",
             "\\cmidrule(lr){3-4}\\cmidrule(lr){5-8}\\cmidrule(lr){9-10}",
             "Reward & Runs & $X$ & isolated & booked $E$ & bracket width & ruins & liquidations & reported & booked \\\\", "\\midrule"]
    for arm, name, _ in ARMS:
        x = df[df.arm == arm]
        if len(x):
            lines.append(f"{name} & {len(x)} & {x.X.mean():.2f} & {x.X_iso.mean():.2f} & {fmt(x.E.mean())} & {x.width_edge.mean():.1f} & "
                         f"{x.edge_ruins.mean():.2f} & {x.edge_liquidations.mean():.1f} & {fmt(x.placebo_reported.mean())} & "
                         f"{fmt(x.placebo_booked.mean())} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (tables / "tab_studyZB_full.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    verdicts = []
    for line in verdict_file.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\|\s*(ZI[ABC]\d+)\s*\|\s*(.*)\|\s*(SUPPORTED|NOT SUPPORTED|NOT TESTABLE)\s*\|\s*$", line)
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
    (tables / "tab_studyZB_hyp.tex").write_text(head + "\n".join(body) + "\n\\bottomrule\n\\end{tabularx}\n", encoding="utf-8")
    print(f"wrote tab_studyZB_full.tex and tab_studyZB_hyp.tex ({len(verdicts)} verdicts)")


def write_figure(df: pd.DataFrame, figures: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    for ax, col, pct, ylabel, title in ((axes[0], "X", True, "Funded probes at max. leverage (%)", "(a) Exploit share"),
                                       (axes[1], "E", False, "Booked edge (% capital / week)", "(b) Booked edge")):
        xt, xl = [], []
        for pos, (arm, name, color) in enumerate(ARMS):
            v = df.loc[df.arm == arm, col].to_numpy() * (100 if pct else 1)
            if not len(v):
                continue
            lo, hi = boot_ci(v)
            ax.bar(pos, v.mean(), 0.7, color=color, alpha=0.85 if color != GREY else 0.6, zorder=2)
            ax.errorbar(pos, v.mean(), yerr=[[v.mean() - lo], [hi - v.mean()]], fmt="none", ecolor=INK, lw=0.8, capsize=2, zorder=4)
            ax.scatter(pos + rng.uniform(-0.2, 0.2, len(v)), v, s=6, color=INK, alpha=0.45, lw=0, zorder=3)
            xt.append(pos)
            xl.append(name)
        ax.set_xticks(xt, xl, rotation=55, ha="right", fontsize=7)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", color=GRID, lw=0.5, zorder=0)
        ax.set_title(title, loc="left", fontsize=9, color=INK)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(figures / "fig_studyZB.pdf")
    fig.savefig(HERE / "fig_studyZB.png", dpi=180)
    print("wrote fig_studyZB.pdf (manuscript figures folder) and fig_studyZB.png (this folder)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(HERE / "runs_zb.csv"))
    ap.add_argument("--verdicts", default=str(HERE / "VERDICTS_ZB.md"))
    ap.add_argument("--tables", default=str(MANUSCRIPT / "sections"))
    ap.add_argument("--figures", default=str(MANUSCRIPT / "figures"))
    a = ap.parse_args()
    df = pd.read_csv(a.runs)
    write_tables(df, Path(a.verdicts), Path(a.tables))
    write_figure(df, Path(a.figures))


if __name__ == "__main__":
    main()
