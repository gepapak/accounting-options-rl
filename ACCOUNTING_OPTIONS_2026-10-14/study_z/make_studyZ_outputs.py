"""Figure and tables for Study Z (written before the full runs; descriptive only): fig_studyZ.pdf / .png, tab_studyZ_full.tex and
tab_studyZ_hyp.tex from runs_z.csv and VERDICTS_Z.md (run aggregate_z.py first).

    python make_studyZ_outputs.py [--runs runs_z.csv] [--verdicts VERDICTS_Z.md] [--tables DIR] [--figures DIR]

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
BLUE, ORANGE, GREY, TEAL, RED, PURPLE = "#2a78d6", "#eb6834", "#a9a7a1", "#1f9d8a", "#b3261e", "#7b5ea7"
plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "legend.frameon": False, "pdf.fonttype": 42})
rng = np.random.default_rng(15)
ARMS = [("REP", "Reported", ORANGE), ("LEDGER", "Ledger", GREY), ("RUINP1", "Portfolio ruin 1", RED), ("RUINP5", "Portfolio ruin 5", RED),
        ("RUINP25", "Portfolio ruin 25", RED), ("RUINA5", "Account ruin 5", PURPLE), ("SPR", "SPR", BLUE), ("TIR", "TIR", TEAL),
        ("P_REP", "Pooled: reported", ORANGE), ("P_RUIN5", "Pooled: ruin 5", PURPLE)]


def boot_ci(x, n=5000):
    x = np.asarray(x, float)
    m = np.array([rng.choice(x, len(x)).mean() for _ in range(n)])
    return np.quantile(m, [0.025, 0.975])


def fmt(v: float) -> str:
    return "--" if not np.isfinite(v) else f"${v:+.1f}$"


def f3(v: float) -> str:
    return "--" if not np.isfinite(v) else f"{v:.2f}"


def write_tables(df: pd.DataFrame, verdict_file: Path, tables: Path) -> None:
    lines = ["\\begin{tabular}{@{}lrrrrrrrrr@{}}", "\\toprule",
             " & & \\multicolumn{3}{c}{Exploit share} & \\multicolumn{2}{c}{Booked edge} & \\multicolumn{3}{c}{Placebo market} \\\\",
             "\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-10}",
             "Reward & Runs & $X$ & both low & one empty & $E$ & from zero & reported & booked & ruins: portfolio / accounts \\\\", "\\midrule"]
    for k, (arm, name, _) in enumerate(ARMS):
        x = df[df.arm == arm]
        if not len(x):
            continue
        if arm == "P_REP":
            lines.append("\\addlinespace")
        lines.append(f"{name} & {len(x)} & {x.X.mean():.2f} & {x.X_sym.mean():.2f} & {f3(x.X_empty.mean())} & {fmt(x.E.mean())} & "
                     f"{fmt(x.Ez.mean())} & {fmt(x.placebo_reported.mean())} & {fmt(x.placebo_booked.mean())} & "
                     f"{x.ruin_portfolio.mean():.1f} / {x.ruin_accounts.mean():.1f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (tables / "tab_studyZ_full.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    verdicts = []
    for line in verdict_file.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\|\s*((?:ZA|ZB)\d+[^|]*)\|\s*(.*)\|\s*(SUPPORTED|NOT SUPPORTED|NOT TESTABLE)\s*\|\s*$", line)
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
    (tables / "tab_studyZ_hyp.tex").write_text(head + "\n".join(body) + "\n\\bottomrule\n\\end{tabularx}\n", encoding="utf-8")
    print(f"wrote tab_studyZ_full.tex and tab_studyZ_hyp.tex ({len(verdicts)} verdicts)")


def write_figure(df: pd.DataFrame, figures: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    for ax, col, pct, ylabel, title in ((axes[0], "X", True, "Low-equity probes at max. leverage (%)", "(a) Exploit share"),
                                       (axes[1], "E", False, "Booked edge (% capital / week)", "(b) Booked edge")):
        xt, xl = [], []
        for pos, (arm, name, color) in enumerate(ARMS):
            v = df.loc[df.arm == arm, col].to_numpy() * (100 if pct else 1)
            if not len(v):
                continue
            p = pos + (0.6 if arm.startswith("P_") else 0.0)
            lo, hi = boot_ci(v)
            ax.bar(p, v.mean(), 0.7, color=color, alpha=0.85 if color != GREY else 0.6, zorder=2,
                   hatch="///" if arm.startswith("P_") else None, edgecolor="white", lw=0)
            ax.errorbar(p, v.mean(), yerr=[[v.mean() - lo], [hi - v.mean()]], fmt="none", ecolor=INK, lw=0.8, capsize=2, zorder=4)
            ax.scatter(p + rng.uniform(-0.2, 0.2, len(v)), v, s=6, color=INK, alpha=0.45, lw=0, zorder=3)
            xt.append(p)
            xl.append(name)
        ax.set_xticks(xt, xl, rotation=55, ha="right", fontsize=7)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", color=GRID, lw=0.5, zorder=0)
        ax.set_title(title, loc="left", fontsize=9, color=INK)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(figures / "fig_studyZ.pdf")
    fig.savefig(HERE / "fig_studyZ.png", dpi=180)                  # a preview next to this script; the manuscript uses the PDF
    print("wrote fig_studyZ.pdf (manuscript figures folder) and fig_studyZ.png (this folder)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(HERE / "runs_z.csv"))
    ap.add_argument("--verdicts", default=str(HERE / "VERDICTS_Z.md"))
    ap.add_argument("--tables", default=str(MANUSCRIPT / "sections"))
    ap.add_argument("--figures", default=str(MANUSCRIPT / "figures"))
    a = ap.parse_args()
    df = pd.read_csv(a.runs)
    write_tables(df, Path(a.verdicts), Path(a.tables))
    write_figure(df, Path(a.figures))


if __name__ == "__main__":
    main()
