"""Main-text table of Study Z, Parts A and B (descriptive, written after the registered analyses): tab_hidden.tex from runs_z.csv and
runs_zb.csv. Exploit share X and booked edge E as a percentage of the ledger-trained arm's, per part.

    python make_tab_hidden.py [--out DIR]      # default: the manuscript repository's sections/ folder
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
MANUSCRIPT = HERE.parents[2] / "accounting-options-rl-latex-latex"
ROWS = [("Reported reward (nothing)", "REP", "REP"), ("Booked ledger (the ledger)", "LEDGER", "LEDGER"),
        ("Ruin penalty 5, portfolio or balance", "RUINP5", "RUINP5"), ("Ruin penalty 25, portfolio or balance", "RUINP25", "RUINP25"),
        ("Penalty 5 per account / per liquidation (the structure)", "RUINA5", "LIQ5"), ("\\textbf{SPR} (clone, sign override)", "SPR", "SPR"),
        ("\\textbf{TIR} (clone, sign override)", "TIR", "TIR"), ("Finder, bracket width (clone, sign override)", None, "FINDER")]


def cell(df, arm, ledger_e, bold):
    if arm is None:
        return "--", "--"
    x = df[df.arm == arm]
    xv, ev = x.X.mean(), 100.0 * x.E.mean() / ledger_e
    xs = f"{xv:.2f}"
    es = f"{ev:.0f}" if ev >= 0 else f"$-{abs(ev):.0f}$"
    if arm == "FINDER":
        es = "$<-1000$"
    return (f"\\textbf{{{xs}}}", f"\\textbf{{{es}}}") if bold else (xs, es)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(MANUSCRIPT / "sections"))
    a = ap.parse_args()
    za = pd.read_csv(HERE.parent / "study_z" / "runs_z.csv")
    zb = pd.read_csv(HERE / "runs_zb.csv")
    la, lb = za[za.arm == "LEDGER"].E.mean(), zb[zb.arm == "LEDGER"].E.mean()
    lines = ["\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{.40\\linewidth}cccc@{}}", "\\toprule",
             " & \\multicolumn{2}{c}{Part A: split accounts} & \\multicolumn{2}{c}{Part B: isolated margin} \\\\",
             "Reward (what it needs) & $X$ & $E$ (\\%) & $X$ & $E$ (\\%) \\\\", "\\midrule"]
    for name, arm_a, arm_b in ROWS:
        bold = arm_a in ("SPR", "TIR")
        xa, ea = cell(za, arm_a, la, bold)
        xb, eb = cell(zb, arm_b, lb, bold)
        lines.append(f"{name} & {xa} & {ea} & {xb} & {eb} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    out = Path(a.out) / "tab_hidden.tex"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out.read_text(encoding="utf-8"))
    print(f"ledger edge: Part A {la:+.2f}, Part B {lb:+.2f} (% K0 per week)")


if __name__ == "__main__":
    main()
