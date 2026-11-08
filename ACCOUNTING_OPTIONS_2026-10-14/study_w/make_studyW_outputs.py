"""Study W tables for the manuscript: reads runs_w.csv, ../study_t/runs_t.csv and VERDICTS_W.md (run aggregate_w.py first) and writes

    tab_studyW_full.tex   per setting and strength: runs, X, booked edge E, TIR-audited and reported edge-market returns (with Study T's references)
    tab_studyW_hyp.tex    the six registered decisions

into the manuscript repository's sections/ folder (default: the sibling accounting-options-rl-latex-latex repository).

    python make_studyW_outputs.py [output_dir]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parents[2] / "accounting-options-rl-latex-latex" / "sections"
SETTINGS = (("T_DK2", "DK2"), ("T_DK1M", "DK1 weak"))
GRID = ("RUIN1", "RUIN5", "RUIN25")
LABEL = {"RUIN1": "Ruin 1", "RUIN5": "Ruin 5", "RUIN25": "Ruin 25", "PLAIN": "Reported (Study T)", "ORACLE": "Ledger (Study T)"}


def fmt(x: float, signed: bool = True) -> str:
    s = f"{x:+.1f}" if signed else f"{x:.2f}"
    return s.replace("-", "$-$")


def main() -> None:
    w = pd.read_csv(HERE / "runs_w.csv")
    t = pd.read_csv(HERE.parent / "study_t" / "runs_t.csv")
    rows = []
    for st, name in SETTINGS:
        ref = t[t.setting == st]
        o_e = ref[ref.arm == "ORACLE"].E.mean()
        for arm in ("PLAIN", "ORACLE"):
            x = ref[ref.arm == arm]
            rows.append(f"{name if arm == 'PLAIN' else ''} & {LABEL[arm]} & {len(x)} & {x.X.mean():.2f} & {fmt(x.E.mean())} & {x.E.mean() / o_e * 100:.0f} & -- & -- \\\\")
        d = w[w.setting == st]
        for arm in GRID:
            x = d[d.arm == arm]
            both = (x.X.mean() <= ref[ref.arm == "ORACLE"].X.mean() + 0.10) and (x.E.mean() >= 0.80 * o_e)
            e_txt = fmt(x.E.mean())
            row = (f" & {LABEL[arm]} & {len(x)} & {x.X.mean():.2f} & {e_txt} & {x.E.mean() / o_e * 100:.0f} & {fmt(x.TA.mean())} & {fmt(x.R.mean())} \\\\")
            rows.append(row.replace(f"& {LABEL[arm]} &", f"& \\textbf{{{LABEL[arm]}}} &") if both else row)
        rows.append("\\midrule")
    rows = rows[:-1]
    head = ("\\begin{tabular}{@{}llrrrrrr@{}}\n\\toprule\nSetting & Reward & Runs & $X$ & $E$ & $E$ (\\%) & TIR audit & Reported \\\\\n\\midrule\n")
    (OUT / "tab_studyW_full.tex").write_text(head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")

    verdicts = []
    for line in (HERE / "VERDICTS_W.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\|\s*(W\d[^|]*)\|\s*(.*)\|\s*(SUPPORTED|NOT SUPPORTED|NOT TESTABLE)\s*\|\s*$", line)
        if m:
            verdicts.append((m.group(1).strip(), m.group(2).strip(), m.group(3).strip()))
    body = []
    for vid, text, verdict in verdicts:
        text = text.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("<=", "$\\le$").replace(">=", "$\\ge$")
        verdict = {"SUPPORTED": "supported", "NOT SUPPORTED": "\\textbf{not supported}", "NOT TESTABLE": "not testable"}[verdict]
        body.append(f"{vid.replace('_', ' ')} & {text} & {verdict} \\\\")
    head2 = "\\begin{tabularx}{\\linewidth}{@{}>{\\raggedright\\arraybackslash}p{0.17\\linewidth}X>{\\raggedright\\arraybackslash}p{0.17\\linewidth}@{}}\n\\toprule\nID & Result & Verdict \\\\\n\\midrule\n"
    (OUT / "tab_studyW_hyp.tex").write_text(head2 + "\n".join(body) + "\n\\bottomrule\n\\end{tabularx}\n", encoding="utf-8")
    print("wrote", OUT / "tab_studyW_full.tex", "and", OUT / "tab_studyW_hyp.tex", f"({len(verdicts)} verdicts)")


if __name__ == "__main__":
    main()
