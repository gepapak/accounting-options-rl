"""Manuscript tables of Study FT, from runs_ft.csv and VERDICTS_FT.md (written after the registered analysis; they compute nothing new). python make_tab_ft.py [folder]"""
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
d = pd.read_csv(HERE / "runs_ft.csv")
parts = (("A", "A: unrealized bucket"), ("B", "B: deficit, half funded"), ("C", "C: deficit only"))
arms = (("REP", "Reported"), ("LEDGER", "Ledger"), ("SPR", "SPR"), ("TIR", "TIR"), ("TSR", "TSR"))
lines = [r"\begin{tabular}{@{}llrrrrrrrrr@{}}", r"\toprule", r"Part & Reward & Agents & $X$ & $E$ & $E_z$ & $E_d$ & Placebo, reported & Placebo, booked & Audit TIR, deficit & Audit SPR, deficit \\", r"\midrule"]
for i, (part, name) in enumerate(parts):
    if i:
        lines.append(r"\addlinespace")
    for arm, label in arms:
        x = d[(d.part == part) & (d.arm == arm)]
        lines.append(f"{name if arm == 'REP' else ''} & {label} & {len(x)} & {x.X.mean() + 1e-12:.2f} & ${x.E.mean():+.1f}$ & ${x.Ez.mean():+.1f}$ & ${x.Ed.mean():+.1f}$ & ${x.placebo_reported.mean():+.1f}$ & ${x.placebo_booked.mean():+.1f}$ & {x.mae_tir_deficit.mean():.1f} & {x.mae_spr_deficit.mean():.1f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(out / "tab_studyFT_full.tex").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
print("\n".join(lines))


def tex(s: str) -> str:
    s = s.replace("\\", r"\textbackslash{}").replace("%", r"\%").replace("_", r"\_").replace("&", r"\&").replace("<=", "$\\le$").replace(">=", "$\\ge$").replace("<", "$<$").replace(">", "$>$")
    return re.sub(r"(\d)e-(\d+)", r"\1e$-$\2", s)


rows = []
for line in (HERE / "VERDICTS_FT.md").read_text(encoding="utf-8").splitlines():
    m = re.match(r"^\|\s*(F[ABDC]\d+)\s*\|\s*(.*?)\s*\|\s*(SUPPORTED|NOT SUPPORTED|NOT TESTABLE)\s*\|\s*$", line)
    if m:
        rows.append(f"{m.group(1)} & {tex(m.group(2))} & {'Supported' if m.group(3) == 'SUPPORTED' else ('Not testable' if m.group(3) == 'NOT TESTABLE' else 'Not supported')} \\\\")
assert len(rows) == 23, len(rows)
hyp = [r"\begin{tabularx}{\linewidth}{@{}l>{\raggedright\arraybackslash}Xl@{}}", r"\toprule", r"ID & Result & Verdict \\", r"\midrule"] + rows + [r"\bottomrule", r"\end{tabularx}"]
(out / "tab_studyFT_hyp.tex").write_text("\n".join(hyp) + "\n", encoding="utf-8", newline="\n")
