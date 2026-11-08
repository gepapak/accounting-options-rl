"""Manuscript tables of Study EXT, from runs_ext.csv and VERDICTS_EXT.md (written after the registered analysis; they compute nothing new). python make_tab_ext.py [folder]"""
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
d = pd.read_csv(HERE / "runs_ext.csv")
names = (("REP", "Reported (the engine's reward)"), ("LEDGER", "Ledger (change of public wealth)"), ("SPR", "SPR"), ("TIR", "TIR"), ("TSR", "TSR"), ("RUIN5", "Ruin penalty 5"), ("RUIN25", "Ruin penalty 25"), ("RUIN100", "Ruin penalty 100"), ("FINDER", "Finder (reward = width)"), ("FINDER_DEF", "Finder, default positions"))
lines = [r"\begin{tabular}{@{}lrrrrrrrrr@{}}", r"\toprule", r"Reward & Agents & $X$ & Bust share & Reported & Booked & Width & Audit, reading & Audit, TIR & Audit, SPR \\", r"\midrule"]
for arm, label in names:
    x = d[d.arm == arm]
    lines.append(f"{label} & {len(x)} & {x.X.mean() + 1e-12:.3f} & {x.bust.mean():.3f} & ${x.reported.mean():+.1f}$ & ${x.booked.mean():+.1f}$ & {x.width.mean():.1f} & {x.mae_line.mean():.0e} & {x.mae_tir.mean():.0e} & {x.mae_spr.mean():.1f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(out / "tab_studyEXT_full.tex").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
print("\n".join(lines))


def tex(s: str) -> str:
    s = s.replace("\\", r"\textbackslash{}").replace("%", r"\%").replace("_", r"\_").replace("&", r"\&").replace("<=", "$\\le$").replace(">=", "$\\ge$").replace("<", "$<$").replace(">", "$>$")
    return re.sub(r"(\d)e-(\d+)", r"\1e$-$\2", s)


rows = []
for line in (HERE / "VERDICTS_EXT.md").read_text(encoding="utf-8").splitlines():
    m = re.match(r"^\|\s*(E[AS]\d+)\s*\|\s*(.*?)\s*\|\s*(SUPPORTED|NOT SUPPORTED|NOT TESTABLE)\s*\|\s*$", line)
    if m:
        rows.append(f"{m.group(1)} & {tex(m.group(2))} & {'Supported' if m.group(3) == 'SUPPORTED' else 'Not supported'} \\\\")
assert len(rows) == 17, len(rows)
hyp = [r"\begin{tabularx}{\linewidth}{@{}l>{\raggedright\arraybackslash}Xl@{}}", r"\toprule", r"ID & Result & Verdict \\", r"\midrule"] + rows + [r"\bottomrule", r"\end{tabularx}"]
(out / "tab_studyEXT_hyp.tex").write_text("\n".join(hyp) + "\n", encoding="utf-8", newline="\n")
