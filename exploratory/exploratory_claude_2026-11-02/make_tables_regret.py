"""Write the manuscript table of the regret-bound terms (sections/tab_regretterms.tex) from regret_bounds_results.json.

    python make_tables_regret.py [output_dir]      # default: the sections/ folder of the manuscript repository next to this repository
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parents[1] / "accounting-options-rl-latex-latex" / "sections"
rows = json.loads((HERE / "regret_bounds_results.json").read_text(encoding="utf-8"))
names = {"S_DK1": "DK1", "T_DK2": "DK2", "T_DK1M": "DK1, weaker edge"}
starts = {"log_uniform": "log-uniform", "zero_half": "half from zero"}
lines = ["\\begin{tabular}{@{}llrrrrrr@{}}", "\\toprule",
         "Setting & Starts & Edge & Reported & SPR & TIR & Credited & Non-ident. \\\\",
         " & & ($E$) & credit (\\%$E$) & (\\%$E$) & (\\%$E$) & steps (\\%) & steps (\\%) \\\\", "\\midrule"]
prev = None
for r in rows:
    e = r["booked_edge_per_episode"]
    label = names[r["setting"]] if r["setting"] != prev else ""
    prev = r["setting"]
    lines.append(f"{label} & {starts[r['start']]} & {e:.1f} & {100 * r['A_reported'] / e:.1f} & {100 * r['A_SPR'] / e:.2f} & "
                 f"{100 * r['A_TIR'] / e:.3f} & {100 * r['credited_step_share']:.2f} & {100 * r['nonidentified_step_share']:.3f} \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
(OUT / "tab_regretterms.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("wrote", OUT / "tab_regretterms.tex")
