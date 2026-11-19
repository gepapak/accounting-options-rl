"""Supplementary table of the cost-shift test's power by violation family, from validate_exact_results.json (run validate_exact.py first).

    python make_costshift_table.py     # writes ../../../accounting-options-rl-latex-latex/sections/tab_costshift_power.tex
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[2] / "accounting-options-rl-latex-latex" / "sections" / "tab_costshift_power.tex"
res = json.loads((HERE / "validate_exact_results.json").read_text(encoding="utf-8"))


def n(v: int) -> str:
    return f"{v:,}".replace(",", "{,}")


lines = ["\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{0.46\\linewidth}rr>{\\centering\\arraybackslash}p{0.17\\linewidth}@{}}", "\\toprule",
         "World (4{,}000 random draws each) & Earlier check & Exact test & Equivalent to a single floor \\\\", "\\midrule"]
for name, v in res["power"].items():
    lines.append(f"{name[0].upper() + name[1:]} & {n(v['flagged_previous_test'])} & {n(v['flagged_exact_test'])} & "
                 f"{'yes' if v['single_floor_equivalent'] else 'no'} \\\\")
fa0, fa1 = res["false_alarms_noise_free"], res["false_alarms_with_noise"]
lines += ["\\midrule",
          f"Single floor, exact arithmetic ({n(fa0['worlds'])} worlds) & {n(fa0['flagged_previous_test'])} & {n(fa0['flagged_exact'])} & yes \\\\",
          f"Single floor, noise $10^{{-13}}\\times$scale ({n(fa1['worlds'])} worlds) & {n(fa1['flagged_previous_test'])} & {n(fa1['flagged_exact'])} & yes \\\\",
          "\\bottomrule", "\\end{tabular}"]
OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
