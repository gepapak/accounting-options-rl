"""Read-only tabulation of probe_actions_dr.json (post hoc, not registered, 16 Nov 2026).

    python tabulate_probe.py > tabulate_probe_out.txt
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
res = json.loads((Path(__file__).resolve().parent / "probe_actions_dr.json").read_text())
order = [("V2", "ORIG"), ("V4", "ORIG"), ("V8", "ORIG"), ("V16", "ORIG"), ("V16", "FLPATCH"), ("L30", "ORIG"), ("L500", "ORIG"),
         ("L500", "FLPATCH")]
print("arm  cond     n | registered learners | collects >1 pt credit at 0.1 or 0.25 | mean credit 0.1  0.25 | mean best-of-two | "
      "traded lot at low equity (median, lots) | refused at e<=0.05")
for arm, cond in order:
    g = [r for r in res if r["arm"] == arm and r["condition"] == cond]
    c1 = np.array([r["low"]["0.1"]["credit_pct"] for r in g])
    c2 = np.array([r["low"]["0.25"]["credit_pct"] for r in g])
    best = np.maximum(c1, c2)
    reg = sum(r["xlow"] >= 0.10 for r in g)
    col = int((best > 1.0).sum())
    lots = [np.median([p["volume"] for p in r["probe"] if p["e"] <= 0.25 and not p["hold"] and not p["refused"]] or [np.nan]) for r in g]
    lots = [x for x in lots if x == x]
    refused = sum(any(p["refused"] for p in r["probe"] if p["e"] <= 0.05) for r in g)
    print(f"{arm:<4} {cond:<7} {len(g):>2} | {reg:>2} | {col:>2} | {c1.mean():+6.2f} {c2.mean():+6.2f} | {best.mean():+6.2f} | "
          f"{('%.2f' % np.median(lots)) if lots else '  - '} (n {len(lots)}) | {refused}")
print()
print("disagreements between the registered label (xlow >= 0.10) and collecting > 1 pt of credit:")
for r in res:
    best = max(r["low"]["0.1"]["credit_pct"], r["low"]["0.25"]["credit_pct"])
    if (r["xlow"] >= 0.10) != (best > 1.0):
        print(f"  {r['arm']} {r['condition']} s{r['seed']}: xlow {r['xlow']:.3f}, credit 0.1 {r['low']['0.1']['credit_pct']:+.2f}, "
              f"0.25 {r['low']['0.25']['credit_pct']:+.2f}")
print()
print("runs ranked by xlow next to their credit (the drifting control marked *):")
rank = sorted(res, key=lambda r: -r["xlow"])[:24]
for r in rank:
    best = max(r["low"]["0.1"]["credit_pct"], r["low"]["0.25"]["credit_pct"])
    mark = "*" if r["condition"] == "FLPATCH" else " "
    print(f" {mark} {r['arm']:<4} {r['condition']:<7} s{r['seed']} xlow {r['xlow']:.3f}  best credit {best:+6.2f}")
