"""Retrospective look at Study V (exploratory; the motivation of Study W, not part of its registration).

For each of the two regimes of Study V (Part A: S_DK1; Part C: S_DK1+Z) and each ruin-penalty strength, it reads the stored 100-episode
arrays of the 8 runs and compares three ways of choosing a strength: by the booked return (needs the ledger), by the TIR-audited return (reward
channel plus a clone at evaluation time) and by the reported return. The data had been analysed before (Study V's verdicts), so this is
procedural separation at best, not blinded confirmation. Writes RETROSPECTIVE_V.json next to this file.

    python retrospective_v.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "study_v" / "results_v"
PARTS = {"A": ("S_DK1", range(150, 158)), "C": ("S_DK1+Z", range(170, 178))}
ARMS = ("RUIN1", "RUIN5", "RUIN25")


def mean(xs):
    return math.fsum(xs) / len(xs)


def main() -> None:
    out = {}
    for part, (setting, seeds) in PARTS.items():
        arrays = {}
        for arm in ARMS:
            rows = [json.loads((RES / f"{setting}__{arm}__s{s}.json").read_text())["audit_edge"]["episodes"] for s in seeds]
            arrays[arm] = {k: [r[k] for r in rows] for k in ("booked", "tir", "reported")}
        views = {"all100": range(100), "odd50": range(0, 100, 2), "even50": range(1, 100, 2)}
        out[part] = {}
        for name, idx in views.items():
            score = {k: {arm: mean([run[i] for run in arrays[arm][k] for i in idx]) for arm in ARMS} for k in ("booked", "tir", "reported")}
            pick = {k: max(ARMS, key=lambda a: score[k][a]) for k in score}
            out[part][name] = {"mean": score, "selected": pick}
        sel_odd = out[part]["odd50"]["selected"]["tir"]
        out[part]["selected_on_odd_by_tir_even_booked"] = out[part]["even50"]["mean"]["booked"][sel_odd]
    (HERE / "RETROSPECTIVE_V.json").write_text(json.dumps(out, indent=1))
    for part in out:
        print(f"Part {part}")
        for name in ("all100", "odd50", "even50"):
            m = out[part][name]
            print(f"  {name}: selected by booked {m['selected']['booked']}, by TIR {m['selected']['tir']}, by reported {m['selected']['reported']}")
            for arm in ARMS:
                print(f"     {arm}: booked {m['mean']['booked'][arm]:+8.3f}  TIR {m['mean']['tir'][arm]:+8.3f}  reported {m['mean']['reported'][arm]:+8.3f}")
        print(f"  TIR-selected on odd episodes, booked on even episodes: {out[part]['selected_on_odd_by_tir_even_booked']:+.3f}")


if __name__ == "__main__":
    main()
