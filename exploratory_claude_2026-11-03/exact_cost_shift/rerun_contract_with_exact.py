"""Re-run the 2 November contract benchmark (exploratory_claude_2026-11-02/contract_test.py) with the exact test next to the previous one.

    python rerun_contract_with_exact.py     # writes contract_exact_results.json next to this file

The sleeve sweeps and the engine's original methods are those of contract_test.py, unchanged (same seeds, same states); only the verdict of every state is
computed twice: by the previous implied-conditions test and by single_floor_fit (the exact test).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
sys.path[:0] = [str(CODE / "exploratory_claude_2026-11-02"), str(HERE)]
import contract_test as ct  # noqa: E402
from exact_cost_shift import single_floor_fit  # noqa: E402

old_fn = ct.contract_test
LOG: list = []


def both(pairs, tol=ct.TOL):
    old = old_fn(pairs, tol)
    ok, _, why = single_floor_fit(pairs)
    LOG.append((bool(old[0]), not ok, why))
    return (not ok), why


ct.contract_test = both


def summarize_both(rows, flags):
    for r, (old, new, _) in zip(rows, flags):
        r["violated_previous"], r["violated_exact"] = old, new
    n = len(rows)
    binds = [r for r in rows if r["binds"]]
    free = [r for r in rows if not r["binds"]]
    inf_b = [r for r in binds if r["informative"]]
    uninf_b = [r for r in binds if not r["informative"]]
    def cnt(sel, key):
        return int(sum(r[key] for r in sel))
    return dict(tested_states=n,
                states_where_a_tested_fee_exceeds_the_account=len(binds),
                informative_among_them=len(inf_b),
                flagged_previous_among_informative=cnt(inf_b, "violated_previous"), flagged_exact_among_informative=cnt(inf_b, "violated_exact"),
                uninformative_among_them=len(uninf_b),
                flagged_previous_among_uninformative=cnt(uninf_b, "violated_previous"), flagged_exact_among_uninformative=cnt(uninf_b, "violated_exact"),
                states_where_no_tested_fee_exceeds_the_account=len(free),
                flagged_previous_among_free=cnt(free, "violated_previous"), flagged_exact_among_free=cnt(free, "violated_exact"))


def main():
    res = {}
    for name, cls in (("single_floor_sleeve", ct.EdgeEnvT), ("multi_floor_sleeve", ct.EdgeEnvMF)):
        LOG.clear()
        rows = ct.sweep_sleeve(cls)
        res[name] = summarize_both(rows, list(LOG))
        print(name, json.dumps(res[name]))
        # which exact reasons occur among the flagged states
        reasons = {}
        for (old, new, why), r in zip(LOG, rows):
            if new:
                key = why.split("(best misfit")[0].strip()
                reasons[key] = reasons.get(key, 0) + 1
        res[name]["reasons_exact"] = reasons
    LOG.clear()
    eng = ct.engine_sweep()
    for row, (old, new, why) in zip(eng, list(LOG)):
        row["violated_previous"], row["violated_exact"], row["why_exact"] = old, new, why
        print("engine cash", row["starting_cash"], "previous:", old, "exact:", new, "|", why)
    res["engine_original_methods"] = eng
    (HERE / "contract_exact_results.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
