"""Exploratory existing-data feasibility: select ruin strength by TIR only.

No training or environment imports. Candidate roster is fixed by the parent
review request. The data and their study-level conclusions have already been
viewed: this procedural separation is not blinded validation or registration.
Episode ordinal 1,3,... selects; ordinal 2,4,... evaluates. Weeks overlap.
"""
from pathlib import Path
import hashlib
import json
import math

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
INPUT = CODE / "ACCOUNTING_OPTIONS_2026-10-14" / "study_v" / "results_v"
PARTS = {"A": ("S_DK1", range(150,158)), "C": ("S_DK1+Z", range(170,178))}
ARMS = ("RUIN1", "RUIN5", "RUIN25")


def mean(xs):
    return math.fsum(xs)/len(xs)


def main():
    stage_path = HERE / "RUIN_SELECTIONS_BY_TIR_ONLY.json"
    result_path = HERE / "retrospective_ruin_selection_results.json"
    if stage_path.exists() or result_path.exists():
        raise RuntimeError("Review files already exist; refusing overwrite")

    # Phase 1 extracts TIR arrays only, no booked values enter this selection.
    tir_data = {}
    input_hashes = {}
    for part, (setting,seeds) in PARTS.items():
        for arm in ARMS:
            tir_data[part,arm] = []
            for seed in seeds:
                p = INPUT / f"{setting}__{arm}__s{seed}.json"
                raw = p.read_bytes()
                input_hashes[p.name] = hashlib.sha256(raw).hexdigest()
                doc = json.loads(raw)
                assert (doc["setting"],doc["arm"],doc["seed"]) == (setting,arm,seed)
                vals = doc["audit_edge"]["episodes"]["tir"]
                assert len(vals) == 100
                assert all(math.isfinite(x) for x in vals)
                tir_data[part,arm].append(vals)
                del doc

    selections = {}
    for part in PARTS:
        selections[part] = {}
        for name, indices in (("all100",range(100)),("odd50",range(0,100,2))):
            scores = {arm:mean([row[i] for row in tir_data[part,arm] for i in indices]) for arm in ARMS}
            chosen = max(ARMS,key=lambda arm:scores[arm])
            selections[part][name] = {"criterion":"highest aggregate mean TIR-audited edge return",
                                      "selected":chosen,"scores":scores,
                                      "tie_order":list(ARMS)}
    stage = {"status":"retrospective TIR-only selections persisted before this script evaluates booked arrays",
             "prior_knowledge":"All study-level results already viewed; no independent confirmation claimed",
             "selections":selections,"input_sha256":input_hashes}
    stage_path.write_text(json.dumps(stage,indent=2)+"\n",encoding="utf-8")
    stage_hash = hashlib.sha256(stage_path.read_bytes()).hexdigest()

    # Phase 2 evaluates booked arrays with fixed, already persisted choices.
    booked_data = {}
    for part,(setting,seeds) in PARTS.items():
        for arm in ARMS:
            booked_data[part,arm] = []
            for seed in seeds:
                p = INPUT / f"{setting}__{arm}__s{seed}.json"
                raw = p.read_bytes()
                assert hashlib.sha256(raw).hexdigest() == input_hashes[p.name]
                doc = json.loads(raw)
                vals = doc["audit_edge"]["episodes"]["booked"]
                assert len(vals) == 100 and all(math.isfinite(x) for x in vals)
                assert abs(mean(vals)-doc["audit_edge"]["booked"]) < 1e-10
                assert abs(mean(vals)-doc["test_edge"]["booked_pct_K0"]) < 1e-6
                booked_data[part,arm].append(vals)

    reports = {}
    for part in PARTS:
        reports[part] = {}
        for view,indices in (("all100",range(100)),("odd50",range(0,100,2)),("even50",range(1,100,2))):
            metrics = {}
            for arm in ARMS:
                ts = [r[i] for r in tir_data[part,arm] for i in indices]
                bs = [r[i] for r in booked_data[part,arm] for i in indices]
                errors = [t-b for t,b in zip(ts,bs)]
                seed_errors = [mean([t[i]-b[i] for i in indices]) for t,b in zip(tir_data[part,arm],booked_data[part,arm])]
                metrics[arm] = {"TIR":mean(ts),"booked":mean(bs),
                                "signed_error_TIR_minus_booked":mean(errors),
                                "episode_MAE":mean([abs(e) for e in errors]),
                                "per_seed_signed_error_min":min(seed_errors),
                                "per_seed_signed_error_max":max(seed_errors),
                                "seeds":8,"episodes_per_seed":len(indices)}
            reports[part][view] = {"metrics":metrics,
                                   "booked_order":sorted(ARMS,key=lambda a:metrics[a]["booked"],reverse=True),
                                   "TIR_order":sorted(ARMS,key=lambda a:metrics[a]["TIR"],reverse=True)}

    conclusions = {}
    for part in PARTS:
        full = selections[part]["all100"]["selected"]
        split = selections[part]["odd50"]["selected"]
        conclusions[part] = {
            "selected_full":full,
            "full_selected_metrics":reports[part]["all100"]["metrics"][full],
            "selected_odd":split,
            "even_selected_metrics":reports[part]["even50"]["metrics"][split],
            "full_selection_matches_booked_max":full==reports[part]["all100"]["booked_order"][0],
            "odd_selection_matches_even_booked_max":split==reports[part]["even50"]["booked_order"][0]}
    out = {"scope":"ONE retrospective existing-result feasibility check; 48 read-only V JSONs; zero training/environment/model calls",
           "units":"percentage points of allocation K0 per test episode/week",
           "selection_stage_sha256":stage_hash,"reports":reports,"conclusions":conclusions,
           "limitations":["Retrospective after study outcomes were viewed; not registered or blinded.",
                          "All100 selection and evaluation reuse identical episodes.",
                          "Odd/even are fixed episode ordinals, with shared/overlapping market windows; no independent confirmation.",
                          "Seed means are conditional on the common market experiment; no iid-market inference or p-values.",
                          "Selects return among the tested grid only; does not certify exploit removal or safety.",
                          "A trainer with a sign twin could train TIR directly; this feasibility check is not a new AI method."]}
    result_path.write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(conclusions,indent=2))


if __name__ == "__main__":
    main()
