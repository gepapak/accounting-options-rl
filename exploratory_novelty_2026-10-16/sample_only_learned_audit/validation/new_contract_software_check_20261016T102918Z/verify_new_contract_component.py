"""Independent saved-event verification of the frozen 240-limit auxiliary.

No frozen auxiliary functions are imported. No simulator, actor or fitted model
is executed. Reading any endpoint requires completed primary/private verification.
"""
from __future__ import annotations
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.dont_write_bytecode = True
VALIDATION = Path(__file__).resolve().parent
HERE = VALIDATION.parent
METHODS = ("raw", "capped_raw", "capped_residual", "paired", "ridge_control", "mlp_control")
DELTA, GAMMA, H, LAG = .05/240, .995, 96, 6


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Checks:
    def __init__(self):
        self.count, self.maximum_error = 0, 0.

    def yes(self, value, message):
        self.count += 1
        if not bool(value):
            raise AssertionError(message)

    def close(self, left, right, message):
        a, b = np.asarray(left, float), np.asarray(right, float)
        self.yes(a.shape == b.shape, message+" shape")
        error = float(np.max(np.abs(a-b), initial=0.))
        self.maximum_error = max(self.maximum_error, error)
        self.yes(np.allclose(a, b, atol=1e-6, rtol=2e-12), message)


def primary_gate(run_id):
    run = HERE/"execution"/run_id
    required = (run/"confirm_stage_complete.json", HERE/("native_completion_"+run_id+".json"))
    if not all(path.is_file() for path in required):
        raise RuntimeError("NOT_READY: primary full completion required before auxiliary endpoint read")
    options = [VALIDATION/("NATIVE_VERIFICATION_"+run_id+suffix+".json") for suffix in ("_critics", "")]
    path = next((p for p in options if p.is_file()), None)
    if path is None:
        raise RuntimeError("NOT_READY: primary private PASS520 required before auxiliary endpoint read")
    primary = read(path)
    if (primary.get("status") != "PASS" or primary.get("bounds_checked") != 520
            or primary.get("result_sha256") != sha(HERE/("native_results_"+run_id+".json"))
            or primary.get("source_freeze_sha256") != sha(run/"SOURCE_FREEZE.json")):
        raise AssertionError("Primary private verification binding failed")
    return run, path


def control(row, method, choice):
    """Vectorized spending independently checks the frozen sequential version."""
    delta = np.asarray(row["delta_"+method.removesuffix("_control")], float)
    coins = np.asarray(row["maturity_coins"], float)
    proposed = np.abs(float(choice["beta"])*delta)
    previous = np.r_[0., np.cumsum(proposed[:-1])] if len(proposed) else np.empty(0)
    amplitudes = np.minimum(proposed, np.maximum(float(choice["budget"])-previous, 0.))
    return float(np.sum(coins*np.sign(delta)*amplitudes))


def sample(rows, method, choice):
    result = []
    for row in rows:
        if method in ("raw", "capped_raw"):
            value = row["gross"]-row["cost"]+row["forgiveness"]
        elif method == "capped_residual":
            value = row["forgiveness"]-row["cost"]
        else:
            value = row["paired_forgiveness"]-row["cost"]
            if method.endswith("_control"):
                value -= control(row, method, choice)
        result.append(value if method == "raw" else min(value, choice["cap"]))
    return np.asarray(result, float)


def certificate(x, bounds, method, choice):
    c, g = float(bounds["discounted_cost_bound"]), float(bounds["discounted_gross_bound"])
    b = float(choice["budget"]) if method.endswith("_control") else 0.
    if method in ("raw", "capped_raw"):
        lo, hi = -c-g, g
    else:
        lo, hi = -c-b, (g if method == "capped_residual" else g/2)+b
    if method != "raw":
        hi = min(hi, float(choice["cap"]))
    pad = max(1e-6, 1e-10*(1+c+g+b)*H)
    mean, variance = float(np.mean(x)), float(np.var(x, ddof=1))
    log = float(np.log(2/DELTA))
    radius = float(np.sqrt(2*variance*log/len(x))+7*(hi-lo+2*pad)*log/(3*(len(x)-1)))
    return dict(n=len(x), mean=mean, variance=variance, radius=radius, lower=mean-radius-pad,
        support_lower=lo-pad, support_upper=hi+pad, numerical_allowance=pad, delta=DELTA)


def verify(run_id):
    run, primary_verification = primary_gate(run_id)
    directory = HERE/"auxiliary_new_contract"/run_id
    if not (directory/"COMPLETION.json").is_file():
        raise RuntimeError("NOT_READY: completed auxiliary analysis required")
    check = Checks()
    completion = read(directory/"COMPLETION.json")
    check.yes(completion["complete"] is True, "complete auxiliary marker")
    check.yes(sha(directory/"results.json") == completion["result_sha256"] and
              sha(directory/"path_summaries.jsonl.gz") == completion["paths_sha256"], "auxiliary artifact hashes")
    freeze_path = HERE/("NEW_CONTRACT_SOURCE_FREEZE_"+run_id+".json")
    frozen = read(freeze_path)
    check.yes(sha(freeze_path) == completion["source_freeze_sha256"], "auxiliary source binding")
    check.yes(sha(run/"SOURCE_FREEZE.json") == frozen["primary_source_freeze_sha256"], "unchanged original primary source freeze")
    for name, digest in frozen["source_sha256"].items():
        check.yes(sha(HERE/name) == sha(HERE/("new_contract_source_snapshot_"+run_id)/name) == digest,
                  "unaltered frozen auxiliary source/snapshot")
    result = read(directory/"results.json")
    check.yes(result["multiplicity"] == dict(family_limits=240, total_delta=.05, delta_per_limit=DELTA,
        joint_primary_auxiliary_95_percent_claim=False), "separate auxiliary error family")
    with gzip.open(directory/"path_summaries.jsonl.gz", "rt", encoding="utf-8") as stream:
        exported = [json.loads(line) for line in stream if line.strip()]
    check.yes(len(exported) == 10240, "complete CONFIRM-only auxiliary paths")
    exported_map = {(row["case_id"], row["episode_id"]): row for row in exported}
    check.yes(len(exported_map) == 10240, "unique auxiliary whole episodes")
    choice_records = read(run/"selection.json")
    choices = {(row["case_id"], row["method"]): row["choice"] for row in choice_records if row["alpha"] == 0.}
    selected = read(run/"SELECTION_FREEZE.json")
    check.yes(sha(run/"selection.json") == selected["selection_sha256"], "original selection unchanged")
    expected_keys = {(i, m) for i in range(40) for m in METHODS}
    outputs = {(row["case_id"], row["method"]): row for row in result["charged_results"]}
    check.yes(set(outputs) == set(choices) == expected_keys and len(result["charged_results"]) == 240,
              "all 240 original case-method combinations")
    cases = read(run/"cases.json")
    check.yes(sha(run/"cases.json") == frozen["cases_sha256"], "original forty-case cohort")
    counts = {method: 0 for method in METHODS}
    profile_counts = {p: {m: 0 for m in METHODS} for p in ("strict", "vulnerable")}
    excluded_events, earliest_confirmation = 0, None
    for case in cases:
        cid = case["case_id"]
        phase = run/f"case_{cid:02d}"/"confirm"
        metadata = read(phase/"metadata.json")
        started = datetime.fromisoformat(metadata["started_utc"])
        check.yes(datetime.fromisoformat(frozen["created_utc"]) < started, "auxiliary freeze before every confirmation start")
        check.yes(datetime.fromisoformat(selected["created_utc"]) < started, "all original choices frozen before confirmation")
        earliest_confirmation = started if earliest_confirmation is None else min(earliest_confirmation, started)
        bounds = metadata["bounds"]
        old_schedule, new_schedule = {}, {}
        for record in metadata["cases_contexts"]:
            context_id, declared = record["context"]["context_id"], record["support"]
            check.yes(declared["burn_in"] == 384 and declared["settlement_lag"] == declared["investment_frequency"] == 6,
                      "actual fixed native timing")
            claims = declared["initial_prepaid_claims"]
            for claim in claims:
                opened, settled = claim["open_native_step"], claim["settle_native_step"]
                check.yes(opened < 384 and settled == opened+6 and 384 <= settled < 390,
                          "old claim opened before audit and settles within first six slots")
            old_schedule[context_id] = {claim["settle_native_step"]-384 for claim in claims}
            new_schedule[context_id] = {i+6 for i, allowed in enumerate(declared["calendar_new_opening_possible"]) if allowed}
            check.yes(all(t >= 6 for t in new_schedule[context_id]), "new contracts cannot mature in old-claim window")
        check.yes(set(old_schedule) == set(range(8)), "all eight contexts retained")
        original = [json.loads(line) for line in (phase/"rows.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        events = np.load(phase/"financial_events.npz", allow_pickle=False)
        offsets = events["episode_offsets"]
        check.yes(len(original) == 256 and len(offsets) == 257, "fixed original confirmation budget")
        rebuilt = []
        for i, primary in enumerate(original):
            row = exported_map[cid, i]
            check.yes((row["phase"], row["episode_id"], row["context_id"]) == ("confirm", i, primary["context_id"]),
                      "unchanged original path identity")
            lo, hi = int(offsets[i]), int(offsets[i+1])
            t = events["event_step"][lo:hi].astype(int)
            coins = events["event_sign_epsilon"][lo:hi].astype(int)
            g, c = events["event_gross"][lo:hi], events["event_cost"][lo:hi]
            cash, base = events["event_cash_before"][lo:hi], events["event_unsigned_base_gross"][lo:hi]
            inj = events["event_injection"][lo:hi]
            check.yes(np.all(c >= 0) and np.all(cash >= -1e-6), "actual ordered solvent financial events")
            check.close(inj, np.maximum(-cash-g+c, 0.), "native actual injection reconstruction")
            pair = .5*(np.maximum(c-cash-base, 0.)+np.maximum(c-cash+base, 0.))
            old = np.array([epsilon != 0 and int(step) < 6 for epsilon, step in zip(coins, t)], dtype=bool)
            check.yes(set(t[old]) <= old_schedule[primary["context_id"]], "exact old event schedule")
            check.yes(set(t[(coins != 0) & ~old]) <= new_schedule[primary["context_id"]], "new maturity schedule")
            check.yes(np.all(c[coins != 0] == 0), "maturity costs are separate; all fee events retained")
            keep, w = (~old).astype(float), GAMMA**t
            expected = dict(gross=float(np.sum(w*g*keep)), cost=float(np.sum(w*c)),
                forgiveness=float(np.sum(w*inj*keep)), paired_forgiveness=float(np.sum(w*pair*keep)),
                excluded_old_maturity_events=int(old.sum()), excluded_old_gross=float(np.sum(w*g*old)),
                excluded_old_forgiveness=float(np.sum(w*inj*old)), excluded_old_paired_forgiveness=float(np.sum(w*pair*old)))
            for key, value in expected.items():
                check.close(row[key], value, "independent retained/excluded component "+key)
            check.close(row["cost"], primary["cost"], "ALL original forward costs retained")
            for key in ("maturity_coins", "delta_ridge", "delta_mlp"):
                check.close(row[key], primary[key], "ALL old and new original innovations retained "+key)
            rebuilt.append(row)
            excluded_events += int(old.sum())
        for method in METHODS:
            saved, choice = outputs[cid, method], choices[cid, method]
            check.yes(saved["choice"] == choice and saved["alpha"] == 0., "original alpha0 choice reused without retuning")
            x = sample(rebuilt, method, choice)
            expected = certificate(x, bounds, method, choice)
            check.yes(len(x) == 256, "RAW and every other method use CONFIRM256")
            for key, value in expected.items():
                check.close(saved["certificate"][key], value, "independent auxiliary EB "+key)
            check.yes(np.min(x) >= expected["support_lower"]-1e-6 and np.max(x) <= expected["support_upper"]+1e-6,
                      "conservative original support retained")
            check.close(saved["booked_expectation_upper"], -bounds["unavoidable_discounted_access_cost"], "analytic booked harm")
            check.yes(saved["positive"] == (expected["lower"] > 0), "auxiliary positive-limit decision")
            counts[method] += int(saved["positive"])
            profile_counts[case["profile"]][method] += int(saved["positive"])
    check.yes(result["method_counts"] == counts and result["profile_counts"] == profile_counts, "all auxiliary counts")
    check.yes(result["excluded_old_maturity_events"] == excluded_events, "complete exclusion count")
    return dict(status="PASS", checks=check.count, bounds_checked=240, n_confirm=256,
        verifier_sha256=sha(__file__),
        maximum_absolute_recomputation_error=check.maximum_error, method_counts=counts,
        source_freeze_sha256=sha(freeze_path), primary_private_verification_sha256=sha(primary_verification),
        result_sha256=sha(directory/"results.json"), paths_sha256=sha(directory/"path_summaries.jsonl.gz"),
        auxiliary_frozen_utc=frozen["created_utc"], earliest_confirmation_utc=earliest_confirmation.isoformat(),
        source_claim_boundary_checked=True, all_primary_innovations_and_choices_retained=True,
        new_simulator_calls=0, actor_inference=0, model_fitting=0,
        interpretation="separate 95% auxiliary component family; all actual inherited state effects and forward costs retained; no joint primary/auxiliary95% claim")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="run_001")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if not args.run_id.startswith("run_") or not args.run_id.replace("_", "").isalnum():
        raise ValueError("Simple original run ID required")
    report = verify(args.run_id)
    if args.write:
        with (VALIDATION/("NEW_CONTRACT_VERIFICATION_"+args.run_id+".json")).open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
