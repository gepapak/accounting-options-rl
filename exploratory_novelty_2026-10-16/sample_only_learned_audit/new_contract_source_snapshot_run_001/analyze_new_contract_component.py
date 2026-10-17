"""Frozen auxiliary component; no simulation, actors, training or import-time analysis."""
from __future__ import annotations
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
FILES = ("EXPLORATORY_NEW_CONTRACT_PLAN.md", "NEW_CONTRACT_INDEPENDENT_REVIEW.md", "analyze_new_contract_component.py")
METHODS = ("raw", "capped_raw", "capped_residual", "paired", "ridge_control", "mlp_control")
H, LAG, GAMMA, DELTA = 96, 6, .995, .05/240


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def freeze_path(run_id):
    return HERE/("NEW_CONTRACT_SOURCE_FREEZE_"+run_id+".json")


def freeze(run_id):
    run = HERE/"execution"/run_id
    require((run/"SOURCE_FREEZE.json").is_file(), "Original primary source freeze required")
    require(not list(run.glob("case_*/confirm")), "Auxiliary definition must precede EVERY confirmation phase")
    require(not (run/"confirm_stage_complete.json").exists(), "Cannot freeze after confirmation")
    path = freeze_path(run_id)
    require(not path.exists(), "Never overwrite an auxiliary freeze")
    snapshot = HERE/("new_contract_source_snapshot_"+run_id)
    snapshot.mkdir(exist_ok=False)
    hashes = {}
    for name in FILES:
        content = (HERE/name).read_bytes()
        with (snapshot/name).open("xb") as stream:
            stream.write(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    # Recheck after all source bytes are captured; a concurrent confirmation
    # start cannot be silently accepted by an earlier directory observation.
    require(not list(run.glob("case_*/confirm")), "Confirmation started during auxiliary freeze")
    record = dict(status="AUXILIARY SOURCE-FROZEN BEFORE CONFIRMATION", run_id=run_id,
        created_utc=datetime.now(timezone.utc).isoformat(), source_sha256=hashes,
        primary_source_freeze_sha256=sha(run/"SOURCE_FREEZE.json"), cases_sha256=sha(run/"cases.json"),
        confirmation_phase_directories_present=False, classification="exploratory auxiliary; not primary-plan revision or preregistration",
        n_cases=40, n_methods=6, n_confirm=256, alpha=0., family_limits=240, delta_per_limit=DELTA,
        primary_family_limits_unchanged=520, no_auxiliary_endpoints_inspected=True)
    write(path, record)
    return dict(status=record["status"], created_utc=record["created_utc"], freeze_sha256=sha(path),
                new_simulator_calls=0, new_actor_inference=0, auxiliary_endpoints_emitted=False)


def control(row, method, choice):
    remaining, terms = float(choice["budget"]), []
    delta = row["delta_"+method.removesuffix("_control")]
    require(len(delta) == len(row["maturity_coins"]), "Original common-branch innovations retained")
    for d, coin in zip(delta, row["maturity_coins"]):
        require(math.isfinite(d) and coin in (-1, 1), "Finite original predictor innovation")
        amplitude = min(abs(float(choice["beta"])*d), remaining)
        terms.append(coin*(1. if d > 0 else -1. if d < 0 else 0.)*amplitude)
        remaining = max(remaining-amplitude, 0.)
    return math.fsum(terms)


def value(row, method, choice):
    if method in ("raw", "capped_raw"):
        result = row["gross"]-row["cost"]+row["forgiveness"]
    elif method == "capped_residual":
        result = row["forgiveness"]-row["cost"]
    else:
        result = row["paired_forgiveness"]-row["cost"]
        if method.endswith("_control"):
            result -= control(row, method, choice)
    return result if method == "raw" else min(result, float(choice["cap"]))


def support(bounds, method, choice):
    c, g = float(bounds["discounted_cost_bound"]), float(bounds["discounted_gross_bound"])
    budget = float(choice["budget"]) if method.endswith("_control") else 0.
    if method == "raw":
        return -g-c, g
    if method == "capped_raw":
        return -g-c, min(float(choice["cap"]), g)
    return -c-budget, min(float(choice["cap"]), (g if method == "capped_residual" else g/2)+budget)


def bound(sample, bounds, method, choice):
    require(len(sample) == 256 and all(math.isfinite(x) for x in sample), "All methods use exactly CONFIRM256")
    budget = float(choice["budget"]) if method.endswith("_control") else 0.
    pad = max(1e-6, 1e-10*(1+bounds["discounted_gross_bound"]+bounds["discounted_cost_bound"]+budget)*H)
    lo, hi = support(bounds, method, choice)
    require(min(sample) >= lo-pad and max(sample) <= hi+pad, "Unchanged conservative primary support")
    mean = math.fsum(sample)/len(sample)
    variance = math.fsum((x-mean)**2 for x in sample)/(len(sample)-1)
    log = math.log(2/DELTA)
    radius = math.sqrt(2*variance*log/len(sample))+7*(hi-lo+2*pad)*log/(3*(len(sample)-1))
    return dict(n=len(sample), mean=mean, variance=variance, radius=radius, lower=mean-radius-pad,
        support_lower=lo-pad, support_upper=hi+pad, numerical_allowance=pad, delta=DELTA)


def analyze(run_id):
    run = HERE/"execution"/run_id
    frozen = read(freeze_path(run_id))
    require(not list((HERE/"auxiliary_new_contract").glob(run_id)) if (HERE/"auxiliary_new_contract").exists() else True,
            "Never overwrite auxiliary analysis outputs")
    require((run/"confirm_stage_complete.json").is_file() and (HERE/("native_completion_"+run_id+".json")).is_file(),
            "No auxiliary numerical analysis before complete primary confirmation and aggregation")
    require(sha(run/"SOURCE_FREEZE.json") == frozen["primary_source_freeze_sha256"] and sha(run/"cases.json") == frozen["cases_sha256"],
            "Original primary source/cases unchanged")
    for name, digest in frozen["source_sha256"].items():
        require(sha(HERE/name) == sha(HERE/("new_contract_source_snapshot_"+run_id)/name) == digest, "Auxiliary source unchanged")
    import numpy as np
    cases = read(run/"cases.json")
    require(len(cases) == 40, "All forty original cases")
    choices = {(r["case_id"], r["method"]): r["choice"] for r in read(run/"selection.json") if float(r["alpha"]) == 0.}
    require(set(choices) == {(i, m) for i in range(40) for m in METHODS}, "All original alpha0 choices unchanged")
    selected = read(run/"SELECTION_FREEZE.json")
    require(sha(run/"selection.json") == selected["selection_sha256"], "Original selected choices binding")
    outputs, path_rows, provenance = [], [], []
    for case in cases:
        cid = int(case["case_id"])
        directory = run/f"case_{cid:02d}"/"confirm"
        metadata, completion = read(directory/"metadata.json"), read(directory/"completion.json")
        require(datetime.fromisoformat(frozen["created_utc"]) < datetime.fromisoformat(metadata["started_utc"]),
                "Auxiliary freeze precedes every confirmation phase")
        require(completion["status"] == metadata["status"] == "COMPLETE", "Complete original confirmation")
        for artifact in completion["artifacts"].values():
            require(sha(artifact["path"]) == artifact["sha256"], "Original confirmation artifact unchanged")
        contexts, bounds = metadata["cases_contexts"], metadata["bounds"]
        initial_due, new_due = {}, {}
        for record in contexts:
            context, declared = record["context"], record["support"]
            context_id = int(context["context_id"])
            require(declared["settlement_lag"] == declared["investment_frequency"] == LAG and declared["burn_in"] == 384,
                    "Fixed native six-step source boundary")
            claims = declared["initial_prepaid_claims"]
            require(all(384 <= int(claim["settle_native_step"]) < 384+LAG for claim in claims),
                    "Every initial prepaid claim settles before relative6")
            initial_due[context_id] = {int(claim["settle_native_step"])-384 for claim in claims}
            new_due[context_id] = {i+LAG for i, allowed in enumerate(declared["calendar_new_opening_possible"]) if allowed}
            require(all(step >= LAG for step in new_due[context_id]), "No audit-created maturity before relative6")
        require(set(initial_due) == set(range(8)), "All eight predetermined contexts checked")
        with (directory/"rows.jsonl").open(encoding="utf-8") as stream:
            original = [json.loads(line) for line in stream if line.strip()]
        require(len(original) == 256 and [r["episode_id"] for r in original] == list(range(256)), "Complete original CONFIRM256")
        events = np.load(directory/"financial_events.npz", allow_pickle=False)
        offsets = events["episode_offsets"]
        require(len(offsets) == 257 and offsets[0] == 0 and np.all(np.diff(offsets) >= 0), "Whole-episode financial-event offsets")
        adjusted = []
        for i, row in enumerate(original):
            lo, hi = int(offsets[i]), int(offsets[i+1])
            step = events["event_step"][lo:hi].astype(int)
            coin = events["event_sign_epsilon"][lo:hi].astype(int)
            gross, cost = events["event_gross"][lo:hi], events["event_cost"][lo:hi]
            injection = events["event_injection"][lo:hi]
            cash, base = events["event_cash_before"][lo:hi], events["event_unsigned_base_gross"][lo:hi]
            require(np.all(cash >= -1e-6) and np.all(cost >= 0) and np.all((step >= 0) & (step < H)), "Actual solvent native event states")
            require(np.allclose(gross, np.where(coin != 0, coin*base, 0.), atol=1e-6, rtol=1e-10), "Actual fair-sign event source")
            require(np.allclose(injection, np.maximum(-cash-gross+cost, 0.), atol=1e-6, rtol=1e-10), "Actual ordered local injection")
            pair = (np.maximum(-cash-base+cost, 0.)+np.maximum(-cash+base+cost, 0.))/2
            old = (coin != 0) & (step < LAG)
            new = (coin != 0) & ~old
            context_id = int(row["context_id"])
            require(set(step[old].tolist()) <= initial_due[context_id], "Old maturity mask matches initial-claim schedule")
            require(set(step[new].tolist()) <= new_due[context_id], "Later maturity mask matches possible audit-created claims")
            require(np.all(cost[coin != 0] == 0), "Maturity and all forward financial fee events are separate")
            weights = GAMMA**step
            for key, array in (("gross", gross), ("cost", cost), ("forgiveness", injection), ("paired_forgiveness", pair)):
                require(math.isclose(float(weights@array), float(row[key]), rel_tol=1e-11, abs_tol=1e-5), "Original path summary reconstruction")
            kept = ~old
            item = {key: row[key] for key in ("case_id", "phase", "episode_id", "context_id", "maturity_coins", "delta_ridge", "delta_mlp")}
            item.update(gross=float(weights[kept]@gross[kept]), cost=float(weights@cost),
                forgiveness=float(weights[kept]@injection[kept]), paired_forgiveness=float(weights[kept]@pair[kept]),
                excluded_old_maturity_events=int(old.sum()),
                excluded_old_gross=float(weights[old]@gross[old]),
                excluded_old_forgiveness=float(weights[old]@injection[old]),
                excluded_old_paired_forgiveness=float(weights[old]@pair[old]))
            require(list(coin[coin != 0]) == row["maturity_coins"], "Primary controls retain ALL original maturity coins")
            adjusted.append(item); path_rows.append(item)
        require(bounds["unavoidable_discounted_access_cost"] > 0, "Strict analytical no-edge booked harm")
        for method in METHODS:
            choice = choices[cid, method]
            sample = [value(row, method, choice) for row in adjusted]
            certificate = bound(sample, bounds, method, choice)
            outputs.append(dict(case_id=cid, policy_index=case["policy_index"], policy_family=case["policy_family"],
                profile=case["profile"], alpha=0., method=method, choice=choice, certificate=certificate,
                positive=certificate["lower"] > 0, booked_expectation_upper=-bounds["unavoidable_discounted_access_cost"],
                bounds={k: bounds[k] for k in ("discounted_cost_bound", "discounted_gross_bound", "unavoidable_discounted_access_cost")}))
        provenance.append(dict(case_id=cid, confirm_completion_sha256=sha(directory/"completion.json"),
            confirm_metadata_sha256=sha(directory/"metadata.json"), financial_events_sha256=sha(directory/"financial_events.npz")))
    require(len(outputs) == 240 and len(path_rows) == 10240, "Complete fixed auxiliary family")
    result = dict(status="COMPLETE SEPARATE EXPLORATORY AUXILIARY", run_id=run_id,
        source_freeze_sha256=sha(freeze_path(run_id)), primary_source_freeze_sha256=sha(run/"SOURCE_FREEZE.json"),
        primary_selection_freeze_sha256=sha(run/"SELECTION_FREEZE.json"),
        multiplicity=dict(family_limits=240, total_delta=.05, delta_per_limit=DELTA, joint_primary_auxiliary_95_percent_claim=False),
        n_cases=40, n_confirm_per_method=256, new_native_queries=0, new_model_fits=0, alpha=0.,
        method_counts={m: sum(r["positive"] for r in outputs if r["method"] == m) for m in METHODS},
        profile_counts={p: {m: sum(r["positive"] for r in outputs if r["method"] == m and r["profile"] == p) for m in METHODS}
                        for p in ("strict", "vulnerable")},
        excluded_old_maturity_events=sum(r["excluded_old_maturity_events"] for r in path_rows),
        charged_results=outputs, input_hashes=provenance,
        interpretation="Direct old-maturity gross/I/A excluded; all forward costs/cost injections and inherited state effects retained. Not a no-old-claim or flat counterfactual, new acquisition, or lifecycle profit. Primary-target choices reused, not retuned.")
    output = HERE/"auxiliary_new_contract"/run_id
    output.mkdir(parents=True, exist_ok=False)
    compressed = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=compressed, mtime=0) as stream:
        for row in path_rows:
            stream.write((json.dumps(row, sort_keys=True, allow_nan=False)+"\n").encode("utf-8"))
    with (output/"path_summaries.jsonl.gz").open("xb") as stream:
        stream.write(compressed.getvalue())
    write(output/"results.json", result)
    write(output/"COMPLETION.json", dict(complete=True, result_sha256=sha(output/"results.json"),
        paths_sha256=sha(output/"path_summaries.jsonl.gz"), source_freeze_sha256=sha(freeze_path(run_id))))
    return dict(status=result["status"], method_counts=result["method_counts"], family_limits=240,
                output="auxiliary_new_contract/"+run_id, new_simulator_calls=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "analyze"))
    parser.add_argument("--run-id", default="run_001")
    args = parser.parse_args()
    require(args.run_id.startswith("run_") and args.run_id.replace("_", "").isalnum(), "Simple original run ID")
    print(json.dumps(freeze(args.run_id) if args.action == "freeze" else analyze(args.run_id), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
