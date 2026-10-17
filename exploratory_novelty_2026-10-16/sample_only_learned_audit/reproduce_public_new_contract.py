"""Standalone standard-library arithmetic replay of the separate 240-limit auxiliary.

The reusable arithmetic was copied from the independent primary reproducer,
not imported from the frozen auxiliary analyzer. No native inputs are required.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
SCHEMA="sample_only_new_contract_public_v1"
METHODS=("raw","capped_raw","capped_residual","paired","ridge_control","mlp_control")
DELTA,HORIZON=.05/240,96
MANDATORY_FILES=frozenset(("cases_and_bounds.json","selection.json","results.json","provenance.json","EXPORT_ARITHMETIC_CHECK.json","paths.jsonl.gz","reproduce_public_new_contract.py","README.md"))

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
        self.count, self.maximum_absolute_error = 0, 0.

    def yes(self, value, description):
        self.count += 1
        if not value:
            raise AssertionError(description)

    def close(self, actual, expected, description):
        self.yes(math.isfinite(float(actual)) and math.isfinite(float(expected)), description+" finite")
        error = abs(float(actual)-float(expected))
        self.maximum_absolute_error = max(self.maximum_absolute_error, error)
        self.yes(error <= 1e-6+2e-12*max(abs(float(actual)), abs(float(expected))), description)


def sequential_control(deltas, coins, beta, budget):
    """Independent sequential reconstruction of the predictable path budget."""
    remaining, terms = float(budget), []
    if remaining < 0 or not math.isfinite(remaining) or beta < 0 or not math.isfinite(beta):
        raise ValueError("Invalid frozen control settings")
    if len(deltas) != len(coins):
        raise ValueError("Mismatched ordered innovation vectors")
    for delta, coin in zip(deltas, coins):
        if not math.isfinite(delta) or coin not in (-1, 1):
            raise ValueError("Invalid stored innovation")
        amplitude = min(abs(beta*delta), remaining)
        terms.append(coin*(1. if delta > 0 else -1. if delta < 0 else 0.)*amplitude)
        remaining = max(remaining-amplitude, 0.)
    return math.fsum(terms)


def path_value(row, method, alpha, choice):
    if method in ("raw", "capped_raw"):
        value = row["gross"]-row["cost"]+(1-alpha)*row["forgiveness"]
    elif method == "capped_residual":
        value = (1-alpha)*row["forgiveness"]-row["cost"]
    else:
        value = (1-alpha)*row["paired_forgiveness"]-row["cost"]
        if method.endswith("_control"):
            model = method.removesuffix("_control")
            value -= sequential_control(row["delta_"+model], row["maturity_coins"],
                                        float(choice["beta"]), float(choice["budget"]))
    return value if method == "raw" else min(value, float(choice["cap"]))


def numerical_pad(bounds, choice=None):
    return max(1e-6, 1e-10*(1+float(bounds["discounted_gross_bound"])+
        float(bounds["discounted_cost_bound"])+float((choice or {}).get("budget", 0.)))*HORIZON)


def deterministic_support(bounds, method, alpha, choice):
    c, g = float(bounds["discounted_cost_bound"]), float(bounds["discounted_gross_bound"])
    if method == "raw":
        return -g-c, g
    if method == "capped_raw":
        return -g-c, min(float(choice["cap"]), g)
    budget = float(choice["budget"]) if method.endswith("_control") else 0.
    upper = (1-alpha)*g*(1. if method == "capped_residual" else .5)+budget
    return -c-budget, min(float(choice["cap"]), upper)


def empirical_bernstein(sample, lower, upper, pad):
    n = len(sample)
    if n < 2 or not all(math.isfinite(x) for x in sample):
        raise ValueError("Finite independent whole-episode sample required")
    mean = math.fsum(sample)/n
    variance = math.fsum((x-mean)**2 for x in sample)/(n-1)
    lo, hi = lower-pad, upper+pad
    if min(sample) < lo-1e-6 or max(sample) > hi+1e-6:
        raise AssertionError("Exported path falls outside deterministic support")
    log = math.log(2/DELTA)
    radius = math.sqrt(2*variance*log/n)+7*(hi-lo)*log/(3*(n-1))
    return dict(n=n, mean=mean, variance=variance, radius=radius, lower=mean-radius-pad,
                support_lower=lo, support_upper=hi, delta=DELTA, numerical_allowance=pad)


def verify_payload(path_rows,case_records,selections,saved_results):
    check=Checks()
    cases={row["case_id"]:row for row in case_records}
    check.yes(len(case_records)==40 and set(cases)==set(range(40)),"all forty original cases")
    check.yes({row["policy_index"] for row in case_records}==set(range(20)),"all twenty frozen policies")
    expected_keys={(cid,method) for cid in range(40) for method in METHODS}
    choices={(row["case_id"],row["method"]):row["choice"] for row in selections}
    check.yes(len(selections)==240 and set(choices)==expected_keys and all(row["alpha"]==0 for row in selections),"all original alpha0 choices")
    outputs={(row["case_id"],row["method"]):row for row in saved_results["charged_results"]}
    check.yes(len(saved_results["charged_results"])==240 and set(outputs)==expected_keys,"all 240 auxiliary lower limits")
    check.yes(saved_results["multiplicity"]==dict(family_limits=240,total_delta=.05,delta_per_limit=DELTA,joint_primary_auxiliary_95_percent_claim=False),"separate240-event error family")
    fields={"case_id","phase","episode_id","context_id","gross","cost","forgiveness","paired_forgiveness","maturity_coins","delta_ridge","delta_mlp"}
    grouped=defaultdict(list);total=0
    for row in path_rows:
        check.yes(set(row)==fields and row["phase"]=="confirm" and row["case_id"] in cases,"CONFIRM-only public field whitelist")
        check.yes(isinstance(row["context_id"],int) and 0<=row["context_id"]<8,"fixed context ID")
        check.yes(all(math.isfinite(float(row[key])) for key in ("gross","cost","forgiveness","paired_forgiveness")),"finite component summaries")
        check.yes(all(row[key]>=0 for key in ("cost","forgiveness","paired_forgiveness")),"nonnegative retained fees and injections")
        check.yes(all(e in (-1,1) for e in row["maturity_coins"]),"all original fair innovations")
        check.yes(len(row["maturity_coins"])==len(row["delta_ridge"])==len(row["delta_mlp"]),"all old/new control differences retained")
        grouped[row["case_id"]].append(row);total+=1
    check.yes(total==10240,"all 10240 confirmation paths retained")
    counts={method:0 for method in METHODS}
    profiles={p:{m:0 for m in METHODS} for p in ("strict","vulnerable")}
    for cid,case in cases.items():
        paths=grouped[cid]
        check.yes(len(paths)==256 and sorted(r["episode_id"] for r in paths)==list(range(256)),"every method uses exactly CONFIRM256")
        check.yes(case["case_id"]==2*case["policy_index"]+case["profile_index"],"original paired case structure")
        bounds=case["bounds"]
        check.yes(all(math.isfinite(float(bounds[k])) and bounds[k]>=0 for k in ("discounted_cost_bound","discounted_gross_bound","unavoidable_discounted_access_cost")),"original conservative scalar supports")
        check.yes(bounds["unavoidable_discounted_access_cost"]>0,"analytic strict expected booked harm")
        for method in METHODS:
            choice=choices[cid,method];saved=outputs[cid,method]
            check.yes(saved["choice"]==choice and saved["alpha"]==0,"unchanged primary choice")
            x=[path_value(row,method,0.,choice) for row in paths]
            lo,hi=deterministic_support(bounds,method,0.,choice)
            expected=empirical_bernstein(x,lo,hi,numerical_pad(bounds,choice))
            for key,value in expected.items():check.close(saved["certificate"][key],value,"auxiliary EB "+key)
            check.close(saved["booked_expectation_upper"],-bounds["unavoidable_discounted_access_cost"],"analytical fee harm")
            check.yes(saved["positive"]==(expected["lower"]>0),"auxiliary lower-limit decision")
            counts[method]+=int(expected["lower"]>0);profiles[case["profile"]][method]+=int(expected["lower"]>0)
    check.yes(saved_results["method_counts"]==counts and saved_results["profile_counts"]==profiles,"all auxiliary detection counts")
    return dict(status="PASS",checks=check.count,bounds_recomputed=240,cases=40,policies=20,whole_paths=10240,n_per_method=256,
        method_counts=counts,profile_counts=profiles,maximum_absolute_arithmetic_difference=check.maximum_absolute_error,
        native_simulator_calls=0,actor_inference=0,model_fitting=0,
        scope="separate95% auxiliary component family conditional on exported summaries and instrumented law; not jointprimary/auxiliary95%, no independent native replay, no no-prepaid counterfactual or new-acquisition claim")

def verify_artifacts(directory, file_sha256):
    """Check every mandatory persisted byte before sealing an export, or later."""
    directory = Path(directory)
    if not isinstance(file_sha256, dict) or set(file_sha256) != MANDATORY_FILES:
        raise ValueError("Exact mandatory public artifact hash set required")
    for name, digest in file_sha256.items():
        p = Path(name)
        if p.is_absolute() or ".." in p.parts or ":" in name or len(p.parts) != 1:
            raise ValueError("Manifest file must be a simple local filename")
        if sha(directory/name) != digest:
            raise AssertionError("Public artifact SHA256 mismatch: "+name)
    with gzip.open(directory/"paths.jsonl.gz", "rt", encoding="utf-8") as stream:
        path_rows = [json.loads(line) for line in stream if line.strip()]
    report = verify_payload(path_rows, read(directory/"cases_and_bounds.json"),
                            read(directory/"selection.json"), read(directory/"results.json"))
    return report


def verify_directory(directory):
    directory = Path(directory)
    manifest = read(directory/"MANIFEST.json")
    if manifest.get("schema") != SCHEMA or manifest.get("complete") is not True:
        raise ValueError("Completed public export manifest required")
    report = verify_artifacts(directory, manifest["file_sha256"])
    report["manifest_sha256"] = sha(directory/"MANIFEST.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path("."))
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    report = verify_directory(args.directory)
    if args.write_report:
        with (args.directory/"REPRODUCTION.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
