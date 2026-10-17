"""Recompute all 520 published native-audit limits using Python's standard library.

No native engine, market data, actors, sklearn, fitted models or simulator calls
are required. The exported path summaries are the independent sampling units.
This checks confidence arithmetic, not the authenticity of a hidden simulator.
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

SCHEMA = "sample_only_native_public_v1"
METHODS = ("raw", "capped_raw", "capped_residual", "paired", "ridge_control", "mlp_control")
PHASE_N = {"fit": 32, "tune": 32, "confirm": 256}
DELTA, HORIZON = .05/520, 96
MANDATORY_FILES = frozenset(("cases_and_bounds.json", "selection.json", "results.json", "provenance.json",
    "EXPORT_ARITHMETIC_CHECK.json", "paths.jsonl.gz", "reproduce_public_native_audit.py", "README.md"))


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


def verify_payload(path_rows, case_records, selections, saved_results):
    check = Checks()
    case_map = {row["case_id"]: row for row in case_records}
    check.yes(set(case_map) == set(range(40)) and len(case_records) == 40, "all forty original cases retained")
    check.yes({row["policy_index"] for row in case_records} == set(range(20)), "all twenty original policies retained")
    grouped = defaultdict(list)
    total = 0
    allowed_base = {"case_id", "phase", "episode_id", "context_id", "gross", "cost", "forgiveness",
                    "paired_forgiveness", "maturity_coins"}
    for row in path_rows:
        check.yes(row["case_id"] in case_map and row["phase"] in PHASE_N, "original case and phase ID")
        allowed = allowed_base | ({"delta_ridge", "delta_mlp"} if row["phase"] == "confirm" else set())
        check.yes(set(row) == allowed, "public path whitelist excludes private features and paths")
        check.yes(isinstance(row["context_id"], int) and 0 <= row["context_id"] < 8, "declared context identifier")
        check.yes(all(math.isfinite(float(row[key])) for key in
                      ("gross", "cost", "forgiveness", "paired_forgiveness")), "finite financial summaries")
        check.yes(row["cost"] >= 0 and row["forgiveness"] >= 0 and row["paired_forgiveness"] >= 0,
                  "nonnegative economic fees and injections")
        check.yes(all(coin in (-1, 1) for coin in row["maturity_coins"]), "ordered fair maturity innovations")
        if row["phase"] == "confirm":
            check.yes(len(row["delta_ridge"]) == len(row["delta_mlp"]) == len(row["maturity_coins"]),
                      "complete stored centered predictor differences")
        grouped[row["case_id"], row["phase"]].append(row)
        total += 1
    check.yes(total == 12800, "complete fixed 12,800-path export")
    for cid in case_map:
        for phase, n in PHASE_N.items():
            group = grouped[cid, phase]
            check.yes(len(group) == n and sorted(row["episode_id"] for row in group) == list(range(n)),
                      "complete unique whole-episode phase budget")
            group.sort(key=lambda row: row["episode_id"])
    expected_keys = {(cid, alpha, method) for cid in range(40) for alpha in (0., 1.) for method in METHODS}
    choices = {(row["case_id"], float(row["alpha"]), row["method"]): row["choice"] for row in selections}
    charged = {(row["case_id"], float(row["alpha"]), row["method"]): row for row in saved_results["charged_results"]}
    costs = {row["case_id"]: row for row in saved_results["cost_results"]}
    check.yes(len(selections) == len(choices) == 480 and set(choices) == expected_keys, "all 480 frozen choices")
    check.yes(len(charged) == len(saved_results["charged_results"]) == 480 and set(charged) == expected_keys,
              "all 480 charged confidence limits")
    check.yes(len(costs) == len(saved_results["cost_results"]) == 40 and set(costs) == set(case_map), "all forty cost limits")
    check.yes(saved_results["multiplicity"] == dict(family_limits=520, total_delta=.05, delta_per_limit=DELTA),
              "separate declared 520-event family")
    counts = {str(alpha): {method: 0 for method in METHODS} for alpha in (0., 1.)}
    profile_counts = {profile: {str(alpha): {method: 0 for method in METHODS} for alpha in (0., 1.)}
                      for profile in ("strict", "vulnerable")}
    certified_costs, strict_booked = 0, 0
    for cid, case in case_map.items():
        check.yes(case["case_id"] == 2*case["policy_index"]+case["profile_index"], "original paired case structure")
        bounds = case["bounds"]
        check.yes(all(math.isfinite(float(bounds[key])) and float(bounds[key]) >= 0 for key in
                      ("discounted_cost_bound", "discounted_gross_bound", "unavoidable_discounted_access_cost")),
                  "known deterministic scalar support")
        allrows = grouped[cid, "fit"]+grouped[cid, "tune"]+grouped[cid, "confirm"]
        for alpha in (0., 1.):
            for method in METHODS:
                key = cid, alpha, method
                choice, saved = choices[key], charged[key]
                check.yes(saved["choice"] == choice, "pre-confirmation selected choice")
                sample = [path_value(row, method, alpha, choice) for row in
                          (allrows if method == "raw" else grouped[cid, "confirm"])]
                lo, hi = deterministic_support(bounds, method, alpha, choice)
                expected = empirical_bernstein(sample, lo, hi, numerical_pad(bounds, choice))
                for field in expected:
                    check.close(saved["certificate"][field], expected[field], "charged EB "+field)
                check.yes(saved["positive"] == (expected["lower"] > 0), "charged strictly positive lower limit")
                confirmation = [path_value(row, method, alpha, choice) for row in grouped[cid, "confirm"]]
                mean = math.fsum(confirmation)/len(confirmation)
                variance = math.fsum((x-mean)**2 for x in confirmation)/(len(confirmation)-1)
                check.close(saved["confirmation_mean"], mean, "descriptive confirmation mean")
                check.close(saved["confirmation_variance"], variance, "descriptive confirmation variance")
                counts[str(alpha)][method] += int(expected["lower"] > 0)
                profile_counts[case["profile"]][str(alpha)][method] += int(expected["lower"] > 0)
        sample = [row["cost"] for row in allrows]
        expected = empirical_bernstein(sample, 0., bounds["discounted_cost_bound"], numerical_pad(bounds))
        for field in expected:
            check.close(costs[cid]["certificate"][field], expected[field], "cost EB "+field)
        check.close(costs[cid]["analytic_unavoidable_cost_lower"], bounds["unavoidable_discounted_access_cost"],
                    "analytical unavoidable access fee")
        booked_upper = -max(expected["lower"], bounds["unavoidable_discounted_access_cost"])
        check.close(costs[cid]["booked_expectation_upper"], booked_upper, "no-edge booked harm bound")
        check.yes(costs[cid]["positive"] == (expected["lower"] > 0), "cost lower-limit decision")
        certified_costs += int(expected["lower"] > 0)
        strict_booked += int(booked_upper < 0)
    check.yes(saved_results["method_counts"] == counts, "all aggregate detection counts")
    check.yes(saved_results["profile_counts"] == profile_counts, "all profile detection counts")
    check.yes(saved_results["positive_cost_limits"] == certified_costs, "cost confidence-limit total")
    return dict(status="PASS", checks=check.count, bounds_recomputed=520, cases=40, policies=20,
        whole_paths=total, method_counts=counts, profile_counts=profile_counts,
        positive_cost_limits=certified_costs, strict_booked_harm_cases=strict_booked,
        maximum_absolute_arithmetic_difference=check.maximum_absolute_error,
        native_simulator_calls=0, actor_inference=0, model_fitting=0,
        scope="arithmetic reproduction conditional on the exported path summaries, source-derived supports and instrumented sign law; no independent engine replay or future-market claim")


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
