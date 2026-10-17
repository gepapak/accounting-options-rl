"""Standalone stdlib arithmetic replay of a completed 340-limit v2 export.

No native module, actor, model, raw market panel or numerical package is used.
This verifies saved-summary arithmetic, not simulator authenticity or supports.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path

SCHEMA = "sample_only_joint_fault_public_v2"
BASE_METHODS = ("raw", "capped_raw", "capped_residual", "paired", "ridge_control", "mlp_control")
PHASE_N = {"fit": 32, "tune": 32, "confirm": 256}
DELTA, HORIZON = .05 / 340, 96
SEEDS = (7, 42, 123, 2025, 3007, 5001, 8102, 9005, 10001, 11202)
MANDATORY_FILES = frozenset(("cases_and_bounds.json", "selection.json", "results.json", "provenance.json",
    "EXPORT_ARITHMETIC_CHECK.json", "paths.jsonl.gz", "reproduce_public_joint_fault.py", "README.md"))
CASE_FIELDS = ("case_id", "policy_index", "policy_family", "checkpoint_seed", "profile",
               "source_run001_case_id", "cash_fraction", "horizon", "burn_in")
BOUND_FIELDS = ("discounted_cost_bound", "discounted_gross_bound", "unavoidable_discounted_access_cost", "cash_inflow_bound")
PATH_FIELDS = ("case_id", "phase", "episode_id", "context_id", "gross", "cost", "forgiveness",
    "paired_forgiveness", "maturity_coins", "excluded_old_gross", "excluded_old_forgiveness",
    "excluded_old_paired_forgiveness", "excluded_old_maturity_events")
CONFIRM_FIELDS = ("delta_ridge", "delta_mlp", "delta_cash_ridge", "delta_cash_mlp", "delta_cash_local")


def methods_for_alpha(alpha):
    if alpha not in (0., 1.):
        raise ValueError("Only the two frozen charge views are defined")
    return BASE_METHODS + (("cash_capped_raw", "cash_ridge_control", "cash_mlp_control", "cash_local_control") if alpha == 0. else ())


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
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
        self.yes(not isinstance(actual, bool) and not isinstance(expected, bool)
                 and math.isfinite(float(actual)) and math.isfinite(float(expected)), description + " finite")
        error = abs(float(actual) - float(expected))
        self.maximum_absolute_error = max(self.maximum_absolute_error, error)
        self.yes(error <= 1e-6 + 2e-12 * max(abs(float(actual)), abs(float(expected))), description)


def sequential_control(deltas, coins, beta, budget):
    remaining, terms = float(budget), []
    if remaining < 0 or not math.isfinite(remaining) or beta < 0 or not math.isfinite(beta):
        raise ValueError("Invalid frozen control settings")
    if len(deltas) != len(coins):
        raise ValueError("Mismatched ordered innovation vectors")
    for delta, coin in zip(deltas, coins):
        if not math.isfinite(delta) or coin not in (-1, 1):
            raise ValueError("Invalid stored innovation")
        amplitude = min(abs(beta * delta), remaining)
        terms.append(coin * (1. if delta > 0 else -1. if delta < 0 else 0.) * amplitude)
        remaining = max(remaining - amplitude, 0.)
    return math.fsum(terms)


def validate_choice(method, alpha, choice):
    if method not in methods_for_alpha(alpha) or set(choice) != {"cap", "beta", "budget"}:
        raise ValueError("Declared method and exact frozen choice fields required")
    for key in ("beta", "budget"):
        value = choice[key]
        if isinstance(value, bool) or not math.isfinite(float(value)) or value < 0:
            raise ValueError("Finite nonnegative control setting required")
    if method == "raw":
        if choice != dict(cap=None, beta=0., budget=0.):
            raise ValueError("RAW must remain unselected and uncapped")
    elif (choice["cap"] is None or isinstance(choice["cap"], bool)
          or not math.isfinite(float(choice["cap"])) or choice["cap"] < 0):
        raise ValueError("Fixed finite nonnegative upper cap required")
    if not method.endswith("_control") and (choice["beta"] != 0 or choice["budget"] != 0):
        raise ValueError("Simple representations cannot silently apply a learned control")


def path_value(row, method, alpha, choice):
    validate_choice(method, alpha, choice)
    if method in ("raw", "capped_raw", "cash_capped_raw", "cash_ridge_control", "cash_mlp_control", "cash_local_control"):
        value = row["gross"] - row["cost"] + (1 - alpha) * row["forgiveness"]
    elif method == "capped_residual":
        value = (1 - alpha) * row["forgiveness"] - row["cost"]
    else:
        value = (1 - alpha) * row["paired_forgiveness"] - row["cost"]
    if method.endswith("_control"):
        model = method.removesuffix("_control")
        value -= sequential_control(row["delta_" + model], row["maturity_coins"],
                                    float(choice["beta"]), float(choice["budget"]))
    return value if method == "raw" else min(value, float(choice["cap"]))


def numerical_pad(bounds, choice=None):
    # Deliberately retains the original G+C numerical access even for cash RAW.
    return max(1e-6, 1e-10 * (1 + float(bounds["discounted_gross_bound"]) +
        float(bounds["discounted_cost_bound"]) + float((choice or {}).get("budget", 0.))) * HORIZON)


def deterministic_support(bounds, method, alpha, choice):
    validate_choice(method, alpha, choice)
    c, g = float(bounds["discounted_cost_bound"]), float(bounds["discounted_gross_bound"])
    if method == "raw":
        return -g - c, g
    if method == "capped_raw":
        return -g - c, min(float(choice["cap"]), g)
    if method.startswith("cash_"):
        budget = float(choice["budget"]) if method.endswith("_control") else 0.
        return -float(bounds["cash_inflow_bound"]) - budget, min(float(choice["cap"]), g + budget)
    budget = float(choice["budget"]) if method.endswith("_control") else 0.
    upper = (1 - alpha) * g * (1. if method == "capped_residual" else .5) + budget
    return -c - budget, min(float(choice["cap"]), upper)


def empirical_bernstein(sample, lower, upper, pad):
    n = len(sample)
    if n < 2 or not all(math.isfinite(x) for x in sample):
        raise ValueError("Finite independent whole-episode sample required")
    if not all(math.isfinite(x) for x in (lower, upper, pad)) or lower > upper or pad < 0:
        raise ValueError("Finite ordered support and nonnegative numerical allowance")
    mean = math.fsum(sample) / n
    variance = math.fsum((x - mean) ** 2 for x in sample) / (n - 1)
    lo, hi = lower - pad, upper + pad
    if min(sample) < lo - 1e-6 or max(sample) > hi + 1e-6:
        raise AssertionError("Exported path falls outside its deterministic support")
    log = math.log(2 / DELTA)
    radius = math.sqrt(2 * variance * log / n) + 7 * (hi - lo) * log / (3 * (n - 1))
    return dict(n=n, mean=mean, variance=variance, radius=radius, lower=mean - radius - pad,
                support_lower=lo, support_upper=hi, delta=DELTA, numerical_allowance=pad)


def verify_payload(path_rows, case_records, selections, saved_results):
    check = Checks()
    case_map = {row["case_id"]: row for row in case_records}
    check.yes(len(case_records) == len(case_map) == 20 and set(case_map) == set(range(20)), "all twenty v2 cases")
    grouped, total = defaultdict(list), 0
    for row in path_rows:
        check.yes(row["case_id"] in case_map and row["phase"] in PHASE_N, "case and phase ID")
        allowed = set(PATH_FIELDS) | (set(CONFIRM_FIELDS) if row["phase"] == "confirm" else set())
        check.yes(set(row) == allowed, "exact public path whitelist")
        check.yes(isinstance(row["context_id"], int) and not isinstance(row["context_id"], bool)
                  and 0 <= row["context_id"] < 8, "fixed context identifier")
        check.yes(all(not isinstance(row[k], bool) and math.isfinite(float(row[k])) for k in
                  ("gross", "cost", "forgiveness", "paired_forgiveness", "excluded_old_gross",
                   "excluded_old_forgiveness", "excluded_old_paired_forgiveness")), "finite component summaries")
        check.yes(all(row[k] >= 0 for k in ("cost", "forgiveness", "paired_forgiveness",
                  "excluded_old_forgiveness", "excluded_old_paired_forgiveness")), "nonnegative fees/forgiveness")
        check.yes(isinstance(row["excluded_old_maturity_events"], int)
                  and not isinstance(row["excluded_old_maturity_events"], bool)
                  and 0 <= row["excluded_old_maturity_events"] <= 6, "old-maturity count within six-step mask")
        check.yes(len(row["maturity_coins"]) <= HORIZON
                  and all(not isinstance(c, bool) and c in (-1, 1) for c in row["maturity_coins"]), "all ordered old/new fair innovations retained")
        check.yes(row["excluded_old_maturity_events"] <= len(row["maturity_coins"]), "excluded events retain their coins")
        if row["phase"] == "confirm":
            check.yes(all(len(row[k]) == len(row["maturity_coins"])
                      and all(math.isfinite(float(x)) for x in row[k]) for k in CONFIRM_FIELDS),
                      "complete finite frozen predictor differences")
        grouped[row["case_id"], row["phase"]].append(row)
        total += 1
    check.yes(total == 6400, "all 6,400 whole paths")
    for cid in case_map:
        for phase, n in PHASE_N.items():
            group = grouped[cid, phase]
            check.yes(len(group) == n and sorted(row["episode_id"] for row in group) == list(range(n)), "complete unique fixed phase budget")
            group.sort(key=lambda row: row["episode_id"])
    expected_keys = {(cid, alpha, method) for cid in range(20) for alpha in (0., 1.) for method in methods_for_alpha(alpha)}
    choices = {(r["case_id"], float(r["alpha"]), r["method"]): r["choice"] for r in selections}
    charged = {(r["case_id"], float(r["alpha"]), r["method"]): r for r in saved_results["charged_results"]}
    costs = {r["case_id"]: r for r in saved_results["cost_results"]}
    check.yes(len(selections) == len(choices) == 320 and set(choices) == expected_keys, "all 320 frozen choices; cash alpha0 only")
    check.yes(len(charged) == len(saved_results["charged_results"]) == 320 and set(charged) == expected_keys, "all 320 charged limits")
    check.yes(len(costs) == len(saved_results["cost_results"]) == 20 and set(costs) == set(case_map), "all twenty cost limits")
    check.yes(saved_results["schema"] == SCHEMA and saved_results["multiplicity"] ==
              dict(family_limits=340, total_delta=.05, delta_per_limit=DELTA), "separate 340-limit family")
    counts = {str(a): {m: 0 for m in methods_for_alpha(a)} for a in (0., 1.)}
    profile_counts = {"joint_fault": {a: dict(v) for a, v in counts.items()}}
    positive_costs, strict_booked = 0, 0
    for cid, case in case_map.items():
        check.yes(set(case) == set(CASE_FIELDS) | {"bounds"}, "exact case whitelist")
        check.yes(case["policy_index"] == cid and case["source_run001_case_id"] == 2 * cid + 1
                  and case["profile"] == "joint_fault" and case["checkpoint_seed"] == SEEDS[cid % 10]
                  and case["policy_family"] == ("plain_mappo" if cid < 10 else "feasible_mappo")
                  and case["horizon"] == 96 and case["burn_in"] == 384 and case["cash_fraction"] == 1e-4,
                  "fixed full policy lineage and intervention")
        bounds = case["bounds"]
        check.yes(set(bounds) == set(BOUND_FIELDS) and all(not isinstance(bounds[k], bool)
                  and math.isfinite(float(bounds[k])) and bounds[k] >= 0 for k in BOUND_FIELDS), "predetermined scalar supports incl K")
        check.yes(bounds["unavoidable_discounted_access_cost"] <= bounds["discounted_cost_bound"], "mandatory fee bounded by cost envelope")
        allrows = grouped[cid, "fit"] + grouped[cid, "tune"] + grouped[cid, "confirm"]
        for alpha in (0., 1.):
            for method in methods_for_alpha(alpha):
                key = cid, alpha, method
                choice, saved = choices[key], charged[key]
                check.yes(saved["choice"] == choice and saved["profile"] == "joint_fault", "frozen choice/profile binding")
                sample = [path_value(r, method, alpha, choice) for r in (allrows if method == "raw" else grouped[cid, "confirm"])]
                lo, hi = deterministic_support(bounds, method, alpha, choice)
                expected = empirical_bernstein(sample, lo, hi, numerical_pad(bounds, choice))
                check.yes(set(saved["certificate"]) == set(expected), "exact charged confidence fields")
                for field in expected:
                    check.close(saved["certificate"][field], expected[field], "charged EB " + field)
                check.yes(isinstance(saved["positive"], bool) and saved["positive"] == (expected["lower"] > 0), "strict charged decision")
                confirmation = [path_value(r, method, alpha, choice) for r in grouped[cid, "confirm"]]
                mean = math.fsum(confirmation) / len(confirmation)
                variance = math.fsum((x - mean) ** 2 for x in confirmation) / (len(confirmation) - 1)
                check.close(saved["confirmation_mean"], mean, "confirmation mean")
                check.close(saved["confirmation_variance"], variance, "confirmation variance")
                counts[str(alpha)][method] += int(expected["lower"] > 0)
                profile_counts["joint_fault"][str(alpha)][method] += int(expected["lower"] > 0)
        expected = empirical_bernstein([r["cost"] for r in allrows], 0., bounds["discounted_cost_bound"], numerical_pad(bounds))
        check.yes(set(costs[cid]["certificate"]) == set(expected), "exact cost confidence fields")
        for field in expected:
            check.close(costs[cid]["certificate"][field], expected[field], "cost EB " + field)
        check.close(costs[cid]["analytic_unavoidable_cost_lower"], bounds["unavoidable_discounted_access_cost"], "mandatory access fee")
        upper = -max(expected["lower"], bounds["unavoidable_discounted_access_cost"])
        check.close(costs[cid]["booked_expectation_upper"], upper, "booked expectation upper bound")
        check.yes(isinstance(costs[cid]["positive"], bool) and costs[cid]["positive"] == (expected["lower"] > 0), "cost decision")
        positive_costs += int(expected["lower"] > 0)
        strict_booked += int(upper < 0)
    check.yes(saved_results["method_counts"] == counts, "all method counts")
    check.yes(saved_results["profile_counts"] == profile_counts, "all profile counts")
    check.yes(saved_results["positive_cost_limits"] == positive_costs, "all positive cost limits")
    return dict(status="PASS", checks=check.count, bounds_recomputed=340, cases=20, policies=20,
        whole_paths=total, method_counts=counts, profile_counts=profile_counts,
        positive_cost_limits=positive_costs, strict_booked_harm_cases=strict_booked,
        maximum_absolute_arithmetic_difference=check.maximum_absolute_error,
        native_simulator_calls=0, actor_inference=0, model_fitting=0,
        scope="arithmetic conditional on exported new-component summaries, all retained innovation controls, source-derived supports and declared hidden-sign law; no independent native replay or whole-fund/acquisition guarantee")


def verify_artifacts(directory, file_sha256):
    directory = Path(directory)
    if not isinstance(file_sha256, dict) or set(file_sha256) != MANDATORY_FILES:
        raise ValueError("Exact mandatory public artifact hash set required")
    for name, digest in file_sha256.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or ":" in name or len(path.parts) != 1:
            raise ValueError("Simple local manifest filename required")
        if sha(directory / name) != digest:
            raise AssertionError("Public artifact SHA256 mismatch: " + name)
    with gzip.open(directory / "paths.jsonl.gz", "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    return verify_payload(rows, read(directory / "cases_and_bounds.json"),
                          read(directory / "selection.json"), read(directory / "results.json"))


def verify_directory(directory):
    directory = Path(directory)
    manifest = read(directory / "MANIFEST.json")
    if manifest.get("schema") != SCHEMA or manifest.get("complete") is not True:
        raise ValueError("Complete v2 public export manifest required")
    if (manifest.get("family_limits") != 340 or manifest.get("path_summary_rows") != 6400
            or manifest.get("total_delta") != .05 or manifest.get("delta_per_limit") != DELTA):
        raise ValueError("Exact v2 family/path metadata required")
    report = verify_artifacts(directory, manifest["file_sha256"])
    provenance = read(directory / "provenance.json")
    if (manifest.get("source_freeze_sha256") != provenance["source_freeze_sha256"]
            or manifest.get("reproducer_sha256") != provenance["reproducer_sha256"]
            or manifest["reproducer_sha256"] != sha(directory / "reproduce_public_joint_fault.py")):
        raise AssertionError("Public manifest/provenance source binding")
    report["manifest_sha256"] = sha(directory / "MANIFEST.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path("."))
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    report = verify_directory(args.directory)
    if args.write_report:
        with (args.directory / "REPRODUCTION.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
