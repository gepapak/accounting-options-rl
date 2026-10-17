"""Held descriptive control ablation on complete independently verified records.

No additional confidence limit, fit, selection, native query or statistical draw.
The original eighteen-file report and every frozen artifact remain unchanged.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
TARGETS = ("primary_alpha0", "primary_alpha1", "auxiliary_alpha0")
MODELS = ("ridge_control", "mlp_control")


def require(value, message):
    if not value:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def control(deltas, coins, beta, budget):
    require(len(deltas) == len(coins), "Ordered innovation vectors agree")
    require(math.isfinite(beta) and beta >= 0 and math.isfinite(budget) and budget >= 0, "Frozen nonnegative settings")
    remaining, terms = budget, []
    for delta, coin in zip(deltas, coins):
        require(math.isfinite(delta) and coin in (-1, 1), "Finite predictable amplitudes and sign innovations")
        amplitude = min(abs(beta * delta), remaining)
        terms.append(coin * (1 if delta > 0 else -1 if delta < 0 else 0) * amplitude)
        remaining = max(0., remaining-amplitude)
    return math.fsum(terms)


def moments(values):
    require(len(values) == 256 and all(math.isfinite(x) for x in values), "Complete finite common-CONFIRM sample")
    mean = math.fsum(values)/len(values)
    return dict(mean=mean, variance=math.fsum((x-mean)**2 for x in values)/(len(values)-1))


def ratio(numerator, denominator):
    require(numerator >= 0 and denominator >= 0, "Nonnegative sample variances")
    if denominator > 0:
        return numerator/denominator, "finite"
    return None, "both_zero" if numerator == 0 else "zero_reference_positive_control"


def public_replay(directory, filename, module_name, count):
    manifest = read(directory/"MANIFEST.json")
    require(manifest.get("complete") is True and manifest.get("family_limits") == count, "Complete public family")
    require(sha(directory/filename) == manifest["file_sha256"][filename], "Archived reproducer identity")
    spec = importlib.util.spec_from_file_location(module_name, directory/filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.verify_directory(directory)
    require(report["status"] == "PASS" and report["bounds_recomputed"] == count, "Fresh independent public arithmetic PASS")
    recorded = read(directory/"REPRODUCTION.json")
    require(recorded.get("status") == "PASS" and recorded.get("manifest_sha256") == sha(directory/"MANIFEST.json"), "Recorded public replay identity")
    return report


def load_inputs(run_id):
    primary, auxiliary = HERE/"results"/run_id, HERE/"results"/(run_id+"_new_contract")
    primary_check = HERE/"validation"/("NATIVE_VERIFICATION_"+run_id+"_critics.json")
    aux_check = HERE/"validation"/("NEW_CONTRACT_VERIFICATION_"+run_id+".json")
    table_marker = HERE/"report_tables"/run_id/"COMPLETION.json"
    required = [primary_check, aux_check, table_marker,
        HERE/("native_completion_"+run_id+".json"), HERE/"execution"/run_id/"confirm_stage_complete.json",
        HERE/"auxiliary_new_contract"/run_id/"COMPLETION.json"]
    required += [directory/name for directory in (primary, auxiliary) for name in ("MANIFEST.json", "REPRODUCTION.json")]
    require(all(p.is_file() for p in required), "NOT_READY: full private/public/report completion gates precede endpoint reads")
    pcheck, acheck = read(primary_check), read(aux_check)
    require(pcheck.get("status") == acheck.get("status") == "PASS" and pcheck.get("bounds_checked") == 520 and
        acheck.get("bounds_checked") == 240 and pcheck.get("nuisance_regressor_prediction_rechecked") is True, "Full private PASS520/240")
    require(pcheck["verifier_sha256"] == sha(HERE/"validation/verify_native_results.py") and
        acheck["verifier_sha256"] == sha(HERE/"validation/verify_new_contract_component.py"), "Executed private-verifier identities")
    tables = read(table_marker)
    require(tables.get("complete") is True and len(tables["file_sha256"]) == 18, "Unchanged completed eighteen-file report")
    require(tables["inputs"]["primary_private_verification_sha256"] == sha(primary_check) and
        tables["inputs"]["auxiliary_private_verification_sha256"] == sha(aux_check), "Completed report/private-verifier binding")
    for name, digest in tables["file_sha256"].items():
        require(Path(name).name == name and sha(table_marker.parent/name) == digest, "Original report artifact identity")
    replays = [public_replay(primary, "reproduce_public_native_audit.py", "control_diag_primary", 520),
        public_replay(auxiliary, "reproduce_public_new_contract.py", "control_diag_auxiliary", 240)]
    require(read(primary/"provenance.json")["private_result_sha256"] == pcheck["result_sha256"] and
        read(auxiliary/"provenance.json")["private_auxiliary_result_sha256"] == acheck["result_sha256"], "Public/private result binding")
    def paths(directory):
        with gzip.open(directory/"paths.jsonl.gz", "rt", encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]
    data = {"primary": (read(primary/"results.json"), paths(primary)),
            "auxiliary": (read(auxiliary/"results.json"), paths(auxiliary))}
    provenance = {"source_sha256": sha(__file__), "private_primary_verifier_sha256": sha(primary_check),
        "private_auxiliary_verifier_sha256": sha(aux_check), "original_report_completion_sha256": sha(table_marker),
        "public_primary_manifest_sha256": sha(primary/"MANIFEST.json"), "public_auxiliary_manifest_sha256": sha(auxiliary/"MANIFEST.json"),
        "public_primary_paths_sha256": sha(primary/"paths.jsonl.gz"), "public_auxiliary_paths_sha256": sha(auxiliary/"paths.jsonl.gz"),
        "fresh_public_replay_bounds": [r["bounds_recomputed"] for r in replays]}
    return data, provenance, table_marker


def calculate(data):
    records = []
    for target in TARGETS:
        alpha = 1. if target == "primary_alpha1" else 0.
        results, paths = data["auxiliary" if target.startswith("auxiliary") else "primary"]
        groups = defaultdict(list)
        for row in paths:
            if row["phase"] == "confirm":
                groups[row["case_id"]].append(row)
        selected = {(r["case_id"], r["method"]): r for r in results["charged_results"] if float(r["alpha"]) == alpha}
        for cid in range(40):
            group = sorted(groups[cid], key=lambda r: r["episode_id"])
            require([r["episode_id"] for r in group] == list(range(256)), "All whole trajectories retained")
            pair = selected[cid, "paired"]
            base = [(1-alpha)*r["paired_forgiveness"]-r["cost"] for r in group]
            base_stats = moments(base)
            for method in MODELS:
                chosen = selected[cid, method]
                choice, pair_choice = chosen["choice"], pair["choice"]
                beta, budget, cap = float(choice["beta"]), float(choice["budget"]), float(choice["cap"])
                model = method.removesuffix("_control")
                controls = [control(r["delta_"+model], r["maturity_coins"], beta, budget) for r in group]
                adjusted = [z-m for z, m in zip(base, controls)]
                adjusted_stats = moments(adjusted)
                capped_zero = [min(z, cap) for z in base]
                capped_control = [min(z, cap) for z in adjusted]
                zero_stats, controlled_stats = moments(capped_zero), moments(capped_control)
                uncapped_ratio, uncapped_status = ratio(adjusted_stats["variance"], base_stats["variance"])
                capped_ratio, capped_status = ratio(controlled_stats["variance"], zero_stats["variance"])
                records.append(dict(target=target, alpha=alpha, profile=chosen["profile"], case_id=cid, method=method, n=256,
                    learned_certified=chosen["positive"], pairing_certified=pair["positive"],
                    learned_only=chosen["positive"] and not pair["positive"],
                    beta=beta, budget=budget, both_settings_nonzero=beta>0 and budget>0,
                    paths_with_nonzero_control=sum(m != 0 for m in controls), learned_cap=cap, pairing_cap=pair_choice["cap"],
                    exact_cap_equality=cap == float(pair_choice["cap"]),
                    uncapped_pair_mean=base_stats["mean"], uncapped_pair_variance=base_stats["variance"],
                    uncapped_control_mean=adjusted_stats["mean"], uncapped_control_variance=adjusted_stats["variance"],
                    uncapped_variance_ratio=uncapped_ratio, uncapped_ratio_status=uncapped_status,
                    same_cap_zero_mean=zero_stats["mean"], same_cap_zero_variance=zero_stats["variance"],
                    same_cap_control_mean=controlled_stats["mean"], same_cap_control_variance=controlled_stats["variance"],
                    same_cap_variance_ratio=capped_ratio, same_cap_ratio_status=capped_status,
                    same_cap_zero_clipped_fraction=sum(z > cap for z in base)/256,
                    same_cap_control_clipped_fraction=sum(z > cap for z in adjusted)/256))
    require(len(records) == 240, "All160 primary and80 auxiliary learned-choice diagnostics")
    summaries = []
    for target in TARGETS:
        for profile in ("strict", "vulnerable"):
            for method in MODELS:
                group = [r for r in records if (r["target"], r["profile"], r["method"]) == (target, profile, method)]
                require(len(group) == 20, "All twenty policies per diagnostic group")
                only = [r for r in group if r["learned_only"]]
                summary = dict(target=target, profile=profile, method=method, cases=20, learned_only=len(only),
                    learned_only_nonzero_settings=sum(r["both_settings_nonzero"] for r in only),
                    learned_only_exact_cap_equal=sum(r["exact_cap_equality"] for r in only),
                    learned_only_zero_settings_equal_cap=sum(not r["both_settings_nonzero"] and r["exact_cap_equality"] for r in only),
                    learned_only_nonzero_settings_equal_cap=sum(r["both_settings_nonzero"] and r["exact_cap_equality"] for r in only))
                for kind in ("uncapped", "same_cap"):
                    ratios = [r[kind+"_variance_ratio"] for r in group if r[kind+"_ratio_status"] == "finite"]
                    summary[kind+"_finite_ratios"] = len(ratios)
                    summary[kind+"_ratio_median"] = statistics.median(ratios) if ratios else None
                    summary[kind+"_both_zero"] = sum(r[kind+"_ratio_status"] == "both_zero" for r in group)
                    summary[kind+"_zero_reference_positive_control"] = sum(r[kind+"_ratio_status"] == "zero_reference_positive_control" for r in group)
                summaries.append(summary)
    return records, summaries


def execute(run_id):
    require(run_id.startswith("run_") and run_id.replace("_", "").isalnum(), "Simple run identifier")
    output = HERE/"control_diagnostics"/run_id
    require(not output.exists(), "Exclusive new descriptive output; preserve prior attempts")
    data, provenance, original_marker = load_inputs(run_id)
    records, summaries = calculate(data)
    output.mkdir(parents=True, exist_ok=False)
    result = dict(status="COMPLETE DESCRIPTIVE CONTROL DIAGNOSTIC", created_utc=datetime.now(timezone.utc).isoformat(),
        provenance=provenance, records=records, summaries=summaries, new_native_queries=0, new_confidence_limits=0,
        interpretation="Common256CONFIRM sample statistics only; no formal superiority test. Uncapped comparison isolates a frozen control on the same paths; same-cap comparison holds upper cap fixed. Primary learned caps were chosen on32TUNE versus PAIR64development paths; exclusive certificates do not establish learned-model benefit. Auxiliary reuses primaryalpha0 choices; all inherited state effects and original coin controls remain.")
    with (output/"RESULTS.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
    with (output/"case_diagnostics.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    with (output/"README.md").open("x", encoding="utf-8") as stream:
        stream.write("# Descriptive frozen-control ablation\n\nAll240 target/case/model combinations use the same256 confirmation trajectories. No new confidence event, retuning, native query or fit. Original18-file report remains intact. Variance ratios report controlled/reference;0/0 andpositive/0 remain undefined, zero/positive is0. Capping frequencies mean strict upper clipping. Uncapped means are descriptive and no finite-sample unbiased-variance superiority claim is made. See RESULTS.json for source/input hashes and complete cap/nonzero-setting cross-tabs.\n")
    require(sha(original_marker) == provenance["original_report_completion_sha256"], "Original report completion remains unchanged")
    hashes = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
    with (output/"COMPLETION.json").open("x", encoding="utf-8") as stream:
        json.dump(dict(complete=True, source_sha256=sha(__file__), file_sha256=hashes, input_sha256=provenance), stream, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps(dict(status=result["status"], records=len(records), summaries=summaries, output=str(output)), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="run_001")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(status="PREPARED_WAITING_COMPLETE_GATES", source_sha256=sha(__file__), endpoint_reads=0, new_queries=0)))
        return 0
    execute(args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
