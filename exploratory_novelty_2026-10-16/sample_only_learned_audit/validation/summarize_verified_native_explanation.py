"""Descriptive explanation after the completed independent/native public gates.

No confidence limit, native query, fitting, sampling or parameter changes.
"""
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    run_id = "run_001"
    required = [HERE/"validation/NATIVE_VERIFICATION_run_001_critics.json", HERE/"validation/NEW_CONTRACT_VERIFICATION_run_001.json",
        HERE/"report_tables/run_001/COMPLETION.json", HERE/"control_diagnostics/run_001/COMPLETION.json"]
    required += [HERE/"results"/name/f for name in ("run_001", "run_001_new_contract") for f in ("MANIFEST.json", "REPRODUCTION.json")]
    assert all(p.is_file() for p in required), "All completed private/public/report/diagnostic gates required"
    private, auxiliary = read(required[0]), read(required[1])
    assert private["status"] == auxiliary["status"] == "PASS" and private["bounds_checked"] == 520 and auxiliary["bounds_checked"] == 240
    report_marker = read(required[2]); diagnostic_marker = read(required[3])
    assert report_marker["complete"] is True and diagnostic_marker["complete"] is True
    for folder, marker in ((required[2].parent, report_marker), (required[3].parent, diagnostic_marker)):
        assert all(sha(folder/name) == value for name, value in marker["file_sha256"].items())
    for name, count in (("run_001", 520), ("run_001_new_contract", 240)):
        directory = HERE/"results"/name
        manifest, replay = read(directory/"MANIFEST.json"), read(directory/"REPRODUCTION.json")
        assert manifest["complete"] is True and manifest["family_limits"] == count and replay["status"] == "PASS"
        assert replay["bounds_recomputed"] == count and replay["manifest_sha256"] == sha(directory/"MANIFEST.json")
        assert all(sha(directory/f) == v for f, v in manifest["file_sha256"].items())
    cases = {r["case_id"]: r for r in read(HERE/"results/run_001/cases_and_bounds.json")}
    report = read(HERE/"report_tables/run_001/DESCRIPTIVE_RESULTS.json")
    grouped = []
    for family in ("plain_mappo", "feasible_mappo"):
        for profile in ("strict", "vulnerable"):
            rows = [r for r in report["native_activity"] if (r["policy_family"],r["profile"]) == (family,profile)]
            assert len(rows) == 10
            grouped.append(dict(policy_family=family,profile=profile,cases=10,confirm_episodes=2560,
                episodes_with_new_maturity=sum(r["episodes_with_new_maturity"] for r in rows),
                new_maturity_events=sum(round(r["mean_new_maturities"]*256) for r in rows),
                episodes_with_initial_prepaid_claims=sum(r["episodes_with_initial_prepaid_claims"] for r in rows),
                terminal_margin_active_episodes=sum(r["terminal_native_margin_active"] for r in rows),
                mean_nonzero_actor_request_steps=sum(r["mean_nonzero_actor_request_steps"] for r in rows)/10,
                max_terminal_open_contracts=max(r["max_terminal_open_contracts"] for r in rows)))
    bundle_stats, limit_ranges = [], []
    for name in ("run_001", "run_001_new_contract"):
        directory = HERE/"results"/name
        with gzip.open(directory/"paths.jsonl.gz", "rt", encoding="utf-8") as stream:
            paths = [json.loads(line) for line in stream if line.strip()]
        for phase in sorted(set(r["phase"] for r in paths)):
            rows = [r for r in paths if r["phase"] == phase]
            bundle_stats.append(dict(bundle=name,phase=phase,paths=len(rows),
                positive_actual_forgiveness_paths=sum(r["forgiveness"] > 0 for r in rows),
                positive_paired_forgiveness_paths=sum(r["paired_forgiveness"] > 0 for r in rows),
                max_actual_forgiveness=max(r["forgiveness"] for r in rows),max_paired_forgiveness=max(r["paired_forgiveness"] for r in rows)))
        result = read(directory/"results.json")
        for alpha in sorted(set(r["alpha"] for r in result["charged_results"])):
            for method in ("raw","capped_raw","capped_residual","paired","ridge_control","mlp_control"):
                rows = [r for r in result["charged_results"] if r["alpha"] == alpha and r["method"] == method]
                limit_ranges.append(dict(bundle=name,alpha=alpha,method=method,cases=40,
                    mean_min=min(r["certificate"]["mean"] for r in rows),mean_max=max(r["certificate"]["mean"] for r in rows),
                    lower_min=min(r["certificate"]["lower"] for r in rows),lower_max=max(r["certificate"]["lower"] for r in rows)))
    choices = report["choices"]
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),
        gate_sha256={p.relative_to(HERE).as_posix():sha(p) for p in required},cash_diagnostics=private["cash_before_diagnostics"],
        activity=grouped,forgiveness=bundle_stats,confidence_and_mean_ranges=limit_ranges,
        all_learned_settings_zero=all(r["nonzero_beta"] == r["nonzero_budget"] == 0 for r in choices),
        actual_work=report["compute"],new_native_queries=0,new_confidence_limits=0,
        interpretation="Observed zero actual and paired injections is descriptive, not a global safety proof. Frozen policy inference did not imply new financial execution: strict guards and feasible-policy native gates suppressed new contracts. The learned tuner selected zero control everywhere, so this run establishes no learned-control performance benefit. Separate520/240 families remain separate; no retuning or sample expansion.")
    output = HERE/"validation/verified_native_explanation_run_001"
    output.mkdir(exist_ok=False)
    with (output/"SUMMARY.json").open("x",encoding="utf-8") as stream: json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False)
    with (output/"README.md").open("x",encoding="utf-8") as stream: stream.write("# Verified native result explanation\n\nSUMMARY.json describes all saved phases, grouped confirmation activity, selected zero controls and exact stored limit/mean ranges after complete private/public checks. No additional inference or simulation was performed. Sampled absence of injections is not a safety proof. Original18-file report and separate control diagnostic remain unchanged.\n")
    print(json.dumps(dict(output=output.relative_to(HERE).as_posix(),activity=grouped,forgiveness=bundle_stats,all_learned_settings_zero=result["all_learned_settings_zero"]),indent=2))


if __name__ == "__main__":
    main()
