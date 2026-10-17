"""Separate public auxiliary export; never modifies the primary eight-file bundle.

Requires complete primary/private PASS520 and independent auxiliary PASS240.
No partial endpoint read/export, native query, actor inference or model fitting.
"""
from __future__ import annotations
from datetime import datetime, timezone
import argparse
import gzip
import importlib.util
import io
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import public_export as primary

SCHEMA = "sample_only_new_contract_public_v1"
DELTA = .05/240


def primary_ready(run_id):
    """Check complete/PASS identities before parsing even the primary result."""
    run = HERE/"execution"/run_id
    if not (run/"confirm_stage_complete.json").is_file() or not (HERE/("native_completion_"+run_id+".json")).is_file():
        raise RuntimeError("NOT_READY: primary completion and private PASS520 precede any endpoint read")
    paths = [HERE/"validation"/("NATIVE_VERIFICATION_"+run_id+suffix+".json") for suffix in ("_critics", "")]
    path = next((p for p in paths if p.is_file()), None)
    if path is None:
        raise RuntimeError("NOT_READY: primary private PASS520 precedes any endpoint read")
    verified = primary.read(path)
    if (verified.get("status") != "PASS" or verified.get("bounds_checked") != 520
            or verified.get("result_sha256") != primary.sha(HERE/("native_results_"+run_id+".json"))
            or verified.get("source_freeze_sha256") != primary.sha(run/"SOURCE_FREEZE.json")):
        raise AssertionError("Primary private verification binding failed")


def _reproducer():
    path = HERE/"reproduce_public_new_contract.py"
    spec = importlib.util.spec_from_file_location("standalone_auxiliary_reproduction", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def export(run_id):
    # These gates execute before reading ANY primary or auxiliary endpoint.
    primary_ready(run_id)
    run, _, _, primary_freeze, primary_selection, _, primary_verification = primary._private_gate(run_id)
    private = HERE/"auxiliary_new_contract"/run_id
    verification_path = HERE/"validation"/("NEW_CONTRACT_VERIFICATION_"+run_id+".json")
    if not (private/"COMPLETION.json").is_file() or not verification_path.is_file():
        raise RuntimeError("NOT_READY: completed independent auxiliary PASS240 required before export")
    verification = primary.read(verification_path)
    if (verification.get("status") != "PASS" or verification.get("bounds_checked") != 240
            or verification.get("result_sha256") != primary.sha(private/"results.json")
            or verification.get("paths_sha256") != primary.sha(private/"path_summaries.jsonl.gz")):
        raise AssertionError("Auxiliary independent verification binding failed")
    freeze_path = HERE/("NEW_CONTRACT_SOURCE_FREEZE_"+run_id+".json")
    frozen = primary.read(freeze_path)
    if verification["source_freeze_sha256"] != primary.sha(freeze_path):
        raise AssertionError("Pre-confirmation auxiliary source binding failed")
    for name, digest in frozen["source_sha256"].items():
        if primary.sha(HERE/name) != primary.sha(HERE/("new_contract_source_snapshot_"+run_id)/name) or primary.sha(HERE/name) != digest:
            raise AssertionError("Frozen auxiliary source changed")
    result = primary.read(private/"results.json")
    with gzip.open(private/"path_summaries.jsonl.gz", "rt", encoding="utf-8") as stream:
        private_paths = [json.loads(line) for line in stream if line.strip()]
    path_keys = ("case_id", "phase", "episode_id", "context_id", "gross", "cost", "forgiveness",
                 "paired_forgiveness", "maturity_coins", "delta_ridge", "delta_mlp")
    public_paths = [{key: row[key] for key in path_keys} for row in private_paths]
    cases, resources = [], []
    for case in primary.read(run/"cases.json"):
        cid = case["case_id"]
        metadata = primary.read(run/f"case_{cid:02d}"/"confirm"/"metadata.json")
        record = {key: case[key] for key in ("case_id", "policy_index", "policy_family", "checkpoint_seed",
                  "profile_index", "profile", "cash_fraction", "horizon", "burn_in")}
        record["bounds"] = {key: metadata["bounds"][key] for key in
                            ("discounted_cost_bound", "discounted_gross_bound", "unavoidable_discounted_access_cost")}
        cases.append(record)
        resources.append(dict(case_id=cid, contexts=[dict(context_id=row["context"]["context_id"],
            resource_sha256=primary.sanitized_hashes(row["metadata"]["resource_sha256"]),
            configuration_sha256=row["metadata"]["config_snapshot_sha256"], warm_state_sha256=row["metadata"]["warm_snapshot_sha256"])
            for row in metadata["cases_contexts"]]))
    selections = [row for row in primary.read(run/"selection.json") if row["alpha"] == 0.]
    charged_fields = ("case_id", "policy_index", "policy_family", "profile", "alpha", "method", "choice",
                      "certificate", "positive", "booked_expectation_upper")
    public_result = dict(schema=SCHEMA, status="COMPLETE SEPARATE EXPLORATORY AUXILIARY", n_cases=40,
        n_policies=20, n_confirm_per_method=256, monetary_units="DKK", alpha=0., new_native_queries=0, new_model_fits=0,
        multiplicity=dict(family_limits=240, total_delta=.05, delta_per_limit=DELTA, joint_primary_auxiliary_95_percent_claim=False),
        method_counts=result["method_counts"], profile_counts=result["profile_counts"],
        excluded_old_maturity_events=result["excluded_old_maturity_events"],
        charged_results=[{key: row[key] for key in charged_fields} for row in result["charged_results"]],
        auxiliary_frozen_utc=verification["auxiliary_frozen_utc"], earliest_confirmation_utc=verification["earliest_confirmation_utc"],
        scope="Old direct maturity G/I/A excluded; all forward financial costs/cost injections and actual inherited state effects retained. Original-objective choices reused. Not a no-prepaid/flat counterfactual, new acquisition, lifecycle profit or future-market performance.")
    reproducer = _reproducer()
    reproduction = reproducer.verify_payload(public_paths, cases, selections, public_result)
    model_manifest = primary.read(run/"models/audit_models_manifest.json")
    provenance = dict(schema=SCHEMA, run_id=run_id,
        auxiliary_source_freeze_sha256=primary.sha(freeze_path), auxiliary_source_sha256=primary.sanitized_hashes(frozen["source_sha256"]),
        primary_source_freeze_sha256=primary.sha(run/"SOURCE_FREEZE.json"),
        primary_selection_freeze_sha256=primary.sha(run/"SELECTION_FREEZE.json"),
        primary_audit_source_sha256=primary.sanitized_hashes(primary_freeze["source_sha256"]),
        primary_native_source_sha256=primary.sanitized_hashes(primary_freeze["native_source_sha256"]),
        checkpoint_input_sha256=primary.sanitized_hashes(primary_freeze["checkpoint_sha256"]),
        auxiliary_independent_verification_sha256=primary.sha(verification_path),
        auxiliary_independent_verifier_source_sha256=verification["verifier_sha256"],
        primary_independent_verification_sha256=primary.sha(primary_verification),
        private_auxiliary_result_sha256=primary.sha(private/"results.json"),
        private_auxiliary_paths_sha256=primary.sha(private/"path_summaries.jsonl.gz"),
        original_selection_sha256=primary.sha(run/"selection.json"),
        fitted_audit_model_bundle_sha256=model_manifest["bundle_sha256"],
        fitted_audit_model_metadata_sha256=model_manifest["metadata_sha256"],
        fitting_input_sha256=primary.sanitized_hashes(primary_selection["fit_input_sha256"]),
        context_input_sha256=resources, auxiliary_private_input_sha256=result["input_hashes"],
        exporter_sha256=primary.sha(__file__), primary_export_helper_sha256=primary.sha(HERE/"public_export.py"),
        reproducer_sha256=primary.sha(HERE/"reproduce_public_new_contract.py"),
        freeze_utc=verification["auxiliary_frozen_utc"], first_confirmation_utc=verification["earliest_confirmation_utc"],
        source_claim_boundary_independently_verified=verification["source_claim_boundary_checked"],
        access_scope="SHA identities only; native data/actors/critic bundles not redistributed; public replay verifies confidence arithmetic, not independent engine/source truth")
    output = HERE/"results"/(run_id+"_new_contract")
    output.mkdir(parents=True, exist_ok=False)
    primary.exclusive_json(output/"cases_and_bounds.json", cases)
    primary.exclusive_json(output/"selection.json", selections)
    primary.exclusive_json(output/"results.json", public_result)
    primary.exclusive_json(output/"provenance.json", provenance)
    primary.exclusive_json(output/"EXPORT_ARITHMETIC_CHECK.json", reproduction)
    compressed = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=compressed, mtime=0) as stream:
        for row in public_paths:
            stream.write((json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode("utf-8"))
    primary.exclusive_bytes(output/"paths.jsonl.gz", compressed.getvalue())
    primary.exclusive_bytes(output/"reproduce_public_new_contract.py", (HERE/"reproduce_public_new_contract.py").read_bytes())
    readme = """# Separate new-contract component: public arithmetic replay

With Python 3.10+ and its standard library only, run:

    python reproduce_public_new_contract.py .

Optional `--write-report` exclusively saves REPRODUCTION.json. The reader first
requires exactly eight mandatory artifact hashes, verifies all persisted bytes,
then reconstructs all 240 one-sided limits, decisions and analytic booked-harm
bounds. No native engine, market panels, actors, sklearn or fitted models.

The 10,240 paths are CONFIRM only: every method, including RAW, uses 256 paths
for each of forty original cases. No development paths are pooled. The allocation
.05/240 is a SEPARATE 95% family; the primary520 remains unchanged, and there is
no jointprimary/auxiliary95% claim. No new native query, fit or tuning was used.

All path gross/forgiveness/paired-forgiveness fields here refer to the narrower
component: direct old prepaid maturity G/I/A removed, ALL forward costs and fee
injections retained. Local pairing uses actual states. Old state effects and ALL
original control coins/differences remain. Original alpha0 primary-objective
caps/beta/budgets are reused without retuning. Conservative original supports
are retained. This is not a no-prepaid or flat counterfactual, acquisition study,
full lifecycle profit, native reward, fund return or future-market claim.

No observations, raw panels, private absolute paths, weights or pickles are
included. Provenance records exact source/input/model identities and the
pre-confirmation auxiliary freeze, not access or redistribution rights. Replay
is conditional on the exported summaries and source-derived supports; it cannot
independently verify the native engine or old-claim attribution. The private
independent verifier checks that attribution directly against compact events.
"""
    primary.exclusive_bytes(output/"README.md", readme.encode("utf-8"))
    file_hashes = {path.name: primary.sha(path) for path in output.iterdir() if path.is_file()}
    final = reproducer.verify_artifacts(output, file_hashes)
    primary.exclusive_json(output/"MANIFEST.json", dict(schema=SCHEMA, complete=True,
        created_utc=datetime.now(timezone.utc).isoformat(), run_id=run_id,
        file_sha256=file_hashes, family_limits=240, total_delta=.05, delta_per_limit=DELTA,
        paths=10240, auxiliary_source_freeze_sha256=primary.sha(freeze_path)))
    return dict(status="COMPLETE SEPARATE PUBLIC AUXILIARY EXPORT", output="results/"+run_id+"_new_contract",
                bounds_recomputed=final["bounds_recomputed"], manifest_sha256=primary.sha(output/"MANIFEST.json"),
                primary_bundle_modified=False, raw_inputs_copied=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="run_001")
    args = parser.parse_args()
    if not args.run_id.startswith("run_") or not args.run_id.replace("_", "").isalnum():
        raise ValueError("Simple original run ID required")
    print(json.dumps(export(args.run_id), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
