"""Export a COMPLETE, independently checked native audit for public CI replay.

No partial financial endpoints are exported. Native traces, actor observations,
market panels, weights, fitted-model pickles and absolute private paths are never
copied. All outputs are exclusive-created; rich internal results stay unchanged.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SCHEMA = "sample_only_native_public_v1"
PHASE_N = {"fit": 32, "tune": 32, "confirm": 256}
METHODS = ("raw", "capped_raw", "capped_residual", "paired", "ridge_control", "mlp_control")
DELTA = .05/520


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def simple_identifier(value):
    text = str(value).replace("\\", "/")
    if ":" in text or text.startswith("/") or ".." in text.split("/"):
        raise ValueError("Public provenance identifiers must be relative")
    return text


def sanitized_hashes(mapping):
    output = {}
    for key, digest in mapping.items():
        name = simple_identifier(key)
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Expected SHA256 provenance")
        if name in output and output[name] != digest:
            raise ValueError("Ambiguous public provenance identifier")
        output[name] = digest
    return output


def exclusive_bytes(path, content):
    path = Path(path)
    partial = path.with_name(path.name+".partial")
    with partial.open("xb") as stream:
        stream.write(content)
        stream.flush()
    if path.exists():
        raise FileExistsError("Never replace an existing public artifact")
    partial.rename(path)


def exclusive_json(path, value):
    content = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n").encode("utf-8")
    # Defensive final scan supplements the field whitelist. Relative source
    # identifiers are permitted; user-home or drive-qualified strings are not.
    if b"C:\\" in content or b"C:/" in content or b"/Users/" in content or b"/home/" in content:
        raise ValueError("Private absolute path leaked into public JSON")
    exclusive_bytes(path, content)


def _load_reproducer():
    path = HERE/"reproduce_public_native_audit.py"
    spec = importlib.util.spec_from_file_location("standalone_public_ci_reproduction", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _private_gate(run_id):
    run = HERE/"execution"/run_id
    result_path = HERE/("native_results_"+run_id+".json")
    marker_path = HERE/("native_completion_"+run_id+".json")
    required = [run/"SOURCE_FREEZE.json", run/"SELECTION_FREEZE.json", run/"confirm_stage_complete.json",
                result_path, marker_path]
    if not all(path.is_file() for path in required):
        raise RuntimeError("NOT_READY: complete confirmation and original aggregation are required before export")
    marker, freeze, selected, result = read(marker_path), read(run/"SOURCE_FREEZE.json"), read(run/"SELECTION_FREEZE.json"), read(result_path)
    if marker.get("complete") is not True or sha(result_path) != marker["results_sha256"]:
        raise AssertionError("Original completed result binding failed")
    if marker["source_freeze_sha256"] != sha(run/"SOURCE_FREEZE.json"):
        raise AssertionError("Original source-freeze completion binding failed")
    if result["source_freeze_sha256"] != sha(run/"SOURCE_FREEZE.json") or result["selection_freeze_sha256"] != sha(run/"SELECTION_FREEZE.json"):
        raise AssertionError("Result source/selection binding failed")
    if sha(run/"selection.json") != selected["selection_sha256"] or sha(run/"models/audit_models_manifest.json") != selected["model_manifest_sha256"]:
        raise AssertionError("Pre-confirmation model/choice binding failed")
    verification_paths = [HERE/"validation"/("NATIVE_VERIFICATION_"+run_id+suffix+".json")
                          for suffix in ("_critics", "")]
    verified = next((path for path in verification_paths if path.is_file()), None)
    if verified is None:
        raise RuntimeError("NOT_READY: independent complete private verification is required before public export")
    verification = read(verified)
    if (verification.get("status") != "PASS" or verification.get("bounds_checked") != 520
            or verification.get("result_sha256") != sha(result_path)
            or verification.get("source_freeze_sha256") != sha(run/"SOURCE_FREEZE.json")):
        raise AssertionError("Independent private verification does not bind this completed result")
    return run, result_path, marker_path, freeze, selected, result, verified


def export(run_id):
    run, result_path, marker_path, freeze, selection_freeze, private_result, verification_path = _private_gate(run_id)
    cases, selections = read(run/"cases.json"), read(run/"selection.json")
    if len(cases) != 40 or len(selections) != 480:
        raise AssertionError("Complete original case/choice cohort required")
    source_hashes = sanitized_hashes(freeze["source_sha256"])
    for name, digest in source_hashes.items():
        if sha(HERE/name) != digest or sha(run/"source_snapshot"/name) != digest:
            raise AssertionError("Frozen source changed before export")
    path_rows, public_cases, phase_provenance, context_resources = [], [], [], []
    for case in cases:
        cid = int(case["case_id"])
        bound = None
        context_identity = None
        for phase, n in PHASE_N.items():
            directory = run/f"case_{cid:02d}"/phase
            completed, metadata = read(directory/"completion.json"), read(directory/"metadata.json")
            if completed.get("status") != "COMPLETE" or metadata["completed_episodes"] != n:
                raise AssertionError("No partial phase may be exported")
            for artifact in completed["artifacts"].values():
                if sha(artifact["path"]) != artifact["sha256"]:
                    raise AssertionError("Private phase artifact changed before export")
            base = rows(directory/"rows.jsonl")
            if len(base) != n:
                raise AssertionError("Fixed full path budget required")
            for row in base:
                item = {key: row[key] for key in ("case_id", "phase", "episode_id", "context_id", "gross", "cost",
                    "forgiveness", "paired_forgiveness", "maturity_coins")}
                if phase == "confirm":
                    item.update({key: row[key] for key in ("delta_ridge", "delta_mlp")})
                path_rows.append(item)
            scalar = {key: metadata["bounds"][key] for key in
                      ("discounted_cost_bound", "discounted_gross_bound", "unavoidable_discounted_access_cost")}
            if bound is None:
                bound = scalar
            elif scalar != bound:
                raise AssertionError("Predraw scalar supports differ between phases")
            resources = [dict(context_id=record["context"]["context_id"],
                              resource_sha256=sanitized_hashes(record["metadata"]["resource_sha256"]),
                              configuration_sha256=record["metadata"]["config_snapshot_sha256"],
                              warm_state_sha256=record["metadata"]["warm_snapshot_sha256"])
                         for record in metadata["cases_contexts"]]
            if context_identity is None:
                context_identity = resources
            elif resources != context_identity:
                raise AssertionError("Context/resource/warm states differ between phases")
            phase_provenance.append(dict(case_id=cid, phase=phase,
                private_completion_sha256=sha(directory/"completion.json"),
                private_metadata_sha256=sha(directory/"metadata.json"),
                private_path_summary_sha256=sha(directory/"rows.jsonl"),
                private_financial_events_sha256=sha(directory/"financial_events.npz"),
                private_branch_features_sha256=sha(directory/"branch_features.npz")))
        public_case = {key: case[key] for key in ("case_id", "policy_index", "policy_family", "checkpoint_seed",
                                                   "profile_index", "profile", "cash_fraction", "horizon", "burn_in")}
        public_case["bounds"] = bound
        public_cases.append(public_case)
        context_resources.append(dict(case_id=cid, contexts=context_identity))
    charged = [{key: row[key] for key in ("case_id", "policy_family", "policy_index", "checkpoint_seed", "profile",
                 "alpha", "method", "choice", "certificate", "positive", "confirmation_mean", "confirmation_variance")}
               for row in private_result["charged_results"]]
    costs = [{key: row[key] for key in ("case_id", "profile", "certificate", "positive",
              "analytic_unavoidable_cost_lower", "booked_expectation_upper")}
             for row in private_result["cost_results"]]
    public_result = dict(schema=SCHEMA, status="COMPLETE EXPLORATORY AUDIT", n_existing_learned_policies=20,
        n_cases=40, n_audit_episodes=12800, n_native_audit_steps=1228800, monetary_units="DKK",
        multiplicity=dict(family_limits=520, total_delta=.05, delta_per_limit=DELTA),
        method_counts=private_result["method_counts"], profile_counts=private_result["profile_counts"],
        positive_cost_limits=private_result["positive_cost_limits"], charged_results=charged, cost_results=costs,
        actual_work=private_result["actual_work"],
        scope="twenty frozen native policies under a declared instrumented fair-sign law and eight fixed contexts, two guard profiles; forward financial component only; not future-market or native-reward performance")
    reproducer = _load_reproducer()
    reproduction = reproducer.verify_payload(path_rows, public_cases, selections, public_result)
    model_manifest = read(run/"models/audit_models_manifest.json")
    provenance = dict(schema=SCHEMA, run_id=run_id, source_freeze_sha256=sha(run/"SOURCE_FREEZE.json"),
        selection_freeze_sha256=sha(run/"SELECTION_FREEZE.json"), private_result_sha256=sha(result_path),
        private_completion_sha256=sha(marker_path), private_verification_sha256=sha(verification_path),
        frozen_case_manifest_sha256=sha(run/"cases.json"), frozen_selection_sha256=sha(run/"selection.json"),
        frozen_search_scores_sha256=sha(run/"search_records.json"),
        audit_source_sha256=source_hashes, native_source_sha256=sanitized_hashes(freeze["native_source_sha256"]),
        checkpoint_input_sha256=sanitized_hashes(freeze["checkpoint_sha256"]),
        fitted_audit_model_bundle_sha256=model_manifest["bundle_sha256"],
        fitted_audit_model_metadata_sha256=model_manifest["metadata_sha256"],
        fitting_input_sha256=sanitized_hashes(selection_freeze["fit_input_sha256"]),
        context_input_identity=context_resources, private_phase_artifact_sha256=phase_provenance,
        exporter_sha256=sha(__file__), reproducer_sha256=sha(HERE/"reproduce_public_native_audit.py"),
        scope="hashes identify original inputs without redistributing private source/data/weights/features/model bundles; arithmetic reproduction does not independently replay the native engine")
    directory = HERE/"results"/run_id
    directory.mkdir(parents=True, exist_ok=False)
    try:
        exclusive_json(directory/"cases_and_bounds.json", public_cases)
        exclusive_json(directory/"selection.json", selections)
        exclusive_json(directory/"results.json", public_result)
        exclusive_json(directory/"provenance.json", provenance)
        exclusive_json(directory/"EXPORT_ARITHMETIC_CHECK.json", reproduction)
        compressed = io.BytesIO()
        # No original filename or timestamp enters the gzip header.
        with gzip.GzipFile(filename="", mode="wb", fileobj=compressed, mtime=0) as stream:
            for row in path_rows:
                stream.write((json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode("utf-8"))
        exclusive_bytes(directory/"paths.jsonl.gz", compressed.getvalue())
        exclusive_bytes(directory/"reproduce_public_native_audit.py", (HERE/"reproduce_public_native_audit.py").read_bytes())
        readme = """# Public arithmetic reproduction of the completed native audit

Run with Python 3.10 or newer, using only its standard library:

    python reproduce_public_native_audit.py .

Optionally add `--write-report` to exclusively create `REPRODUCTION.json`. The
program first checks every public artifact SHA256, then independently recomputes
all 520 one-sided empirical-Bernstein limits and decisions. No NumPy, sklearn,
native simulator, raw market panel, actor checkpoint or fitted model is required.

`paths.jsonl.gz` contains all 12,800 whole-episode summaries from the three fixed
phases and 40 cases. RAW pools 320 same-law paths per case; selected/capped methods
confirm on 256 paths, after 32 FIT and 32 TUNE paths. Cost limits pool 320 paths.
Both charge views share native histories. The 480 charged and 40 cost limits use
the separate family allocation delta=.05/520; other experiment families are not
included. Sequential absolute-amplitude controls are reconstructed from ordered
coin/predictor-difference vectors, without a fitted model. Deterministic scalar
supports, numerical allowances, choices and every original confidence number are
in the accompanying JSON files. Booked harm uses the larger of the cost lower
limit and the source-proved unavoidable access-fee lower bound.

This reproduces confidence arithmetic conditional on the exported records and
the declared instrumented fair-sign law. It does not independently validate the
native engine, actor interfaces, source-derived supports or sign intervention.
The result concerns the forward financial-event component for twenty frozen
policies under eight fixed contexts and two guard profiles, not native reward,
complete fund/lifecycle return, future markets or retrained full-booking policies.
The strict profile is guard-restored and may retain prepaid claims; inactivity
or zero forgiveness is not assumed. All abstentions and full-booking views remain.

No actor observations, native raw panels, absolute private paths, weights,
pickles or private feature rows are included. `provenance.json` supplies source,
checkpoint, data/context and model SHA256 identities without supplying access or
redistribution rights. Full engine reproduction remains a separate dependency
and access requirement. Original internal results are preserved unchanged.
"""
        exclusive_bytes(directory/"README.md", readme.encode("utf-8"))
        names = sorted(path.name for path in directory.iterdir() if path.is_file())
        file_hashes = {name: sha(directory/name) for name in names}
        # Reopen, hash, gunzip and reproduce the exact persisted bytes BEFORE
        # publishing the only completion marker. Any failure leaves no manifest.
        final = reproducer.verify_artifacts(directory, file_hashes)
        manifest = dict(schema=SCHEMA, complete=True, created_utc=datetime.now(timezone.utc).isoformat(),
            run_id=run_id, file_sha256=file_hashes,
            family_limits=520, total_delta=.05, delta_per_limit=DELTA,
            path_summary_rows=12800, source_freeze_sha256=provenance["source_freeze_sha256"],
            exporter_sha256=sha(__file__), reproducer_sha256=provenance["reproducer_sha256"])
        exclusive_json(directory/"MANIFEST.json", manifest)
        return dict(status="COMPLETE PUBLIC EXPORT", output="results/"+run_id,
                    manifest_sha256=sha(directory/"MANIFEST.json"), bounds_recomputed=final["bounds_recomputed"],
                    checks=final["checks"], raw_inputs_copied=False, original_results_modified=False)
    except BaseException:
        # No prior artifact is deleted or rewritten. An absent MANIFEST means
        # incomplete export; preserve the directory for diagnosis.
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="run_001")
    args = parser.parse_args()
    if not args.run_id.startswith("run_") or not args.run_id.replace("_", "").isalnum():
        raise ValueError("Simple frozen run ID required")
    print(json.dumps(export(args.run_id), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
