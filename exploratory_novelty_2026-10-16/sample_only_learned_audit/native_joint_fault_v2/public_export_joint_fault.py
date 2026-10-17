"""Held stdlib-only export of a complete independently verified v2 audit.

Default invocation reads no outcome. Root may later use --execute only after
completion, source/selection freezes and independent private PASS340 exist.
No native imports, actors, model loading, fits or statistical draws occur here.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent
SCHEMA = "sample_only_joint_fault_public_v2"
PHASE_N = {"fit": 32, "tune": 32, "confirm": 256}
DELTA = .05 / 340
ORIGINAL_FREEZE_SHA256 = "6152949392bd8de0f494bc21ffb5fd7d9be4a16a6c8866f4eb5595254da523d4"
REPRODUCER = "reproduce_public_joint_fault.py"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def simple_identifier(value):
    text = str(value).replace("\\", "/")
    if not text or ":" in text or text.startswith("/") or any(p in ("", ".", "..") for p in text.split("/")):
        raise ValueError("Public source identifiers must be relative and unambiguous")
    return text


def sanitized_hashes(mapping):
    output = {}
    for key, digest in mapping.items():
        name = simple_identifier(key)
        require(isinstance(digest, str) and len(digest) == 64
                and all(c in "0123456789abcdef" for c in digest), "SHA256 provenance required")
        require(name not in output, "Duplicate public provenance identifier")
        output[name] = digest
    return output


def numeric_work(value):
    """Copy numeric accounting only; never arbitrary private strings/paths."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        require(math.isfinite(value), "Finite reported computation counter")
        return value
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            require(isinstance(key, str) and re.fullmatch(r"[A-Za-z0-9_]+", key), "Plain computation field name")
            if not isinstance(item, str):
                output[key] = numeric_work(item)
        return output
    if isinstance(value, (tuple, list)):
        return [numeric_work(item) for item in value if not isinstance(item, str)]
    raise ValueError("Unsupported computation metadata")


def exclusive_bytes(path, content):
    path = Path(path)
    partial = path.with_name(path.name + ".partial")
    with partial.open("xb") as stream:
        stream.write(content)
        stream.flush()
    require(not path.exists(), "Never replace a prior public artifact")
    partial.rename(path)


def exclusive_json(path, value):
    content = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    require(not any(token in content for token in (b"C:\\", b"C:/", b"/Users/", b"/home/")), "No private absolute path in public JSON")
    exclusive_bytes(path, content)


def _load_reproducer():
    spec = importlib.util.spec_from_file_location("standalone_joint_fault_arithmetic", HERE / REPRODUCER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _private_gate(run_id):
    run = HERE / "execution" / run_id
    result_path = HERE / ("joint_fault_results_" + run_id + ".json")
    marker_path = HERE / ("joint_fault_completion_" + run_id + ".json")
    verification_path = HERE / "validation" / ("JOINT_FAULT_VERIFICATION_" + run_id + ".json")
    verifier_path = HERE / "validation" / "verify_joint_fault_results.py"
    required = [result_path, marker_path, verification_path, verifier_path, run / "SOURCE_FREEZE.json",
        run / "SELECTION_FREEZE.json", run / "selection.json", run / "cases.json", run / "confirm_stage_complete.json",
        run / "models" / "audit_models_manifest.json", run / "models_raw" / "audit_models_manifest.json", HERE / REPRODUCER]
    required += [run / f"case_{cid:02d}" / phase / name for cid in range(20) for phase in PHASE_N
                 for name in ("completion.json", "metadata.json", "rows.jsonl")]
    require(all(path.is_file() for path in required), "NOT_READY: full v2 completion/private-verification/source gates precede export")
    marker = read(marker_path)
    freeze = read(run / "SOURCE_FREEZE.json")
    selected = read(run / "SELECTION_FREEZE.json")
    verification = read(verification_path)
    require(marker.get("complete") is True and marker["results_sha256"] == sha(result_path)
            and marker["source_freeze_sha256"] == sha(run / "SOURCE_FREEZE.json"), "Complete result/source binding")
    require(verification.get("status") == "PASS" and verification.get("bounds_checked") == 340
            and verification.get("nuisance_regressor_prediction_rechecked") is True
            and verification["verifier_sha256"] == sha(verifier_path)
            and verification["result_sha256"] == sha(result_path)
            and verification["source_freeze_sha256"] == sha(run / "SOURCE_FREEZE.json"), "Independent complete PASS340 with all predictor replay")
    require(freeze["cases_sha256"] == sha(run / "cases.json") and freeze["family_limits"] == 340
            and freeze["delta_per_limit"] == DELTA, "Frozen complete v2 case/family manifest")
    require(selected["selection_sha256"] == sha(run / "selection.json")
            and selected["model_manifest_sha256"] == sha(run / "models" / "audit_models_manifest.json")
            and selected["raw_model_manifest_sha256"] == sha(run / "models_raw" / "audit_models_manifest.json"), "Pre-confirmation choices/paired/raw models")
    sources = sanitized_hashes(freeze["source_sha256"])
    require({Path(__file__).name, REPRODUCER, "validation/verify_joint_fault_results.py"} <= set(sources), "Export/replay/verifier sources frozen before outcomes")
    for name, digest in sources.items():
        require(sha(HERE / name) == sha(run / "source_snapshot" / name) == digest, "Unchanged frozen v2 source: " + name)
    original_freeze_path = ORIGINAL / "execution" / "run_001" / "SOURCE_FREEZE.json"
    require(freeze["original_freeze_sha256"] == sha(original_freeze_path) == ORIGINAL_FREEZE_SHA256, "Canonical original source freeze preserved")
    original_freeze = read(original_freeze_path)
    dependencies = sanitized_hashes(freeze["original_source_sha256"])
    for name, digest in dependencies.items():
        require(original_freeze["source_sha256"][name] == sha(ORIGINAL / name)
                == sha(ORIGINAL / "execution" / "run_001" / "source_snapshot" / name) == digest,
                "Exact frozen original dependency: " + name)
    private_result = read(result_path)
    require(private_result["source_freeze_sha256"] == sha(run / "SOURCE_FREEZE.json")
            and private_result["selection_freeze_sha256"] == sha(run / "SELECTION_FREEZE.json"), "Result/source/selection chronology binding")
    return run, result_path, marker_path, verification_path, freeze, selected, private_result


def export(run_id):
    require(re.fullmatch(r"run_[A-Za-z0-9_]+", run_id) is not None, "Simple run ID")
    directory = HERE / "results" / run_id
    require(not directory.exists(), "Exclusive new v2 export; never overwrite an earlier attempt")
    run, result_path, marker_path, verification_path, freeze, selection_freeze, private_result = _private_gate(run_id)
    reproducer = _load_reproducer()
    cases, private_choices = read(run / "cases.json"), read(run / "selection.json")
    require(len(cases) == 20 and len(private_choices) == 320, "All v2 cases and frozen choices")
    selections = [{k: row[k] for k in ("case_id", "alpha", "method", "choice")} for row in private_choices]
    path_rows, public_cases, phase_provenance, context_resources = [], [], [], []
    bound_inputs = {p: sha(p) for p in (result_path, marker_path, verification_path,
        run / "SOURCE_FREEZE.json", run / "SELECTION_FREEZE.json", run / "selection.json", run / "cases.json")}
    for case in cases:
        cid, bound, context_identity = int(case["case_id"]), None, None
        for phase, n in PHASE_N.items():
            phase_dir = run / f"case_{cid:02d}" / phase
            completion, metadata = read(phase_dir / "completion.json"), read(phase_dir / "metadata.json")
            require(completion.get("status") == "COMPLETE" and metadata["completed_episodes"] == n, "Complete fixed phase")
            phase_hashes = {}
            for artifact in completion["artifacts"].values():
                path = Path(artifact["path"]).resolve()
                require(path.is_relative_to(phase_dir.resolve()) and path.is_file(), "Private artifact stays inside its phase")
                require(sha(path) == artifact["sha256"], "Bound private phase artifact unchanged")
                identifier = simple_identifier(path.relative_to(phase_dir.resolve()).as_posix())
                phase_hashes[identifier] = artifact["sha256"]
                bound_inputs[path] = artifact["sha256"]
            require({"rows.jsonl", "metadata.json"} <= set(phase_hashes), "Phase completion binds rows and support metadata")
            base = rows(phase_dir / "rows.jsonl")
            require(len(base) == n, "Fixed complete whole-path budget")
            for row in base:
                item = {k: row[k] for k in reproducer.PATH_FIELDS}
                if phase == "confirm":
                    item.update({k: row[k] for k in reproducer.CONFIRM_FIELDS})
                path_rows.append(item)
            scalar = {k: metadata["bounds"][k] for k in reproducer.BOUND_FIELDS}
            if bound is None:
                bound = scalar
            else:
                require(scalar == bound, "Same pre-draw scalar supports across phases")
            resources = [dict(context_id=r["context"]["context_id"],
                resource_sha256=sanitized_hashes(r["metadata"]["resource_sha256"]),
                configuration_sha256=r["metadata"]["config_snapshot_sha256"],
                warm_state_sha256=r["metadata"]["warm_snapshot_sha256"])
                for r in metadata["cases_contexts"]]
            require(len(resources) == 8 and {r["context_id"] for r in resources} == set(range(8)), "Complete fixed context identity")
            if context_identity is None:
                context_identity = resources
            else:
                require(resources == context_identity, "Preserved warm/context/resource identities across phases")
            bound_inputs[phase_dir / "completion.json"] = sha(phase_dir / "completion.json")
            phase_provenance.append(dict(case_id=cid, phase=phase, private_completion_sha256=sha(phase_dir / "completion.json"),
                                        private_artifact_sha256=phase_hashes))
        public_case = {k: case[k] for k in reproducer.CASE_FIELDS}
        public_case["bounds"] = bound
        public_cases.append(public_case)
        context_resources.append(dict(case_id=cid, contexts=context_identity))
    case_map = {case["case_id"]: case for case in public_cases}
    charged = []
    for row in private_result["charged_results"]:
        case = case_map[row["case_id"]]
        identity = {k: case[k] for k in ("case_id", "policy_family", "policy_index", "checkpoint_seed", "profile")}
        require(all(row.get(k, v) == v for k, v in identity.items()), "Private/public charged identity")
        charged.append(dict(**identity, **{k: row[k] for k in ("alpha", "method", "choice", "certificate", "positive",
                                                              "confirmation_mean", "confirmation_variance")}))
    costs = [{k: row[k] for k in ("case_id", "profile", "certificate", "positive", "analytic_unavoidable_cost_lower", "booked_expectation_upper")}
             for row in private_result["cost_results"]]
    public_result = dict(schema=SCHEMA, status="COMPLETE EXPLORATORY JOINT-FAULT AUDIT", n_existing_learned_policies=20,
        n_cases=20, n_audit_episodes=6400, n_native_audit_steps=614400, monetary_units="DKK",
        multiplicity=dict(family_limits=340, total_delta=.05, delta_per_limit=DELTA),
        method_counts=private_result["method_counts"], profile_counts=private_result["profile_counts"],
        positive_cost_limits=private_result["positive_cost_limits"], charged_results=charged, cost_results=costs,
        actual_work_numeric=numeric_work(private_result["actual_work"]),
        scope="new financial-component target excluding direct old maturities under inherited state and fresh hidden fair signs;20archivedpolicies;L3-style audit-only cap removal,not historical redeployment;no whole-fund/acquisition claim")
    reproduction = reproducer.verify_payload(path_rows, public_cases, selections, public_result)
    model_manifest = read(run / "models" / "audit_models_manifest.json")
    raw_model_manifest = read(run / "models_raw" / "audit_models_manifest.json")
    provenance = dict(schema=SCHEMA, run_id=run_id, source_freeze_sha256=sha(run / "SOURCE_FREEZE.json"),
        selection_freeze_sha256=sha(run / "SELECTION_FREEZE.json"), private_result_sha256=sha(result_path),
        private_completion_sha256=sha(marker_path), private_verification_sha256=sha(verification_path),
        frozen_case_manifest_sha256=sha(run / "cases.json"), frozen_selection_sha256=sha(run / "selection.json"),
        audit_source_sha256=sanitized_hashes(freeze["source_sha256"]),
        original_source_sha256=sanitized_hashes(freeze["original_source_sha256"]), original_freeze_sha256=freeze["original_freeze_sha256"],
        native_source_sha256=sanitized_hashes(freeze["native_source_sha256"]),
        checkpoint_input_sha256=sanitized_hashes(freeze["checkpoint_sha256"]),
        fitted_audit_model_bundle_sha256=model_manifest["bundle_sha256"],
        fitted_audit_model_metadata_sha256=model_manifest["metadata_sha256"],
        fitted_raw_audit_model_bundle_sha256=raw_model_manifest["bundle_sha256"],
        fitted_raw_audit_model_metadata_sha256=raw_model_manifest["metadata_sha256"],
        fitting_input_sha256=sanitized_hashes(selection_freeze["fit_input_sha256"]),
        raw_fitting_input_sha256=sanitized_hashes(selection_freeze["raw_fit_input_sha256"]),
        context_input_identity=context_resources, private_phase_artifact_sha256=phase_provenance,
        exporter_sha256=sha(__file__), reproducer_sha256=sha(HERE / REPRODUCER),
        scope="input identities only;no private data/model/source redistribution or anonymous/full-pipeline/clean-install claim;reproduction checks arithmetic conditional on summaries and source supports")
    directory.mkdir(parents=True, exist_ok=False)
    exclusive_json(directory / "cases_and_bounds.json", public_cases)
    exclusive_json(directory / "selection.json", selections)
    exclusive_json(directory / "results.json", public_result)
    exclusive_json(directory / "provenance.json", provenance)
    exclusive_json(directory / "EXPORT_ARITHMETIC_CHECK.json", reproduction)
    compressed = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=compressed, mtime=0) as stream:
        for row in path_rows:
            stream.write((json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8"))
    exclusive_bytes(directory / "paths.jsonl.gz", compressed.getvalue())
    exclusive_bytes(directory / REPRODUCER, (HERE / REPRODUCER).read_bytes())
    readme = """# Standalone arithmetic reproduction: joint-fault v2

Use Python3.10+ and its standard library only:

    python reproduce_public_joint_fault.py .

Add --write-report to exclusively create REPRODUCTION.json. The program checks all public SHA256 identities and recomputes every340 original one-sided limit and decision. No NumPy, sklearn, native simulator, actor, fitted model or raw market data is used. This is saved-record arithmetic reproduction, not independent native-engine replay or a clean-install/full-pipeline claim.

All6400 whole paths are retained:20policies ×(32FIT+32TUNE+256CONFIRM), with fixed8-context and96-step law. Case p follows original vulnerable case2p+1 but uses new statistical paths. Direct inherited maturity gross/forgiveness is excluded from this target while all forward financial costs and inherited state effects remain. Excluded old summaries are reported separately; ALL ordered old/new coins and frozen predictor differences remain in the controls. This is not no-prepaid deployment, complete lifecycle profit or an executable flat-policy contrast.

The family is320charged limits plus20cost=340: ten methods at alpha0 (including cash_capped_raw,cash_ridge_control,cash_mlp_control,cash_local_control), six at alpha1. RAW/cost use320 paths; all capped/selected methods use256CONFIRM paths. Every cap, scale, absolute control budget and nuisance model was frozen before confirmation. Cash RAW uses the SAME upper-capped RAW statistic with a separately source-proved cash-inflow lower endpoint -K, only at alpha0. Cash learned controls subtract a predictable absolute-budget martingale from RAW, using support[-K-B,min(cap,Gbar+B)]. Their stored coefficients combine the known current new-financial RAW half-difference (zero for direct old maturities) with separately FIT-only trained RAW-future predictors; ALL coins remain. The analytic cash_local_control uses that current coefficient with future predictor zero and the same32TUNE/cap/beta/B procedure. It tests whether known current accounting, rather than learned continuation, explains a benefit. The original paired-future models are unchanged. These source-reviewed comparisons were fixed before new statistical sampling, not added after a v2 outcome. The implemented numerical allowance STILL uses G+C support; no removal of all numerical support access is claimed.

The separate95% family has delta=.05/340. It is not pooled with older experiments for paper-wide coverage. All abstentions and full-charge known-null checks are retained. A positive charged-component limit plus strict booked-harm upper bound is a scoped accounting witness, not whole-fund advantage, learned exploit acquisition, generic AI-method superiority or future-market coverage. A nonpositive limit is inconclusive. Supports, sign-law validity, K and simulator authenticity are assumptions validated elsewhere, not inferred from this export.

Only whitelisted scalar summaries, choices, numerical limits and ordered coin/difference vectors are supplied. Private observations, native cash traces, raw panels, checkpoints, model pickles and absolute private paths are excluded. Input/source/model hashes provide identities without access or redistribution rights. The original run001 and its exported bundles remain unchanged.
"""
    exclusive_bytes(directory / "README.md", readme.encode("utf-8"))
    file_hashes = {p.name: sha(p) for p in directory.iterdir() if p.is_file()}
    final = reproducer.verify_artifacts(directory, file_hashes)
    require(all(sha(path) == digest for path, digest in bound_inputs.items()), "Private inputs unchanged throughout export")
    # Recheck all current/snapshotted source and completion bindings before seal.
    _private_gate(run_id)
    manifest = dict(schema=SCHEMA, complete=True, created_utc=datetime.now(timezone.utc).isoformat(), run_id=run_id,
        file_sha256=file_hashes, family_limits=340, total_delta=.05, delta_per_limit=DELTA, path_summary_rows=6400,
        source_freeze_sha256=provenance["source_freeze_sha256"], exporter_sha256=sha(__file__),
        reproducer_sha256=provenance["reproducer_sha256"])
    exclusive_json(directory / "MANIFEST.json", manifest)
    return dict(status="COMPLETE PUBLIC V2 EXPORT", output="results/" + run_id,
        manifest_sha256=sha(directory / "MANIFEST.json"), bounds_recomputed=final["bounds_recomputed"], checks=final["checks"],
        native_simulator_calls=0, actor_inference=0, model_fitting=0, original_bundles_modified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="run_001")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(status="PREPARED_WAITING_COMPLETE_PRIVATE_PASS340", exporter_sha256=sha(__file__),
                              endpoint_reads=0, native_simulator_calls=0, actor_inference=0, model_fitting=0)))
        return 0
    print(json.dumps(export(args.run_id), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
