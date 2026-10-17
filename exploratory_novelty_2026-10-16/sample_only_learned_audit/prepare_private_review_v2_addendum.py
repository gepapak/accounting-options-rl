"""Prepare a private additive review packet only after all completion gates.

Default invocation is held and performs no archive, endpoint or native work.
No native code, actor, audit critic, learner or scientific simulator is executed.
Only the exported standard-library arithmetic reproducers may be executed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import time
import zipfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PREFIX = PurePosixPath("native_review_v2_addendum")
BASE_ZIP_SHA256 = "d96cf47f5d6746cf760608293bfbe5e62a5720f2b98c30702a89debf5f6b8299"
AUDIT_RELATIVE = PurePosixPath("github/accounting-options-rl/exploratory_novelty_2026-10-16/sample_only_learned_audit")
GOLDEN_NAMES = ("plain7_strict", "feasible7_vulnerable")
PACKET_REPLAY_WRAPPER = "replay_private_bundle_attempt4.py"
CONTROL_DIAGNOSTIC_FILES = {"RESULTS.json", "case_diagnostics.csv", "README.md"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_member(name):
    value = PurePosixPath(str(name).replace("\\", "/"))
    require(not value.is_absolute() and value.parts and
            all(part not in (".", "..") and ":" not in part for part in value.parts),
            "Unsafe relative packet member: " + str(name))
    return value


def inside(path, root):
    return Path(path).resolve().is_relative_to(Path(root).resolve())


def require_files(paths):
    missing = [str(path) for path in paths if not path.is_file()]
    require(not missing, "NOT_READY: required completion/verification/replay files absent: " + ", ".join(missing))


def gate_paths(here, run_id, replay):
    run = here / "execution" / run_id
    auxiliary = here / "auxiliary_new_contract" / run_id
    primary_public, auxiliary_public = here / "results" / run_id, here / "results" / (run_id + "_new_contract")
    return {
        "primary_completion": here / ("native_completion_" + run_id + ".json"),
        "confirmation_completion": run / "confirm_stage_complete.json",
        "primary_verification": here / "validation" / ("NATIVE_VERIFICATION_" + run_id + "_critics.json"),
        "auxiliary_completion": auxiliary / "COMPLETION.json",
        "auxiliary_verification": here / "validation" / ("NEW_CONTRACT_VERIFICATION_" + run_id + ".json"),
        "primary_public_manifest": primary_public / "MANIFEST.json",
        "primary_public_reproduction": primary_public / "REPRODUCTION.json",
        "auxiliary_public_manifest": auxiliary_public / "MANIFEST.json",
        "auxiliary_public_reproduction": auxiliary_public / "REPRODUCTION.json",
        "packet_replay": replay / "replay_report.json",
        "descriptive_tables": here / "report_tables" / run_id / "COMPLETION.json",
        "control_diagnostic_completion": here / "control_diagnostics" / run_id / "COMPLETION.json",
    }


def control_diagnostic_gate(here, run_id, gates):
    """Bind the separate post-gate descriptive layer; create no new CI family."""
    directory = here / "control_diagnostics" / run_id
    marker = read(directory / "COMPLETION.json")
    require(marker.get("complete") is True and set(marker["file_sha256"]) == CONTROL_DIAGNOSTIC_FILES,
            "Exact three-artifact completed descriptive control diagnostic required")
    tool = here / "post_completion_control_diagnostics.py"
    plan = here / "POST_COMPLETION_CONTROL_DIAGNOSTIC_PLAN.md"
    require(marker["source_sha256"] == sha(tool), "Executed descriptive diagnostic source identity")
    inputs = marker["input_sha256"]
    expected = {
        "source_sha256": sha(tool), "plan_sha256": sha(plan),
        "private_primary_verifier_sha256": sha(gates["primary_verification"]),
        "private_auxiliary_verifier_sha256": sha(gates["auxiliary_verification"]),
        "original_report_completion_sha256": sha(gates["descriptive_tables"]),
        "public_primary_manifest_sha256": sha(gates["primary_public_manifest"]),
        "public_auxiliary_manifest_sha256": sha(gates["auxiliary_public_manifest"]),
        "public_primary_paths_sha256": sha(here / "results" / run_id / "paths.jsonl.gz"),
        "public_auxiliary_paths_sha256": sha(here / "results" / (run_id + "_new_contract") / "paths.jsonl.gz"),
    }
    require(all(inputs.get(key) == digest for key, digest in expected.items()) and
            inputs.get("fresh_public_replay_bounds") == [520, 240], "Descriptive diagnostic source/plan/private/public/report bindings")
    for name, digest in marker["file_sha256"].items():
        require(sha(directory / name) == digest, "Completed diagnostic artifact identity: " + name)
    result = read(directory / "RESULTS.json")
    require(result.get("status") == "COMPLETE DESCRIPTIVE CONTROL DIAGNOSTIC" and
            result.get("provenance") == inputs and result.get("new_native_queries") == 0 and
            result.get("new_confidence_limits") == 0, "Post-gate diagnostic remains descriptive, with no new queries/limits")
    records, summaries = result["records"], result["summaries"]
    expected_records = {(target, cid, model) for target in ("primary_alpha0", "primary_alpha1", "auxiliary_alpha0")
                        for cid in range(40) for model in ("ridge_control", "mlp_control")}
    require(len(records) == 240 and {(row["target"], row["case_id"], row["method"]) for row in records} == expected_records and
            all(row["n"] == 256 for row in records), "Complete240 fixed common-CONFIRM diagnostic records")
    expected_summaries = {(target, profile, model) for target in ("primary_alpha0", "primary_alpha1", "auxiliary_alpha0")
                          for profile in ("strict", "vulnerable") for model in ("ridge_control", "mlp_control")}
    require(len(summaries) == 12 and {(row["target"], row["profile"], row["method"]) for row in summaries} == expected_summaries and
            all(row["cases"] == 20 for row in summaries), "All twelve complete descriptive control summary groups")
    return marker


def verify_public(directory, filename, count, name):
    manifest = read(directory / "MANIFEST.json")
    saved = read(directory / "REPRODUCTION.json")
    require(manifest.get("complete") is True and manifest.get("family_limits") == count,
            "Completed public family manifest required")
    require(saved.get("status") == "PASS" and saved.get("bounds_recomputed") == count and
            saved.get("manifest_sha256") == sha(directory / "MANIFEST.json"),
            "Recorded standalone public replay must bind this complete family")
    require(sha(directory / filename) == manifest["file_sha256"][filename], "Archived public reproducer hash")
    spec = importlib.util.spec_from_file_location(name, directory / filename)
    require(spec is not None and spec.loader is not None, "Cannot load exact public arithmetic reproducer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # This is confidence arithmetic over completed public summaries, not native
    # actor/model inference or new scientific trajectories.
    report = module.verify_directory(directory)
    require(report.get("status") == "PASS" and report.get("bounds_recomputed") == count,
            "Fresh standard-library public arithmetic replay failed")
    require(set(manifest["file_sha256"]) == set(module.MANDATORY_FILES) and len(module.MANDATORY_FILES) == 8,
            "Exact eight-file public artifact whitelist required")
    return manifest, report


def verify_base(base, archive):
    require(sha(archive) == BASE_ZIP_SHA256, "Preserved complete v1 ZIP identity")
    manifest_path = base / "BUNDLE_MANIFEST.json"
    manifest = read(manifest_path)
    expected = {str(safe_member(row["path_in_bundle"])): row["sha256"] for row in manifest["copied_artifacts"]}
    require(len(expected) == len(manifest["copied_artifacts"]), "Unique v1 base member names")
    expected["BUNDLE_MANIFEST.json"] = sha(manifest_path)
    require(len(expected) == 155, "Original155-file complete v1 base required")
    with zipfile.ZipFile(archive) as zipped:
        require(len(zipped.namelist()) == len(set(zipped.namelist())) and set(zipped.namelist()) == set(expected),
                "Base ZIP exact member set")
        for name, digest in expected.items():
            path = base / Path(safe_member(name))
            require(inside(path, base) and path.is_file() and sha(path) == digest, "Extracted base artifact identity: " + name)
            check = hashlib.sha256()
            with zipped.open(name) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    check.update(block)
            require(check.hexdigest() == digest, "Every decompressed base ZIP member identity: " + name)
    return manifest, expected


def completion_gates(here, run_id, replay, base_manifest, base_manifest_sha256):
    paths = gate_paths(here, run_id, replay)
    # No private/public endpoint is parsed or hashed before all existence gates.
    require_files(list(paths.values()))
    primary, auxiliary = read(paths["primary_verification"]), read(paths["auxiliary_verification"])
    require(primary.get("status") == "PASS" and primary.get("bounds_checked") == 520,
            "Complete primary private PASS520 required")
    require(auxiliary.get("status") == "PASS" and auxiliary.get("bounds_checked") == 240,
            "Complete auxiliary private PASS240 required")
    require(primary.get("nuisance_regressor_prediction_rechecked") is True, "Full saved-critic primary verification required")
    require(primary.get("verifier_sha256") == sha(here / "validation/verify_native_results.py") and
            auxiliary.get("verifier_sha256") == sha(here / "validation/verify_new_contract_component.py"),
            "Executed full private-verifier source hashes must match packaged sources")
    run, private_aux = here / "execution" / run_id, here / "auxiliary_new_contract" / run_id
    freeze = run / "SOURCE_FREEZE.json"
    auxiliary_freeze = here / ("NEW_CONTRACT_SOURCE_FREEZE_" + run_id + ".json")
    marker = read(paths["primary_completion"])
    aux_marker = read(paths["auxiliary_completion"])
    require(marker.get("complete") is True and aux_marker.get("complete") is True,
            "Original aggregation and auxiliary completion markers required")
    require(sha(freeze) == marker["source_freeze_sha256"] == primary["source_freeze_sha256"] == base_manifest["source_freeze_sha256"],
            "Primary/base freeze identity")
    require(sha(here / ("native_results_" + run_id + ".json")) == marker["results_sha256"] == primary["result_sha256"],
            "Primary completed result identity")
    require(sha(auxiliary_freeze) == aux_marker["source_freeze_sha256"] == auxiliary["source_freeze_sha256"],
            "Auxiliary freeze identity")
    require(sha(private_aux / "results.json") == aux_marker["result_sha256"] == auxiliary["result_sha256"] and
            sha(private_aux / "path_summaries.jsonl.gz") == aux_marker["paths_sha256"] == auxiliary["paths_sha256"],
            "Auxiliary completed result/path identities (hash-only, not copied)")
    frozen = read(freeze)
    for name, digest in frozen["source_sha256"].items():
        require(sha(here / Path(safe_member(name))) == sha(run / "source_snapshot" / Path(safe_member(name))) == digest,
                "Original frozen numerical auditor remains unchanged")
    aux_frozen = read(auxiliary_freeze)
    for name, digest in aux_frozen["source_sha256"].items():
        require(sha(here / Path(safe_member(name))) == sha(here / ("new_contract_source_snapshot_" + run_id) / Path(safe_member(name))) == digest,
                "Original frozen auxiliary source remains unchanged")
    public = {}
    for suffix, count, filename, tag in (("", 520, "reproduce_public_native_audit.py", "primary"),
                                          ("_new_contract", 240, "reproduce_public_new_contract.py", "auxiliary")):
        directory = here / "results" / (run_id + suffix)
        manifest, report = verify_public(directory, filename, count, "addendum_" + tag + "_reproducer")
        provenance = read(directory / "provenance.json")
        if tag == "primary":
            require(provenance["private_result_sha256"] == primary["result_sha256"], "Public primary/private binding")
        else:
            require(provenance["private_auxiliary_result_sha256"] == auxiliary["result_sha256"], "Public auxiliary/private binding")
        public[tag] = {"directory": directory, "manifest": manifest, "fresh_reproduction": report}
    replay_report = read(paths["packet_replay"])
    require(replay_report.get("status") == "PASS" and replay_report.get("native_audit_steps") == 192 and
            replay_report.get("deterministic_warm_steps") == 768 and not replay_report.get("blocked_access") and
            replay_report.get("original_native_root_used") is False,
            "Released exact two-path packet-only software replay PASS required")
    require(replay_report.get("packet_manifest_sha256") == base_manifest_sha256,
            "Packet replay must bind preserved complete v1 base manifest")
    require(replay_report.get("native_source_sha256") == frozen["native_source_sha256"] and
            replay_report.get("checkpoint_sha256") == frozen["checkpoint_sha256"] and
            bool(replay_report.get("loaded_native_module_origins")), "Packet replay native/model/module provenance")
    cases = replay_report.get("cases", [])
    require({case.get("case") for case in cases} == set(GOLDEN_NAMES) and len(cases) == 2 and
            all(case.get("status") == "PASS" and case.get("terminal_queue_empty") is True and
                case.get("parent_and_config_immutable") is True for case in cases),
            "Both fixed packet software fixtures must pass")
    require(sha(replay / PACKET_REPLAY_WRAPPER) == sha(here / PACKET_REPLAY_WRAPPER) == replay_report["wrapper_sha256"],
            "Explicit passed attempt4 wrapper snapshot/current source identity")
    require(sha(replay / "CLEAN_BUNDLE_REPLAY_PLAN.md") == replay_report["plan_sha256"], "Passed packet replay plan snapshot identity")
    for name in GOLDEN_NAMES:
        require(sha(here / "native_validation/warm_attempt_2" / (name + "_trace.npz")) == replay_report["golden_trace_sha256"][name],
                "Packet replay binds original private golden fixture")
    table_marker = read(paths["descriptive_tables"])
    require(table_marker.get("complete") is True, "Completed descriptive tables required")
    require(table_marker["inputs"]["primary_private_verification_sha256"] == sha(paths["primary_verification"]) and
            table_marker["inputs"]["auxiliary_private_verification_sha256"] == sha(paths["auxiliary_verification"]),
            "Descriptive tables bind current full private verification reports")
    control_diagnostic_gate(here, run_id, paths)
    return paths, public, frozen, aux_frozen


def add_entry(entries, source, destination, category, expected=None):
    destination = safe_member(destination)
    require(destination.parts[0] == str(PREFIX), "Every addendum member uses its distinct top-level prefix")
    require(str(destination) not in entries, "Duplicate addendum destination")
    source = Path(source).resolve()
    require(source.is_file(), "Required addendum source missing: " + str(source))
    require(source.suffix.lower() not in (".pkl", ".pickle", ".joblib", ".zip", ".keras"),
            "No model/checkpoint pickle/archive is added")
    if source.suffix.lower() == ".npz":
        require(category == "private_original_software_fixture" and source.name in {name + "_trace.npz" for name in GOLDEN_NAMES},
                "Only the two original private software NPZ fixtures are allowed")
    digest = sha(source)
    require(expected is None or digest == expected, "Addendum source differs from its verified hash")
    entries[str(destination)] = {"source": source, "category": category, "sha256": digest, "bytes": source.stat().st_size}


def collect_entries(here, run_id, replay, gates, public, frozen, aux_frozen):
    entries = {}
    def add(relative, destination, category, expected=None):
        add_entry(entries, here / relative, PREFIX / destination, category, expected)
    tools = {
        "prepare_private_review_v2_addendum.py": "post_freeze_addendum_builder",
        "prepare_packet_forecast_cache.py": "post_freeze_portable_cache_helper",
        "replay_private_bundle.py": "post_freeze_packet_replay_wrapper",
        "replay_private_bundle_attempt4.py": "released_packet_replay_wrapper_exact_passed_version",
        "validation/verify_native_results.py": "independent_private_verifier_source",
        "validation/verify_new_contract_component.py": "independent_private_verifier_source",
        "validation/verify_sample_core.py": "independent_core_verifier_source",
        "validation/check_portable_cache_wrapper.py": "literal_static_check_source",
        "validation/check_private_v2_builder.py": "literal_static_check_source",
        "public_export.py": "post_freeze_public_export_source",
        "public_export_new_contract.py": "post_freeze_public_export_source",
        "reproduce_public_native_audit.py": "standalone_public_arithmetic_source",
        "reproduce_public_new_contract.py": "standalone_public_arithmetic_source",
        "post_completion_native_tables.py": "post_completion_descriptive_report_source",
        "post_completion_control_diagnostics.py": "post_gate_descriptive_control_ablation_source",
        "audit_native_state_sources.py": "source_only_state_review_helper",
        "validation/audit_native_chronology.py": "source_only_chronology_helper",
        "validation/inventory_software_compute.py": "retained_software_compute_inventory_source",
        "prematerialize_native_forecast_views.py": "historical_cache_preparation_helper",
        "validate_native_support.py": "saved_software_trace_support_helper",
        "validate_runtime_provenance.py": "source_input_provenance_helper",
        "validate_native_adapter.py": "historical_software_validation_helper",
        "validate_native_warm_clone.py": "historical_software_validation_helper",
        "build_public_reproducibility_manifest.py": "historical_packaging_inventory_helper",
        "prepare_private_review_bundle.py": "historical_private_packet_builder",
        "prepare_complete_private_review_bundle.py": "historical_private_packet_builder",
    }
    for relative, category in tools.items():
        add(relative, PurePosixPath("source") / relative, category)
    documents = ("README.md", "PUBLIC_REPRODUCIBILITY.md", "REPRODUCTION_ACCESS_PLAN.md", "PORTABLE_CACHE_PREPARATION.md",
                 "INTERPRETATION_NOTE.md", "POST_RUN_REVIEW_CHECKLIST.md", "REVIEW_BUNDLE_V2_ADDENDUM_PLAN.md",
                 "PRIVATE_V2_ADDENDUM_BUILDER.md", "CLEAN_BUNDLE_REPLAY_PLAN.md", "RUNTIME_TRANSPORT_NOTE.md",
                  "POST_COMPLETION_CONTROL_DIAGNOSTIC_PLAN.md", "native_validation/UNRESTRICTED_NEGATIVE_TAIL_BOUNDARY.md",
                  "native_validation/PACKET_PORTABILITY_VALIDATION.md",
                 "NATIVE_SOFTWARE_VALIDATION_PLAN.md", "NATIVE_WARM_CLONE_VALIDATION_PLAN.md",
                 "validation/README.md", "validation/CHECKPOINT_INTERFACE_REVIEW.md", "validation/PROPOSED_FIXED_PROTOCOL.md",
                 "validation/RUNTIME_AND_POWER_NOTE.md", "validation/support_review.md",
                 "validation/NO_NEW_CLAIMS_REFERENCE_COROLLARY_REVIEW.md", "validation/NATIVE_CHRONOLOGY_REVIEW.md",
                 "validation/NATIVE_EPISODE_STATE_REVIEW.md")
    for relative in documents:
        add(relative, PurePosixPath("documents/current_audit") / relative, "current_addendum_document_not_base_replacement")
    for relative in ("public_reproducibility_manifest.json", "checkpoint_inventory.json", "validation/NATIVE_CHRONOLOGY_STATIC.json",
                     "validation/NATIVE_STATE_STATIC_INVENTORY.json", "exact_sample_only_checks.json", "validation/SAMPLE_CORE_VERIFICATION.json"):
        add(relative, PurePosixPath("verification/source_and_core") / relative, "source_or_core_verification_record")
    inventories = sorted((here / "validation").glob("SOFTWARE_COMPUTE_INVENTORY_*.json"))
    require(bool(inventories), "Latest retained software compute inventory required")
    for path in (inventories[-1], inventories[-1].with_suffix(".md")):
        add_entry(entries, path, PREFIX / "verification/software_inventory" / path.name,
                  "latest_software_inventory_as_of_its_timestamp")
    for key, path in gates.items():
        add_entry(entries, path, PREFIX / "verification/completion_gates" / (key + ".json"), "complete_verified_gate_record")
    for name in ("SOURCE_FREEZE.json", "SELECTION_FREEZE.json", "LAUNCH.json", "development_stage_complete.json", "confirm_stage_complete.json"):
        add("execution/" + run_id + "/" + name, PurePosixPath("provenance/primary") / name, "primary_fixed_metadata_only_no_phase_data")
    for relative, digest in aux_frozen["source_sha256"].items():
        relative = str(safe_member(relative))
        add(relative, PurePosixPath("frozen_auxiliary/current") / relative, "original_frozen_auxiliary_source_or_plan", digest)
        add("new_contract_source_snapshot_" + run_id + "/" + relative,
            PurePosixPath("frozen_auxiliary/original_snapshot") / relative, "original_frozen_auxiliary_snapshot", digest)
    add("NEW_CONTRACT_SOURCE_FREEZE_" + run_id + ".json", "frozen_auxiliary/SOURCE_FREEZE.json", "original_auxiliary_source_freeze")
    for tag, record in public.items():
        directory, manifest = record["directory"], record["manifest"]
        for name, digest in manifest["file_sha256"].items():
            add_entry(entries, directory / name, PREFIX / "public_arithmetic" / tag / name, "complete_public_arithmetic_family_" + tag, digest)
        for name in ("MANIFEST.json", "REPRODUCTION.json"):
            add_entry(entries, directory / name, PREFIX / "public_arithmetic" / tag / name, "complete_public_arithmetic_manifest_or_replay_" + tag)
    tables = here / "report_tables" / run_id
    table_manifest = read(tables / "COMPLETION.json")
    expected_table_names = {"DESCRIPTIVE_RESULTS.json", "all_lower_bounds.csv", "confirmation_variance_ratios.csv", "native_activity.csv",
        "counts.tex", "matched_certification.tex", "variance_ratios.tex", "chosen_controls.tex", "compute.tex",
        "software_validation_compute.tex", "native_activity.tex", "README.md"}
    expected_table_names.update(target + "_" + profile + "_bounds.tex"
        for target in ("primary_alpha0", "primary_alpha1", "auxiliary_alpha0") for profile in ("strict", "vulnerable"))
    require(set(table_manifest["file_sha256"]) == expected_table_names, "Exact generated descriptive artifact whitelist")
    for name, digest in table_manifest["file_sha256"].items():
        member = safe_member(name)
        require(len(member.parts) == 1 and member.suffix.lower() in (".json", ".csv", ".tex", ".md"),
                "Descriptive fragment whitelist; no raw panels or private phase data")
        add_entry(entries, tables / name, PREFIX / "descriptive_reports" / name, "complete_post_run_descriptive_report", digest)
    add_entry(entries, tables / "COMPLETION.json", PREFIX / "descriptive_reports/COMPLETION.json", "descriptive_report_completion")
    diagnostic = here / "control_diagnostics" / run_id
    diagnostic_marker = read(diagnostic / "COMPLETION.json")
    require(set(diagnostic_marker["file_sha256"]) == CONTROL_DIAGNOSTIC_FILES, "Separate exact descriptive control output whitelist")
    for name, digest in diagnostic_marker["file_sha256"].items():
        add_entry(entries, diagnostic / name, PREFIX / "descriptive_control_diagnostics" / name,
                  "post_gate_control_ablation_no_new_confidence_family", digest)
    add_entry(entries, diagnostic / "COMPLETION.json", PREFIX / "descriptive_control_diagnostics/COMPLETION.json",
              "post_gate_descriptive_control_completion")
    fixtures = here / "native_validation/warm_attempt_2"
    for name in GOLDEN_NAMES:
        add_entry(entries, fixtures / (name + "_trace.npz"), PREFIX / "private_software_fixtures" / (name + "_trace.npz"),
                  "private_original_software_fixture")
    for name in ("validation_report.json", "NATIVE_WARM_CLONE_VALIDATION_PLAN.md", "native_engine_adapter.py", "validate_native_warm_clone.py"):
        add_entry(entries, fixtures / name, PREFIX / "private_software_fixtures" / name, "private_original_software_fixture_provenance")
    for name in ("replay_report.json", PACKET_REPLAY_WRAPPER, "CLEAN_BUNDLE_REPLAY_PLAN.md"):
        add_entry(entries, replay / name, PREFIX / "packet_software_replay" / name, "released_two_path_packet_replay_record")
    for directory, report_name in (("calendar_support_attempt_1", "validation_report.json"),
                                   ("forecast_cache_prep_attempt_1", "preparation_report.json"),
                                   ("portable_cache_packet_attempt_1", "preparation_report.json")):
        source_dir = here / "native_validation" / directory
        report = read(source_dir / report_name)
        require(report.get("status") == "PASS", "Historical calendar/cache software validation must pass")
        for path in sorted(source_dir.iterdir()):
            if path.is_file() and (path.name == report_name or path.suffix.lower() in (".py", ".md")):
                add_entry(entries, path, PREFIX / "retained_software_validation" / directory / path.name, "historical_software_source_or_pass_record")
    # Failed software attempts remain provenance; no extra trajectory NPZ is
    # copied beyond the two explicitly authorized golden reference fixtures.
    history_dirs = [here / "native_validation/warm_attempt_1"]
    history_dirs.extend(sorted((here / "native_validation").glob("clean_bundle_replay_attempt_*")))
    for directory in history_dirs:
        if directory.is_dir():
            for path in sorted(directory.iterdir()):
                if path.is_file() and path.suffix.lower() in (".py", ".md", ".json"):
                    add_entry(entries, path, PREFIX / "retained_software_history" / directory.name / path.name,
                              "preserved_software_attempt_source_or_report_no_extra_traces")
    patterns = ("portable_cache_static_attempt_*", "v2_addendum_static_attempt_*", "public_export_software_check_*",
                "new_contract_software_check_*", "post_completion_tables_software_check_*", "control_diagnostics_software_check_*")
    for pattern in patterns:
        for directory in sorted((here / "validation").glob(pattern)):
            require(directory.is_dir(), "Expected literal/static attempt directory")
            # Immediate files only: never traverse synthetic/private fixture trees.
            for path in sorted(directory.iterdir()):
                if path.is_file() and path.suffix.lower() in (".py", ".json", ".md"):
                    add_entry(entries, path, PREFIX / "literal_static_history" / directory.name / path.name,
                              "retained_literal_static_source_or_record")
    for path in sorted((here / "validation/source_versions").glob("*")):
        if path.is_file() and path.suffix.lower() in (".py", ".json", ".md"):
            add_entry(entries, path, PREFIX / "literal_static_history/source_versions" / path.name, "preserved_unfrozen_verifier_history")
    # Current documentation is namespaced and never overwrites base READMEs.
    code, paper = here.parents[1], here.parents[3]
    external_docs = [(code / name, PurePosixPath("code_repository") / name) for name in ("README.md", "DESCRIPTION.md", "CHECKLIST.md")]
    external_docs += [(paper / "github/accounting-options-rl-latex-latex/README.md", PurePosixPath("manuscript_repository/README.md")),
                      (paper / "CODEX_REVIEW.md", PurePosixPath("paper_review/CODEX_REVIEW.md")),
                      (paper / "DEEP_NOVELTY_REVIEW.md", PurePosixPath("paper_review/DEEP_NOVELTY_REVIEW.md"))]
    for path, destination in external_docs:
        if path.is_file():
            add_entry(entries, path, PREFIX / "documents" / destination, "current_context_document_private_addendum_only")
    return entries


def manifest_records(entries):
    return [{"path_in_addendum": name, "category": row["category"], "bytes": row["bytes"],
             "sha256": row["sha256"], "source_path_private": str(row["source"])}
            for name, row in sorted(entries.items())]


def publish_nonreplacing(partial, target):
    require(not target.exists(), "Refuse existing archive publication target")
    if os.name == "nt":
        os.rename(partial, target)
    else:
        os.link(partial, target)
        partial.unlink()


def build(args):
    here, base, base_zip, replay, output, archive, completion = [path.resolve() for path in
        (args.audit_root, args.base_folder, args.base_zip, args.packet_replay_dir, args.output_root, args.archive_path, args.completion_path)]
    run_id = args.run_id
    require(run_id.startswith("run_") and run_id.replace("_", "").isalnum(), "Simple run identifier required")
    require(bool(args.coordinator_release.strip()), "Explicit post-run coordinator release note required")
    completion_partial = completion.with_name(completion.name + ".partial")
    require(not any(path.exists() for path in (output, archive, completion, completion_partial, archive.with_name(archive.name + ".partial"))),
            "All output/archive/completion/partial paths must be fresh; preserve previous attempts")
    require(not inside(output, base) and not inside(base, output) and not inside(replay, output),
            "Output cannot overlap preserved base or passed replay")
    require(all(not inside(path, base) and not inside(path, replay) for path in (output, archive, completion)),
            "No packaging output may modify preserved base or passed replay trees")
    private_root = here / "review_bundle_private"
    require(all(inside(path, private_root) for path in (output, archive, completion)),
            "All packaging outputs remain in ignored local review_bundle_private")
    require(not inside(archive, output) and not inside(completion, output),
            "Archive and external completion seal must remain outside the contents directory")
    # Check readiness before hashing base/public/private endpoint artifacts.
    require_files(list(gate_paths(here, run_id, replay).values()))
    wall, cpu = time.perf_counter(), time.process_time()
    base_manifest, base_members = verify_base(base, base_zip)
    gates, public, frozen, aux_frozen = completion_gates(here, run_id, replay, base_manifest, sha(base / "BUNDLE_MANIFEST.json"))
    entries = collect_entries(here, run_id, replay, gates, public, frozen, aux_frozen)
    require(not (set(entries) & set(base_members)), "Addendum must have zero base ZIP member collisions")
    require(all(str(safe_member(name)).startswith(str(PREFIX) + "/") for name in entries), "Distinct addendum namespace")
    output.mkdir(parents=True, exist_ok=False)
    for name, row in sorted(entries.items()):
        target = output / Path(safe_member(name))
        require(inside(target, output), "Copy destination remains in fresh addendum output")
        target.parent.mkdir(parents=True, exist_ok=True)
        require(sha(row["source"]) == row["sha256"], "Source changed after pre-copy inventory")
        with row["source"].open("rb") as old, target.open("xb") as new:
            shutil.copyfileobj(old, new, 1024 * 1024)
        require(sha(target) == sha(row["source"]) == row["sha256"], "Exact copy and unchanged source")
    generated_root = output / Path(PREFIX)
    readme = """# Local private v2 review addendum

Unpack this archive beside a verified, unchanged complete v1 packet. Every member
uses `native_review_v2_addendum/`; no base member is overwritten. Current docs
are addendum documents rather than replacements for original v1 READMEs. The
base remains the exact155-file complete v1 ZIP bound in MANIFEST.json. This is an
addendum, not a new combined tree or a clean dependency installation.

The source/plan freezes retain their original scope. New verification, export,
software replay, cache and packaging helpers are post-freeze tools. Public
confidence-arithmetic families are separate: primary520 and auxiliary240 each
have95% coverage under their assumptions; joint coverage is at least90%, not95%.
The builder reran only standard-library arithmetic over completed public records.
It made no native query, actor inference, critic query, fit or new scientific draw.

The separate descriptive control diagnostic reconstructs the original frozen
control on the same256 confirmation paths, comparing uncapped statistics and
zero-control versus controlled statistics at the same learned cap. It adds no
confidence limit/family and does not retune or establish superiority. Its source,
plan, exact three output artifacts and completion record remain separate from
the original eighteen-file report. The negative-tail boundary note is an
established fixed-sample limitation with its accounting-specific scope stated.

The two NPZ fixtures under `private_software_fixtures/` are the exact historical
software checks and include actor observations; they are PRIVATE and excluded
from the public numeric exports. Packet replay is exact two-path equivalence in
the existing dependency environment, not all40cases or a fresh dependency install.
The conditional no-new-claims corollary does not establish existence/executability
of a native flat/no-new-claims reference policy. Native full-scenario risk/context
summaries are retained; this is no ordinary-market deployment/profitability claim.

No private scientific execution trees, raw market panels, fitted critic/model
pickles or additional weights are copied here. The preserved private v1 packet
already contains exact engine/checkpoints/configs and authorized runtime inputs.
Original metadata and this private manifest retain absolute paths and require a
separate anonymity review before any anonymous delivery. No upload, push, public
weight release or artifact-license assignment is performed.

MANIFEST.json lists exact contents and source hashes, but is not a completion
seal. Accept the archive only with its separate final COMPLETION JSON, which
binds its ZIP hash after every decompressed member was independently rehashed.
An absent completion record means an incomplete/failed attempt; retain it.
"""
    readme_path = generated_root / "README.md"
    with readme_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(readme)
    manifest = {"schema": "private_native_review_v2_addendum_v1", "status": "CONTENTS_MANIFEST_NOT_COMPLETION_SEAL",
        "created_utc": datetime.now(timezone.utc).isoformat(), "run_id": run_id,
        "coordinator_post_run_release": args.coordinator_release,
        "base_zip_sha256": BASE_ZIP_SHA256, "base_zip_files": 155,
        "base_bundle_manifest_sha256": sha(base / "BUNDLE_MANIFEST.json"),
        "builder_sha256": sha(__file__), "namespace": str(PREFIX), "base_member_collisions": 0,
        "primary_source_freeze_sha256": sha(here / "execution" / run_id / "SOURCE_FREEZE.json"),
        "auxiliary_source_freeze_sha256": sha(here / ("NEW_CONTRACT_SOURCE_FREEZE_" + run_id + ".json")),
        "gate_record_sha256": {key: sha(path) for key, path in gates.items()},
        "copied_artifacts": manifest_records(entries),
        "generated_readme_sha256": sha(readme_path),
        "private_software_npz_files": 2, "public_families": {"primary": 520, "auxiliary": 240},
        "additional_descriptive_control_diagnostic": {"records": 240, "summaries": 12, "confirmation_paths_per_case": 256,
            "new_confidence_limits": 0, "new_native_queries": 0, "retuning": False},
        "fresh_public_arithmetic_replays": {tag: record["fresh_reproduction"] for tag, record in public.items()},
        "native_steps": 0, "actor_calls": 0, "audit_critic_queries": 0, "model_fits": 0,
        "scientific_execution_trees_copied": False, "raw_market_panels_copied": False, "model_pickles_copied": False,
        "privacy": "LOCAL_PRIVATE_ONLY; original metadata and source paths retained; no distribution",
        "inventory_scope": "Latest software inventory is as-of its recorded timestamp; separate packet PASS may postdate it."}
    manifest_path = generated_root / "MANIFEST.json"
    with manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    expected = {name: row["sha256"] for name, row in entries.items()}
    expected[str(PREFIX / "README.md")] = sha(readme_path)
    expected[str(PREFIX / "MANIFEST.json")] = sha(manifest_path)
    actual = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    require(actual == set(expected), "Exact fresh addendum directory member set")
    require(not (actual & set(base_members)), "Generated metadata also has zero v1 collisions")
    archive.parent.mkdir(parents=True, exist_ok=True)
    partial = archive.with_name(archive.name + ".partial")
    with zipfile.ZipFile(partial, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as zipped:
        for name in sorted(expected):
            source = output / Path(safe_member(name))
            require(sha(source) == expected[name], "Addendum file immutable before compression")
            zipped.write(source, arcname=name, compress_type=zipfile.ZIP_STORED if source.suffix.lower() in (".gz", ".npz") else zipfile.ZIP_DEFLATED,
                         compresslevel=1)
    with zipfile.ZipFile(partial) as zipped:
        require(len(zipped.namelist()) == len(set(zipped.namelist())) and set(zipped.namelist()) == set(expected), "Exact ZIP member set")
        for name, digest in expected.items():
            check = hashlib.sha256()
            with zipped.open(name) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    check.update(block)
            require(check.hexdigest() == digest, "Decompressed ZIP member hash: " + name)
    for row in entries.values():
        require(sha(row["source"]) == row["sha256"], "Original source/report changed during packaging")
    require(sha(base_zip) == BASE_ZIP_SHA256, "Original v1 ZIP remains byte-exact")
    # Recheck numerical source identity without parsing private endpoint content.
    for name, digest in frozen["source_sha256"].items():
        require(sha(here / Path(safe_member(name))) == digest, "Frozen numerical source unchanged after packaging")
    for name, digest in aux_frozen["source_sha256"].items():
        require(sha(here / Path(safe_member(name))) == digest, "Frozen auxiliary source unchanged after packaging")
    partial_sha = sha(partial)
    publish_nonreplacing(partial, archive)
    require(sha(archive) == partial_sha, "Published archive is the validated exact bytes")
    final = {"schema": "private_native_review_v2_addendum_completion_v1", "complete": True,
        "status": "PASS_BYTE_EXACT_LOCAL_PRIVATE_ADDENDUM", "created_utc": datetime.now(timezone.utc).isoformat(),
        "archive_path_private": str(archive), "archive_sha256": partial_sha, "archive_bytes": archive.stat().st_size,
        "base_zip_sha256": BASE_ZIP_SHA256, "base_original_unchanged": True,
        "manifest_sha256": sha(manifest_path), "archive_files": len(expected),
        "uncompressed_bytes": sum((output / Path(safe_member(name))).stat().st_size for name in expected),
        "all_decompressed_members_sha256_checked": True, "base_path_collisions": 0,
        "frozen_numerical_sources_unchanged": True, "native_steps": 0, "actor_calls": 0,
        "wall_seconds": time.perf_counter()-wall, "cpu_seconds": time.process_time()-cpu}
    completion.parent.mkdir(parents=True, exist_ok=True)
    # This is the sole COMPLETE seal and the last artifact published.
    payload = (json.dumps(final, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    with completion_partial.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    publish_nonreplacing(completion_partial, completion)
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--audit-root", type=Path, default=HERE)
    parser.add_argument("--run-id", default="run_001")
    parser.add_argument("--base-folder", type=Path, default=HERE / "review_bundle_private/native_review_complete_v1")
    parser.add_argument("--base-zip", type=Path, default=HERE / "review_bundle_private/native_review_complete_v1.zip")
    parser.add_argument("--packet-replay-dir", type=Path)
    parser.add_argument("--output-root", type=Path, default=HERE / "review_bundle_private/native_review_v2_addendum")
    parser.add_argument("--archive-path", type=Path, default=HERE / "review_bundle_private/native_review_v2_addendum.zip")
    parser.add_argument("--completion-path", type=Path, default=HERE / "review_bundle_private/native_review_v2_addendum_COMPLETION.json")
    parser.add_argument("--coordinator-release", default="")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"status": "PREPARED_NOT_EXECUTED_WAITING_POST_RUN_RELEASE",
                          "builder_sha256": sha(__file__), "archive_created": False,
                          "endpoint_reads": 0, "native_steps": 0, "actor_calls": 0}))
        return 0
    if args.packet_replay_dir is None or not args.coordinator_release.strip():
        parser.error("--packet-replay-dir and --coordinator-release are required with --execute")
    return build(args)


if __name__ == "__main__":
    raise SystemExit(main())
