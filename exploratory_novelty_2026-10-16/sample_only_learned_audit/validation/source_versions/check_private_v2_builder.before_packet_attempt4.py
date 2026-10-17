"""Literal/static packaging checks only; creates no ZIP and reads no endpoints."""
from __future__ import annotations
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "prepare_private_review_v2_addendum.py"


def main():
    output = HERE / "v2_addendum_static_attempt_2"
    output.mkdir(exist_ok=False)
    payload = SOURCE.read_bytes()
    tree = ast.parse(payload)
    checks = []
    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)
    check("syntax", isinstance(tree, ast.Module))
    spec = importlib.util.spec_from_file_location("private_v2_builder_literal_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    check("import_no_native_or_numeric_packages", not any(name in sys.modules for name in
          ("native_engine_adapter", "forecast_prior", "numpy", "pandas", "torch", "sklearn")))
    for invalid in ("../escape", "/absolute", "C:/absolute", "C:\\absolute", "nested/../../escape"):
        try:
            module.safe_member(invalid)
        except ValueError:
            checks.append("unsafe_member_rejected:" + invalid)
        else:
            raise AssertionError("Unsafe member accepted")
    with tempfile.TemporaryDirectory(dir=output) as directory:
        root = Path(directory)
        gates = module.gate_paths(root, "run_literal", root / "replay")
        check("twelve_completion_gates", len(gates) == 12)
        check("exact_descriptive_diagnostic_output_whitelist", module.CONTROL_DIAGNOSTIC_FILES == {"RESULTS.json","case_diagnostics.csv","README.md"})
        original_read = module.read
        def no_read(path):
            raise AssertionError("Unexpected endpoint/manifest read")
        module.read = no_read
        try:
            module.completion_gates(root, "run_literal", root / "replay", {}, "literal_hash")
        except ValueError as error:
            check("missing_markers_refuse_before_endpoint_reads", "NOT_READY" in str(error))
        finally:
            module.read = original_read
        source = root / "document.md"
        source.write_text("synthetic document\n", encoding="utf-8")
        entries = {}
        module.add_entry(entries, source, module.PREFIX / "documents/document.md", "literal_document")
        record = module.manifest_records(entries)[0]
        check("addendum_namespace", record["path_in_addendum"] == "native_review_v2_addendum/documents/document.md")
        check("individual_source_hash_recorded", record["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest())
        check("category_and_size_recorded", record["category"] == "literal_document" and record["bytes"] == source.stat().st_size)
        try:
            module.add_entry(entries, source, module.PREFIX / "documents/document.md", "duplicate")
        except ValueError:
            checks.append("duplicate_destination_rejected")
        else:
            raise AssertionError("Duplicate destination accepted")
        try:
            module.add_entry({}, source, "Prototype5/document.md", "unsafe_base_collision")
        except ValueError:
            checks.append("base_namespace_rejected")
        else:
            raise AssertionError("Base namespace accepted")
        for suffix in (".pkl", ".pickle", ".joblib", ".zip", ".keras", ".npz"):
            forbidden = root / ("forbidden" + suffix)
            # These are literal bytes, not archives, panels, models or pickles.
            forbidden.write_bytes(b"synthetic forbidden-extension fixture")
            try:
                module.add_entry({}, forbidden, module.PREFIX / forbidden.name, "wrong_category")
            except ValueError:
                checks.append("unapproved_binary_rejected:" + suffix)
            else:
                raise AssertionError("Unapproved binary accepted")
        check("namespace_disjoint_from_simulated_base", not (set(entries) & {"Prototype5/config.py", "README.md", "provenance/SOURCE_FREEZE.json"}))
    held = subprocess.run([sys.executable, "-B", str(SOURCE)], capture_output=True, text=True, check=True)
    held_report = json.loads(held.stdout)
    check("default_builder_held", held_report["status"] == "PREPARED_NOT_EXECUTED_WAITING_POST_RUN_RELEASE")
    check("default_no_archive_no_endpoints", held_report["archive_created"] is False and held_report["endpoint_reads"] == 0)
    missing = subprocess.run([sys.executable, "-B", str(SOURCE), "--execute"], capture_output=True, text=True)
    check("execute_requires_replay_and_coordinator_release", missing.returncode == 2 and "--packet-replay-dir" in missing.stderr)
    calls = {node.func.id if isinstance(node.func, ast.Name) else node.func.attr for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))}
    check("no_native_actor_or_model_calls", not (calls & {"NativeRuntime", "NativeScenario", "new_episode", "step", "predict", "fit", "load_models", "load_model", "pickle_load"}))
    text = payload.decode("utf-8")
    check("completion_gates_precede_output_creation", text.index("gates, public, frozen, aux_frozen = completion_gates") < text.index("output.mkdir(parents=True, exist_ok=False)"))
    check("ZIP_member_validation_precedes_archive_publication", text.index('"Decompressed ZIP member hash: "') < text.index("publish_nonreplacing(partial, archive)"))
    check("completion_seal_last", text.index("publish_nonreplacing(partial, archive)") < text.index("publish_nonreplacing(completion_partial, completion)"))
    check("both_unproved_native_reference_and_private_fixture_scope_documented", "does not establish existence/executability" in text and "PRIVATE and excluded" in text)
    check("diagnostic_source_plan_and_zero_new_family_bound", '"plan_sha256": sha(plan)' in text and 'result.get("new_confidence_limits") == 0' in text and 'result.get("new_native_queries") == 0' in text)
    check("original_eighteen_file_report_whitelist_retained", 'set(table_manifest["file_sha256"]) == expected_table_names' in text)
    report = {"status": "PASS", "scope": "literal/static checks only; no real archive creation, native queries, model inference, endpoint reads or numerical reruns",
              "created_utc": datetime.now(timezone.utc).isoformat(), "builder_sha256": hashlib.sha256(payload).hexdigest(),
              "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "check_count": len(checks), "checks": checks, "archive_created": False, "native_steps": 0, "actor_calls": 0,
              "real_endpoint_reads": 0}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "checks": len(checks), "builder_sha256": report["builder_sha256"]}))


if __name__ == "__main__":
    main()
