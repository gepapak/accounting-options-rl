"""Stdlib-only literal/static checks; never imports native code or data."""
from __future__ import annotations
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "prepare_packet_forecast_cache.py"


def main():
    output = HERE / "portable_cache_static_attempt_3"
    output.mkdir(exist_ok=False)
    source = SOURCE.read_bytes()
    checks = []
    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)
    tree = ast.parse(source)
    check("syntax", isinstance(tree, ast.Module))
    check("no_run_id_binding", b"run_001" not in source)
    spec = importlib.util.spec_from_file_location("portable_cache_literal_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    check("module_import_has_no_native_or_array_packages",
          not any(name in sys.modules for name in ("native_engine_adapter", "forecast_prior", "numpy", "pandas", "torch")))
    for invalid in ("../escape", "/absolute", "C:/absolute", "C:\\absolute", "x/../../escape"):
        try:
            module.safe_relative(invalid)
        except ValueError:
            checks.append("relative_path_rejected:" + invalid)
        else:
            raise AssertionError("Unsafe relative path accepted")
    check("windows_manifest_path_mapping", module.safe_relative("folder\\input.csv") == Path("folder/input.csv"))
    try:
        module.verify_inputs(output, output.parent, output, output / "MISSING", output / "MISSING")
    except ValueError as error:
        check("containment_rejected_before_any_manifest_read", "inside packet" in str(error))
    else:
        raise AssertionError("Out-of-packet source root accepted")
    with tempfile.TemporaryDirectory(dir=output) as folder:
        root = Path(folder)
        payload = b"a,b\n1,2\n"
        target = root / "cache.csv"
        action, digest = module.atomic_publish(target, payload)
        check("exclusive_atomic_publication", action == "atomic_exclusive_publication" and target.read_bytes() == payload)
        check("published_hash", digest == hashlib.sha256(payload).hexdigest())
        check("partial_removed_only_after_success", not target.with_name(target.name + ".portable-preparation.partial").exists())
        action, _ = module.atomic_publish(target, payload)
        check("complete_existing_verified", action == "existing_complete_bytes_verified")
        try:
            module.atomic_publish(target, b"wrong\n")
        except ValueError:
            checks.append("wrong_existing_target_rejected")
        else:
            raise AssertionError("Wrong target accepted")
        check("wrong_existing_target_preserved", target.read_bytes() == payload)
        failed = root / "failed.csv"
        partial = failed.with_name(failed.name + ".portable-preparation.partial")
        partial.write_bytes(b"previous failed bytes")
        try:
            module.atomic_publish(failed, payload)
        except FileExistsError:
            checks.append("previous_partial_rejected")
        else:
            raise AssertionError("Previous partial overwritten")
        check("previous_partial_preserved", partial.read_bytes() == b"previous failed bytes" and not failed.exists())
        data = root / "market.csv"
        data.write_text("wind,solar,hydro,price,load\n1,2,3,4,5\n", encoding="utf-8")
        check("literal_native_row_count", module.row_count(data) == 1)
        data.write_text("wind,solar,hydro,price,load\n1,nan,3,4,5\n", encoding="utf-8")
        try:
            module.row_count(data)
        except ValueError:
            checks.append("native_dropped_row_fails_closed")
        else:
            raise AssertionError("Dropped-row source accepted")
    held = subprocess.run([sys.executable, "-B", str(SOURCE)], capture_output=True, text=True, check=True)
    status = json.loads(held.stdout)
    check("default_execution_held", status["status"] == "PREPARED_NOT_EXECUTED_WAITING_COORDINATOR_RELEASE")
    check("default_zero_native_counts", all(status[key] == 0 for key in
          ("native_imports", "actor_calls", "environments_created", "audit_rng_draws", "native_steps")))
    missing = subprocess.run([sys.executable, "-B", str(SOURCE), "--execute"], capture_output=True, text=True)
    check("execute_requires_explicit_paths", missing.returncode == 2 and "--packet-root" in missing.stderr)
    forbidden_calls = {"NativeRuntime", "NativeScenario", "NativeEpisode", "load_models", "new_episode", "step", "default_rng", "SeedSequence"}
    names = {node.func.id if isinstance(node.func, ast.Name) else node.func.attr
             for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))}
    check("no_actor_environment_or_rng_calls_in_source", not (forbidden_calls & names))
    report = {"status": "PASS", "scope": "stdlib literal/static checks only; no native queries, imports, inputs, actors, audit RNG or endpoints",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "wrapper_sha256": hashlib.sha256(source).hexdigest(),
              "test_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "checks": checks, "check_count": len(checks), "native_steps": 0, "actor_calls": 0}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "checks": len(checks), "wrapper_sha256": report["wrapper_sha256"]}))


if __name__ == "__main__":
    main()
