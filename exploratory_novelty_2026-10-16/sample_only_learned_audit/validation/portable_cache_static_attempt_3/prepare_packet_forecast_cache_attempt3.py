"""Portable deterministic forecast-view preparation; no actors or environments.

Default invocation prints PREPARED_NOT_EXECUTED. Use --execute only after the
ongoing native experiment has ended and its coordinator releases this helper.
Frozen adapter/engine bytes are never changed. Existing installed dependencies
are required; this is neither a clean install nor a new scientific replication.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import time
import traceback

sys.dont_write_bytecode = True
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

AUDIT_RELATIVE = Path("github/accounting-options-rl/exploratory_novelty_2026-10-16/sample_only_learned_audit")
TARGET_NAME = "precomputed_forecasts_native_view.csv"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inside(path, root):
    return Path(path).resolve().is_relative_to(Path(root).resolve())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_relative(name):
    relative = PurePosixPath(str(name).replace("\\", "/"))
    require(not relative.is_absolute() and relative.parts and
            all(part not in (".", "..") and ":" not in part for part in relative.parts),
            "Unsafe manifest relative path: " + str(name))
    return Path(*relative.parts)


def checked_path(root, name):
    path = (root / safe_relative(name)).resolve()
    require(inside(path, root), "Manifest path escapes explicit root")
    return path


def atomic_publish(target, expected):
    """Keep all existing files intact; publish exclusively, preserving failures."""
    target = Path(target)
    expected_sha = hashlib.sha256(expected).hexdigest()
    if target.exists():
        require(target.is_file() and sha(target) == expected_sha,
                "Existing cache is incomplete or differs; it was not overwritten: " + str(target))
        return "existing_complete_bytes_verified", expected_sha
    partial = target.with_name(target.name + ".portable-preparation.partial")
    # Refuse a previous failed publication rather than erase its evidence.
    with partial.open("xb") as stream:
        stream.write(expected)
        stream.flush()
        os.fsync(stream.fileno())
    require(sha(partial) == expected_sha, "Partial-file byte verification failed")
    try:
        # Windows rename refuses an existing target and is atomic within this
        # directory. POSIX rename replaces targets, so use a non-replacing hard
        # link there instead. Never replace an existing target on either OS.
        if os.name == "nt":
            os.rename(partial, target)
        else:
            os.link(partial, target)
    except FileExistsError:
        require(sha(target) == expected_sha,
                "Concurrent cache differs; both existing file and partial retained")
        partial.unlink()
        return "concurrent_complete_bytes_verified", expected_sha
    if partial.exists():
        partial.unlink()
    require(sha(target) == expected_sha, "Published-file byte verification failed")
    return "atomic_exclusive_publication", expected_sha


def verify_inputs(packet, audit, engine, freeze_path, manifest_path):
    require(inside(audit, packet) and inside(engine, packet), "Explicit sources must be inside packet root")
    require(inside(freeze_path, packet) and inside(manifest_path, packet), "Provenance inputs must be packet-local")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    require(manifest.get("schema") == "complete_private_native_review_bundle_v1",
            "A complete private v1 base manifest is required")
    require(sha(freeze_path) == manifest.get("source_freeze_sha256"),
            "Source-freeze bytes differ from complete packet manifest")
    require(len(frozen["native_source_sha256"]) == 34 and len(frozen["source_sha256"]) == 11,
            "Unexpected native/frozen-auditor source set")
    checked = {}
    for label, root, mapping in (("native", engine, frozen["native_source_sha256"]),
                                 ("auditor", audit, frozen["source_sha256"]),
                                 ("checkpoint_config", engine, frozen["checkpoint_sha256"])):
        for name, digest in mapping.items():
            path = checked_path(root, name)
            require(path.is_file() and sha(path) == digest, "Frozen hash mismatch: " + str(path))
            checked[label + "/" + str(name).replace("\\", "/")] = digest
    runtime = [row for row in manifest["copied_artifacts"]
               if row["category"] == "exact_frozen_runtime_csv"]
    require(len(runtime) == 10 and len({row["path_in_bundle"] for row in runtime}) == 10,
            "Exactly ten unique frozen runtime CSVs are required")
    runtime_paths = {}
    for row in runtime:
        relative = PurePosixPath(row["path_in_bundle"])
        require(relative.parts[0] == "Prototype5", "Unexpected native-input manifest prefix")
        path = checked_path(engine, str(PurePosixPath(*relative.parts[1:])))
        require(path.is_file() and path.stat().st_size == row["bytes"] and sha(path) == row["sha256"],
                "Frozen runtime-input hash/size mismatch: " + str(path))
        runtime_paths[path.relative_to(engine).as_posix()] = row["sha256"]
    return frozen, checked, runtime_paths


def row_count(path):
    # Match native generator.load_energy_data's dropna(required) row contract.
    # Frozen input should have no dropped rows; fail closed otherwise. No price
    # normalization, MW conversion, actor or environment is needed for a count.
    import math
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = ("wind", "solar", "hydro", "price", "load")
        require(all(name in (reader.fieldnames or []) for name in required), "Missing native market columns")
        total = 0
        for row in reader:
            require(all(math.isfinite(float(row[name])) for name in required),
                    "Market input would alter native row count; unsupported packet")
            total += 1
    return total


def load_exact_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "Cannot load explicit frozen module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def execute(args):
    packet, audit, engine, freeze, manifest, output, cache, provenance = [
        path.resolve() for path in (args.packet_root, args.audit_root, args.engine_root,
                                   args.source_freeze, args.input_manifest, args.output_root,
                                   args.cache_root, args.provenance_output)]
    require(packet.is_dir() and audit.is_dir() and engine.is_dir(), "Missing explicit packet/source root")
    require(inside(manifest, packet) and inside(freeze, packet), "Provenance inputs must be packet-local")
    require(cache == output / "runtime_cache" / "forecast_views",
            "Cache must equal output-root/runtime_cache/forecast_views, matching adapter HERE semantics")
    require(not inside(output, engine) and not inside(output, audit) and
            not inside(provenance, engine) and not inside(provenance, audit),
            "Writes must not touch frozen native/auditor trees")
    require(not inside(engine, output) and not inside(audit, output), "Output must not contain frozen sources")
    provenance.mkdir(parents=True, exist_ok=False)
    report = {"status": "RUNNING", "started_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "exact deterministic forecast views only; no scientific outcomes or native episodes",
              "wrapper_sha256": sha(__file__), "packet_manifest_sha256": sha(manifest),
              "source_freeze_sha256": sha(freeze), "contexts": [], "blocked_access": [],
              "actor_calls": 0, "environments_created": 0, "audit_rng_draws": 0,
              "native_steps": 0, "audit_steps": 0,
              "path_roots": {"packet": str(packet), "auditor": str(audit), "engine": str(engine),
                             "output": str(output), "cache": str(cache), "provenance": str(provenance)},
              "limitation": "existing dependency environment; source/input preparation only, not clean install or full-run replay"}
    wall, cpu = time.perf_counter(), time.process_time()
    shutil.copy2(__file__, provenance / Path(__file__).name)
    try:
        frozen, source_hashes, runtime_hashes = verify_inputs(packet, audit, engine, freeze, manifest)
        report["verified_source_model_config_sha256"] = source_hashes
        report["runtime_input_sha256"] = runtime_hashes
        # No native directory enters normal import search. The only two native
        # source files executed below are adapter (stdlib-only module import)
        # and forecast_prior (NumPy + lazy pandas loader), via explicit paths.
        native_names = {Path(name).stem for name in frozen["native_source_sha256"]}
        require(not (native_names & set(sys.modules)), "Native module already imported before isolation")
        dependency_roots = {Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()}
        if os.environ.get("WINDIR"):
            dependency_roots.add(Path(os.environ["WINDIR"]).resolve())
        sys.path[:] = [entry for entry in sys.path if entry and
                       any(inside(Path(entry), root) for root in dependency_roots)]
        reads = [packet, output, provenance, *dependency_roots]
        writes = [cache, provenance]
        wrapper_source = Path(__file__).resolve()

        def guard(event, event_args):
            if event not in ("open", "os.listdir", "os.scandir", "os.mkdir", "os.remove", "os.rename", "os.link"):
                return
            if event == "os.rename" or event == "os.link":
                paths = event_args[:2]
            else:
                paths = event_args[:1]
            for value in paths:
                if value is None or isinstance(value, int):
                    continue
                if isinstance(value, bytes):
                    value = os.fsdecode(value)
                if not isinstance(value, (str, os.PathLike)) or str(value).startswith("<"):
                    continue
                path = Path(value).resolve()
                writing = event in ("os.mkdir", "os.remove", "os.rename", "os.link")
                if event == "open":
                    mode = event_args[1] if len(event_args) > 1 else None
                    flags = event_args[2] if len(event_args) > 2 else 0
                    writing = (isinstance(mode, str) and any(mark in mode for mark in "wax+")) or bool(
                        isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
                # Allow only ancestors needed by mkdir(parents=True) for cache.
                allowed_write = any(inside(path, root) for root in writes) or (
                    event == "os.mkdir" and inside(cache, path) and inside(path, output))
                allowed_read = path == wrapper_source or any(inside(path, root) for root in reads)
                if not allowed_read or (writing and not allowed_write):
                    report["blocked_access"].append({"event": event, "path": str(path), "write": writing})
                    raise PermissionError("Explicit packet/cache-only access required: " + str(path))
        sys.addaudithook(guard)
        import pandas as pd
        import numpy as np
        report["dependency_versions"] = {"python": sys.version, "pandas": pd.__version__, "numpy": np.__version__}
        adapter = load_exact_module("portable_frozen_native_adapter", audit / "native_engine_adapter.py")
        adapter.ENGINE, adapter.HERE = engine, output  # in-memory path overrides only
        forecast = load_exact_module("portable_frozen_forecast_prior", engine / "forecast_prior.py")
        cache.mkdir(parents=True, exist_ok=True)
        for zone in ("original", "v2"):
            stem = "unseendata" + ("_v2" if zone == "v2" else "")
            market = engine / "evaluation_dataset_ffill" / (stem + ".csv")
            source_rows = row_count(market)
            last = source_rows - 480
            require(last >= 0, "Native context too short")
            suffix = "full" if zone == "original" else "unseendata_v2"
            directory = engine / "forecast_cache_settlement_hourly_v2" / "forecast_cache_eval_episode20_2025" / (
                "forecast_cache_eval_episode20_2025-" + suffix)
            files = [path for path in directory.glob("precomputed_forecasts_*.csv") if "metadata" not in path.name]
            require(len(files) == 1, "Native forecast source is not unique")
            source = files[0].resolve()
            require(source.relative_to(engine).as_posix() in runtime_hashes,
                    "Forecast source is absent from the exact ten-file runtime manifest")
            features = forecast.load_forecast_prior_features(str(directory), 0)
            require(features is not None and len(features) == source_rows, "Forecast/context row mismatch")
            for quantile, start in enumerate(np.linspace(0, last, 4, dtype=int)):
                start = int(start)
                view = features[start:start + 480]
                require(len(view) == 480, "Incomplete native view")
                key = adapter.state_sha({"source": str(source), "start": start, "H": 480})[:20]
                destination = cache / key
                destination.mkdir(parents=True, exist_ok=True)
                expected = pd.DataFrame(view, columns=forecast.FORECAST_PRIOR_ALL_COLS).to_csv(index=False).encode("utf-8")
                action, digest = atomic_publish(destination / TARGET_NAME, expected)
                report["contexts"].append({"context_id": len(report["contexts"]), "zone": zone,
                    "quantile_id": quantile, "start_row": start, "source_rows": source_rows,
                    "rows": 480, "columns": len(forecast.FORECAST_PRIOR_ALL_COLS),
                    "forecast_source": source.relative_to(engine).as_posix(),
                    "forecast_source_sha256": runtime_hashes[source.relative_to(engine).as_posix()],
                    "cache_key": key, "cache_path": str(destination / TARGET_NAME),
                    "cache_sha256": digest, "action": action})
        require(len(report["contexts"]) == 8, "Exactly eight native views required")
        # Recheck all sources/runtime inputs after preparation, not sample values.
        verify_inputs(packet, audit, engine, freeze, manifest)
        loaded_native = {}
        for name, module in tuple(sys.modules.items()):
            origin = getattr(module, "__file__", None)
            if origin and inside(Path(origin), engine):
                loaded_native[name] = Path(origin).resolve().relative_to(engine).as_posix()
        require(set(loaded_native.values()) == {"forecast_prior.py"},
                "Unexpected engine/actor/environment import during cache-only preparation")
        report["loaded_native_source_modules"] = loaded_native
        report["adapter_in_memory_roots"] = {"ENGINE": str(adapter.ENGINE), "HERE": str(adapter.HERE)}
        require(not report["blocked_access"], "Blocked fallback/write attempt was observed")
        report["status"] = "PASS"
    except BaseException as error:
        report["status"] = "SOFTWARE_FAILURE"
        report["error"] = repr(error)
        report["traceback"] = traceback.format_exc()
    report["wall_seconds"] = time.perf_counter() - wall
    report["cpu_seconds"] = time.process_time() - cpu
    with (provenance / "preparation_report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps({key: report[key] for key in ("status", "native_steps", "actor_calls", "wall_seconds", "cpu_seconds")}))
    return 0 if report["status"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    for name in ("packet-root", "audit-root", "engine-root", "source-freeze", "input-manifest",
                 "output-root", "cache-root", "provenance-output"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"status": "PREPARED_NOT_EXECUTED_WAITING_COORDINATOR_RELEASE",
                          "wrapper_sha256": sha(__file__), "native_imports": 0, "actor_calls": 0,
                          "environments_created": 0, "audit_rng_draws": 0, "native_steps": 0}))
        return 0
    required = ("packet_root", "audit_root", "engine_root", "source_freeze", "input_manifest",
                "output_root", "cache_root", "provenance_output")
    for name in required:
        if getattr(args, name) is None:
            parser.error("--" + name.replace("_", "-") + " is required with --execute")
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
