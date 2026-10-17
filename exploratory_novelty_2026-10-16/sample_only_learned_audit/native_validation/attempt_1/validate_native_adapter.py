"""Exactly two deterministic native software paths; no statistical endpoints."""
from __future__ import annotations

import contextlib
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def main():
    root = HERE / "native_validation"
    root.mkdir(exist_ok=True)
    attempt_number = len(list(root.glob("attempt_*"))) + 1
    output = root / ("attempt_" + str(attempt_number))
    output.mkdir()
    for name in ("native_engine_adapter.py", "validate_native_adapter.py", "NATIVE_SOFTWARE_VALIDATION_PLAN.md"):
        shutil.copy2(HERE / name, output / name)
    report = {"status": "RUNNING", "purpose": "two deterministic software paths, no statistical endpoints",
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in
                  ("native_engine_adapter.py", "validate_native_adapter.py", "NATIVE_SOFTWARE_VALIDATION_PLAN.md")},
              "cases": []}
    cpu, wall = time.process_time(), time.perf_counter()
    try:
        with ((output / "native_stdout_stderr.log").open("w", encoding="utf-8") as log,
              contextlib.redirect_stdout(log), contextlib.redirect_stderr(log)):
            from native_engine_adapter import NativeRuntime, EpisodeSpec, NativeScenario, ENGINE, jsonable
            runtime_start = time.perf_counter()
            runtime = NativeRuntime()
            runtime_seconds = time.perf_counter() - runtime_start
            choices = [
                ("plain7", ENGINE / "batch_tier_phase_runs/prototype5_mappo_marl_final_v1/seed7/tier1_seed7/final_models"),
                ("feasible7", ENGINE / "Ablations/batch_tier_phase_runs/prototype5_mechanism_ablations_final_v1/feasible_action_full/seed7/tier1_forecast_utilization_seed7/final_models")]
            for name, checkpoint in choices:
                prepare = time.perf_counter()
                scenario = NativeScenario(runtime, EpisodeSpec(str(checkpoint), horizon=96))
                preparation_seconds = time.perf_counter()-prepare
                signs = runtime.np.ones(96, dtype=runtime.np.int8)
                if name == "feasible7":
                    signs[1::2] = -1
                construct = time.perf_counter()
                episode = scenario.new_episode(signs)
                construction_seconds = time.perf_counter()-construct
                original_probe = episode.env._horizon_unit_payoff(100.0, 175.0)
                assert original_probe == 75.0
                roll = time.perf_counter()
                for _ in range(96):
                    episode.step()
                rollout_seconds = time.perf_counter()-roll
                assert episode.env._horizon_unit_payoff(100.0, 175.0) == original_probe
                assert not episode.env._horizon_contracts
                assert episode.summary()["immutable_config_snapshot_unchanged"]
                runtime.assert_sources_unchanged()
                arrays = episode.trace_arrays()
                runtime.np.savez_compressed(output / (name+"_trace.npz"), **arrays)
                report["cases"].append(jsonable({
                    "case": name, "status": "PASS", "metadata": scenario.metadata,
                    "summary": episode.summary(), "runtime_import_seconds": runtime_seconds,
                    "scenario_preparation_seconds": preparation_seconds,
                    "fresh_environment_construction_seconds": construction_seconds,
                    "rollout_including_actor_seconds": rollout_seconds,
                    "seconds_per_step_including_actor": rollout_seconds/96,
                    "checks": ["native actor observation/action contract", "native saved weights used",
                        "cashflow and injection reconciliation on every financial event",
                        "common-sign mature aggregate", "all signs applied after action mapping",
                        "configured/full-context cost and payoff support", "native helper restored outside settlement",
                        "terminal contract queue empty", "fresh immutable config preserved", "native source hashes unchanged"]}))
            report["status"] = "PASS"
    except Exception as error:
        report["status"] = "SOFTWARE_FAILURE"
        report["error"] = repr(error)
        report["traceback"] = traceback.format_exc()
    report["cpu_seconds"] = time.process_time()-cpu
    report["wall_seconds"] = time.perf_counter()-wall
    (output / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "completed_software_paths": len(report["cases"]),
                      "cpu_seconds": report["cpu_seconds"], "wall_seconds": report["wall_seconds"],
                      "output": str(output)}))
    if report["status"] != "PASS":
        print(report["traceback"])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
