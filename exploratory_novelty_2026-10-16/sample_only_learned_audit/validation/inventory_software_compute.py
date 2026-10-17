"""Inventory retained software-validation compute, never statistical endpoints.

Reads only native_validation reports/source snapshots and mathematical-test
reports. No execution/ statistical result or episode-row file is opened. Native
counts are distinguished from timing measurements and incomplete evidence.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
VALIDATION = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timing(data):
    return dict(cpu_seconds=data.get("cpu_seconds"),
                wall_seconds=data.get("wall_seconds", data.get("elapsed_seconds")),
                wall_source_field="wall_seconds" if "wall_seconds" in data else "elapsed_seconds(perf_counter)" if "elapsed_seconds" in data else None)


def record(path, kind, warm, audit, count_basis, unknown=False):
    data = read(path)
    return dict(report=path.relative_to(HERE).as_posix(), report_sha256=sha(path),
        status=data["status"], category=kind, completed_native_warm_transitions=warm,
        completed_native_audit_transitions=audit, learned_actor_forward_calls=2*(warm+audit),
        count_basis=count_basis, unknown_additional_native_work=unknown, **timing(data))


def inventory():
    root = HERE/"native_validation"
    rows = []
    cold = root/"attempt_1/validation_report.json"
    data = read(cold)
    assert len(data["cases"]) == 2 and all(case["summary"]["steps"] == 96 for case in data["cases"])
    assert all(case["metadata"]["spec"].get("burn_in", 0) == 0 for case in data["cases"])
    rows.append(record(cold, "two completed cold deterministic software paths", 0, 192,
        "audit steps recorded in both case summaries; two learned checkpoints per native step, source-derived actor count"))
    failed = root/"warm_attempt_1/validation_report.json"
    data = read(failed)
    assert data["status"] == "SOFTWARE_FAILURE" and len(data["cases"]) == 1
    first = data["cases"][0]
    assert first["summary"]["steps"] == 96 and first["metadata"]["burn_in"]["native_steps"] == 384
    adapter = (failed.parent/"native_engine_adapter.py").read_text(encoding="utf-8")
    validator = (failed.parent/"validate_native_warm_clone.py").read_text(encoding="utf-8")
    assert "digest = native_state_sha(env)" in data["traceback"]
    assert adapter.index("for _ in range(self.spec.burn_in):") < adapter.index("digest = native_state_sha(env)")
    assert 'feasible7_vulnerable' in validator and validator.index('plain7_strict') < validator.index('feasible7_vulnerable')
    assert "Unsupported address-bearing native buffer" in data["error"]
    rows.append(record(failed, "failed second-case fingerprint after completed warm loop", 768, 96,
        "first case explicitly records384warm+96audit; traceback reaches second case native_state_sha AFTER its384-step loop. Second384 warm source/control-flow-derived, not separately timed. Both actors per step source-derived."))
    rows[-1]["failure_stage"] = "second case warm fingerprint; zero second-case audit steps"
    passed = root/"warm_attempt_2/validation_report.json"
    data = read(passed)
    assert len(data["cases"]) == 2
    assert all(c["summary"]["steps"] == 96 and c["metadata"]["burn_in"]["native_steps"] == 384 for c in data["cases"])
    rows.append(record(passed, "two completed warm-clone deterministic software paths", 768, 192,
        "two case summaries each explicitly record384warm+96audit; two learned actors per step source-derived"))
    calendar = root/"calendar_support_attempt_1/validation_report.json"
    data = read(calendar)
    assert data["deterministic_native_warm_transitions"] == 768 and data["new_audit_transitions"] == 0
    rows.append(record(calendar, "fresh initial-state support plus reused old trace arithmetic", 768, 0,
        "report explicitly records768newwarm and0newaudit; reused traces add ZERO native queries; actor count source-derived"))
    forecast = root/"forecast_cache_prep_attempt_1/preparation_report.json"
    data = read(forecast)
    assert data["native_steps"] == data["audit_steps"] == data["actor_forward_calls"] == 0
    rows.append(record(forecast, "deterministic forecast cache materialization only", 0, 0,
        "report explicitly records zero native steps and zero actor calls; parsing/import/cache CPU still measured"))
    guards, unique_metadata = [], set()
    for path in sorted(root.glob("runtime_guard_*/provenance_report.json")):
        data = read(path)
        assert "financial outcomes never loaded" in data["scope"]
        checks = data["checks"]
        assert all(item["financial_outcome_rows_read"] is False for item in checks)
        unique_metadata.update((item["case_id"], item["phase"]) for item in checks)
        guards.append(dict(report=path.relative_to(HERE).as_posix(), report_sha256=sha(path),
            status=data["status"], metadata_checks=len(checks), **timing(data)))
        rows.append(record(path, "metadata/input hashing only; no financial outcome rows", 0, 0,
            "source performs no native calls; reports completed-metadata checks, not newly generated episodes"))
    mathematical = []
    exact = HERE/"exact_sample_only_checks.json"
    exact_data = read(exact)
    assert exact_data["monte_carlo_draws"] == exact_data["learners_trained"] == 0
    assert exact_data["floor_certificate_cases"] == len(exact_data["floor_cases"]) == 864
    assert exact_data["terminal_certificate_cases"] == len(exact_data["terminal_cases"]) == 81
    # Published counts are fixed identities, not simulated whole episodes.
    mathematical.append(dict(report=exact.relative_to(HERE).as_posix(), report_sha256=sha(exact),
        category="exact Fraction identities", native_transitions=0, learned_actor_forward_calls=0,
        floor_certificate_cases=exact_data["floor_certificate_cases"], terminal_cases=exact_data["terminal_certificate_cases"], timing=timing(exact_data),
        count_basis="exact enumerated mathematical cases; no native engine"))
    core = VALIDATION/"SAMPLE_CORE_VERIFICATION.json"
    core_data = read(core)
    assert core_data["new_simulator_steps"] == core_data["new_checkpoint_inference"] == core_data["new_fitted_predictors"] == 0
    mathematical.append(dict(report=core.relative_to(HERE).as_posix(), report_sha256=sha(core),
        category="retained current core arithmetic verification", checks=core_data["checks"],
        native_transitions=0, learned_actor_forward_calls=0, timing=timing(core_data),
        count_basis="stored software traces are reused; current report is not a complete invocation/time ledger"))
    for path in sorted(VALIDATION.glob("EARLY_PHASE_VERIFICATION_*.json")):
        data = read(path)
        assert data["new_simulator_steps"] == data["new_actor_inference"] == data["new_fits"] == 0
        mathematical.append(dict(report=path.relative_to(HERE).as_posix(), report_sha256=sha(path),
            category="saved development-record arithmetic only", checks=data["checks"],
            native_transitions=0, learned_actor_forward_calls=0, timing=timing(data)))
    for path in sorted(VALIDATION.glob("*_software_check_*/verification.json")):
        data = read(path)
        mathematical.append(dict(report=path.relative_to(HERE).as_posix(), report_sha256=sha(path),
            category="fixed-literal/source/gate or synthetic arithmetic software checks", native_transitions=0,
            learned_actor_forward_calls=0, literal_checks=data.get("checks", data.get("literal_checks")),
            synthetic_reproducer_checks=data.get("synthetic_reproducer_checks"), timing=timing(data)))
    warm = sum(r["completed_native_warm_transitions"] for r in rows)
    audit = sum(r["completed_native_audit_transitions"] for r in rows)
    measured_records = rows+[dict(**item["timing"]) for item in mathematical]
    cpu = [r["cpu_seconds"] for r in measured_records if r.get("cpu_seconds") is not None]
    wall = [r["wall_seconds"] for r in measured_records if r.get("wall_seconds") is not None]
    return dict(status="READ-ONLY SOFTWARE INVENTORY", created_utc=datetime.now(timezone.utc).isoformat(),
        inventory_source_sha256=sha(__file__), retained_native_attempts=rows, mathematical_checks=mathematical,
        metadata_guard_checks=sum(r["metadata_checks"] for r in guards), unique_metadata_phase_records=len(unique_metadata),
        completed_native_work=dict(warm_transitions=warm, audit_transitions=audit, native_transitions=warm+audit,
            learned_actor_forward_calls=sum(r["learned_actor_forward_calls"] for r in rows)),
        reported_timing_subtotals=dict(cpu_seconds=sum(cpu), cpu_records=len(cpu), wall_seconds=sum(wall), wall_records=len(wall),
            scope="sum of retained reports with these fields; NOT complete total CPU or elapsed wall time; unknown/unrecorded invocations excluded"),
        pending_packet_replay=dict(status="HELD NOT EXECUTED", planned_native_warm_transitions=768,
            planned_native_audit_transitions=192, planned_total_native_transitions=960, planned_learned_actor_forward_calls=1920,
            executed_native_transitions_included_here=0),
        exclusions=["running primary statistical experiment planned1474560native steps is separate and not read here",
            "same-path auxiliary analysis adds no native queries", "earlier study/novelty simulations and historical RL training are outside this software-stage inventory",
            "report generation, source reads, syntax checks, unlogged arithmetic invocations and metadata guards with no timers have unknown CPU/wall",
            "failed statistical-pipeline startup is outside this software-stage inventory; empty logs are not a measured CPU/wall total",
            "source-derived actor counts mean two learned checkpoints(investor+battery) per completednative step, not all rule-controller operations"],
        statistical_financial_endpoints_read=False, new_native_queries=0, frozen_files_modified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.parse_args()
    result = inventory()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = VALIDATION/("SOFTWARE_COMPUTE_INVENTORY_"+stamp)
    with stem.with_suffix(".json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")
    lines = ["# Retained sample-only software-validation compute", "", result["created_utc"], "",
        "Read-only inventory; no statistical financial endpoint or running episode record was opened.", "",
        "| Retained attempt | Completed warm | Completed audit | Learned actor calls | CPU s | Wall s |",
        "|---|---:|---:|---:|---:|---:|"]
    for row in result["retained_native_attempts"]:
        fmt = lambda x: "unrecorded" if x is None else f"{x:.6f}"
        lines.append(f"| {row['report']} | {row['completed_native_warm_transitions']} | {row['completed_native_audit_transitions']} | {row['learned_actor_forward_calls']} | {fmt(row['cpu_seconds'])} | {fmt(row['wall_seconds'])} |")
    lines += ["", "All mathematical/arithmetic rows below add ZERO native transitions and ZERO learned actor calls.", "",
        "| Retained mathematical/software check | Assertions/cases | CPU s | Wall s |", "|---|---|---:|---:|"]
    for row in result["mathematical_checks"]:
        count = str(row.get("checks", row.get("literal_checks", "not separately counted")))
        if "floor_certificate_cases" in row:
            count = f"{row['floor_certificate_cases']} floor + {row['terminal_cases']} terminal exact cases"
        if row.get("synthetic_reproducer_checks") is not None:
            count += f" literal; {row['synthetic_reproducer_checks']} synthetic arithmetic checks"
        fmt = lambda x: "unrecorded" if x is None else f"{x:.6f}"
        lines.append(f"| {row['report']} | {count} | {fmt(row['timing']['cpu_seconds'])} | {fmt(row['timing']['wall_seconds'])} |")
    total = result["completed_native_work"]
    lines += ["", f"Retained completed software work: {total['warm_transitions']} warm + {total['audit_transitions']} audit = {total['native_transitions']} native transitions; {total['learned_actor_forward_calls']} learned actor calls.", "",
        "The failed first warm attempt completed one384+96 path, then the second384-step warm loop before fingerprint failure. The latter count follows the retained traceback and source control flow; there were zero second-case audit steps. Its per-case time is not independently recorded.", "",
        "Calendar support validation reused saved audit traces: count768fresh warm steps and ZERO fresh audit steps. Forecast-cache materialization and metadata/arithmetic checks add no native transitions, though some measured CPU/wall is nonzero.", "",
        "The timing subtotal in JSON sums only records containing timers; it is neither a complete compute ledger nor total elapsed time. Missing timers and unlogged/overwritten arithmetic invocations remain unknown. The planned primary1474560native transitions are separate. No historical learner training is included.", "",
        f"Metadata guards inspected {result['metadata_guard_checks']} records across retained invocations, covering {result['unique_metadata_phase_records']} unique phase records. Repeated inspection is not new episode generation.", "",
        "Packet replay: HELD, NOT EXECUTED. Planned768warm+192audit=960native steps and1920actor calls are excluded from executed totals.", "",
        "Every input report and this inventory source are hashed in the public-safe JSON. No observations, raw panels, private absolute paths, weights or financial profits are copied."]
    with stem.with_suffix(".md").open("x", encoding="utf-8") as stream: stream.write("\n".join(lines)+"\n")
    print(json.dumps(dict(status=result["status"], output=stem.name, completed_native_work=total,
        statistical_financial_endpoints_read=False, pending_packet_replay="HELD NOT EXECUTED")))


if __name__ == "__main__":
    main()
