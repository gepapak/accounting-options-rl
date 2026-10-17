"""Read timestamp/feature provenance only; no native module imports or queries."""
from __future__ import annotations
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
ENGINE = AUDIT.parents[3] / "Prototype5"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    report = {"status": "SOURCE_METADATA_REVIEW", "created_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "timestamps, forecast metadata and source provenance only; no native queries, actor inference or scientific endpoint reads",
              "native_steps": 0, "actor_calls": 0, "input_chronology": {}, "source_sha256": {}}
    paths = ["rolling_past_history_dataset_ffill/history_019.csv", "rolling_past_history_dataset_ffill/history_020.csv",
             "evaluation_dataset_ffill/unseendata.csv", "evaluation_dataset_ffill/unseendata_v2.csv"]
    for relative in paths:
        path = ENGINE / relative
        stamps, risks = [], set()
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                stamps.append(row["timestamp"])
                if "risk" in row:
                    risks.add(row["risk"])
        record = {"rows": len(stamps), "first_timestamp": stamps[0], "last_timestamp": stamps[-1], "sha256": sha(path)}
        if "evaluation_dataset" in relative:
            record["risk_unique_count"] = len(risks)
            record["windows"] = [{"start_row": start, "warm_start": stamps[start],
                "audit_start": stamps[start+384], "audit_end": stamps[start+479]}
                for start in (0, 12944, 25888, 38832)]
        report["input_chronology"][relative] = record
    metadata_path = ENGINE / "forecast_models_settlement_hourly_v2/episode_20/price_short_experts/ANN/ann_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    keys = ("training_start", "training_end", "validation_end", "test_end", "look_back", "horizon_steps",
            "forecast_information_set", "ann_input_includes_entry_price", "entry_reference_column")
    report["forecast_metadata"] = {key: metadata[key] for key in keys}
    report["forecast_metadata_sha256"] = sha(metadata_path)
    for relative in ("environment.py", "evaluation.py", "generator.py", "forecast_prior.py", "forecast_price_experts.py", "risk.py",
                     "observation_builder.py", "scripts/precompute_hourly_forecast_cache_v2.py",
                     "scripts/build_ffill_training_protocol.py", "scripts/complete_evaluation_window.py",
                     "dataset generation/datagenerationforevaluationyear2025.py",
                     "dataset generation/datagenerationforevaluationyear2025_unseendata_v2.py"):
        report["source_sha256"][relative] = sha(ENGINE / relative)
    report["source_sha256"]["frozen_auditor/native_engine_adapter.py"] = sha(AUDIT / "native_engine_adapter.py")
    report["conclusion"] = "2024 normalization prime and declared ANN fit chronology precede all2025 contexts; full-scenario native risk/context summaries preclude a broad real-time leak-free-interface claim; fresh hidden sign independence is a distinct assumption."
    with (HERE / "NATIVE_CHRONOLOGY_STATIC.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
    print(json.dumps({"status": report["status"], "native_steps": 0, "actor_calls": 0, "inputs": 4}))


if __name__ == "__main__":
    main()
