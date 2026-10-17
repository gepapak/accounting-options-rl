"""Authorized local source/model copy; no data CSV, inference or experiment."""
from __future__ import annotations
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
CODE=HERE.parents[1]
ENGINE=HERE.parents[3]/"Prototype5"

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):digest.update(block)
    return digest.hexdigest()

def main():
    wall,cpu=time.perf_counter(),time.process_time()
    frozen=json.loads((HERE/"execution/run_001/SOURCE_FREEZE.json").read_text(encoding="utf-8"))
    inventory=json.loads((HERE/"public_reproducibility_manifest.json").read_text(encoding="utf-8"))
    root=HERE/"review_bundle_private/native_review_v1"
    root.mkdir(parents=True,exist_ok=False)
    copied=[]
    def copy(source,destination,category,expected=None):
        source,destination=Path(source),Path(destination)
        assert source.suffix.lower() not in (".csv",".gz",".npz",".nc",".xlsx"),"No raw/panel/feature input copy"
        before=sha(source)
        if expected is not None:assert before==expected,"Frozen source/model identity"
        destination.parent.mkdir(parents=True,exist_ok=True)
        with source.open("rb") as old,destination.open("xb") as new:shutil.copyfileobj(old,new,1024*1024)
        assert sha(destination)==before==sha(source),"Source unchanged and copy byte exact"
        copied.append({"path_in_bundle":destination.relative_to(root).as_posix(),"category":category,
            "bytes":destination.stat().st_size,"sha256":before})
    for name,digest in frozen["native_source_sha256"].items():
        copy(ENGINE/name,root/"Prototype5"/name,"exact_frozen_native_source",digest)
    for name,digest in frozen["checkpoint_sha256"].items():
        copy(ENGINE/name,root/"Prototype5"/name,"exact_frozen_checkpoint_or_config",digest)
    reconstruction=("scripts/build_real_settlement_protocol_v2.py","scripts/build_real_liquidity_volume_protocol_v1.py",
        "scripts/build_ffill_training_protocol.py","scripts/complete_evaluation_window.py","scripts/finalize_protocol_provenance.py",
        "scripts/precompute_hourly_forecast_cache_v2.py","scripts/train_hourly_forecast_models_v2.py",
        "dataset generation/datagenerationforMARLtraining2015-2024.py","dataset generation/datagenerationforevaluationyear2025.py",
        "dataset generation/datagenerationforevaluationyear2025_unseendata_v2.py","dataset generation/wind speed data/convert_wind_nc_to_excel.py")
    for name in reconstruction:copy(ENGINE/name,root/"Prototype5"/name,"additional_reconstruction_source")
    for name in ("LICENSE","LICENSE-docs.md","THIRD_PARTY_LICENSES.md","requirements-engine.txt"):
        copy(CODE/name,root/"provenance"/name,"existing_license_or_runtime_notes")
    for name in ("public_reproducibility_manifest.json","REPRODUCTION_ACCESS_PLAN.md","PUBLIC_REPRODUCIBILITY.md"):
        copy(HERE/name,root/"provenance"/name,"dependency_and_access_provenance")
    copy(HERE/"execution/run_001/SOURCE_FREEZE.json",root/"provenance/SOURCE_FREEZE.json","original_experiment_source_freeze")
    copy(Path(__file__),root/"provenance/prepare_private_review_bundle.py","metadata_copy_builder")
    minimal=[row for row in inventory["market_history_forecast_resources"] if
        not row["path_from_native_engine"].startswith("rolling_past_history") or
        row["path_from_native_engine"].endswith(("history_019.csv","history_020.csv"))]
    assert len(minimal)==10 and sum(row["bytes"] for row in minimal)==53910563
    public_engine_comparison=[]
    for row in inventory["full_frozen_native_module_bundle"]:
        candidate=CODE/"engine"/row["path_from_native_engine"]
        public_engine_comparison.append({"file":row["path_from_native_engine"],"status":
            "missing" if not candidate.is_file() else "byte_identical" if sha(candidate)==row["sha256"] else "different"})
    metadata={"schema":"private_native_review_bundle_v1","created_utc":datetime.now(timezone.utc).isoformat(),
        "status":"COMPLETE_LOCAL_PRIVATE_COPY_NO_DISTRIBUTION","source_freeze_sha256":sha(HERE/"execution/run_001/SOURCE_FREEZE.json"),
        "builder_sha256":sha(__file__),"copied_files":copied,"copied_bytes":sum(row["bytes"] for row in copied),
        "core_native_modules":34,"policy_zip_files":40,"training_config_files":20,
        "additional_reconstruction_source_files":len(reconstruction),"market_history_forecast_csvs_copied":0,
        "minimum_runtime_input_dependencies_not_copied":minimal,"minimum_runtime_input_bytes":53910563,
        "public_engine_comparison":public_engine_comparison,
        "optional_forecast_regeneration_bank_not_copied":{"root":"forecast_models_settlement_hourly_v2/episode_20","files":5,"bytes":1950415},
        "license_scope":"Source identified as authors MIT in existing THIRD_PARTY_LICENSES; policy/data artifact public license needs explicit release labeling. Local private review only; original configs preserved byte-for-byte including paths.",
        "no_simulation_or_inference":True,"wall_seconds":time.perf_counter()-wall,"cpu_seconds":time.process_time()-cpu}
    with (root/"BUNDLE_MANIFEST.json").open("x",encoding="utf-8") as stream:json.dump(metadata,stream,indent=2,sort_keys=True,allow_nan=False);stream.write("\n")
    readme="""# Local private native review bundle

Exact frozen native Python source and all20 investor/battery checkpoint pairs,
with saved training/runtime configs. No market/history/forecast CSV is copied.
See provenance/REPRODUCTION_ACCESS_PLAN.md for exact input identities, sizes,
license evidence, reconstruction scripts and the remaining replay/access gap.
Place Prototype5 at <Paper1>/Prototype5 next to the public audit repository at
<Paper1>/github/accounting-options-rl. Do not replace the running reference tree.
Inputs must be supplied separately and matched to the manifest before replay.
The scripts are included as SOURCE; they were not executed by this bundle job.
This packet is uncompressed, local and private. It is not anonymized or released.
All copied artifact hashes are in BUNDLE_MANIFEST.json; original files unchanged.
"""
    with (root/"README.md").open("x",encoding="utf-8") as stream:stream.write(readme)
    assert not list(root.rglob("*.csv")) and not list(root.rglob("*.npz"))
    print(json.dumps({"status":metadata["status"],"output":str(root),"copied_files":len(copied),"copied_bytes":metadata["copied_bytes"],"csvs_copied":0,"native_steps":0,"actor_calls":0,"wall_seconds":metadata["wall_seconds"]}))

if __name__=="__main__":main()
