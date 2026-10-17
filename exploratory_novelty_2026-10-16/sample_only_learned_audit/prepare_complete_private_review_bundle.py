"""Authorized complete local private packet; exact copies, no model/data execution."""
from __future__ import annotations
from datetime import datetime,timezone
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ENGINE=HERE.parents[3]/"Prototype5"

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):digest.update(block)
    return digest.hexdigest()

def main():
    wall,cpu=time.perf_counter(),time.process_time()
    parent=HERE/"review_bundle_private"
    source_subset=parent/"native_review_v1"
    subset=json.loads((source_subset/"BUNDLE_MANIFEST.json").read_text(encoding="utf-8"))
    frozen=json.loads((HERE/"execution/run_001/SOURCE_FREEZE.json").read_text(encoding="utf-8"))
    root=parent/"native_review_complete_v1";root.mkdir(parents=True,exist_ok=False)
    copied=[]
    def copy(source,target,category,expected=None):
        source,target=Path(source),Path(target);before=sha(source)
        if expected is not None:assert before==expected,"Original frozen artifact identity"
        target.parent.mkdir(parents=True,exist_ok=True)
        with source.open("rb") as old,target.open("xb") as new:shutil.copyfileobj(old,new,1024*1024)
        assert sha(target)==before==sha(source),"Copy exact and original unchanged"
        copied.append({"path_in_bundle":target.relative_to(root).as_posix(),"category":category,
            "bytes":target.stat().st_size,"sha256":before})
    for row in subset["copied_files"]:
        copy(source_subset/row["path_in_bundle"],root/row["path_in_bundle"],row["category"],row["sha256"])
    copy(source_subset/"BUNDLE_MANIFEST.json",root/"provenance/SOURCE_CHECKPOINT_SUBSET_MANIFEST.json","historical_source_only_subset_manifest")
    input_records=[]
    for row in subset["minimum_runtime_input_dependencies_not_copied"]:
        path=ENGINE/row["path_from_native_engine"]
        copy(path,root/"Prototype5"/row["path_from_native_engine"],"exact_frozen_runtime_csv",row["sha256"])
        input_records.append(row)
    bank=ENGINE/"forecast_models_settlement_hourly_v2/episode_20"
    ann_files=sorted(path for path in bank.rglob("*") if path.is_file())
    assert len(ann_files)==5 and sum(path.stat().st_size for path in ann_files)==1950415
    for path in ann_files:copy(path,root/"Prototype5"/path.relative_to(ENGINE),"optional_exact_frozen_forecast_bank")
    # Supply the exact frozen auditor in the relative layout expected by its API.
    audit_relative=Path("github/accounting-options-rl/exploratory_novelty_2026-10-16/sample_only_learned_audit")
    for name,digest in frozen["source_sha256"].items():
        copy(HERE/"execution/run_001/source_snapshot"/name,root/audit_relative/name,"exact_frozen_audit_source",digest)
    # Existing source-side provenance records, not market observations.
    metadata=("evaluation_window_completion_manifest.json","real_settlement_protocol_v2_manifest.json","real_liquidity_volume_protocol_v1_manifest.json",
        "evaluation_dataset_ffill/unseendata.csv.metadata.json","evaluation_dataset_ffill/unseendata_v2.csv.metadata.json",
        "evaluation_dataset_ffill/unseendata_liquidity_volume_real_v1.csv.metadata.json","evaluation_dataset_ffill/unseendata_v2_liquidity_volume_real_v1.csv.metadata.json",
        "forecast_cache_settlement_hourly_v2/hourly_forecast_cache_v2_manifest.json",
        "forecast_cache_settlement_hourly_v2/forecast_cache_eval_episode20_2025/forecast_cache_eval_episode20_2025-full/precomputed_forecasts_20250101_to_20250930_39312rows_metadata.json",
        "forecast_cache_settlement_hourly_v2/forecast_cache_eval_episode20_2025/forecast_cache_eval_episode20_2025-unseendata_v2/precomputed_forecasts_20250101_to_20250930_39312rows_metadata.json")
    for name in metadata:copy(ENGINE/name,root/"Prototype5"/name,"original_input_provenance_metadata")
    with (ENGINE/"rolling_past_history_dataset_ffill/history_manifest.csv").open(encoding="utf-8",newline="") as stream:
        history=[row for row in csv.DictReader(stream) if row["history_file"] in ("history_019.csv","history_020.csv")]
    assert len(history)==2
    with (root/"provenance/USED_HISTORY_PROVENANCE.json").open("x",encoding="utf-8") as stream:json.dump(history,stream,indent=2)
    copy(Path(__file__),root/"provenance/prepare_complete_private_review_bundle.py","complete_private_packet_builder")
    readme="""# Complete local private native-policy review packet

Status: exact-copy local review packet; NOT uploaded, distributed, anonymized or
tested in a clean installation. The original running source/data are unchanged.

Included: all34 frozen native Python modules; all20 investor policies and their
matching20battery policies; all20 original training/runtime configs; the exact
frozen auditor source;11extra reconstruction SOURCE scripts; exactly10 operational
runtime CSVs; optional frozen episode20ANN model/scaler/metadata/history bank5files.
No raw eSett records, statistical phase traces, actor observation arrays, ongoing
financial endpoint results or fitted audit nuisance critics are copied.

## Replay layout

This folder has the expected <Paper1>/Prototype5 and
<Paper1>/github/accounting-options-rl/exploratory_novelty_2026-10-16/sample_only_learned_audit
layout. Install the recorded native dependencies in a fresh environment. The
source/checkpoint/data bytes match the manifest; clean-install execution remains
a separate validation step. Reproduction should use a NEW run-id/source freeze,
not replay the reference's absolute path-dependent SOURCE_FREEZE in another tree.
Before parallel feasible-policy execution, pre-materialize the8 deterministic
480-row forecast cache views with fresh bound provenance as described in the
source-side documentation. The frozen runner does not perform that engineering
preparation automatically. Do not modify the ongoing reference execution.

## Ten exact runtime CSVs

Two native original/v2 evaluation contexts, two aligned settlement panels, two
aligned liquidity panels, only history019/020(the trailing365-day normalization
window), and two episode20original/v2forecast caches. Total53,910,563bytes.
The larger30-file public inventory is a conservative superset, not a minimum.
The older2014bootstrap and earlier19histories are unnecessary for this audit's
frozen-policy replay. The optional ANN bank is1,950,415bytes and is unnecessary
when supplied forecast caches are used. No RL or forecast retraining is needed.

## Provenance and release state

Native day-ahead, settlement and liquidity data trace to Energi Data Service,
not eSett. Their official terms are CC BY4.0: provide attribution and indicate
alignment/resampling/transformation, with no suggestion of Energinet endorsement.
Source: Energinet(www.energidataservice.dk).
https://www.energidataservice.dk/terms-and-conditions

Wind comes from documented Copernicus ERA5 100m u/v point inputs; the current
official ERA5 single-levels catalogue identifies a CC-BY licence, DOI
10.24381/cds.adbb2d47. Credit Copernicus/C3S/ECMWF and describe wind interpolation
and the author's power-curve transformation. Solar/hydro/load and ANN forecasts
are authors' modeled/derived components, not proprietary forecast-provider data.
https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview

The source is identified as author MIT code by the existing third-party-license
record; copied LICENSE and THIRD_PARTY_LICENSES are retained. Existing artifact
license scopes do not explicitly name all policy ZIPs/new forecast/data bundles:
assign explicit author release labels and preserve third-party attribution before
public release. Exact runtime identity is established by hashes; the full
historical stochastic physical-generation pipeline has not been clean-room
regenerated. Frozen exact inputs solve replay access without claiming that
current download/retraining scripts recreate every original byte.

IMPORTANT FOR ANONYMOUS SUBMISSION: original JSON and model/config metadata retain
absolute user/project paths, because original bytes and hashes are preserved.
This is a PRIVATE review packet. An anonymized/public derivative needs a separate
explicit review and separate hashes; do not silently edit original artifacts.

provenance/REPRODUCTION_ACCESS_PLAN.md is the earlier source-only-stage plan;
its no-CSV statement describes that preserved historical subset. This README and
BUNDLE_MANIFEST.json supersede that statement for this complete packet.
"""
    with (root/"README.md").open("x",encoding="utf-8") as stream:stream.write(readme)
    copied.append({"path_in_bundle":"README.md","category":"complete_packet_readme","bytes":(root/"README.md").stat().st_size,"sha256":sha(root/"README.md")})
    csvs=list(root.rglob("*.csv"));assert len(csvs)==10 and sum(path.stat().st_size for path in csvs)==53910563
    manifest={"schema":"complete_private_native_review_bundle_v1","created_utc":datetime.now(timezone.utc).isoformat(),
        "status":"COMPLETE_LOCAL_PRIVATE_COPY_NO_DISTRIBUTION","copied_artifacts":copied,
        "exact_native_modules":34,"investor_zip_count":20,"battery_zip_count":20,"training_configs":20,
        "runtime_csv_count":10,"runtime_csv_bytes":53910563,"optional_ann_files":5,"optional_ann_bytes":1950415,
        "minimum_runtime_input_dependencies":input_records,"source_freeze_sha256":sha(HERE/"execution/run_001/SOURCE_FREEZE.json"),
        "historical_subset_manifest_sha256":sha(source_subset/"BUNDLE_MANIFEST.json"),
        "builder_sha256":sha(__file__),"actor_calls":0,"native_steps":0,"statistical_endpoints_copied":False,
        "anonymity_status":"private, original absolute metadata paths retained byte-exact",
        "public_release_blockers":["Explicit author artifact-license labels and third-party attributions must accompany release.","Original absolute user/project paths and serialized metadata need separate anonymization review.","Fresh clean-install replay and automatic run-id-bound forecast-cache setup are not yet validated."],
        "unidentified_esett_or_proprietary_forecast_dependency":False}
    manifest_path=root/"BUNDLE_MANIFEST.json"
    with manifest_path.open("x",encoding="utf-8") as stream:json.dump(manifest,stream,indent=2,sort_keys=True,allow_nan=False);stream.write("\n")
    archive=parent/"native_review_complete_v1.zip";partial=archive.with_name(archive.name+".partial")
    assert not archive.exists()
    with zipfile.ZipFile(partial,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as zipped:
        for path in sorted(root.rglob("*")):
            if path.is_file():zipped.write(path,arcname=path.relative_to(root).as_posix(),
                compress_type=zipfile.ZIP_STORED if path.suffix.lower() in (".zip",".keras") else zipfile.ZIP_DEFLATED,compresslevel=1)
    partial.rename(archive)
    # Validate the compressed packet against each byte-exact folder artifact.
    expected={path.relative_to(root).as_posix():sha(path) for path in root.rglob("*") if path.is_file()}
    with zipfile.ZipFile(archive) as zipped:
        assert set(zipped.namelist())==set(expected)
        for name,digest in expected.items():
            actual=hashlib.sha256()
            with zipped.open(name) as stream:
                for block in iter(lambda:stream.read(1024*1024),b""):actual.update(block)
            assert actual.hexdigest()==digest
    for name,digest in frozen["native_source_sha256"].items():assert sha(ENGINE/name)==digest
    for name,digest in frozen["checkpoint_sha256"].items():assert sha(ENGINE/name)==digest
    integrity={"status":"PASS_BYTE_EXACT_PRIVATE_ARCHIVE","archive_sha256":sha(archive),"archive_bytes":archive.stat().st_size,
        "bundle_manifest_sha256":sha(manifest_path),"folder_files":len(expected),"folder_bytes":sum(path.stat().st_size for path in root.rglob("*") if path.is_file()),
        "runtime_csv_count":10,"optional_ann_files":5,"all_archive_members_sha256_checked":True,
        "original_frozen_sources_and_checkpoints_unchanged":True,"actor_calls":0,"native_steps":0,
        "wall_seconds":time.perf_counter()-wall,"cpu_seconds":time.process_time()-cpu}
    with (parent/"native_review_complete_v1_INTEGRITY.json").open("x",encoding="utf-8") as stream:json.dump(integrity,stream,indent=2,sort_keys=True)
    print(json.dumps(integrity));print(str(archive))

if __name__=="__main__":main()
