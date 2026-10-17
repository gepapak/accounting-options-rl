"""HELD engineering preparation: eight exact forecast views, zero native steps."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent
ENGINE = ORIGINAL.parents[3]/"Prototype5"
FREEZE = ORIGINAL/"execution/run_001/SOURCE_FREEZE.json"
INPUT_MANIFEST = ORIGINAL/"review_bundle_private/native_review_complete_v1/BUNDLE_MANIFEST.json"
FREEZE_SHA = "6152949392bd8de0f494bc21ffb5fd7d9be4a16a6c8866f4eb5595254da523d4"
INPUT_MANIFEST_SHA = "3e290f4984d629284e640fccad7798fb79839db71956a8607eecc1e6e8c75035"


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024),b""): digest.update(block)
    return digest.hexdigest()


def require(value,message):
    if not bool(value): raise AssertionError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_original_inputs():
    require(sha(FREEZE)==FREEZE_SHA,"Original source freeze unchanged")
    require(sha(INPUT_MANIFEST)==INPUT_MANIFEST_SHA,"Exact private input manifest")
    frozen,manifest=read(FREEZE),read(INPUT_MANIFEST)
    checked={}
    for category,root in (("source_sha256",ORIGINAL),("native_source_sha256",ENGINE),("checkpoint_sha256",ENGINE)):
        for name,digest in frozen[category].items():
            path=root/name
            require(sha(path)==digest,"Original fixed dependency unchanged: "+str(path))
            checked[str(path)]=digest
    runtime=[r for r in manifest["copied_artifacts"] if r["category"]=="exact_frozen_runtime_csv"]
    require(len(runtime)==10,"All ten native runtime input CSVs")
    for row in runtime:
        path=ENGINE.parent/row["path_in_bundle"]
        require(sha(path)==row["sha256"],"Original fixed runtime input unchanged: "+str(path))
        checked[str(path)]=row["sha256"]
    return checked


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def atomic_view(target,payload):
    expected=hashlib.sha256(payload).hexdigest()
    if target.exists():
        require(target.is_file() and sha(target)==expected,"Preserve wrong/partial existing cache: "+str(target))
        return "verified_existing",expected
    partial=target.with_name(target.name+".joint-fault-preparation.partial")
    with partial.open("xb") as stream:
        stream.write(payload);stream.flush();os.fsync(stream.fileno())
    require(sha(partial)==expected,"Exact partial cache bytes")
    if os.name=="nt": os.rename(partial,target)
    else:
        os.link(partial,target);partial.unlink()
    require(sha(target)==expected,"Published exact cache bytes")
    return "atomic_exclusive_publication",expected


def execute(output,approval_note):
    require(bool(approval_note.strip()),"Coordinator engineering release required")
    output=output.resolve();output.mkdir(parents=True,exist_ok=False)
    start,cpu=time.perf_counter(),time.process_time()
    report=dict(status="RUNNING",started_utc=datetime.now(timezone.utc).isoformat(),approval_note=approval_note,
        source_sha256=sha(__file__),original_freeze_sha256=FREEZE_SHA,input_manifest_sha256=INPUT_MANIFEST_SHA,
        actor_calls=0,environments_created=0,native_steps=0,audit_rng_draws=0,contexts=[])
    shutil.copy2(__file__,output/Path(__file__).name)
    try:
        before=verify_original_inputs();report["verified_input_sha256"]=before
        import numpy as np
        import pandas as pd
        adapter=load_module("joint_fault_cache_original_adapter",ORIGINAL/"native_engine_adapter.py")
        forecast=load_module("joint_fault_cache_native_forecast_reader",ENGINE/"forecast_prior.py")
        report["new_adapter_sha256"]=sha(HERE/"joint_fault_adapter.py")
        root=ENGINE/"forecast_cache_settlement_hourly_v2/forecast_cache_eval_episode20_2025"
        for zone,suffix in (("original","full"),("v2","unseendata_v2")):
            directory=root/("forecast_cache_eval_episode20_2025-"+suffix)
            features=forecast.load_forecast_prior_features(str(directory),0)
            require(features is not None and len(features)==39312,"Fixed complete native forecast rows")
            sources=[p for p in directory.glob("precomputed_forecasts_*.csv") if "metadata" not in p.name]
            require(len(sources)==1,"Unique original precomputed forecast source")
            source=sources[0]
            for q,start_row in enumerate(np.linspace(0,len(features)-480,4,dtype=int)):
                start_row=int(start_row)
                key=adapter.state_sha(dict(source=str(source),start=start_row,H=480))[:20]
                target=HERE/"runtime_cache/forecast_views"/key/"precomputed_forecasts_native_view.csv"
                target.parent.mkdir(parents=True,exist_ok=True)
                payload=pd.DataFrame(features[start_row:start_row+480],columns=forecast.FORECAST_PRIOR_ALL_COLS).to_csv(index=False).encode("utf-8")
                action,digest=atomic_view(target,payload)
                report["contexts"].append(dict(context_id=len(report["contexts"]),zone=zone,quantile_id=q,
                    start_row=start_row,rows=480,columns=len(forecast.FORECAST_PRIOR_ALL_COLS),
                    forecast_source=str(source),forecast_source_sha256=sha(source),cache_key=key,
                    cache_path=str(target),cache_sha256=digest,action=action))
        require(len(report["contexts"])==8,"All eight exact forecast views before parallel native workers")
        require(before==verify_original_inputs(),"All original dependencies unchanged after engineering preparation")
        report["status"]="PASS"
    except BaseException as error:
        report["status"]="SOFTWARE_FAILURE";report["error"]=repr(error);report["traceback"]=traceback.format_exc()
    report["wall_seconds"]=time.perf_counter()-start;report["cpu_seconds"]=time.process_time()-cpu
    with (output/"preparation_report.json").open("x",encoding="utf-8") as stream:json.dump(report,stream,indent=2,allow_nan=False)
    print(json.dumps({k:report[k] for k in ("status","wall_seconds","cpu_seconds","native_steps","actor_calls")}))
    return 0 if report["status"]=="PASS" else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute",action="store_true");p.add_argument("--approval-note",default="")
    p.add_argument("--output-dir",type=Path,default=HERE/"validation/cache_preparation_attempt_1")
    args=p.parse_args()
    if not args.execute:
        print(json.dumps(dict(status="PREPARED_HELD",native_imports=0,native_steps=0,actor_calls=0)));return 0
    return execute(args.output_dir,args.approval_note)


if __name__=="__main__":raise SystemExit(main())
