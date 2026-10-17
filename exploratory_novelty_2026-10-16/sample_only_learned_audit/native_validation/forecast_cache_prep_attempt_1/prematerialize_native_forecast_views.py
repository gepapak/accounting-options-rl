"""Deterministic cache preparation only: no actor, environment or audit draw."""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
from datetime import datetime,timezone
import contextlib
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    output=HERE/"native_validation"/"forecast_cache_prep_attempt_1"
    output.mkdir(parents=True,exist_ok=False)
    report={"status":"RUNNING","scope":"deterministic exact forecast-cache pre-materialization; zero actors, environment/audit steps, random draws or statistical endpoints", "started_utc":datetime.now(timezone.utc).isoformat(),"contexts":[],"actor_forward_calls":0,"native_steps":0,"audit_steps":0}
    for name in ("prematerialize_native_forecast_views.py","native_engine_adapter.py"):
        shutil.copy2(HERE/name,output/name)
    report["source_sha256"]={name:sha(HERE/name) for name in ("prematerialize_native_forecast_views.py","native_engine_adapter.py")}
    frozen=json.loads((HERE/"execution/run_001/SOURCE_FREEZE.json").read_text(encoding="utf-8"))
    assert report["source_sha256"]["native_engine_adapter.py"]==frozen["source_sha256"]["native_engine_adapter.py"]
    wall,cpu=time.perf_counter(),time.process_time()
    try:
        with ((output/"native_stdout_stderr.log").open("w",encoding="utf-8") as log,
              contextlib.redirect_stdout(log),contextlib.redirect_stderr(log)):
            from native_engine_adapter import NativeRuntime,state_sha,fixed_context_grid
            runtime=NativeRuntime()
            for context in fixed_context_grid(runtime,burn_in=384,H=96):
                features,source=runtime.forecasts(context["zone"])
                start=int(context["start_row"]);H=480
                view=features[start:start+H]
                assert len(view)==H
                cache_key=state_sha({"source":str(source),"start":start,"H":H})[:20]
                directory=HERE/"runtime_cache/forecast_views"/cache_key
                directory.mkdir(parents=True,exist_ok=True)
                target=directory/"precomputed_forecasts_native_view.csv"
                expected=runtime.pd.DataFrame(view,columns=runtime.forecast.FORECAST_PRIOR_ALL_COLS).to_csv(index=False).encode("utf-8")
                expected_sha=hashlib.sha256(expected).hexdigest()
                if target.exists():
                    assert sha(target)==expected_sha, "Existing cache differs from full deterministic native source view: "+str(target)
                    action="existing_complete_bytes_validated"
                else:
                    partial=target.with_name(target.name+".preparation.partial")
                    with partial.open("xb") as stream:
                        stream.write(expected);stream.flush();os.fsync(stream.fileno())
                    assert not target.exists(),"Concurrent cache publication; preserve partial and investigate"
                    partial.rename(target)
                    assert sha(target)==expected_sha
                    action="atomic_complete_publication"
                loaded=runtime.pd.read_csv(target)
                assert len(loaded)==480 and list(loaded.columns)==list(runtime.forecast.FORECAST_PRIOR_ALL_COLS)
                report["contexts"].append({"context_id":int(context["context_id"]),"zone":context["zone"],"start_row":start,
                    "rows":480,"columns":len(loaded.columns),"source_path":str(source),"source_sha256":sha(source),
                    "cache_path":str(target),"cache_sha256":sha(target),"action":action})
            runtime.assert_sources_unchanged()
            assert len(report["contexts"])==8
            assert sha(HERE/"native_engine_adapter.py")==frozen["source_sha256"]["native_engine_adapter.py"]
            report["status"]="PASS"
    except BaseException as error:
        report["status"]="FAILURE";report["error"]=repr(error);report["traceback"]=traceback.format_exc()
    report["wall_seconds"]=time.perf_counter()-wall;report["cpu_seconds"]=time.process_time()-cpu
    (output/"preparation_report.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({key:report[key] for key in ("status","actor_forward_calls","native_steps","audit_steps","wall_seconds","cpu_seconds")}))
    print(str(output))
    if report["status"]!="PASS": print(report["traceback"]);return 1
    return 0

if __name__=="__main__": raise SystemExit(main())
