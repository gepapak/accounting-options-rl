"""Read-only source/support/provenance guard; never reads financial row outcomes."""
from __future__ import annotations
from datetime import datetime,timezone
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ENGINE=HERE.parents[3]/"Prototype5"

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""): digest.update(block)
    return digest.hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def require(value,message):
    if not value: raise AssertionError(message)

def check_metadata(path,case,freeze):
    data=read(path);phase=data["phase"]
    require(data["status"]=="COMPLETE" and data["case"]==case,"Frozen complete case metadata")
    n={"fit":32,"tune":32,"confirm":256}[phase]
    require(data["completed_episodes"]==data["planned_episodes"]==n,"Fixed complete episode count")
    require(data["horizon"]==96 and data["burn_in"]==384 and data["gamma"]==.995,"Native clock and discount")
    require(data["query_counts"]["audit_native_transitions"]==n*96,"Complete native query count")
    require(data["query_counts"]["actor_forward_calls"]==2*n*96,"Both authentic actors queried")
    for name,digest in data["source_sha256"].items():
        require(digest==freeze["source_sha256"][name],"Frozen numerical source "+name)
    contexts=data["cases_contexts"];require(len(contexts)==8,"All eight predeclared contexts")
    require([row["context"]["context_id"] for row in contexts]==list(range(8)),"Stable context IDs")
    fingerprints=[]
    for cid,row in enumerate(contexts):
        context,metadata,bounds=row["context"],row["metadata"],row["support"]
        require(context["zone"]==("original" if cid<4 else "v2"),"Declared dataset order")
        require(context["start_row"]==(0,12944,25888,38832)[cid%4],"Declared quantile start")
        spec=metadata["spec"]
        require(spec["zone"]==context["zone"] and spec["start_row"]==context["start_row"],"Native context agreement")
        for name in ("checkpoint_dir","profile","cash_fraction","horizon","burn_in"):
            require(spec[name]==case[name],"Frozen native spec "+name)
        require(spec["native_seed"]==case["checkpoint_seed"],"Authentic saved checkpoint seed")
        intervention=metadata["profile_intervention"]
        require(intervention["cash_fraction_after_burn_in"]==1e-4,"Same stressed initial cash")
        require(intervention["preserve_original_sizing_base"] and intervention["preserve_native_halt_flags"],"Preserved original sizing/history flags")
        require(intervention["enable_trading_sleeve_margin"]==(case["profile"]=="strict"),"Native strict/vulnerable margin")
        expected_exit=.15 if case["profile"]=="strict" else "disabled_inf"
        require(intervention["mtm_loss_exit_threshold_pct"]==expected_exit,"Native loss-exit intervention")
        require(metadata["burn_in"]["native_steps"]==384 and row["support_initialization_native_transitions"]==0,"Native warm-up and zero-step support preparation")
        require(bounds["gamma"]==.995 and bounds["horizon"]==96 and bounds["burn_in"]==384,"Support clock")
        require(bounds["n_payoff_moment_queries"]==0,"No conditional payoff moment oracle")
        require(bounds["source_sha256"]==freeze["source_sha256"]["native_support.py"],"Frozen calendar support source")
        require(bounds["allocation_upper_bound"]==1 and bounds["configured_meta_cap_max_descriptive"]==.8,"Correct native allocation envelope")
        require(bounds["investment_frequency"]==bounds["settlement_lag"]==6,"Native fixed six-step timing")
        require(math.isclose(bounds["original_initial_trading_sleeve"],662068965.5172415,rel_tol=1e-12),"Same ORIGINAL sleeve F")
        arrays={name:bounds[name] for name in ("calendar_cost_bound","calendar_gross_bound","calendar_roll_cost_bound","calendar_access_cost_bound","calendar_collateral_cost_bound")}
        for name,values in arrays.items():require(len(values)==96 and all(math.isfinite(x) and x>=0 for x in values),"Finite deterministic support "+name)
        allowed=[t>0 and t%6==0 and t+6<480 for t in range(384,480)]
        require(bounds["calendar_new_opening_possible"]==allowed and bounds["possible_new_openings"]==sum(allowed),"All and only possible native opening clocks")
        require(all(allowed[t] or arrays["calendar_roll_cost_bound"][t]==0 for t in range(96)),"No roll charge on impossible opening clock")
        for key,array in (("discounted_cost_bound","calendar_cost_bound"),("discounted_gross_bound","calendar_gross_bound")):
            expected=sum((.995**t)*value for t,value in enumerate(bounds[array]))
            require(bounds[key]>=expected-1e-7 and math.isclose(bounds[key],expected,rel_tol=2e-11,abs_tol=1e-5),"Calendar scalar support "+key)
        require(bounds["unavoidable_discounted_access_cost"]>0,"Strict deterministic booked-cost harm")
        for name,digest in metadata["source_sha256"].items():require(digest==freeze["native_source_sha256"][name],"Frozen authentic engine source "+name)
        # These records reveal resource identity, never actor observations or cash outcomes.
        resources=metadata["resource_sha256"]
        for name,digest in resources.items():
            if name=="training_config":resource=Path(case["checkpoint_dir"])/"training_config.json"
            elif name in ("native_forecast_source","forecast_view"):continue
            else:resource=ENGINE/name
            require(sha(resource)==digest,"Actual external native input changed: "+name)
        fingerprints.append({"context_id":cid,"resource_sha256":resources,"config_snapshot_sha256":metadata["config_snapshot_sha256"],
            "warm_snapshot_sha256":metadata["warm_snapshot_sha256"],"calendar_support":bounds})
    union=data["bounds"]
    for key in ("discounted_cost_bound","discounted_gross_bound"):
        require(union[key]==max(row["support"][key] for row in contexts),"Full predetermined context union "+key)
    require(union["n_contexts"]==8,"Union covers complete eight-context law")
    return {"case_id":case["case_id"],"phase":phase,"status":"PASS","all_eight_contexts_checked":True,
        "fixed_complete_episode_budget_checked":True,"financial_outcome_rows_read":False,"fingerprints":fingerprints}

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--run-id",default="run_001");args=parser.parse_args()
    run=HERE/"execution"/args.run_id
    freeze=read(run/"SOURCE_FREEZE.json");cases={row["case_id"]:row for row in read(run/"cases.json")}
    timestamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output=HERE/"native_validation"/("runtime_guard_"+timestamp);output.mkdir(parents=True,exist_ok=False)
    report={"status":"RUNNING","run_id":args.run_id,"source_freeze_sha256":sha(run/"SOURCE_FREEZE.json"),"validator_sha256":sha(__file__),
        "scope":"read-only completed metadata, support and input hashes only; financial outcomes never loaded", "checks":[],"warnings":[],
        "startup_note":"Earlier detached PID2984 attempt produced zero-byte logs and no LAUNCH record; root reports zero queries. Managed execution began with actual PID26404, LAUNCH09:43:18UTC; no scientific stream/sample change."}
    try:
        require(sha(run/"cases.json")==freeze["cases_sha256"],"Frozen case list")
        for name,digest in freeze["source_sha256"].items():require(sha(HERE/name)==digest,"Frozen numerical source changed: "+name)
        for name,digest in freeze["native_source_sha256"].items():require(sha(ENGINE/name)==digest,"Frozen native source changed: "+name)
        paths=sorted(run.glob("case_*/[ftc]*/metadata.json"))
        by_case={}
        for path in paths:
            cid=int(path.parent.parent.name.split("_")[-1])
            checked=check_metadata(path,cases[cid],freeze)
            old=by_case.get(cid)
            if old is not None:
                for before,after in zip(old["fingerprints"],checked["fingerprints"]):
                    require(before["resource_sha256"]==after["resource_sha256"],"Resource identity differs between phases")
                    require(before["calendar_support"]==after["calendar_support"],"Source support differs between phases")
                    require(before["config_snapshot_sha256"]==after["config_snapshot_sha256"],"Native config differs between phases")
                    if before["warm_snapshot_sha256"]!=after["warm_snapshot_sha256"]:
                        report["warnings"].append({"case_id":cid,"context_id":before["context_id"],"kind":"warm_recursive_fingerprint_differs_between_stages; economic-state cause must be examined"})
            by_case[cid]=checked
            report["checks"].append(checked)
        report["completed_phase_metadata_checked"]=len(paths)
        report["status"]="PASS" if paths else "PENDING_FIRST_COMPLETE_PHASE"
    except BaseException as error:
        import traceback
        report["status"]="FAILURE";report["error"]=repr(error);report["traceback"]=traceback.format_exc()
    (output/"provenance_report.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({"status":report["status"],"completed_phase_metadata_checked":report.get("completed_phase_metadata_checked",len(report["checks"])),"warnings":len(report["warnings"]),"output":str(output)}))
    if report["status"]=="FAILURE":print(report["traceback"]);return 1
    return 0

if __name__=="__main__":raise SystemExit(main())
