"""Arithmetic support check on two ALREADY SAVED software traces, zero new audit steps."""
from __future__ import annotations
import contextlib
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
PASSED_ADAPTER="da93030da39c1f2a4d0eaef3666177fc6628b7da97161ed892d1b9abfa38c879"

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    output=HERE/"native_validation"/"calendar_support_attempt_1"
    output.mkdir(parents=True,exist_ok=False)
    report={"status":"RUNNING","purpose":"existing two software traces plus fresh initial-state-only cash contract; no new audit trajectories or confidence endpoints", "started_utc":datetime.now(timezone.utc).isoformat(),"cases":[],"new_audit_transitions":0}
    for name in ("native_engine_adapter.py","native_support.py","validate_native_support.py"):
        shutil.copy2(HERE/name,output/name)
    report["source_sha256"]={name:sha(output/name) for name in ("native_engine_adapter.py","native_support.py","validate_native_support.py")}
    wall,cpu=time.perf_counter(),time.process_time()
    try:
        assert sha(HERE/"native_engine_adapter.py")==PASSED_ADAPTER
        with ((output/"native_stdout_stderr.log").open("w",encoding="utf-8") as log,
              contextlib.redirect_stdout(log),contextlib.redirect_stderr(log)):
            import numpy as np
            from native_engine_adapter import NativeRuntime,NativeScenario,EpisodeSpec,ENGINE,jsonable
            from native_support import tightened_calendar_support,check_trace_support
            runtime=NativeRuntime()
            choices=[("plain7_strict","strict",ENGINE/"batch_tier_phase_runs/prototype5_mappo_marl_final_v1/seed7/tier1_seed7/final_models"),
                     ("feasible7_vulnerable","vulnerable",ENGINE/"Ablations/batch_tier_phase_runs/prototype5_mechanism_ablations_final_v1/feasible_action_full/seed7/tier1_forecast_utilization_seed7/final_models")]
            for name,profile,path in choices:
                scenario=NativeScenario(runtime,EpisodeSpec(str(path),horizon=96,burn_in=384,profile=profile,cash_fraction=1e-4))
                episode=scenario.new_episode(np.ones(96,dtype=np.int8))
                assert not episode.step_records and episode.env.t==384
                assert episode.env.budget==1e-4*episode.env._initial_trading_sleeve_dkk()
                assert episode.env._get_investor_sizing_base_mode()=="initial_trading_sleeve"
                assert bool(episode.env.config.enable_trading_sleeve_margin)==(profile=="strict")
                if profile=="vulnerable": assert np.isinf(episode.env.config.mtm_loss_exit_threshold_pct)
                bounds=tightened_calendar_support(episode)
                saved=HERE/"native_validation"/"warm_attempt_2"/(name+"_trace.npz")
                with np.load(saved,allow_pickle=False) as archive:
                    trace={key:archive[key] for key in archive.files}
                checked=check_trace_support(trace,bounds)
                assert sha(HERE/"native_engine_adapter.py")==PASSED_ADAPTER
                runtime.assert_sources_unchanged()
                report["cases"].append(jsonable({"case":name,"profile":profile,"status":"PASS","new_audit_transitions":0,
                    "saved_trace":str(saved),"saved_trace_sha256":sha(saved),"saved_trace_cash_fraction":.1,
                    "fresh_initial_state_cash_fraction":1e-4,"initial_cash_contract_pass":True,
                    "native_actor_observation_dim":int(episode.env._fill_obs().shape[0]) if False else scenario.metadata["training_contract"],
                    "support":bounds,"saved_trace_support_check":checked,
                    "support_scope":"same warmed physical prefix, cash-independent capacity/cost bound; old .1 cash trace covered without changing or replaying audit"}))
            report["status"]="PASS"
            report["deterministic_native_warm_transitions"]=2*384
    except BaseException as error:
        report["status"]="SOFTWARE_FAILURE";report["error"]=repr(error);report["traceback"]=traceback.format_exc()
    report["wall_seconds"]=time.perf_counter()-wall;report["cpu_seconds"]=time.process_time()-cpu
    (output/"validation_report.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({key:report[key] for key in ("status","new_audit_transitions","wall_seconds","cpu_seconds")}))
    print(str(output))
    if report["status"]!="PASS": print(report["traceback"]);return 1
    return 0

if __name__=="__main__": raise SystemExit(main())
