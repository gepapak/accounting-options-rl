"""Two fixed warm-clone software paths, no statistical endpoint evaluation."""
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


def memory():
    import psutil
    return psutil.Process().memory_info()._asdict()


def main():
    root = HERE / "native_validation"
    output = root / ("warm_attempt_"+str(len(list(root.glob("warm_attempt_*")))+1))
    output.mkdir(parents=True)
    names = ("native_engine_adapter.py", "validate_native_warm_clone.py", "NATIVE_WARM_CLONE_VALIDATION_PLAN.md")
    for name in names:
        shutil.copy2(HERE/name,output/name)
    report = {"status":"RUNNING", "purpose":"two deterministic warm-clone software paths; no statistical endpoints",
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": {name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in names}, "cases":[]}
    wall,cpu=time.perf_counter(),time.process_time()
    try:
        with ((output/"native_stdout_stderr.log").open("w",encoding="utf-8") as log,
              contextlib.redirect_stdout(log),contextlib.redirect_stderr(log)):
            from native_engine_adapter import NativeRuntime,EpisodeSpec,NativeScenario,ENGINE,fixed_context_grid,jsonable
            runtime=NativeRuntime()
            report["memory_after_import"]=memory()
            report["fixed_context_grid"]=fixed_context_grid(runtime)
            choices=[("plain7_strict","strict",ENGINE/"batch_tier_phase_runs/prototype5_mappo_marl_final_v1/seed7/tier1_seed7/final_models"),
                     ("feasible7_vulnerable","vulnerable",ENGINE/"Ablations/batch_tier_phase_runs/prototype5_mechanism_ablations_final_v1/feasible_action_full/seed7/tier1_forecast_utilization_seed7/final_models")]
            for name,profile,path in choices:
                prepare=time.perf_counter()
                scenario=NativeScenario(runtime,EpisodeSpec(str(path),horizon=96,burn_in=384,
                                                            profile=profile,cash_fraction=.1))
                prepare_s=time.perf_counter()-prepare
                after_warm=memory()
                signs=runtime.np.ones(96,dtype=runtime.np.int8)
                if profile=="vulnerable": signs[1::2]=-1
                episode=scenario.new_episode(signs)
                after_clone=memory()
                assert episode.env.t==384
                assert episode.env.budget==.1*episode.env._initial_trading_sleeve_dkk()
                assert episode.env._get_investor_sizing_base_mode()=="initial_trading_sleeve"
                assert episode.env._horizon_unit_payoff(100,175)==75
                rollout=time.perf_counter()
                for _ in range(96): episode.step()
                rollout_s=time.perf_counter()-rollout
                assert episode.env._horizon_unit_payoff(100,175)==75
                summary=episode.summary()
                assert summary["immutable_warm_snapshot_unchanged"]
                assert summary["immutable_config_snapshot_unchanged"]
                assert summary["clone_all_field_hash_equal_before_intervention"]
                runtime.assert_sources_unchanged()
                arrays=episode.trace_arrays()
                assert runtime.np.array_equal(arrays["path_step"],runtime.np.arange(96))
                assert runtime.np.array_equal(arrays["path_native_step"],runtime.np.arange(384,480))
                runtime.np.savez_compressed(output/(name+"_trace.npz"),**arrays)
                report["cases"].append(jsonable({"case":name,"status":"PASS","metadata":scenario.metadata,
                    "summary":summary,"preparation_including_warmup_s":prepare_s,
                    "audit_rollout_including_actor_s":rollout_s,"seconds_per_audit_step":rollout_s/96,
                    "memory_after_warmup":after_warm,"memory_after_clone":after_clone,"memory_after_audit":memory()}))
            report["status"]="PASS"
    except Exception as error:
        report["status"]="SOFTWARE_FAILURE"
        report["error"]=repr(error)
        report["traceback"]=traceback.format_exc()
    report["wall_seconds"]=time.perf_counter()-wall
    report["cpu_seconds"]=time.process_time()-cpu
    (output/"validation_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({"status":report["status"],"completed_software_paths":len(report["cases"]),
                      "wall_seconds":report["wall_seconds"],"cpu_seconds":report["cpu_seconds"],"output":str(output)}))
    if report["status"]!="PASS":
        print(report["traceback"])
        return 1
    return 0


if __name__=="__main__":
    raise SystemExit(main())
