"""HELD two fixed technical paths; no inferential endpoint or adaptive choice."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def require(value,message):
    if not bool(value):raise AssertionError(message)


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))


def execute(output,cache_report,approval_note):
    require(bool(approval_note.strip()),"Explicit coordinator technical release required")
    require(cache_report.is_file(),"Completed eight-view source preparation must precede any native object")
    prepared=read(cache_report)
    require(prepared.get("status")=="PASS" and len(prepared["contexts"])==8 and
            prepared.get("native_steps")==prepared.get("actor_calls")==prepared.get("environments_created")==0,
            "Exact eight-view zero-native preparation PASS")
    require(prepared["source_sha256"]==sha(HERE/"prepare_joint_fault_cache.py"),"Executed cache-preparation helper")
    require(prepared["new_adapter_sha256"]==sha(HERE/"joint_fault_adapter.py"),"Same reviewed adapter as prepared cache")
    for context in prepared["contexts"]:
        require(sha(Path(context["cache_path"]))==context["cache_sha256"],"All eight exact cache bytes remain complete")
    output=output.resolve();output.mkdir(parents=True,exist_ok=False)
    sources=("validate_joint_fault_smoke.py","prepare_joint_fault_cache.py","joint_fault_adapter.py",
             "joint_fault_support.py","TECHNICAL_SMOKE_PLAN.md","SOURCE_SUPPORT_REVIEW_DRAFT.md")
    hashes={name:sha(HERE/name) for name in sources}
    for name in sources:shutil.copy2(HERE/name,output/name)
    start,cpu=time.perf_counter(),time.process_time()
    report=dict(status="RUNNING",started_utc=datetime.now(timezone.utc).isoformat(),approval_note=approval_note,
        source_sha256=hashes,cache_preparation_sha256=sha(cache_report),cases=[],completed_native_steps=0,
        attempted_native_step_calls=0,failed_native_step_calls=0,actor_predict_calls=0,audit_steps=0,
        scope="Two fixed deterministic technical paths, not statistical evidence; no positivity gate or tuning")
    episode=None
    try:
        from prepare_joint_fault_cache import verify_original_inputs
        before=verify_original_inputs();report["original_dependency_sha256"]=before
        import numpy as np
        import joint_fault_adapter as adapter
        import joint_fault_support as support
        runtime=adapter.NativeRuntime()
        original_step=runtime.environment.RenewableMultiAgentEnv.step
        def counted_native_step(env,actions):
            report["attempted_native_step_calls"]+=1
            try:result=original_step(env,actions)
            except BaseException:
                report["failed_native_step_calls"]+=1;raise
            report["completed_native_steps"]+=1
            return result
        runtime.environment.RenewableMultiAgentEnv.step=counted_native_step
        load_models=runtime.load_models
        counted_models=set();active_episode=[None];last_prediction={}
        def checked_load_models(checkpoint):
            models=load_models(checkpoint)
            for agent,model in models.items():
                if id(model) in counted_models:continue
                counted_models.add(id(model));predict=model.predict
                def counted_predict(obs,*args,_predict=predict,_agent=agent,**kwargs):
                    report["actor_predict_calls"]+=1
                    ep=active_episode[0]
                    if ep is not None:
                        require(np.all(np.isfinite(obs)),"Finite actual native actor observation")
                        require(np.array_equal(obs,ep.observations[_agent]) and
                                np.array_equal(obs,ep.env._obs_buf[_agent]),"Actor receives authentic current native observation")
                    value=_predict(obs,*args,**kwargs)
                    if ep is not None:last_prediction[_agent]=np.asarray(value[0]).copy()
                    return value
                model.predict=counted_predict
            return models
        runtime.load_models=checked_load_models
        registry=read(HERE.parent/"execution/run_001/cases.json")
        source_cases={int(row["case_id"]):row for row in registry}
        for name,old_case,alternating in (("plain7_joint_fault",1,False),("feasible7_joint_fault",21,True)):
            source=source_cases[old_case]
            require(source["checkpoint_seed"]==7 and source["profile"]=="vulnerable","Fixed seed7 policy lineage")
            spec=adapter.EpisodeSpec(source["checkpoint_dir"],zone="original",start_row=0,horizon=96,
                native_seed=7,profile="joint_fault",cash_fraction=1e-4,burn_in=384)
            scenario=adapter.NativeScenario(runtime,spec)
            require(float(scenario.config_snapshot.liquidity_participation_cap_fraction)==.25,"Warm config remains capped")
            signs=np.ones(96,dtype=np.int8)
            if alternating:signs[1::2]=-1
            episode=scenario.new_episode(signs)
            require(float(episode.env.config.liquidity_participation_cap_fraction)==0.,"Fresh audit clone is uncapped")
            require(float(episode.env.budget)==1e-4*float(episode.env._initial_trading_sleeve_dkk()),"Fixed initial cash stress")
            bounds=support.tightened_calendar_support(episode,gamma=.995)
            initial_actor_obs={k:np.asarray(v).copy() for k,v in episode.observations.items()}
            require(all(np.array_equal(v,episode.env._obs_buf[k]) for k,v in initial_actor_obs.items()),"Authentic rebuilt first observation")
            actual_actions={agent:[] for agent in scenario.models}
            native_episode_step=episode.env.step
            def checked_audit_step(actions):
                for agent in scenario.models:
                    expected=runtime.evaluation._coerce_action_for_space(last_prediction[agent],episode.env.action_space(agent))
                    require(np.array_equal(np.asarray(actions[agent]),np.asarray(expected)),"Native action equals actual actor prediction/coercion")
                    actual_actions[agent].append(np.asarray(actions[agent]).copy())
                return native_episode_step(actions)
            episode.env.step=checked_audit_step
            active_episode[0]=episode
            for _ in range(96):
                episode.step();report["audit_steps"]+=1
            active_episode[0]=None
            trace=episode.trace_arrays();summary=episode.summary()
            support.check_trace_support(trace,bounds)
            require(summary["terminal_open_contracts"]==0,"Empty terminal financial queue")
            require(summary["immutable_warm_snapshot_unchanged"] and summary["immutable_config_snapshot_unchanged"]
                and summary["clone_all_field_hash_equal_before_intervention"],"Warm parent/config and initial clone identity")
            for key,array in trace.items():
                if np.asarray(array).dtype.kind in "fc":require(np.all(np.isfinite(array)),"Finite saved native trace: "+key)
            raw_actions=np.asarray(actual_actions["investor_0"]).reshape(96,-1)[:,0]
            require(np.array_equal(raw_actions,np.asarray(trace["path_actor_action"])),"All investor actions equal source-to-trace pipeline")
            old=(np.asarray(trace["event_sign_epsilon"])!=0)&(np.asarray(trace["event_step"])<6)
            weights=.995**np.asarray(trace["event_step"])
            G,C,I=(np.asarray(trace["event_"+key],float) for key in ("gross","cost","injection"))
            new_gross=float(weights[~old]@G[~old]);all_cost=float(weights@C);new_I=float(weights[~old]@I[~old])
            alpha0=new_gross-all_cost+new_I
            direct=float(weights[~old]@(G[~old]-C[~old]+I[~old]))
            tolerance=1e-6+1e-10*(bounds["discounted_gross_bound"]+bounds["discounted_cost_bound"])
            require(abs(alpha0-direct)<=tolerance,"New target removes old G/I only and keeps all costs")
            require(alpha0>=-bounds["cash_inflow_bound"]-tolerance,"Whole-path cash endpoint coverage")
            for agent,values in actual_actions.items():trace["smoke_native_action_"+agent]=np.asarray(values)
            np.savez_compressed(output/(name+"_trace.npz"),**trace)
            report["cases"].append(dict(case=name,status="PASS",old_case_id=old_case,steps=96,warm_steps=384,
                cache_context_id=0,initial_actor_shapes={k:list(v.shape) for k,v in initial_actor_obs.items()},
                source_target_old_mask_events=int(old.sum()),source_support=bounds,native_summary=summary,
                target_identity_error=abs(alpha0-direct),target_cash_lower_coverage=True,
                source_actor_observation_action_equality=True,terminal_queue_empty=True,
                no_positivity_or_activity_requirement=True))
            runtime.assert_sources_unchanged();episode=None
        require(report["completed_native_steps"]==960 and report["audit_steps"]==192,"Exact two fixed software paths only")
        require(report["actor_predict_calls"]==1920,"Exactly two actual learned actors per warm/audit step; no extra inference")
        require(before==verify_original_inputs(),"All original dependencies unchanged")
        require(all(sha(HERE/name)==digest for name,digest in hashes.items()),"Reviewed sources unchanged during smoke")
        report["status"]="PASS"
    except BaseException as error:
        report["status"]="SOFTWARE_FAILURE";report["error"]=repr(error);report["traceback"]=traceback.format_exc()
        if episode is not None:
            try:
                import numpy as np
                np.savez_compressed(output/"failed_partial_trace.npz",**episode.trace_arrays())
            except BaseException as partial_error:report["partial_trace_error"]=repr(partial_error)
    report["wall_seconds"]=time.perf_counter()-start;report["cpu_seconds"]=time.process_time()-cpu
    with (output/"smoke_report.json").open("x",encoding="utf-8") as stream:json.dump(report,stream,indent=2,allow_nan=False)
    print(json.dumps({k:report[k] for k in ("status","completed_native_steps","audit_steps","actor_predict_calls","wall_seconds","cpu_seconds")}))
    return 0 if report["status"]=="PASS" else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute",action="store_true");p.add_argument("--approval-note",default="")
    p.add_argument("--cache-report",type=Path,default=HERE/"validation/cache_preparation_attempt_1/preparation_report.json")
    p.add_argument("--output-dir",type=Path,default=HERE/"validation/technical_smoke_attempt_1")
    args=p.parse_args()
    if not args.execute:
        print(json.dumps(dict(status="PREPARED_HELD",native_imports=0,native_steps=0,actor_calls=0)));return 0
    return execute(args.output_dir,args.cache_report,args.approval_note)


if __name__=="__main__":raise SystemExit(main())
