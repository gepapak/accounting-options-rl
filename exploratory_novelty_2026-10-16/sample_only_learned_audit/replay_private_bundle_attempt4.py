"""Prepared external packet-path wrapper; no native execution without --execute."""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
import argparse
import contextlib
from datetime import datetime,timezone
import hashlib
import importlib.util
import importlib.machinery
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
AUDIT_RELATIVE=Path("github/accounting-options-rl/exploratory_novelty_2026-10-16/sample_only_learned_audit")

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):digest.update(block)
    return digest.hexdigest()

def inside(path,root):
    try:return path.is_relative_to(root)
    except ValueError:return False

def execute(bundle,references,output):
    bundle,references,output=bundle.resolve(),references.resolve(),output.resolve()
    native=bundle/"Prototype5";audit=bundle/AUDIT_RELATIVE
    assert native.is_dir() and audit.is_dir(),"Complete native/auditor packet layout required"
    output.mkdir(parents=True,exist_ok=False)
    runtime_root=output/"runtime";runtime_root.mkdir()
    shutil.copy2(Path(__file__),output/Path(__file__).name)
    shutil.copy2(HERE/"CLEAN_BUNDLE_REPLAY_PLAN.md",output/"CLEAN_BUNDLE_REPLAY_PLAN.md")
    report={"status":"RUNNING","purpose":"exact two historical deterministic software paths; not statistical evidence", "started_utc":datetime.now(timezone.utc).isoformat(),
        "wrapper_sha256":sha(__file__),"plan_sha256":sha(HERE/"CLEAN_BUNDLE_REPLAY_PLAN.md"),"cases":[],"native_reads":{},"blocked_access":[],
        "original_native_root_used":False,"scope":"packet-only source/data/model replay in existing installed environment; no new incentive estimates or confidence bounds"}
    wall,cpu=time.perf_counter(),time.process_time()
    try:
        # Read explicit golden comparison traces BEFORE the engine-only I/O guard.
        import numpy as np
        expected={};reference_hashes={}
        for name in ("plain7_strict","feasible7_vulnerable"):
            path=references/(name+"_trace.npz")
            reference_hashes[name]=sha(path)
            with np.load(path,allow_pickle=False) as saved:expected[name]={key:saved[key] for key in saved.files}
        report["golden_trace_sha256"]=reference_hashes
        frozen=json.loads((bundle/"provenance/SOURCE_FREEZE.json").read_text(encoding="utf-8"))
        for name,digest in frozen["native_source_sha256"].items():assert sha(native/name)==digest,"Exact native source "+name
        for name,digest in frozen["checkpoint_sha256"].items():assert sha(native/name)==digest,"Exact paired model/config "+name
        for name,digest in frozen["source_sha256"].items():assert sha(audit/name)==digest,"Unmodified frozen auditor "+name
        manifest=json.loads((bundle/"BUNDLE_MANIFEST.json").read_text(encoding="utf-8"))
        for record in manifest["copied_artifacts"]:
            if record["category"] in ("exact_frozen_runtime_csv","optional_exact_frozen_forecast_bank"):
                assert sha(bundle/record["path_in_bundle"])==record["sha256"],"Exact packet runtime input"
        report["packet_manifest_sha256"]=sha(bundle/"BUNDLE_MANIFEST.json")
        report["native_source_sha256"]=frozen["native_source_sha256"]
        report["checkpoint_sha256"]=frozen["checkpoint_sha256"]
        report["runtime_inputs_sha256"]={row["path_in_bundle"]:row["sha256"] for row in manifest["copied_artifacts"] if row["category"]=="exact_frozen_runtime_csv"}
        original_native=(HERE.parents[3]/"Prototype5").resolve()
        assert not inside(native,original_native) and native!=original_native,"Replay must not point at original native tree"
        # Clear original-workspace import paths; installed Python dependency roots remain.
        sys.path[:]=[str(audit),str(native)]+[entry for entry in sys.path if entry and
            not inside(Path(entry).resolve(),HERE.parents[3].resolve())]
        native_names={Path(name).stem for name in frozen["native_source_sha256"]}
        assert not (native_names & set(sys.modules)),"Native module already loaded before packet isolation"
        for name in native_names:
            spec=importlib.machinery.PathFinder.find_spec(name,[str(native)])
            assert spec and spec.origin and inside(Path(spec.origin).resolve(),native),"All frozen native modules resolve inside packet"
        for name in ("MPLCONFIGDIR","NUMBA_CACHE_DIR","TORCH_HOME","KERAS_HOME","HF_HOME","XDG_CACHE_HOME","TEMP","TMP"):
            target=runtime_root/"runtime_cache"/name.lower();target.mkdir(parents=True,exist_ok=True);os.environ[name]=str(target)
        import tempfile
        tempfile.tempdir=os.environ["TEMP"]
        os.chdir(runtime_root)
        read_roots=[bundle,output,Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve()]
        # Explicit host dependency/introspection reads authorized after retained
        # attempt2; no original-engine tree or broad user-directory allowance.
        import site
        user_site=Path(site.getusersitepackages()).resolve()
        read_roots.append(user_site)
        # LOCALAPPDATA may be sandbox-redirected although Windows font discovery
        # still probes the installed user's real local-font directory.
        read_roots.append((Path.home()/"AppData/Local/Microsoft/Windows/Fonts").resolve())
        for variable in ("LOCALAPPDATA", "APPDATA"):
            if os.environ.get(variable):
                read_roots.append((Path(os.environ[variable])/"Microsoft/Windows/Fonts").resolve())
        windows=os.environ.get("WINDIR")
        if windows:read_roots.append(Path(windows).resolve())
        wrapper_source=Path(__file__).resolve()
        hardware_probe=Path("C:/sys/bus/pci/devices").resolve()
        hardware_probe_exists=hardware_probe.exists()
        report["allowed_readonly_wrapper_source"]=str(wrapper_source)
        report["optional_hardware_probe"]={"path":str(hardware_probe),"existed_before_guard":hardware_probe_exists,
            "allowed_only_if_absent":True}
        report["null_device_exception"]="Windows NUL or \\\\.\\NUL only; no filesystem tree"
        allowed_dependencies=[str(path) for path in read_roots[2:]]
        report["allowed_external_dependency_roots"]=allowed_dependencies
        def guard(event,args):
            if event not in ("open","os.listdir","os.scandir"):return
            value=args[0] if args else None
            if value is None or isinstance(value,int):return
            if isinstance(value,bytes):value=os.fsdecode(value)
            if not isinstance(value,(str,os.PathLike)) or str(value).startswith("<"):return
            # Installed torch/dill probes this OS sink before loading models.
            # Permit only the canonical Windows null device, never a file tree.
            device=str(value).lower().replace("/", "\\").rstrip("\\")
            if os.name=="nt" and device in ("nul", "\\\\.\\nul"):return
            path=Path(value).resolve()
            writing=False
            if event=="open":
                mode=args[1] if len(args)>1 else None;flags=args[2] if len(args)>2 else 0
                writing=(isinstance(mode,str) and any(mark in mode for mark in ("w","a","x","+"))) or bool(isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            allowed_readonly=(not writing) and (path==wrapper_source or
                (path==hardware_probe and not hardware_probe_exists))
            denied=inside(path,original_native) or (not allowed_readonly and not any(inside(path,root) for root in read_roots)) or (writing and not inside(path,output))
            if denied:
                report["blocked_access"].append({"event":event,"path":str(path),"write":writing})
                raise PermissionError("Packet replay disallows non-packet native source/data/model or outside-output write: "+str(path))
            if inside(path,native) and event=="open" and not writing:
                key=path.relative_to(native).as_posix();report["native_reads"][key]=report["native_reads"].get(key,0)+1
        sys.addaudithook(guard)
        path=audit/"native_engine_adapter.py"
        spec=importlib.util.spec_from_file_location("packet_native_engine_adapter",path)
        adapter=importlib.util.module_from_spec(spec);sys.modules[spec.name]=adapter;spec.loader.exec_module(adapter)
        # Only the external path roots change. Every numerical source byte stays frozen.
        adapter.ENGINE=native;adapter.HERE=runtime_root;adapter.CODE=bundle/"github/accounting-options-rl";adapter.PAPER1=bundle
        def native_origins():
            origins={}
            for name in native_names:
                module=sys.modules.get(name)
                if module is not None:
                    origin=Path(module.__file__).resolve();assert inside(origin,native),"Native module leaked outside packet: "+name
                    origins[name]=origin.relative_to(native).as_posix()
            for name,module in tuple(sys.modules.items()):
                origin=getattr(module,"__file__",None)
                if origin:assert not inside(Path(origin).resolve(),original_native),"Original native module fallback: "+name
            return origins
        with ((output/"native_stdout_stderr.log").open("w",encoding="utf-8") as log,
              contextlib.redirect_stdout(log),contextlib.redirect_stderr(log)):
            runtime=adapter.NativeRuntime();native_origins()
            choices=[("plain7_strict","strict",native/"batch_tier_phase_runs/prototype5_mappo_marl_final_v1/seed7/tier1_seed7/final_models"),
                ("feasible7_vulnerable","vulnerable",native/"Ablations/batch_tier_phase_runs/prototype5_mechanism_ablations_final_v1/feasible_action_full/seed7/tier1_forecast_utilization_seed7/final_models")]
            for name,profile,checkpoint in choices:
                scenario=adapter.NativeScenario(runtime,adapter.EpisodeSpec(str(checkpoint),horizon=96,burn_in=384,profile=profile,cash_fraction=.1))
                signs=np.ones(96,dtype=np.int8)
                if profile=="vulnerable":signs[1::2]=-1
                episode=scenario.new_episode(signs)
                assert episode.env.t==384 and episode.env.budget==.1*episode.env._initial_trading_sleeve_dkk()
                for _ in range(96):episode.step()
                arrays=episode.trace_arrays();summary=episode.summary()
                assert summary["terminal_open_contracts"]==0 and summary["immutable_warm_snapshot_unchanged"] and summary["immutable_config_snapshot_unchanged"] and summary["clone_all_field_hash_equal_before_intervention"]
                assert set(arrays)==set(expected[name]),"Same complete trace schema"
                comparisons=[]
                for key in sorted(arrays):
                    if key=="path_step_seconds_excluding_actor_inference":continue
                    actual,golden=np.asarray(arrays[key]),np.asarray(expected[name][key]);assert actual.shape==golden.shape,"Trace shape "+key
                    if actual.dtype.kind in "biuUS":
                        assert np.array_equal(actual,golden),"Exact integer/string trace "+key
                        error=0.;rtol=atol=0.
                    else:
                        assert np.all(np.isfinite(actual)) and np.all(np.isfinite(golden)),"Finite trace "+key
                        rtol,atol=(1e-7,1e-7) if "actor_observation" in key or "actor_action" in key else (1e-10,1e-6)
                        error=float(np.max(np.abs(actual-golden),initial=0.))
                        assert np.allclose(actual,golden,rtol=rtol,atol=atol),"Numerical trajectory mismatch "+key
                    comparisons.append({"array":key,"shape":list(actual.shape),"max_absolute_error":error,"rtol":rtol,"atol":atol})
                np.savez_compressed(output/(name+"_packet_trace.npz"),**arrays)
                resource_keys=scenario.metadata["resource_sha256"]
                report["cases"].append({"case":name,"status":"PASS","compared_arrays":comparisons,"native_actor_contract":summary["model_contract_checks"],
                    "resource_sha256":resource_keys,"terminal_queue_empty":True,"parent_and_config_immutable":True,"native_audit_steps":96,"deterministic_warm_steps":384})
                runtime.assert_sources_unchanged();native_origins()
            report["loaded_native_module_origins"]=native_origins()
            report["status"]="PASS";report["native_audit_steps"]=192;report["deterministic_warm_steps"]=768
        for name,digest in frozen["native_source_sha256"].items():assert sha(native/name)==digest
        for name,digest in frozen["source_sha256"].items():assert sha(audit/name)==digest
        assert not report["blocked_access"],"No denied/original fallback access"
    except BaseException as error:
        report["status"]="SOFTWARE_FAILURE";report["error"]=repr(error);report["traceback"]=traceback.format_exc()
    report["wall_seconds"]=time.perf_counter()-wall;report["cpu_seconds"]=time.process_time()-cpu
    with (output/"replay_report.json").open("x",encoding="utf-8") as stream:json.dump(report,stream,indent=2,sort_keys=True,allow_nan=False)
    print(json.dumps({key:report[key] for key in ("status","wall_seconds","cpu_seconds")}));print(str(output))
    if report["status"]!="PASS":print(report["traceback"]);return 1
    return 0

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--execute",action="store_true")
    parser.add_argument("--bundle",type=Path,default=HERE/"review_bundle_private/native_review_complete_v1")
    parser.add_argument("--reference-dir",type=Path,default=HERE/"native_validation/warm_attempt_2")
    parser.add_argument("--output-dir",type=Path,default=HERE/"native_validation/clean_bundle_replay_attempt_4")
    args=parser.parse_args()
    if not args.execute:
        print(json.dumps({"status":"PREPARED_NOT_EXECUTED_WAITING_ROOT_RELEASE","wrapper_sha256":sha(__file__),"native_imports":0,"actor_calls":0,"native_steps":0}));return 0
    return execute(args.bundle,args.reference_dir,args.output_dir)

if __name__=="__main__":raise SystemExit(main())
