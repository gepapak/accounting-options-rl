"""Native whole-episode phase generation for the frozen sample-only audit.

Importing this module performs no simulation, checkpoint inference or fitting.
Only the parent frozen runner may call run_case. Complete/partial files are never
overwritten. Progress contains counts, never financial endpoints. Native actors
remain fixed; audit critics are loaded only for CONFIRM and are never refitted.
"""
from __future__ import annotations

import os
for _thread_key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_key] = "1"
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SCHEMA = "native_sample_only_phase_v1"
PHASE_BANDS = {"fit": 26300000, "tune": 27300000, "confirm": 28300000}
PHASE_N = {"fit": 32, "tune": 32, "confirm": 256}
HORIZON, BURN_IN, CASH_FRACTION, GAMMA = 96, 384, 1e-4, .995
MONEY_SCALE = 662068965.5172415
CHECKPOINT_SEEDS = (7, 42, 123, 2025, 3007, 5001, 8102, 9005, 10001, 11202)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n").encode("utf-8")


def publish_bytes(path, content):
    """Write an exclusive partial, then publish without replacing any artifact."""
    path = Path(path)
    partial = path.with_name(path.name+".partial")
    if path.exists() or partial.exists():
        raise FileExistsError("Preserve prior/partial artifact: "+str(path))
    with partial.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        raise FileExistsError("Concurrent artifact publication: "+str(path))
    # Windows Path.rename refuses an existing destination. Each phase has one
    # owner, and its exclusive directory creation prevents a competing writer.
    partial.rename(path)


def publish_json(path, value):
    publish_bytes(path, json_bytes(value))


def publish_npz(path, **arrays):
    import numpy as np
    path = Path(path)
    partial = path.with_name(path.name+".partial")
    if path.exists() or partial.exists():
        raise FileExistsError("Preserve prior/partial artifact: "+str(path))
    with partial.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        raise FileExistsError("Concurrent NPZ publication: "+str(path))
    partial.rename(path)


def validate_case(case):
    required = ("case_id", "policy_index", "policy_family", "checkpoint_seed", "checkpoint_dir",
                "profile_index", "profile", "cash_fraction", "horizon", "burn_in")
    if any(key not in case for key in required):
        raise ValueError("Incomplete frozen case schema")
    cid, policy, profile = int(case["case_id"]), int(case["policy_index"]), int(case["profile_index"])
    if not (0 <= policy < 20 and profile in (0, 1) and cid == 2*policy+profile):
        raise ValueError("Case/policy/profile indices disagree")
    if case["profile"] != ("strict", "vulnerable")[profile]:
        raise ValueError("Profile name disagrees with index")
    if case["policy_family"] != ("plain_mappo" if policy < 10 else "feasible_mappo"):
        raise ValueError("Policy family disagrees with inventory order")
    if int(case["checkpoint_seed"]) != CHECKPOINT_SEEDS[policy % 10]:
        raise ValueError("Checkpoint seed disagrees with frozen inventory order")
    if (float(case["cash_fraction"]) != CASH_FRACTION or int(case["horizon"]) != HORIZON
            or int(case["burn_in"]) != BURN_IN):
        raise ValueError("Unfrozen cash/horizon/warm-history intervention")
    directory = Path(case["checkpoint_dir"])
    if not directory.is_absolute() or directory.name != "final_models":
        raise ValueError("Expected absolute authentic final_models directory")
    for name in ("investor_0_policy.zip", "battery_operator_0_policy.zip", "training_config.json"):
        if not (directory/name).is_file():
            raise FileNotFoundError(directory/name)


def _feature_options(case):
    return dict(money_scale=MONEY_SCALE, horizon=HORIZON,
                policy_index=int(case["policy_index"]), profile_index=int(case["profile_index"]))


def _source_hashes():
    names = ("native_phase_worker.py", "native_engine_adapter.py", "native_support.py",
             "sample_estimators.py", "audit_models.py", "checkpoint_inventory.json")
    return {name: sha(HERE/name) for name in names}


def _native_activity(episode, trace):
    import numpy as np
    actions = np.asarray(trace["path_actor_action"], float)
    return dict(actor_forward_calls=2*HORIZON,
                nonzero_investor_action_steps=int(np.count_nonzero(np.abs(actions) > 1e-12)),
                maximum_abs_investor_delta=float(np.max(np.abs(actions), initial=0.)),
                maturity_events=sum(event["kind"] == "maturity" for event in episode.events),
                roll_cost_events=sum(event["kind"] == "roll_cost" for event in episode.events),
                terminal_open_contracts=len(episode.env._horizon_contracts),
                native_trading_enabled=bool(episode.env.reward_calculator.trading_enabled),
                native_margin_active=bool(getattr(episode.env, "_trading_sleeve_margin_active", False)))


def _prepare(runtime, case):
    """Only pre-draw context/support preparation; no endpoint-based decisions."""
    import numpy as np
    from native_engine_adapter import EpisodeSpec, NativeScenario, fixed_context_grid
    from native_support import tightened_calendar_support, union_context_support
    contexts = fixed_context_grid(runtime, burn_in=BURN_IN, H=HORIZON)
    if len(contexts) != 8 or [int(row["context_id"]) for row in contexts] != list(range(8)):
        raise AssertionError("Exactly eight fixed context IDs are required")
    scenarios, context_records, supports = [], [], []
    cpu, wall = time.process_time(), time.perf_counter()
    for context in contexts:
        spec = EpisodeSpec(str(case["checkpoint_dir"]), zone=context["zone"],
            start_row=int(context["start_row"]), horizon=HORIZON, burn_in=BURN_IN,
            native_seed=int(case["checkpoint_seed"]), profile=case["profile"], cash_fraction=CASH_FRACTION)
        scenario = NativeScenario(runtime, spec)
        # The initialization reveals no financial endpoint and takes zero native
        # transitions. Context support depends on declared arrays and warm state.
        probe = scenario.new_episode(np.ones(HORIZON, dtype=np.int8))
        original_f = float(probe.env._initial_trading_sleeve_dkk())
        if not np.isclose(original_f, MONEY_SCALE, rtol=1e-12, atol=1e-6):
            raise AssertionError("Original sizing base differs from frozen feature scale")
        support = tightened_calendar_support(probe, gamma=GAMMA)
        supports.append(support)
        scenarios.append(scenario)
        context_records.append(dict(context=context, metadata=scenario.metadata, support=support,
            original_sizing_base=original_f, support_initialization_native_transitions=0))
        del probe
    union = union_context_support(supports)
    if not all(np.isfinite(float(union[key])) and float(union[key]) >= 0
               for key in ("discounted_cost_bound", "discounted_gross_bound")):
        raise AssertionError("Invalid complete-context support union")
    return scenarios, context_records, union, dict(
        cpu_seconds=time.process_time()-cpu, wall_seconds=time.perf_counter()-wall,
        warm_native_transitions=len(contexts)*BURN_IN,
        warm_actor_forward_calls=2*len(contexts)*BURN_IN,
        zero_step_support_initializations=len(contexts))


def _one_phase(runtime, case, phase, directory, n, scenarios, contexts, bounds, models, source_hashes):
    import numpy as np
    from native_engine_adapter import jsonable
    import sample_estimators as estimators
    from native_support import check_trace_support

    if phase not in PHASE_BANDS or int(n) != PHASE_N[phase]:
        raise ValueError("Phase and complete whole-episode budget must match frozen grid")
    directory.mkdir(parents=True, exist_ok=False)
    started, cpu, wall = utc_now(), time.process_time(), time.perf_counter()
    rows_partial = directory/"rows.jsonl.partial"
    plus_all, minus_all, actual_all, targets_all, coins_all = [], [], [], [], []
    event_fields = ("step", "cash_before", "cash_after", "gross", "cost", "injection",
                    "sign_epsilon", "unsigned_base_gross")
    event_arrays = {key: [] for key in event_fields}
    event_offsets = [0]
    offsets = [0]
    predictions = {name+"_"+branch: [] for name in ("ridge", "mlp") for branch in ("plus", "minus")}
    completed = 0
    query = dict(audit_native_transitions=0, actor_forward_calls=0,
                 branch_injection_arithmetic_evaluations=0, critic_branch_rows_per_model=0,
                 critic_predict_calls=0, fresh_audit_clones=0)
    try:
        if phase == "confirm" and (models is None or set(models) != {"ridge", "mlp"}):
            raise ValueError("CONFIRM requires both completed frozen nuisance predictors")
        trace_directory = directory/"private_traces"
        if phase in ("fit", "tune"):
            trace_directory.mkdir()
        with rows_partial.open("x", encoding="utf-8", newline="\n") as row_stream:
            for episode_id in range(int(n)):
                seed_vector = [20261016, PHASE_BANDS[phase], int(case["case_id"]), episode_id]
                rng = np.random.default_rng(np.random.SeedSequence(seed_vector))
                context_id = int(rng.integers(0, len(scenarios)))
                signs = (2*rng.integers(0, 2, HORIZON, dtype=np.int8)-1).astype(np.int8)
                scenario = scenarios[context_id]
                episode = scenario.new_episode(signs)
                initial_pending = len(episode.env._horizon_contracts)
                query["fresh_audit_clones"] += 1
                for _ in range(HORIZON):
                    episode.step()
                trace = episode.trace_arrays()
                check_trace_support(trace, contexts[context_id]["support"])
                native = episode.summary()
                if (native["steps"] != HORIZON or native["terminal_open_contracts"] != 0
                        or not native["immutable_warm_snapshot_unchanged"]
                        or not native["immutable_config_snapshot_unchanged"]
                        or not native["clone_all_field_hash_equal_before_intervention"]):
                    raise AssertionError("Native complete-episode/isolation gate failed")
                summarized = estimators.summarize(trace)
                for key in event_fields:
                    event_arrays[key].append(np.asarray(trace["event_"+key]))
                event_offsets.append(event_offsets[-1]+summarized["events"])
                options = _feature_options(case)
                plus = estimators.branch_features(trace, summarized, branch=1, **options)
                minus = estimators.branch_features(trace, summarized, branch=-1, **options)
                actual, target = estimators.fit_rows(trace, summarized, **options)
                active = summarized["active_indices"]
                coins = summarized["coin"][active].astype(np.int8)
                plus_all.append(plus); minus_all.append(minus)
                actual_all.append(actual); targets_all.append(target); coins_all.append(coins)
                offsets.append(offsets[-1]+len(coins))
                row = dict(schema=SCHEMA, case_id=int(case["case_id"]), phase=phase,
                    episode_id=episode_id, context_id=context_id, seed_vector=seed_vector,
                    hidden_signs=signs.tolist(), gross=summarized["gross"], cost=summarized["cost"],
                    forgiveness=summarized["forgiveness"], paired_forgiveness=summarized["paired_forgiveness"],
                    event_count=summarized["events"], maturity_coins=coins.tolist(),
                    maturity_events=summarized["maturity_events"],
                    initial_pending_prepaid_contracts=initial_pending,
                    terminal_open_contracts=int(native["terminal_open_contracts"]),
                    excluded_native_cash_change=float(native["excluded_native_cash_change"]),
                    identity_error=summarized["identity_error"],
                    branch_start=offsets[-2], branch_stop=offsets[-1],
                    native_activity=_native_activity(episode, trace),
                    clone_seconds=float(native["clone_seconds"]),
                    native_shaped_reward_descriptive=float(native["native_shaped_reward_descriptive"]),
                    native_nav_change_descriptive=float(native["native_fund_nav_change_descriptive"]))
                if models is not None:
                    for name in ("ridge", "mlp"):
                        p = np.asarray(models[name].predict(plus), float)
                        m = np.asarray(models[name].predict(minus), float)
                        if p.shape != coins.shape or m.shape != coins.shape or not np.all(np.isfinite(np.r_[p, m])):
                            raise AssertionError("Frozen common-branch predictor shape/finite gate")
                        predictions[name+"_plus"].append(p); predictions[name+"_minus"].append(m)
                        row["delta_"+name] = ((p-m)/2).tolist()
                    query["critic_branch_rows_per_model"] += 2*len(coins)
                    query["critic_predict_calls"] += 4
                if phase in ("fit", "tune"):
                    trace_path = trace_directory/f"episode_{episode_id:04d}.npz"
                    publish_npz(trace_path, **trace)
                    row["private_trace_path"] = str(trace_path.resolve())
                    row["private_trace_sha256"] = sha(trace_path)
                row_stream.write(json.dumps(jsonable(row), sort_keys=True, allow_nan=False)+"\n")
                row_stream.flush()
                query["audit_native_transitions"] += HORIZON
                query["actor_forward_calls"] += 2*HORIZON
                query["branch_injection_arithmetic_evaluations"] += 2*summarized["events"]
                completed += 1
                if completed % 8 == 0 or completed == n:
                    print(json.dumps(dict(event="phase_progress", case_id=int(case["case_id"]),
                                          phase=phase, completed_episodes=completed, planned_episodes=n)), flush=True)
                del episode, trace
            os.fsync(row_stream.fileno())
        if completed != n or query["audit_native_transitions"] != n*HORIZON:
            raise AssertionError("Incomplete phase cannot publish completion")
        runtime.assert_sources_unchanged()
        if source_hashes != _source_hashes():
            raise AssertionError("Audit source changed during phase")
        row_path = directory/"rows.jsonl"
        rows_partial.rename(row_path)
        concatenate = lambda values, width=None: (np.concatenate(values, axis=0) if values else
            np.empty((0, width), float) if width is not None else np.empty(0, float))
        branch_path = directory/"branch_features.npz"
        branch_arrays = dict(plus=concatenate(plus_all, 32), minus=concatenate(minus_all, 32),
            coins=concatenate(coins_all).astype(np.int8), episode_offsets=np.asarray(offsets, dtype=np.int64))
        if models is not None:
            branch_arrays.update({"q_"+name: concatenate(values) for name, values in predictions.items()})
            for name in ("ridge", "mlp"):
                branch_arrays["delta_"+name] = (branch_arrays["q_"+name+"_plus"]-branch_arrays["q_"+name+"_minus"])/2
        publish_npz(branch_path, **branch_arrays)
        event_path = directory/"financial_events.npz"
        publish_npz(event_path, **{"event_"+name: concatenate(values)
                                   for name, values in event_arrays.items()},
                    episode_offsets=np.asarray(event_offsets, dtype=np.int64))
        fit_path = None
        if phase in ("fit", "tune"):
            fit_path = directory/"fit_rows.npz"
            publish_npz(fit_path, X=concatenate(actual_all, 32), y=concatenate(targets_all),
                        episode_offsets=np.asarray(offsets, dtype=np.int64))
        metadata = dict(schema=SCHEMA, status="COMPLETE", case=case, phase=phase,
            started_utc=started, completed_utc=utc_now(), planned_episodes=n, completed_episodes=completed,
            seed_contract="default_rng(SeedSequence([20261016,phase_band,case_id,episode_id])); uniform context then 96 independent fair signs",
            phase_band=PHASE_BANDS[phase], horizon=HORIZON, burn_in=BURN_IN, gamma=GAMMA,
            money_scale=MONEY_SCALE, cases_contexts=contexts, bounds=bounds, support_union=bounds, query_counts=query,
            source_sha256=source_hashes, native_source_sha256=runtime.source_sha256,
            cpu_seconds=time.process_time()-cpu, wall_seconds=time.perf_counter()-wall,
            strict_profile_scope="guard-restored; warm prepaid claims can mature before margin checks; inactivity is not assumed",
            all_nuisance_models_frozen_before_confirmation=phase == "confirm",
            feature_contract="32 common-state plus/minus branch features, fixed SAME predictor; targets globally discounted after-event future paired contribution",
            private_trace_scope="development only" if phase != "confirm" else "confirmation branch features and predictions cached; native actors not needed to verify bounds")
        metadata_path = directory/"metadata.json"
        publish_json(metadata_path, jsonable(metadata))
        artifact_paths = [row_path, branch_path, event_path, metadata_path]+([fit_path] if fit_path is not None else [])
        completion = dict(schema=SCHEMA, status="COMPLETE", case_id=int(case["case_id"]), phase=phase,
            completed_episodes=completed, artifacts={path.name: dict(path=str(path.resolve()), sha256=sha(path))
                                                   for path in artifact_paths})
        completion_path = directory/"completion.json"
        publish_json(completion_path, completion)
        return dict(case_id=int(case["case_id"]), phase=phase, completion_manifest_path=str(completion_path.resolve()),
                    summary_path=str(row_path.resolve()), feature_path=str(branch_path.resolve()),
                    fit_rows_path=None if fit_path is None else str(fit_path.resolve()),
                    financial_events_path=str(event_path.resolve()), metadata_path=str(metadata_path.resolve()))
    except BaseException as error:
        failure = dict(schema=SCHEMA, status="INCOMPLETE", case_id=int(case["case_id"]), phase=phase,
            planned_episodes=n, completed_episodes=completed, failed_utc=utc_now(), error=repr(error),
            traceback=traceback.format_exc(), query_counts=query,
            cpu_seconds=time.process_time()-cpu, wall_seconds=time.perf_counter()-wall)
        failure_path = directory/"failure.json"
        if not failure_path.exists() and not failure_path.with_name(failure_path.name+".partial").exists():
            publish_json(failure_path, failure)
        raise


def run_case(case, phase, output_dir, n, models_dir=None):
    """Generate frozen FIT32+TUNE32 or CONFIRM256, with no import-time draw.

    development creates case_XX/fit and case_XX/tune in one native process and
    shares only its pre-draw immutable warm snapshots. Parent must fit all critics
    and freeze every selection before it dispatches any confirm case.
    """
    validate_case(case)
    if phase not in ("development", "confirm"):
        raise ValueError("Parent phase must be development or confirm")
    if int(n) != (32 if phase == "development" else 256):
        raise ValueError("Complete frozen phase budget is mandatory")
    if (phase == "confirm") != (models_dir is not None):
        raise ValueError("Only CONFIRM loads frozen audit critics")
    output = Path(output_dir).resolve()/f"case_{int(case['case_id']):02d}"
    phases = ("fit", "tune") if phase == "development" else ("confirm",)
    output.mkdir(parents=True, exist_ok=True)
    if any((output/name).exists() for name in phases):
        raise FileExistsError("Never overwrite or resume a partial/completed phase")
    source_hashes = _source_hashes()
    from native_engine_adapter import NativeRuntime
    runtime = NativeRuntime()
    try:
        models = None
        if phase == "confirm":
            from audit_models import load_models
            models = load_models(models_dir)
        scenarios, contexts, bounds, preparation = _prepare(runtime, case)
        results = [_one_phase(runtime, case, name, output/name, PHASE_N[name], scenarios,
                              contexts, bounds, models, source_hashes) for name in phases]
        manifest_path = output/(phase+"_completion.json")
        publish_json(manifest_path, dict(schema=SCHEMA, status="COMPLETE", phase=phase,
            case_id=int(case["case_id"]), case=case, preparation=preparation, bounds=bounds,
            phases=results, source_sha256=source_hashes, completed_utc=utc_now()))
        return dict(case_id=int(case["case_id"]), phase=phase,
                    completion_manifest_path=str(manifest_path.resolve()), phases=results, bounds=bounds)
    finally:
        # A worker may later receive another case. Keep imported native modules,
        # not forty cases' economic warm buffers, actor weights or market views.
        runtime.warm_cache.clear()
        runtime.model_cache.clear()
        runtime.data_cache.clear()
        runtime.forecast_cache.clear()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-json", type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job_json.read_text(encoding="utf-8"))
    result = run_case(job["case"], job["phase"], job["output_dir"], job["n"], job.get("models_dir"))
    print(json.dumps(dict(event="case_complete", case_id=result["case_id"], phase=result["phase"],
                         completion_manifest_path=result["completion_manifest_path"])), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
