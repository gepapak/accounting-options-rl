"""Read-only arithmetic verification of completed native sample-only records.

No engine import, simulator draw, learned actor inference or fitting. Optional
--check-critics applies only the trusted saved audit regressors to stored branch
features after their manifests and hashes have been checked. The default checks
saved common-branch predictions and all confidence arithmetic independently.
Outputs, if requested, are written exclusively inside this validation directory.
"""
from __future__ import annotations
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

sys.dont_write_bytecode = True
VALIDATION = Path(__file__).resolve().parent
HERE = VALIDATION.parent
CODE = HERE.parents[2]
ENGINE = CODE.parents[1]/"Prototype5"
METHODS = ("raw", "capped_raw", "capped_residual", "paired", "ridge_control", "mlp_control", "cash_capped_raw", "cash_ridge_control", "cash_mlp_control", "cash_local_control")
PHASE_N = {"fit": 32, "tune": 32, "confirm": 256}
PHASE_BAND = {"fit": 36300000, "tune": 37300000, "confirm": 38300000}
DELTA, HORIZON, GAMMA = .05/340, 96, .995


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(array):
    value = np.ascontiguousarray(array, dtype="<f8")
    header = (json.dumps(dict(shape=list(value.shape), dtype="little-endian float64"),
                         sort_keys=True, indent=2, allow_nan=False)+"\n").encode("utf-8")
    return hashlib.sha256(header+value.tobytes(order="C")).hexdigest()


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


class Checks:
    def __init__(self):
        self.count = 0
        self.max_error = {}

    def yes(self, condition, message):
        self.count += 1
        if not bool(condition):
            raise AssertionError(message)

    def close(self, a, b, message, atol=1e-6, rtol=2e-12, category="arithmetic"):
        left, right = np.asarray(a, float), np.asarray(b, float)
        self.yes(left.shape == right.shape, message+" shape")
        error = float(np.max(np.abs(left-right), initial=0.))
        self.max_error[category] = max(error, self.max_error.get(category, 0.))
        self.yes(np.allclose(left, right, atol=atol, rtol=rtol), message)


def pad(bounds, choice=None):
    return max(1e-6, 1e-10*(1+float(bounds["discounted_gross_bound"])+
        float(bounds["discounted_cost_bound"])+float((choice or {}).get("budget", 0.)))*HORIZON)


def support(bounds, method, alpha, choice):
    c, g = float(bounds["discounted_cost_bound"]), float(bounds["discounted_gross_bound"])
    if method.startswith("cash_"):
        if alpha != 0.:raise ValueError("Cash lower endpoint is alpha0 only")
        extra=float(choice['budget']) if method.endswith('_control') else 0.
        return -float(bounds["cash_inflow_bound"])-extra, min(float(choice["cap"]),g+extra)
    if method == "raw":
        return -g-c, g
    if method == "capped_raw":
        return -g-c, min(float(choice["cap"]), g)
    budget = float(choice["budget"]) if method.endswith("_control") else 0.
    upper = ((1-alpha)*g if method == "capped_residual" else (1-alpha)*g/2)+budget
    return -c-budget, min(float(choice["cap"]), upper)


def control(deltas, coins, beta, budget):
    """Sequential remaining-budget reconstruction, independent of core cumsum."""
    remaining, result = float(budget), 0.
    for delta, coin in zip(deltas, coins):
        amplitude = min(abs(float(beta)*float(delta)), remaining)
        result += float(coin)*np.sign(float(delta))*amplitude
        remaining = max(remaining-amplitude, 0.)
    return float(result)


def values(rows, method, alpha, choice):
    result = []
    for row in rows:
        if method in ("raw", "capped_raw", "cash_capped_raw", "cash_ridge_control", "cash_mlp_control", "cash_local_control"):
            x = row["gross"]-row["cost"]+(1-alpha)*row["forgiveness"]
            if method.endswith('_control'):
                x-=control(row['delta_'+method.removesuffix('_control')],row['maturity_coins'],choice['beta'],choice['budget'])
        elif method == "capped_residual":
            x = (1-alpha)*row["forgiveness"]-row["cost"]
        else:
            x = (1-alpha)*row["paired_forgiveness"]-row["cost"]
            if method.endswith("_control"):
                x -= control(row["delta_"+method.removesuffix("_control")],
                             row["maturity_coins"], choice["beta"], choice["budget"])
        result.append(x if method == "raw" else min(x, choice["cap"]))
    return np.asarray(result, float)


def eb(x, lower, upper, allowance):
    x = np.asarray(x, float)
    n = len(x)
    mean, variance = float(np.sum(x)/n), float(np.sum((x-np.mean(x))**2)/(n-1))
    expanded_lower, expanded_upper = lower-allowance, upper+allowance
    log = np.log(2/DELTA)
    radius = float(np.sqrt(2*variance*log/n)+7*(expanded_upper-expanded_lower)*log/(3*(n-1)))
    return dict(n=n, mean=mean, variance=variance, radius=radius, lower=mean-radius-allowance,
                support_lower=expanded_lower, support_upper=expanded_upper,
                delta=DELTA, numerical_allowance=allowance)


def verify_certificate(check, saved, expected, x, prefix):
    check.yes(saved["n"] == expected["n"], prefix+" independent whole-path n")
    for key in expected:
        if key != "n":
            check.close(saved[key], expected[key], prefix+" "+key, atol=1e-6, category="confidence")
    check.yes(np.isfinite(x).all(), prefix+" finite sample")
    check.yes(float(x.min()) >= expected["support_lower"]-1e-6 and
              float(x.max()) <= expected["support_upper"]+1e-6, prefix+" deterministic support")


def verify_external_resources(check, run_dir, case_list):
    """Hash each actual external file once per stable size/mtime identity."""
    public_path = HERE.parent/"public_reproducibility_manifest.json"
    preparation_path = HERE.parent/"native_validation/forecast_cache_prep_attempt_1/preparation_report.json"
    public, prepared = read(public_path), read(preparation_path)
    check.yes(public["source_freeze_sha256"] == read(run_dir/"SOURCE_FREEZE.json")["original_freeze_sha256"], "public external-input manifest source binding")
    check.yes(prepared["status"] == "PASS" and prepared["audit_steps"] == prepared["actor_forward_calls"] == 0,
              "forecast view preparation used no episode or actor query")
    known = {row["path_from_native_engine"]: row["sha256"] for key in
             ("checkpoint_artifacts", "market_history_forecast_resources") for row in public[key]}
    forecast = {int(row["context_id"]): row for row in prepared["contexts"]}
    check.yes(set(forecast) == set(range(8)), "eight complete prepared forecast views")
    cache = {}
    def cached_sha(path):
        path = Path(path).resolve()
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        if key not in cache:
            cache[key] = sha(path)
        return cache[key]
    contexts_checked = 0
    for case in case_list:
        fingerprints = None
        for phase in PHASE_N:
            metadata = read(run_dir/f"case_{int(case['case_id']):02d}"/phase/"metadata.json")
            current = []
            for record in metadata["cases_contexts"]:
                context, native = record["context"], record["metadata"]
                cid = int(context["context_id"])
                resources = native["resource_sha256"]
                current.append((context, resources, native["config_snapshot_sha256"],
                                native["warm_snapshot_sha256"], record["support"]))
                for name, digest in resources.items():
                    if name == "training_config":
                        actual = Path(case["checkpoint_dir"])/"training_config.json"
                        check.yes(digest == known[actual.relative_to(ENGINE).as_posix()], "published training-config hash")
                    elif name == "native_forecast_source":
                        actual = Path(forecast[cid]["source_path"])
                        check.yes(digest == forecast[cid]["source_sha256"] == known[actual.relative_to(ENGINE).as_posix()],
                                  "published native forecast-source hash")
                    elif name == "forecast_view":
                        key = hashlib.sha256(json.dumps(dict(source=forecast[cid]["source_path"],start=int(context["start_row"]),H=480),sort_keys=True).encode()).hexdigest()[:20]
                        actual = HERE/"runtime_cache/forecast_views"/key/"precomputed_forecasts_native_view.csv"
                        check.yes(digest == forecast[cid]["cache_sha256"], "exact prepared forecast-view bytes")
                    else:
                        actual = ENGINE/name
                        check.yes(digest == known[name.replace("\\", "/")], "published native data/history/checkpoint hash")
                    check.yes(cached_sha(actual) == digest, "actual frozen external resource identity")
                contexts_checked += 1
            if fingerprints is None:
                fingerprints = current
            else:
                check.yes(current == fingerprints, "FIT/TUNE/CONFIRM resource/config/warm/support identity")
    return dict(context_records_checked=contexts_checked, unique_stable_file_identities_hashed=len(cache),
                public_manifest_sha256=sha(public_path), forecast_preparation_sha256=sha(preparation_path))


def verify_phase(check, run_dir, case, phase, critics=None):
    cid = int(case["case_id"])
    directory = run_dir/f"case_{cid:02d}"/phase
    completion, metadata = read(directory/"completion.json"), read(directory/"metadata.json")
    check.yes(completion["status"] == metadata["status"] == "COMPLETE", "complete phase gate")
    for artifact in completion["artifacts"].values():
        check.yes(sha(artifact["path"]) == artifact["sha256"], "phase artifact hash")
    rows = load_rows(directory/"rows.jsonl")
    check.yes(len(rows) == PHASE_N[phase] == metadata["completed_episodes"], "whole phase budget")
    check.yes(metadata["case"] == case, "phase case contract")
    contexts = metadata["cases_contexts"]
    check.yes(len(contexts) == 8, "eight predetermined contexts")
    for record in contexts:
        declared = record["support"]
        weights=GAMMA**np.arange(HORIZON)
        battery=np.asarray(declared['calendar_battery_positive_credit_bound'],float)
        oldcredit=np.asarray(declared['calendar_old_positive_credit_bound'],float)
        check.yes(battery.shape==oldcredit.shape==(HORIZON,) and np.all(battery>=0) and np.all(oldcredit>=0),'source cash-inflow calendar support')
        exact_K=float(declared['cash_inflow_initial_cash'])+float(weights@(battery+oldcredit))
        check.yes(declared['cash_inflow_bound']>=exact_K,'source K covers initial plus every excluded-positive allowance')
        check.yes(declared['cash_inflow_battery_credit_bound']>=float(weights@battery) and
            declared['cash_inflow_old_maturity_credit_bound']>=float(weights@oldcredit),'cash budget components rounded outward')
        burn, lag = int(declared["burn_in"]), int(declared["settlement_lag"])
        check.yes(burn == 384 and lag == 6, "source fixed warm-history and settlement clocks")
        for claim in declared["initial_prepaid_claims"]:
            opened, settles = int(claim["open_native_step"]), int(claim["settle_native_step"])
            check.yes(opened < burn and settles == opened+lag and burn <= settles < burn+lag,
                      "initial prepaid claim source boundary for separate auxiliary component")
        check.yes(all(burn+i+lag >= burn+lag for i, allowed in
                      enumerate(declared["calendar_new_opening_possible"]) if allowed),
                  "all possible audit-created maturities after initial-claim boundary")
    bounds = metadata["bounds"]
    saved_config = read(Path(case["checkpoint_dir"])/"training_config.json")
    execution = saved_config["runtime_contract"]["execution"]
    final_config = saved_config["final_config"]
    hours = float(final_config.get("time_step_hours", 10./60.))
    check.close(hours, 10./60., "source-default native ten-minute clock")
    annual = float(execution["annual_market_access_fee_dkk"])
    allocation = float(execution["market_access_fee_allocation_fraction"])
    check.close(final_config["annual_market_access_fee_dkk"], annual, "saved access-fee contract coherence")
    check.close(final_config["market_access_fee_allocation_fraction"], allocation, "saved fee-allocation coherence")
    exact_fee = max(annual, 0.)*min(max(allocation, 0.), 1.)*max(hours, 0.)/(365.25*24.)
    check.yes(exact_fee > 0., "positive mandatory native access fee")
    exact_access_path = float(exact_fee*np.sum(GAMMA**np.arange(HORIZON)))
    check.yes(0. < bounds["unavoidable_discounted_access_cost"] <= exact_access_path,
              "analytical access LOWER bound does not exceed exact mandatory fee")
    for key in ("discounted_cost_bound", "discounted_gross_bound", "cash_inflow_bound"):
        check.close(bounds[key], max(record["support"][key] for record in contexts), "support mixture maximum")
    events = np.load(directory/"financial_events.npz", allow_pickle=False)
    cashflow = np.load(directory/"cash_flow_checks.npz", allow_pickle=False)
    branches = np.load(directory/"branch_features.npz", allow_pickle=False)
    eoff, boff = events["episode_offsets"], branches["episode_offsets"]
    check.yes(len(eoff) == len(boff) == len(rows)+1 and eoff[0] == boff[0] == 0, "per-episode offsets")
    check.yes(np.all(np.diff(eoff) >= 0) and np.all(np.diff(boff) >= 0), "monotone offsets")
    check.yes(branches["plus"].shape == branches["minus"].shape == (int(boff[-1]), 32), "branch feature dimensions")
    # Only the hypothetical coin, resulting cash and resulting injection differ.
    common = [j for j in range(32) if j not in (23, 24, 25)]
    check.close(branches["plus"][:, common], branches["minus"][:, common], "same common branch information")
    check.close(branches["plus"][:, 23], np.ones(int(boff[-1])), "positive hypothetical sign")
    check.close(branches["minus"][:, 23], -np.ones(int(boff[-1])), "negative hypothetical sign")
    if phase == "confirm":
        for model in ("ridge", "mlp", "cash_ridge", "cash_mlp"):
            expected_delta = (branches["q_"+model+"_plus"]-branches["q_"+model+"_minus"])/2
            if model.startswith('cash_'):expected_delta+=branches['known_current_raw_delta']
            check.close(branches["delta_"+model], expected_delta, "common-model centered "+model)
            if critics is not None:
                check.close(critics[model].predict(branches["plus"]), branches["q_"+model+"_plus"],
                            "frozen critic positive branch "+model)
                check.close(critics[model].predict(branches["minus"]), branches["q_"+model+"_minus"],
                            "frozen critic negative branch "+model)
    diagnostics = dict(events=0, maturities=0, pending_prepaid_episodes=0,
                       minimum_financial_cash_before=None, strictly_negative_cash_before=0,
                       negative_cash_before_within_engineering_tolerance=0)
    future_targets,raw_future_targets = [],[]
    for i, row in enumerate(rows):
        check.yes((row["case_id"], row["phase"], row["episode_id"]) == (cid, phase, i), "episode identity")
        seed = [20261016, PHASE_BAND[phase], cid, i]
        check.yes(row["seed_vector"] == seed, "independent phase stream identifier")
        # Reconstruct the prescribed RNG outputs, not a new simulated episode.
        rng = np.random.default_rng(np.random.SeedSequence(seed))
        context_id = int(rng.integers(0, 8))
        signs = (2*rng.integers(0, 2, HORIZON, dtype=np.int8)-1).astype(np.int8)
        check.yes(row["context_id"] == context_id, "predeclared context draw replay")
        check.yes(np.array_equal(row["hidden_signs"], signs), "predeclared sign-slot replay")
        lo, hi = int(eoff[i]), int(eoff[i+1])
        blo, bhi = int(boff[i]), int(boff[i+1])
        step = events["event_step"][lo:hi].astype(int)
        cash = events["event_cash_before"][lo:hi]
        after = events["event_cash_after"][lo:hi]
        gross, cost = events["event_gross"][lo:hi], events["event_cost"][lo:hi]
        actual_i = events["event_injection"][lo:hi]
        coin = events["event_sign_epsilon"][lo:hi].astype(int)
        base = events["event_unsigned_base_gross"][lo:hi]
        check.yes(np.isfinite(cash).all(), "finite pre-event financial cash before exact-sign diagnostics")
        if len(cash):
            exact_minimum = float(np.min(cash))
            diagnostics["minimum_financial_cash_before"] = (exact_minimum if diagnostics["minimum_financial_cash_before"] is None
                else min(diagnostics["minimum_financial_cash_before"], exact_minimum))
            diagnostics["strictly_negative_cash_before"] += int(np.count_nonzero(cash < 0.))
            diagnostics["negative_cash_before_within_engineering_tolerance"] += int(np.count_nonzero((cash < 0.) & (cash >= -1e-6)))
        check.yes(np.all((0 <= step) & (step < HORIZON)) and np.all(np.diff(step) >= 0), "relative calendar clock")
        first = np.flatnonzero(np.r_[True, np.diff(step) != 0])
        check.yes(np.array_equal(step[first], np.arange(HORIZON)), "mandatory first financial event in all 96 calendar rows")
        check.yes(np.all(coin[first] == 0), "first event is deterministic access fee by native source order")
        check.close(gross[first], np.zeros(HORIZON), "mandatory access event has no financial gross")
        check.close(base[first], np.zeros(HORIZON), "mandatory access event has no fair-sign magnitude")
        check.close(cost[first], np.full(HORIZON, exact_fee), "full economic access fee returned even at floored cash")
        check.yes(np.all(cash >= 0) and np.all(after >= 0) and np.all(cost >= 0), "solvent local cash and nonnegative economic cost")
        check.yes(np.all(np.isin(coin, [-1, 0, 1])), "fair sign or deterministic event")
        check.close(gross, np.where(coin != 0, coin*base, 0.), "signed native maturity")
        check.close(actual_i, np.maximum(-cash-gross+cost, 0.), "separate native event injection", category="event")
        check.close(after, np.maximum(cash+gross-cost, 0.), "separate native event cash", category="event")
        active = coin != 0
        check.yes(np.array_equal(coin[active], signs[step[active]]), "post-action sign slots")
        check.yes(len(set(step[active].tolist())) == int(active.sum()), "one aggregate fair coin per calendar")
        local_pair = (np.maximum(-cash-base+cost, 0.)+np.maximum(-cash+base+cost, 0.))/2
        check.close(local_pair[~active], actual_i[~active], "cost events paired deterministically")
        w = GAMMA**step
        old = active&(step<6)
        target_gross=np.where(old,0.,gross); target_i=np.where(old,0.,actual_i);target_pair=np.where(old,0.,local_pair)
        reconstructed = dict(gross=float(w@target_gross),cost=float(w@cost),forgiveness=float(w@target_i),paired_forgiveness=float(w@target_pair),
            excluded_old_gross=float(w@np.where(old,gross,0.)),excluded_old_forgiveness=float(w@np.where(old,actual_i,0.)),
            excluded_old_paired_forgiveness=float(w@np.where(old,local_pair,0.)),excluded_old_maturity_events=int(old.sum()))
        contribution = w*(target_pair-cost)
        # Reverse accumulation is independent of the core's total-minus-cumsum.
        future = np.r_[np.cumsum(contribution[:0:-1])[::-1], 0.]
        future_targets.append(future[active])
        raw_contribution=w*(target_gross-cost+target_i)
        raw_future=np.r_[np.cumsum(raw_contribution[:0:-1])[::-1],0.]
        raw_future_targets.append(raw_future[active])
        known=np.where(old,0.,w*(base+(np.maximum(-cash-base+cost,0.)-np.maximum(-cash+base+cost,0.))/2))
        check.close(branches['known_current_raw_delta'][blo:bhi],known[active],'known current RAW branch half-difference')
        check.close(row['delta_cash_local'],known[active],'local-only CASH baseline innovations')
        for key, value in reconstructed.items():
            check.close(row[key], value, "independent financial summary "+key, category="summary")
        check.yes(row["event_count"] == hi-lo and row["maturity_events"] == int(active.sum()), "event counts")
        check.yes(row["terminal_open_contracts"] == 0, "no censored terminal queue")
        check.yes(np.array_equal(row["maturity_coins"], coin[active]) and
                  np.array_equal(branches["coins"][blo:bhi], coin[active]), "ordered innovation vectors")
        context = contexts[context_id]["support"]
        tlo,thi=int(cashflow['transfer_offsets'][i]),int(cashflow['transfer_offsets'][i+1])
        plo,phi=int(cashflow['path_offsets'][i]),int(cashflow['path_offsets'][i+1])
        check.yes(phi-plo==HORIZON,"complete cash-bound path calendar")
        check.yes(np.all(cashflow['path_native_cash_before'][plo:phi]>=0) and
            np.all(cashflow['path_native_cash_after'][plo:phi]>=0),"source cash-bound step nonnegativity")
        transfer_step=cashflow['transfer_step'][tlo:thi]
        kinds=cashflow['transfer_kind'][tlo:thi]
        inside=cashflow['transfer_cash_change_inside_method'][tlo:thi]
        returned=cashflow['transfer_returned_cashflow'][tlo:thi]
        check.yes(not np.any(kinds=='margin_conversion'),"no excluded margin recapitalization")
        for t in range(HORIZON):
            battery=(transfer_step==t)&(kinds=='battery');distribution=(transfer_step==t)&(kinds=='distribution')
            check.yes(int(battery.sum())==int(distribution.sum())==1,"one source battery/distribution call per row")
            check.yes(abs(float(inside[battery][0]))<=1e-7 and float(inside[distribution][0])<=1e-7,"excluded cash-write classification")
            positive=max(float(returned[battery][0]),0.)
            check.yes(positive<=context['calendar_battery_positive_credit_bound'][t]+1e-5,"source battery credit support")
            check.yes(float(cashflow['path_excluded_native_cash_change'][plo+t])<=positive+1e-4,"no unbudgeted positive excluded flow")
        old_expected=context['old_maturity_unsigned_batch_gross']
        for old_step in step[old]:
            mask=old&(step==old_step)
            check.close(base[mask],np.full(int(mask.sum()),old_expected[str(int(old_step))]),"source-known inherited batch amplitude")
        old_credit=float(np.sum(w[old]*np.maximum(gross[old]+actual_i[old],0.)))
        check.yes(old_credit<=context['cash_inflow_old_maturity_credit_bound']+1e-4,"removed inherited positive cash budget")
        raw_new=reconstructed['gross']-reconstructed['cost']+reconstructed['forgiveness']
        check.yes(raw_new>=-context['cash_inflow_bound']-1e-3,"alpha0 cash-funded target lower endpoint")
        by_cost = np.bincount(step, weights=cost, minlength=HORIZON)
        by_gross = np.bincount(step, weights=np.abs(gross), minlength=HORIZON)
        check.yes(np.all(by_cost <= np.asarray(context["calendar_cost_bound"])+2e-5), "native calendar cost support")
        check.yes(np.all(by_gross <= np.asarray(context["calendar_gross_bound"])+2e-5), "native calendar gross support")
        check.yes(row["cost"] <= bounds["discounted_cost_bound"]+2e-5, "context-union path cost support")
        if phase == "confirm":
            for model in ("ridge", "mlp", "cash_ridge", "cash_mlp"):
                check.close(row["delta_"+model], branches["delta_"+model][blo:bhi], "cached per-episode "+model)
        else:
            check.yes(sha(row["private_trace_path"]) == row["private_trace_sha256"], "private development trace hash")
            private = np.load(row["private_trace_path"], allow_pickle=False)
            access_ids = private["event_kind"] == "access_cost"
            check.yes(np.array_equal(private["event_step"][access_ids], np.arange(HORIZON)),
                      "exactly one explicitly identified access event per native row")
            check.close(private["event_cost"][access_ids], np.full(HORIZON, exact_fee),
                        "private native access event identity")
        diagnostics["events"] += hi-lo
        diagnostics["maturities"] += int(active.sum())
        diagnostics["pending_prepaid_episodes"] += row["initial_pending_prepaid_contracts"] > 0
    q = metadata["query_counts"]
    check.yes(q["audit_native_transitions"] == PHASE_N[phase]*HORIZON, "actual audit transition count")
    check.yes(q["actor_forward_calls"] == 2*PHASE_N[phase]*HORIZON, "both fixed actor queries counted")
    check.yes(q["fresh_audit_clones"] == PHASE_N[phase], "fresh whole-episode clones")
    check.yes(q["branch_injection_arithmetic_evaluations"] == 2*diagnostics["events"], "local branch arithmetic count")
    check.yes(q['critic_predict_calls']==(8*PHASE_N[phase] if phase=='confirm' else 0),'four regressors times two branch calls counted')
    check.yes(q['critic_branch_rows_per_model']==(2*diagnostics['maturities'] if phase=='confirm' else 0),'per-regressor common branch rows counted')
    if phase in ("fit", "tune"):
        fit = np.load(directory/"fit_rows.npz", allow_pickle=False)
        actual_features = np.where(branches["coins"][:, None] == 1, branches["plus"], branches["minus"])
        check.close(fit["X"], actual_features, "actual-branch supervised inputs")
        check.close(fit["y"], np.concatenate(future_targets), "GLOBAL discounted after-event targets", category="future_target")
        check.close(fit['y_raw'],np.concatenate(raw_future_targets),'GLOBAL after-event NEW RAW future targets',category='raw_future_target')
        check.yes(np.array_equal(fit["episode_offsets"], boff), "whole-episode supervised offsets")
    return rows, bounds, diagnostics


def verify(run_dir, result_path, check_critics=False):
    check = Checks()
    freeze, case_list = read(run_dir/"SOURCE_FREEZE.json"), read(run_dir/"cases.json")
    check.yes(sha(run_dir/"cases.json") == freeze["cases_sha256"], "frozen case manifest")
    check.yes(len(case_list) == 20 and [case["case_id"] for case in case_list] == list(range(20)), "complete twenty-case family")
    for group, root in (("source_sha256", HERE), ("native_source_sha256", ENGINE), ("checkpoint_sha256", ENGINE),
                        ('runtime_input_sha256',ENGINE),('forecast_view_sha256',HERE)):
        for name, digest in freeze[group].items():
            check.yes(sha(root/name) == digest, "current source/input freeze hash "+name)
            if group == "source_sha256":
                check.yes(sha(run_dir/"source_snapshot"/name) == digest, "pre-endpoint source snapshot")
    for name,digest in freeze['original_source_sha256'].items():
        check.yes(sha(HERE.parent/name)==digest,"unchanged protected original source")
    check.yes(sha(HERE.parent/'execution/run_001/SOURCE_FREEZE.json')==freeze['original_freeze_sha256'],"original frozen experiment preserved")
    check.yes(sha(HERE.parent/'execution/run_001/cases.json')==freeze['original_cases_sha256'],'original fixed actor registry identity')
    check.yes(sha(HERE/freeze['cache_preparation_report_path'])==freeze['cache_preparation_report_sha256'],'new cache preparation PASS report identity')
    check.yes(sha(HERE/'prepare_joint_fault_cache.py')==freeze['cache_preparation_source_sha256'],'new exact-view preparation source identity')
    check.yes(sha(HERE/freeze['technical_smoke_report_path'])==freeze['technical_smoke_report_sha256'],'two fixed engineering smoke provenance identity')
    external_provenance = verify_external_resources(check, run_dir, case_list)
    selection_freeze = read(run_dir/"SELECTION_FREEZE.json")
    check.yes(selection_freeze["source_freeze_sha256"] == sha(run_dir/"SOURCE_FREEZE.json"), "selection source binding")
    for filename, key in (("selection.json", "selection_sha256"), ("search_records.json", "search_sha256"),
                          ("models/audit_models_manifest.json", "model_manifest_sha256"),
                          ("models_raw/audit_models_manifest.json", "raw_model_manifest_sha256")):
        check.yes(sha(run_dir/filename) == selection_freeze[key], "pre-confirmation selection/model binding")
    model_manifest = read(run_dir/"models/audit_models_manifest.json")
    for filename, key in ((model_manifest["bundle_file"], "bundle_sha256"),
                          (model_manifest["metadata_file"], "metadata_sha256")):
        check.yes(sha(run_dir/"models"/filename) == model_manifest[key], "trusted nuisance model hash")
    model_metadata = read(run_dir/"models"/model_manifest["metadata_file"])
    raw_manifest=read(run_dir/'models_raw/audit_models_manifest.json')
    for filename,key in ((raw_manifest['bundle_file'],'bundle_sha256'),(raw_manifest['metadata_file'],'metadata_sha256')):
        check.yes(sha(run_dir/'models_raw'/filename)==raw_manifest[key],'trusted RAW nuisance model hash')
    raw_metadata=read(run_dir/'models_raw'/raw_manifest['metadata_file'])
    check.yes(model_metadata['seed']==39300000 and raw_metadata['seed']==40300000,'distinct predeclared regressor seeds')
    fit_inputs,fit_targets,raw_targets=[],[],[]
    for case in case_list:
        fit_path = run_dir/f"case_{int(case['case_id']):02d}"/"fit"/"fit_rows.npz"
        check.yes(sha(fit_path) == selection_freeze["fit_input_sha256"][str(case["case_id"])], "frozen FIT input binding")
        check.yes(sha(fit_path)==selection_freeze['raw_fit_input_sha256'][str(case['case_id'])],'RAW uses same pooledFIT input file')
        data = np.load(fit_path, allow_pickle=False)
        fit_inputs.append(data["X"]); fit_targets.append(data["y"])
        raw_targets.append(data['y_raw'])
    check.yes(array_digest(np.concatenate(fit_inputs)) == model_metadata["training_input_sha256"], "nuisance model used pooled FIT inputs only")
    check.yes(array_digest(np.concatenate(fit_targets)) == model_metadata["raw_target_sha256"], "nuisance model used pooled FIT targets only")
    check.yes(array_digest(np.concatenate(fit_inputs))==raw_metadata['training_input_sha256'],'RAW nuisance used same pooledFIT X')
    check.yes(array_digest(np.concatenate(raw_targets))==raw_metadata['raw_target_sha256'],'RAW nuisance used pooledFIT RAW future labels')
    critics = None
    if check_critics:
        sys.path.insert(0, str(HERE))
        from audit_models import load_models
        critics = load_models(run_dir/"models")
        critics.update({'cash_'+name:model for name,model in load_models(run_dir/'models_raw').items()})
    choices = read(run_dir/"selection.json")
    choice_map = {(r["case_id"], float(r["alpha"]), r["method"]): r["choice"] for r in choices}
    expected_keys = {(i,a,m) for i in range(20) for a in (0.,1.) for m in METHODS if not m.startswith("cash_") or a==0.}
    check.yes(len(choices) == len(choice_map) == 320 and set(choice_map) == expected_keys, "all 320 frozen choices")
    results = read(result_path)
    check.yes(results["n_cases"] == 20 and results["multiplicity"]["family_limits"] == 340, "complete reported family")
    check.close(results["multiplicity"]["delta_per_limit"], DELTA, "family error allocation")
    charged = {(r["case_id"], float(r["alpha"]), r["method"]): r for r in results["charged_results"]}
    cost_saved = {r["case_id"]: r for r in results["cost_results"]}
    check.yes(len(charged) == 320 and set(charged) == expected_keys and set(cost_saved) == set(range(20)), "all 340 output bounds")
    counts = {str(a): {method: 0 for method in METHODS if not method.startswith("cash_") or a==0.} for a in (0., 1.)}
    phase_diagnostics, audit_steps = [], 0
    work = {key: 0 for key in results["actual_work"]}
    for case in case_list:
        cid = int(case["case_id"])
        phases = {}
        common_bounds = None
        for phase in PHASE_N:
            phase_rows, bounds, diagnostic = verify_phase(check, run_dir, case, phase, critics)
            if phase == "confirm":
                metadata = read(run_dir/f"case_{cid:02d}"/phase/"metadata.json")
                check.yes(metadata["started_utc"] > selection_freeze["created_utc"], "every confirmation starts after all model/choice freezes")
            phases[phase] = phase_rows
            if common_bounds is None:
                common_bounds = bounds
            else:
                check.yes(bounds == common_bounds, "identical pre-draw support across phases")
            phase_diagnostics.append(dict(case_id=cid, phase=phase, **diagnostic))
            audit_steps += PHASE_N[phase]*HORIZON
            metadata = read(run_dir/f"case_{cid:02d}"/phase/"metadata.json")
            for key, value in metadata["query_counts"].items():
                work[key] += value
            work["phase_cpu_seconds"] += metadata["cpu_seconds"]
        for stage in ("development", "confirm"):
            preparation = read(run_dir/f"case_{cid:02d}"/(stage+"_completion.json"))["preparation"]
            for key in ("warm_native_transitions", "warm_actor_forward_calls", "zero_step_support_initializations", "known_old_payoff_point_evaluations"):
                work[key] += preparation[key]
            work["preparation_cpu_seconds"] += preparation["cpu_seconds"]
        allrows = phases["fit"]+phases["tune"]+phases["confirm"]
        for alpha in (0., 1.):
            for method in METHODS:
                if method.startswith("cash_") and alpha!=0.:continue
                key = (cid, alpha, method)
                choice, saved = choice_map[key], charged[key]
                check.yes(saved["choice"] == choice and saved["bounds"] == common_bounds, "frozen choice/support used")
                x = values(allrows if method == "raw" else phases["confirm"], method, alpha, choice)
                lo, hi = support(common_bounds, method, alpha, choice)
                expected = eb(x, lo, hi, pad(common_bounds, choice))
                verify_certificate(check, saved["certificate"], expected, x, "charged")
                positive = expected["lower"] > 0
                check.yes(saved["positive"] == positive, "strict positive lower bound decision")
                counts[str(alpha)][method] += int(positive)
                confirmation = values(phases["confirm"], method, alpha, choice)
                check.close(saved["confirmation_mean"], float(confirmation.mean()), "descriptive common-confirmation mean")
                check.close(saved["confirmation_variance"], float(confirmation.var(ddof=1)), "descriptive common-confirmation variance")
        x = np.asarray([row["cost"] for row in allrows])
        expected = eb(x, 0., common_bounds["discounted_cost_bound"], pad(common_bounds))
        verify_certificate(check, cost_saved[cid]["certificate"], expected, x, "cost")
        analytic_cost_lower = float(common_bounds["unavoidable_discounted_access_cost"])
        check.close(cost_saved[cid]["analytic_unavoidable_cost_lower"], analytic_cost_lower,
                    "deterministic unavoidable native access cost lower bound")
        check.close(cost_saved[cid]["booked_expectation_upper"],
                    -max(expected["lower"], analytic_cost_lower), "no-edge booked harm upper limit")
    check.yes(audit_steps == 614400, "equal shared native audit query total")
    check.yes(work["warm_native_transitions"] == 122880, "two-stage deterministic warm query total")
    for key, value in work.items():
        check.close(value, results["actual_work"][key], "reported actual work "+key, category="work")
    check.yes(counts == results["method_counts"], "all reported detection counts")
    cash_diagnostics = dict(
        minimum_financial_cash_before=min(row["minimum_financial_cash_before"] for row in phase_diagnostics
                                          if row["minimum_financial_cash_before"] is not None),
        strictly_negative_cash_before=sum(row["strictly_negative_cash_before"] for row in phase_diagnostics),
        negative_cash_before_within_engineering_tolerance=sum(row["negative_cash_before_within_engineering_tolerance"] for row in phase_diagnostics),
        financial_events_checked=sum(row["events"] for row in phase_diagnostics),
        phases_checked=len(phase_diagnostics), engineering_acceptance_threshold_unchanged=0.,
        theoretical_nonnegative_cash_requirement=0., scope="exact saved finite pre-event cash diagnostics only; no tolerance/engine/CI change or native-flat-executability claim")
    return dict(status="PASS", checked_utc=datetime.now(timezone.utc).isoformat(), checks=check.count,
        verifier_sha256=sha(__file__),
        source_freeze_sha256=sha(run_dir/"SOURCE_FREEZE.json"), result_sha256=sha(result_path),
        bounds_checked=340, maximum_recomputation_errors=check.max_error, method_counts=counts,
        external_provenance=external_provenance,
        phase_diagnostics=phase_diagnostics, native_audit_steps_verified=audit_steps,
        cash_before_diagnostics=cash_diagnostics,
        actual_work_verified=work,
        additional_parent_tune_critic_predict_calls=160,
        compute_scope="worker counters; parent selection inference and supervised fitting are additional work; phase/preparation CPU excludes process imports/model loading",
        nuisance_regressor_prediction_rechecked=bool(check_critics),
        new_simulator_steps=0, new_actor_inference=0, new_model_fits=0,
        scope="conditional fixed-context sample-only experiment; private saved native events; no independent simulator replay",
        caveat="This verifier checks saved records/provenance/arithmetic; source review and exact theory justify conditional sign centering and deterministic support derivation.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="run_001")
    parser.add_argument("--check-critics", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if not args.run_id.startswith("run_") or not args.run_id.replace("_", "").isalnum():
        raise ValueError("Simple frozen run ID required")
    run_dir = HERE/"execution"/args.run_id
    result_path = HERE/("joint_fault_results_"+args.run_id+".json")
    required = (run_dir/"confirm_stage_complete.json", result_path,HERE/("joint_fault_completion_"+args.run_id+".json"))
    if not all(path.exists() for path in required):
        print(json.dumps(dict(status="NOT_READY", reason="complete confirmation and aggregation are required",
                              missing=[str(path) for path in required if not path.exists()])))
        return 2
    cpu, wall = time.process_time(), time.perf_counter()
    report = verify(run_dir, result_path, args.check_critics)
    report.update(cpu_seconds=time.process_time()-cpu, wall_seconds=time.perf_counter()-wall)
    if args.write:
        path = VALIDATION/("JOINT_FAULT_VERIFICATION_"+args.run_id+".json")
        with path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key != "phase_diagnostics"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
