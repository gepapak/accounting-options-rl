"""EXPLORATORY simulator-only witness search; no DP, RL or Study R inputs.

Reusable finite empirical kernel, feedback simulator and independent fixed-n
certificate functions. See EXPLORATORY_PLAN.md before interpreting the outputs.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GAMMA, K0, FEE = .99, 20000., 1.
SEARCH_PATH_BUDGET, CONFIRM_PATHS = 16384, 32768
CAPS = (32., 128., 512., 2048.)
METHODS = ("one_action", "forced_prefix_myopic", "fixed_barrier", "structured_feedback", "random_parameter_search")
CASES = tuple(itertools.product(("DK1", "DK2"), (.1, .5, 1.), (4, 16, 48)))
DELTA = .05 / (len(CASES) * len(METHODS) * 6)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def empirical_bernstein(x, lower, upper, delta):
    """One-sided M&P 2009 Thm4 bounds, fixed independent sample of whole paths."""
    x = np.asarray(x, float)
    if len(x) < 2 or not 0 < delta < 1 or upper < lower:
        raise ValueError("invalid fixed-sample certificate inputs")
    if x.min() < lower - 1e-6 or x.max() > upper + 1e-6:
        raise AssertionError((float(x.min()), float(x.max()), lower, upper))
    variance, n = float(x.var(ddof=1)), len(x)
    log = np.log(2 / delta)
    radius = float(np.sqrt(2 * variance * log / n) + 7 * (upper - lower) * log / (3 * (n - 1)))
    mean = float(x.mean())
    return dict(mean=mean, variance=variance, radius=radius, lower=mean-radius,
                upper=mean+radius, range_lower=float(lower), range_upper=float(upper), n=n, delta=delta)


class EmpiricalKernel:
    """Finite empirical TRAIN rows, independent fair signs, known no-edge costs."""
    def __init__(self, spread_magnitude, cap_mwh, fee=FEE):
        self.spread = np.asarray(spread_magnitude, float)
        self.cap = np.asarray(cap_mwh, float)
        self.fee = float(fee)
        self.n = len(self.cap)
        self.stats = {}
        for lev in (.25, 1, 4, 16):
            q = np.minimum(lev * .25, self.cap)
            magnitude, cost = q*self.spread, fee*q
            loss = np.sort(np.r_[cost-magnitude, cost+magnitude])
            suffix = np.r_[np.cumsum(loss[::-1])[::-1], 0.]
            self.stats[lev] = dict(mean_cost=float(cost.mean()), losses=loss, suffix=suffix,
                                   min_x=float((-magnitude-cost).min()), max_x=float((magnitude-cost).max()))
        self.max_loss = -self.stats[16]["min_x"]
        self.max_gain = max(0., self.stats[16]["max_x"])

    def h(self, lev, ledger):
        s = self.stats[lev]
        i = np.searchsorted(s["losses"], ledger, side="right")
        return (s["suffix"][i]-(len(s["losses"])-i)*ledger)/len(s["losses"])

    def one_action(self, ledger, horizon, gamma=GAMMA, lev=16):
        d = gamma**(horizon-1)
        return float(d*(-self.stats[lev]["mean_cost"] + self.h(lev, ledger)))

    def best_one_action(self, ledger, horizon, gamma=GAMMA):
        values={0:0., **{lev:self.one_action(ledger,horizon,gamma,lev) for lev in self.stats}}
        lev=max(values,key=values.get)
        return values[lev], lev, max(values[l] for l in self.stats)


def action(policy, ledger, step, horizon, initial, kernel, gamma=GAMMA):
    """Nonanticipative feedback on observed booked ledger and remaining clock."""
    if policy["kind"] == "one_action":
        return np.full(ledger.shape, 16 if step == horizon-1 else 0, dtype=np.int8)
    if step == 0:
        return np.full(ledger.shape, 16, dtype=np.int8)
    obs = np.clip(ledger, -3*K0, 5*K0)
    if policy["kind"] == "myopic":
        best = np.zeros(ledger.shape)
        chosen = np.zeros(ledger.shape, dtype=np.int8)
        d = gamma**(horizon-step-1)
        for lev in (4, 16):
            value = -kernel.stats[lev]["mean_cost"] + d*(kernel.h(lev, obs)-np.maximum(-obs, 0))
            improve = value > best
            chosen[improve], best[improve] = lev, value[improve]
        return chosen
    if policy["kind"] == "barrier":
        return np.where(obs <= policy["barrier"]*initial, 16, 0).astype(np.int8)
    late = step >= policy["switch"]*horizon
    lev = policy["late_lev"] if late else policy["early_lev"]
    barrier = policy["late_barrier"] if late else policy["early_barrier"]
    return np.where(obs <= barrier*initial, lev, 0).astype(np.int8)


def rollout(kernel, policy, indices, signs, initial, gamma=GAMMA):
    """Vectorized paired ledgers. Each row is one independent whole trajectory."""
    n, horizon = indices.shape
    ledger = np.full(n, initial, float)
    discounted_cost = np.zeros(n)
    discounted_gross = np.zeros(n)
    for t in range(horizon):
        lev = action(policy, ledger, t, horizon, initial, kernel, gamma)
        q = np.minimum(lev*.25, kernel.cap[indices[:, t]])
        gross = q*kernel.spread[indices[:, t]]*signs[:, t]
        cost = q*kernel.fee
        ledger += gross-cost
        discounted_cost += gamma**t*cost
        discounted_gross += gamma**t*gross
    forgiveness = gamma**(horizon-1)*np.maximum(-ledger, 0)
    booked = discounted_gross-discounted_cost
    raw = booked+forgiveness
    residual = forgiveness-discounted_cost
    error = float(np.max(np.abs(raw-discounted_gross-residual)))
    return dict(raw=raw, residual=residual, forgiveness=forgiveness, cost=discounted_cost,
                booked=booked, gross=discounted_gross, identity_error=error)


def path_bounds(kernel, policy, initial, horizon, gamma=GAMMA):
    """Finite-support bounds, including a tighter terminal-owner raw-return bound."""
    d, w = gamma**(horizon-1), float(np.sum(gamma**np.arange(horizon)))
    if policy["kind"] == "one_action":
        low_x, high_x = kernel.stats[16]["min_x"], kernel.stats[16]["max_x"]
        return dict(cost_max=d*4*kernel.fee,
                    raw_lower=d*(max(initial+low_x, 0)-initial),
                    raw_upper=d*(max(initial+high_x, 0)-initial),
                    forgiveness_max=d*max(-initial-low_x, 0),
                    booked_mean_upper=-d*kernel.stats[16]["mean_cost"])
    # Summation by parts: raw = d*max(L_H,0)-L0+(1-gamma)*sum gamma^(t-1)*L_t.
    weighted_time = float(np.sum(np.arange(1,horizon)*gamma**np.arange(horizon-1)))
    return dict(cost_max=4*kernel.fee*w,
                raw_lower=-d*initial-(1-gamma)*kernel.max_loss*weighted_time,
                raw_upper=kernel.max_gain*w,
                forgiveness_max=d*max(horizon*kernel.max_loss-initial, 0),
                booked_mean_upper=-kernel.stats[16]["mean_cost"])


def selected_certificate(kernel, policy, cap, raw_cap, draws, initial, delta=DELTA):
    """Reusable split-confirmation certificate, with no optional sample stopping."""
    horizon = draws[0].shape[1]
    r = rollout(kernel, policy, *draws, initial)
    b = path_bounds(kernel, policy, initial, horizon)
    raw = empirical_bernstein(r["raw"], b["raw_lower"], b["raw_upper"], delta)
    residual = empirical_bernstein(r["residual"], -b["cost_max"], b["forgiveness_max"], delta)
    clipped = empirical_bernstein(np.minimum(r["residual"], cap), -b["cost_max"], cap, delta)
    clipped_raw = empirical_bernstein(np.minimum(r["raw"], raw_cap), b["raw_lower"],
                                     min(raw_cap, b["raw_upper"]), delta)
    forg = empirical_bernstein(np.minimum(r["forgiveness"], cap), 0, cap, delta)
    cost = empirical_bernstein(r["cost"], 0, b["cost_max"], delta)
    lf, uc = max(0., forg["lower"]), min(b["cost_max"], cost["upper"])
    alpha_upper = max(0., 1-uc/lf) if lf > 0 else 0.
    return dict(raw=raw, residual=residual, clipped=clipped, clipped_raw=clipped_raw, forgiveness=forg, cost=cost,
                booked_mean_upper=b["booked_mean_upper"],
                certified_raw=raw["lower"] > 0, certified_residual=residual["lower"] > 0,
                certified_clipped=clipped["lower"] > 0,
                certified_clipped_raw=clipped_raw["lower"] > 0,
                partial_terminal_alpha_strict_upper=float(alpha_upper),
                identity_error=r["identity_error"],
                simulated_booked_mean=float(r["booked"].mean()))


def candidate_library(method, rng):
    if method in ("one_action", "forced_prefix_myopic"):
        return [dict(kind="myopic" if method=="forced_prefix_myopic" else method)]
    if method == "fixed_barrier":
        return [dict(kind="barrier", barrier=b) for b in (1.,1.5,2.,1e100)]
    if method == "structured_feedback":
        return [dict(kind="two_stage", early_lev=e, late_lev=l, early_barrier=b,
                     late_barrier=c, switch=s)
                for e,l,b,c,s in itertools.product((4,16),(4,16),(1.,2.),(.5,2.),(.5,.75))]
    return [dict(kind="two_stage", early_lev=int(rng.choice((0,4,16))),
                 late_lev=int(rng.choice((0,4,16))), early_barrier=float(rng.uniform(.5,4)),
                 late_barrier=float(rng.uniform(.5,4)), switch=float(rng.uniform(.1,.9))) for _ in range(32)]


def search(kernel, policies, draws, initial):
    n = SEARCH_PATH_BUDGET//len(policies)
    horizon = draws[0].shape[1]
    indices, signs = draws[0][:n], draws[1][:n]
    records, selected = [], None
    for i, policy in enumerate(policies):
        r = rollout(kernel, policy, indices, signs, initial)
        b = path_bounds(kernel, policy, initial, horizon)
        for cap in CAPS:
            # A heuristic only: its adaptively selected search score is NOT a confidence claim.
            z = empirical_bernstein(np.minimum(r["residual"], cap), -b["cost_max"], cap, .1)
            zraw = empirical_bernstein(np.minimum(r["raw"], cap), b["raw_lower"], min(cap,b["raw_upper"]), .1)
            record = dict(candidate=i, policy=policy, cap=cap, search_score=z["lower"],
                          search_clipped_mean=z["mean"], search_paths=n,
                          search_clipped_raw_score=zraw["lower"])
            records.append(record)
            if selected is None or record["search_score"] > selected["search_score"]:
                selected = record
    selected = dict(selected)
    selected["raw_cap"] = max((r for r in records if r["candidate"]==selected["candidate"]),
                              key=lambda r:r["search_clipped_raw_score"])["cap"]
    return selected, records, len(policies)*n*horizon


def draws(rng, kernel, n, horizon):
    return (rng.integers(kernel.n, size=(n,horizon), dtype=np.int32),
            (2*rng.integers(2, size=(n,horizon), dtype=np.int8)-1))


def check_frozen_environment(kernel):
    """Feedback-path reward comparison against unchanged frozen terminal environment."""
    sys.path[:0] = [str(ROOT/"LIABILITY_STUDY_2026-09-29"),
                   str(ROOT/"ACCOUNTING_OPTIONS_2026-10-14/study_n")]
    from envs_b import LEV
    from envs_n import LagEnv
    rng = np.random.default_rng(10817)
    policies = [candidate_library(m, rng)[0] for m in METHODS]
    policies += [candidate_library("structured_feedback", rng)[-1]]
    max_error, cases = 0., 0
    for initial in (.1*K0, .5*K0):
        for policy in policies:
            idx, sign = draws(rng, kernel, 2, 16)
            result = rollout(kernel, policy, idx, sign, initial)
            for j in range(2):
                market = SimpleNamespace(n=1000, spread=np.zeros(1000), cap_mwh=np.zeros(1000))
                market.spread[200:216] = kernel.spread[idx[j]]*sign[j]
                market.cap_mwh[200:216] = kernel.cap[idx[j]]
                arm_values = {}
                for arm in ("TFG", "TBK"):
                    env = LagEnv(market, arm, train=False)
                    env.reset(options=dict(e0=initial/K0,tau0=16,t0=200))
                    total = 0.
                    for t in range(16):
                        lev = int(action(policy, np.array([env.L]), t, 16, initial, kernel)[0])
                        _, reward, _, _, _ = env.step(int(np.flatnonzero(LEV == lev)[0]))
                        total += GAMMA**t*reward*K0/100
                    arm_values[arm] = total
                err = max(abs(arm_values["TFG"]-result["raw"][j]),
                          abs(arm_values["TBK"]-result["booked"][j]))
                max_error = max(max_error, err)
                assert err < 1e-7, (policy, initial, err)
                cases += 1
    return dict(feedback_paths=cases, arm_path_checks=2*cases, max_abs_error_eur=max_error)


def main():
    cpu0, wall0 = time.process_time(), time.perf_counter()
    kernels, input_hashes = {}, {}
    for zone in ("DK1", "DK2"):
        p = ROOT/f"GENERALIZATION_STUDY_2026-09-29/data_panel/panel_{zone}.pkl"
        df = pd.read_pickle(p)
        train = df.loc[df.split=="train"].iloc[200:]
        kernels[zone] = EmpiricalKernel(np.abs(train.imb.to_numpy(float)-train.spot.to_numpy(float)),
                                        .25*.25*train.bal_mw.to_numpy(float))
        input_hashes[str(p.relative_to(ROOT))] = sha(p)
    for p in (ROOT/"LIABILITY_STUDY_2026-09-29/envs_b.py",
              ROOT/"ACCOUNTING_OPTIONS_2026-10-14/study_n/envs_n.py"):
        input_hashes[str(p.relative_to(ROOT))] = sha(p)
    check = check_frozen_environment(kernels["DK1"])
    rows, all_search, selected_policies, status = [], [], [], "complete"
    for case_id, (zone,e,horizon) in enumerate(CASES):
        if time.process_time()-cpu0 > 110:
            status = "partial_cpu_guard"
            break
        k, initial = kernels[zone], e*K0
        exact_best, exact_lev, exact_nonflat = k.best_one_action(initial,horizon)
        # Independent phase RNGs. Within-phase shared paths do not change IID path units.
        search_draws = draws(np.random.default_rng(701000+case_id), k, SEARCH_PATH_BUDGET, horizon)
        confirmation = draws(np.random.default_rng(902000+case_id), k, CONFIRM_PATHS, horizon)
        for method_id, method in enumerate(METHODS):
            if time.process_time()-cpu0 > 110:
                status = "partial_cpu_guard"
                break
            policies = candidate_library(method, np.random.default_rng(501000+case_id*10+method_id))
            selected, trace, calls = search(k, policies, search_draws, initial)
            all_search.extend(dict(case_id=case_id,zone=zone,equity_multiple=e,horizon=horizon,
                                   method=method,**t) for t in trace)
            certificate = selected_certificate(k, selected["policy"], selected["cap"], selected["raw_cap"], confirmation, initial)
            record = dict(case_id=case_id,zone=zone,equity_multiple=e,horizon=horizon,method=method,
                          candidate_count=len(policies),selected_candidate=selected["candidate"],
                          cap=selected["cap"],raw_cap=selected["raw_cap"],search_logical_transitions=calls,
                          confirmation_logical_transitions=CONFIRM_PATHS*horizon,
                          exact_max16_one_action_eur=k.one_action(initial,horizon),
                          exact_best_one_action_eur=exact_best,exact_best_one_action_leverage=exact_lev,
                          exact_best_nonflat_one_action_eur=exact_nonflat,
                          one_action_exact_positive=exact_best>0,
                          booked_mean_upper=certificate["booked_mean_upper"],
                          full_charge_same_action_positive=False,
                          partial_terminal_alpha_strict_upper=certificate["partial_terminal_alpha_strict_upper"],
                          identity_error=certificate["identity_error"],
                          simulated_booked_mean=certificate["simulated_booked_mean"])
            for representation in ("raw","residual","clipped","clipped_raw","forgiveness","cost"):
                for key,value in certificate[representation].items():
                    record[f"{representation}_{key}"] = value
            for representation in ("raw","residual","clipped","clipped_raw"):
                record[f"certified_{representation}"] = certificate[f"certified_{representation}"]
            rows.append(record)
            selected_policies.append(dict(case_id=case_id,method=method,selected=selected,certificate=certificate))
        if status != "complete":
            break
    pd.DataFrame(rows).to_csv(HERE/"exploratory_empirical_certificates.csv",index=False)
    pd.DataFrame([{**r,"policy":json.dumps(r["policy"],sort_keys=True)} for r in all_search]).to_csv(
        HERE/"exploratory_search_candidates.csv",index=False)
    summaries=[]
    frame=pd.DataFrame(rows)
    for method,d in frame.groupby("method",sort=False):
        summaries.append(dict(method=method,cases=len(d),
            one_action_exact_positive_cases=int(d.one_action_exact_positive.sum()),
            raw_certified=int(d.certified_raw.sum()),residual_certified=int(d.certified_residual.sum()),
            clipped_certified=int(d.certified_clipped.sum()),
            clipped_raw_certified=int(d.certified_clipped_raw.sum()),
            clipped_certified_where_exact_one_action_fails=int((d.certified_clipped & ~d.one_action_exact_positive).sum()),
            median_residual_over_raw_variance=float(np.median(d.residual_variance/d.raw_variance)),
            residual_variance_exceeds_raw_cases=int((d.residual_variance>d.raw_variance).sum()),
            search_logical_transitions=int(d.search_logical_transitions.sum()),
            confirmation_logical_transitions=int(d.confirmation_logical_transitions.sum())))
    result=dict(status="EXPLORATORY_"+status,confirmed_case_method_pairs=len(rows),planned_case_method_pairs=90,
        delta_per_bound=DELTA,familywise_error_budget=.05,planned_bounds=540,
        confidence_bound="Maurer-Pontil2009Theorem4; fixed n; SEARCH and CONFIRM independent",
        environment_check=check,summary_by_method=summaries,selected_policies=selected_policies,
        max_accounting_identity_error=float(frame.identity_error.max()),
        cpu_seconds=time.process_time()-cpu0,wall_seconds=time.perf_counter()-wall0,
        numeric_threads=1,R_files_read=False,registered_files_modified=False,dp_solved=False,learners_trained=0,
        input_sha256=input_hashes,script_sha256=sha(__file__),plan_sha256=sha(HERE/"EXPLORATORY_PLAN.md"),
        kernel_rows={z:k.n for z,k in kernels.items()},
        kernel_preprocessing="all four allowed nonzero magnitude/cost levels, full empirical sorting; not simulator calls",
        interpretation="Positive bound certifies a fixed policy's empirical IID model incentive with exact booked harm; abstention does not certify safety or learnability. No universal variance reduction claim.")
    (HERE/"exploratory_empirical_summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="selected_policies"},indent=2))


if __name__ == "__main__":
    main()
