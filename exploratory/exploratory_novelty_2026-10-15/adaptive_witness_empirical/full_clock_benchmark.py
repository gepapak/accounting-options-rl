"""EXPLORATORY final full-clock extension; every earlier output is preserved."""
from __future__ import annotations
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import benchmark_adaptive_witness as base

HERE,ROOT=base.HERE,base.ROOT
EQUITIES=(.02,.05,.1,.25,.5,1.,2.)
DURATIONS=(1,4,16,48,96,192)
HORIZONS=(1,4,16,48,96,192)
BUDGET,NCONFIRM=262144,32768
DELTA=.05/(2*len(EQUITIES)*len(base.METHODS)*6)


def predicted_score(values,lower,upper):
    """SEARCH-only prediction at prespecified confirmation n; never a claim."""
    log=np.log(2/DELTA)
    return float(values.mean()-np.sqrt(2*values.var(ddof=1)*log/NCONFIRM)
                 -7*(upper-lower)*log/(3*(NCONFIRM-1)))


def search_latest(kernel,method,rng,search_draws,initial):
    controller=base.candidate_library(method,rng)
    candidates=[(t,p) for t in ((1,) if method=="one_action" else DURATIONS) for p in controller]
    n=BUDGET//sum(t for t,p in candidates)
    records,selected=[],None
    for candidate,(duration,policy) in enumerate(candidates):
        r=base.rollout(kernel,policy,search_draws[0][:n,:duration],search_draws[1][:n,:duration],initial)
        b=base.path_bounds(kernel,policy,initial,duration)
        for cap in base.CAPS:
            value=predicted_score(np.minimum(r["residual"],cap),-b["cost_max"],cap)
            rawvalue=predicted_score(np.minimum(r["raw"],cap),b["raw_lower"],min(cap,b["raw_upper"]))
            row=dict(candidate=candidate,duration=duration,policy=policy,cap=cap,
                     predicted_confirmation_score=value,predicted_raw_score=rawvalue,search_paths=n)
            records.append(row)
            if selected is None or value>selected["predicted_confirmation_score"]:
                selected=row
    selected=dict(selected)
    selected["raw_cap"]=max((r for r in records if r["candidate"]==selected["candidate"]),
                            key=lambda r:r["predicted_raw_score"])["cap"]
    return selected,records,n*sum(t for t,p in candidates)


def compile_calendar(certificate,duration,horizon,initial,kernel):
    """Scale one local certificate to compatible deadlines, without new draws."""
    if horizon<duration:
        return dict(compatible=False,status="ABSTAIN_incompatible_duration")
    factor=base.GAMMA**(horizon-duration)
    c=certificate
    row=dict(compatible=True,status="compatible",factor=factor,
             late_charged_sample_mean=factor*c["raw"]["mean"],
             early_charged_sample_mean=c["raw"]["mean"]-(1-factor)*c["forgiveness"]["mean"],
             late_residual_sample_mean=factor*c["residual"]["mean"],
             early_residual_sample_mean=factor*c["forgiveness"]["mean"]-c["cost"]["mean"],
             paired_residual_delay_improvement=(1-factor)*c["cost"]["mean"],
             exact_expected_delay_improvement_lower=(1-factor)*kernel.stats[16]["mean_cost"],
             booked_mean_upper=factor*c["booked_mean_upper"],
             partial_terminal_alpha_strict_upper=c["partial_terminal_alpha_strict_upper"],
             exact_best_one_action_eur=kernel.best_one_action(initial,horizon)[0])
    for representation in ("raw","residual","clipped_raw","clipped"):
        row[representation+"_scaled_lower"]=factor*c[representation]["lower"]
        row["certified_"+representation]=c["certified_"+representation]
    return row


def check_calendar_compiler(kernel):
    """Actual frozen environment check of delayed action/reward compilation."""
    sys.path[:0]=[str(ROOT/"LIABILITY_STUDY_2026-09-29"),
                   str(ROOT/"ACCOUNTING_OPTIONS_2026-10-14/study_n")]
    from envs_b import LEV
    from envs_n import LagEnv
    from types import SimpleNamespace
    rng=np.random.default_rng(2401000)
    error,checks=0.,0
    for initial in (.1*base.K0,.5*base.K0):
        for duration in (4,192):
            for method in base.METHODS:
                policy=base.candidate_library(method,rng)[-1]
                idx,sign=base.draws(rng,kernel,1,duration)
                local=base.rollout(kernel,policy,idx,sign,initial)
                horizon,start=192,192-duration
                market=SimpleNamespace(n=1000,spread=np.zeros(1000),cap_mwh=np.zeros(1000))
                market.spread[200+start:200+horizon]=kernel.spread[idx[0]]*sign[0]
                market.cap_mwh[200+start:200+horizon]=kernel.cap[idx[0]]
                for arm,key in (("TFG","raw"),("TBK","booked")):
                    env=LagEnv(market,arm,train=False)
                    env.reset(options=dict(e0=initial/base.K0,tau0=horizon,t0=200))
                    total=0.
                    for t in range(horizon):
                        lev=0 if t<start else int(base.action(policy,np.array([env.L]),t-start,
                                                              duration,initial,kernel)[0])
                        _,reward,_,_,_=env.step(int(np.flatnonzero(LEV==lev)[0]))
                        total+=base.GAMMA**t*reward*base.K0/100
                    err=abs(total-base.GAMMA**start*local[key][0]);error=max(error,err)
                    assert err<1e-7,(method,duration,arm,err)
                    checks+=1
    return dict(delayed_arm_path_checks=checks,outer_horizon=192,max_abs_error_eur=error)


CPU_DEADLINE=float("inf")
TRANSITION_COUNTER=0
class CpuBudgetExceeded(Exception):
    pass

def bounded_rollout(kernel, policy, indices, signs, initial, gamma=base.GAMMA):
    """Vectorized paired ledgers. Each row is one independent whole trajectory."""
    global TRANSITION_COUNTER
    n, horizon = indices.shape
    ledger = np.full(n, initial, float)
    discounted_cost = np.zeros(n)
    discounted_gross = np.zeros(n)
    for t in range(horizon):
        if time.process_time()>CPU_DEADLINE:
            raise CpuBudgetExceeded()
        TRANSITION_COUNTER += n
        lev = base.action(policy, ledger, t, horizon, initial, kernel, gamma)
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

base.rollout=bounded_rollout


def main():
    global CPU_DEADLINE
    cpu0,wall0=time.process_time(),time.perf_counter()
    CPU_DEADLINE=cpu0+107
    kernels,hashes={},{}
    for zone in ("DK1","DK2"):
        path=ROOT/f"GENERALIZATION_STUDY_2026-09-29/data_panel/panel_{zone}.pkl"
        d=pd.read_pickle(path);d=d.loc[d.split=="train"].iloc[200:]
        kernels[zone]=base.EmpiricalKernel(np.abs(d.imb.to_numpy(float)-d.spot.to_numpy(float)),
                                           .25*.25*d.bal_mw.to_numpy(float))
        hashes[str(path.relative_to(ROOT))]=base.sha(path)
    for path in (ROOT/"LIABILITY_STUDY_2026-09-29/envs_b.py",
                 ROOT/"ACCOUNTING_OPTIONS_2026-10-14/study_n/envs_n.py"):
        hashes[str(path.relative_to(ROOT))]=base.sha(path)
    check=base.check_frozen_environment(kernels["DK1"])
    calendar_check=check_calendar_compiler(kernels["DK1"])
    records,calendars,selected_records,traces=[],[],[],[]
    status="complete"
    for state_id,(zone,equity) in enumerate(itertools.product(("DK1","DK2"),EQUITIES)):
        k,initial=kernels[zone],equity*base.K0
        search_draws=base.draws(np.random.default_rng(2101000+state_id),k,BUDGET,max(DURATIONS))
        confirm=base.draws(np.random.default_rng(2201000+state_id),k,NCONFIRM,max(DURATIONS))
        exact_best,exact_lev,exact_nonflat=k.best_one_action(initial,1)
        for method_id,method in enumerate(base.METHODS):
            if time.process_time()-cpu0>110:
                status="partial_cpu_guard";break
            try:
                selected,trace,calls=search_latest(k,method,np.random.default_rng(2301000+state_id*10+method_id),
                                                   search_draws,initial)
                duration=selected["duration"]
                c=base.selected_certificate(k,selected["policy"],selected["cap"],selected["raw_cap"],
                    (confirm[0][:,:duration],confirm[1][:,:duration]),initial,delta=DELTA)
            except CpuBudgetExceeded:
                status="partial_cpu_guard"
                break
            row=dict(state_id=state_id,zone=zone,equity_multiple=equity,method=method,duration=duration,
                     selected_candidate=selected["candidate"],cap=selected["cap"],raw_cap=selected["raw_cap"],
                     exact_best_one_action_local1_eur=exact_best,exact_best_one_action_leverage=exact_lev,
                     exact_best_nonflat_one_action_local1_eur=exact_nonflat,
                     exact_one_action_positive=exact_best>0,booked_mean_upper=c["booked_mean_upper"],
                     search_active_transitions=calls,confirmation_active_transitions=NCONFIRM*duration,
                     full_charge_same_action_positive=False,identity_error=c["identity_error"],
                     partial_terminal_alpha_strict_upper=c["partial_terminal_alpha_strict_upper"])
            for rep in ("raw","residual","clipped_raw","clipped","forgiveness","cost"):
                row.update({rep+"_"+key:value for key,value in c[rep].items()})
            for rep in ("raw","residual","clipped_raw","clipped"):
                row["certified_"+rep]=c["certified_"+rep]
            records.append(row)
            selected_records.append(dict(state_id=state_id,zone=zone,equity_multiple=equity,method=method,
                                          selected=selected,certificate=c))
            traces.extend(dict(state_id=state_id,method=method,**r) for r in trace)
            for horizon in HORIZONS:
                calendars.append(dict(state_id=state_id,zone=zone,equity_multiple=equity,method=method,
                    active_duration=duration,outer_horizon=horizon,**compile_calendar(c,duration,horizon,initial,k)))
        if status!="complete":break
    frame=pd.DataFrame(records)
    frame.to_csv(HERE/"full_clock_certificates.csv",index=False)
    pd.DataFrame(calendars).to_csv(HERE/"full_clock_calendar.csv",index=False)
    pd.DataFrame([{**r,"policy":json.dumps(r["policy"],sort_keys=True)} for r in traces]).to_csv(
        HERE/"full_clock_candidates.csv",index=False)
    summaries=[]
    for method,d in frame.groupby("method",sort=False):
        summaries.append(dict(method=method,states=len(d),one_action_exact_positive=int(d.exact_one_action_positive.sum()),
            raw_certified=int(d.certified_raw.sum()),residual_certified=int(d.certified_residual.sum()),
            clipped_raw_certified=int(d.certified_clipped_raw.sum()),clipped_residual_certified=int(d.certified_clipped.sum()),
            new_clipped_certificates_beyond_one_action=int((d.certified_clipped & ~d.exact_one_action_positive).sum()),
            search_active_transitions=int(d.search_active_transitions.sum()),
            confirmation_active_transitions=int(d.confirmation_active_transitions.sum())))
    result=dict(status="EXPLORATORY_"+status,state_method_pairs=len(records),planned_state_method_pairs=70,
        calendar_rows=len(calendars),calendar_rows_not_independent_trials=True,
        delta_per_bound=DELTA,familywise_error_budget=.05,planned_bounds=420,
        summary_by_method=summaries,selected_policies=selected_records,environment_check=check,
        calendar_compiler_environment_check=calendar_check,
        max_accounting_identity_error=float(frame.identity_error.max()),
        cpu_seconds=time.process_time()-cpu0,wall_seconds=time.perf_counter()-wall0,numeric_threads=1,
        actual_vectorized_transitions_all_phases=TRANSITION_COUNTER,
        incomplete_method_transition_work_in_counter=(status!="complete"),
        R_files_read=False,registered_files_modified=False,dp_solved=False,learners_trained=0,
        input_sha256=hashes,script_sha256=base.sha(__file__),kernel_script_sha256=base.sha(base.__file__),
        plan_sha256=base.sha(HERE/"EXPLORATORY_FULL_CLOCK_PLAN.md"),
        compiler="latest local-controller placement; algebraic gamma^(H-T) scaling for H>=T; no extra simulator calls",
        interpretation="All fixed states/methods retained. Positive means policy-specific empirical IID incentive, not acquisition or global safety; abstain outside compatible horizons or failed certificates.")
    (HERE/"full_clock_summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="selected_policies"},indent=2))


if __name__=="__main__":main()
