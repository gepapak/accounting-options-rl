"""EXPLORATORY selected-case fresh test; old benchmark rates stay unchanged."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import full_clock_benchmark as guard
base=guard.base
HERE,ROOT=base.HERE,base.ROOT
NSEARCH,NCONFIRM,BATCH,DELTA=8192,524288,8192,.01
CAPS=(2048.,8192.,32768.,131072.)
SEARCH_SEED,CONFIRM_SEED,CHECK_SEED=3401000,3501000,3601000


def select_old_controller():
    old=pd.read_csv(HERE/"full_clock_certificates.csv")
    eligible=old.loc[~old.exact_one_action_positive].copy()
    eligible["selection_projected_score"]=(eligible.residual_mean-
        np.sqrt(2*eligible.residual_variance*np.log(2/DELTA)/NCONFIRM))
    # Stable tie resolution preserves source-row order, with no confirmation input.
    eligible=eligible.sort_values("selection_projected_score",ascending=False,kind="stable")
    source=json.loads((HERE/"full_clock_summary.json").read_text())
    policies={(r["state_id"],r["method"]):r["selected"]["policy"] for r in source["selected_policies"]}
    eligible["frozen_policy_json"]=[json.dumps(policies[(r.state_id,r.method)],sort_keys=True)
                                      for r in eligible.itertuples()]
    eligible.to_csv(HERE/"selected_case_selection.csv",index=False)
    r=eligible.iloc[0]
    return r,policies[(int(r.state_id),str(r.method))],len(eligible)


def source_check(kernel,policy,initial,duration):
    sys.path[:0]=[str(ROOT/"LIABILITY_STUDY_2026-09-29"),
                   str(ROOT/"ACCOUNTING_OPTIONS_2026-10-14/study_n")]
    from envs_b import LEV
    from envs_n import LagEnv
    idx,sign=base.draws(np.random.default_rng(CHECK_SEED),kernel,2,duration)
    paired=base.rollout(kernel,policy,idx,sign,initial)
    maxerror=0.
    for j in range(2):
        market=SimpleNamespace(n=1000,spread=np.zeros(1000),cap_mwh=np.zeros(1000))
        market.spread[200:200+duration]=kernel.spread[idx[j]]*sign[j]
        market.cap_mwh[200:200+duration]=kernel.cap[idx[j]]
        for arm,key in (("TFG","raw"),("TBK","booked")):
            env=LagEnv(market,arm,train=False)
            env.reset(options=dict(e0=initial/base.K0,tau0=duration,t0=200))
            total=0.
            for t in range(duration):
                lev=int(base.action(policy,np.array([env.L]),t,duration,initial,kernel)[0])
                _,reward,_,_,_=env.step(int(np.flatnonzero(LEV==lev)[0]))
                total+=base.GAMMA**t*reward*base.K0/100
            error=abs(total-paired[key][j]);maxerror=max(maxerror,error)
            assert error<1e-7,(arm,error)
    return dict(arm_path_checks=4,max_abs_error_eur=maxerror,validation_seed=CHECK_SEED)


def projected_cap_score(values,lower,upper):
    log=np.log(2/DELTA)
    return float(values.mean()-np.sqrt(2*values.var(ddof=1)*log/NCONFIRM)
        -7*(upper-lower)*log/(3*(NCONFIRM-1)))


def main():
    cpu0,wall0=time.process_time(),time.perf_counter()
    guard.CPU_DEADLINE=cpu0+97
    guard.TRANSITION_COUNTER=0
    chosen,policy,eligible_count=select_old_controller()
    duration=int(chosen.duration);initial=float(chosen.equity_multiple)*base.K0;zone=str(chosen.zone)
    panel=ROOT/f"GENERALIZATION_STUDY_2026-09-29/data_panel/panel_{zone}.pkl"
    d=pd.read_pickle(panel);d=d.loc[d.split=="train"].iloc[200:]
    kernel=base.EmpiricalKernel(np.abs(d.imb.to_numpy(float)-d.spot.to_numpy(float)),
                                .25*.25*d.bal_mw.to_numpy(float))
    check=source_check(kernel,policy,initial,duration)
    costmax=4*base.FEE*float(np.sum(base.GAMMA**np.arange(duration)))
    s=base.rollout(kernel,policy,*base.draws(np.random.default_rng(SEARCH_SEED),kernel,NSEARCH,duration),initial)
    scores=[]
    for cap in CAPS:
        values=np.minimum(s["residual"],cap)
        scores.append(dict(cap=cap,search_clipped_mean=float(values.mean()),
                           search_clipped_variance=float(values.var(ddof=1)),
                           projected_confirmation_lower=projected_cap_score(values,-costmax,cap)))
    selected=max(scores,key=lambda r:r["projected_confirmation_lower"])
    cap=selected["cap"]
    confirm_rng=np.random.default_rng(CONFIRM_SEED)
    clipped=np.empty(NCONFIRM)
    sums={k:0. for k in ("raw","residual","booked","gross","cost","forgiveness")}
    completed,max_identity,status=0,0.,"COMPLETE"
    for batch in range(NCONFIRM//BATCH):
        try:
            paired=base.rollout(kernel,policy,*base.draws(confirm_rng,kernel,BATCH,duration),initial)
        except guard.CpuBudgetExceeded:
            status="INCOMPLETE_cpu_guard";break
        clipped[completed:completed+BATCH]=np.minimum(paired["residual"],cap)
        for key in sums:sums[key]+=float(paired[key].sum())
        completed+=BATCH
        max_identity=max(max_identity,paired["identity_error"])
    if completed!=NCONFIRM:status="INCOMPLETE_cpu_guard"
    certificate=base.empirical_bernstein(clipped,-costmax,cap,DELTA) if status=="COMPLETE" else None
    booked_upper=-kernel.stats[16]["mean_cost"]
    result=dict(status="EXPLORATORY_selected_case_"+status,selection_eligible_controllers=eligible_count,
        selected_old_record={k:(v.item() if hasattr(v,"item") else v) for k,v in chosen.to_dict().items()},
        frozen_policy=policy,zone=zone,initial_ledger_eur=initial,duration=duration,
        search_seed=SEARCH_SEED,confirmation_seed=CONFIRM_SEED,n_search=NSEARCH,
        planned_confirmation_paths=NCONFIRM,completed_confirmation_paths=completed,batch_paths=BATCH,
        cap_candidates=scores,selected_cap=cap,certificate=certificate,delta_single_new_test=DELTA,
        positive_witness_confirmed=(certificate["lower"]>0) if certificate else None,
        expected_booked_return_upper=booked_upper,full_charge_same_action_positive=False,
        exact_all_level_one_action_value=kernel.best_one_action(initial,duration)[0],
        exact_best_nonflat_one_action_value=kernel.best_one_action(initial,duration)[2],
        descriptive_confirmation_means={k:v/completed for k,v in sums.items()} if completed else None,
        source_check=check,max_accounting_identity_error=max_identity,
        actual_vectorized_transitions_all_phases=guard.TRANSITION_COUNTER,
        cpu_seconds=time.process_time()-cpu0,wall_seconds=time.perf_counter()-wall0,numeric_threads=1,
        input_sha256={str(p.relative_to(ROOT)):base.sha(p) for p in (panel,
            ROOT/"LIABILITY_STUDY_2026-09-29/envs_b.py",ROOT/"ACCOUNTING_OPTIONS_2026-10-14/study_n/envs_n.py",
            HERE/"full_clock_certificates.csv",HERE/"full_clock_summary.json")},
        script_sha256=base.sha(__file__),kernel_script_sha256=base.sha(base.__file__),
        guard_script_sha256=base.sha(guard.__file__),plan_sha256=base.sha(HERE/"SELECTED_CASE_FOLLOWUP_PLAN.md"),
        R_files_read=False,registered_files_modified=False,dp_solved=False,learners_trained=0,
        interpretation="Fresh independent selected-case conditional99%certificate, not comparative discoveryrate/newmarketconfirmation; all old benchmark results unchanged. Incomplete confirmation yields no verdict.")
    (HERE/"selected_case_followup.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
