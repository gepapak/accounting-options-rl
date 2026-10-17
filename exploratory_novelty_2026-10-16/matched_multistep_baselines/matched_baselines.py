"""EXPLORATORY fixed matched baselines; no policy search/training or old writes."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[name]="1"
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=ROOT/"exploratory_novelty_2026-10-15/adaptive_witness_empirical"
sys.path.insert(0,str(OLD))
import benchmark_adaptive_witness as original

HORIZON,INITIAL,FEE,GAMMA=96,10000.,1.,.99
NSEARCH,NCONFIRM,BATCH=8192,524288,8192
CAPS=(2048.,8192.,32768.,131072.)
SEARCH_SEED,CONFIRM_SEED,CHECK_SEED=2026101601,2026101602,2026101603
POLICIES=[dict(name="frozen_adaptive",kind="adaptive",duration=96,lev=16.)]
POLICIES += [dict(name=f"constant{lev:g}_T96",kind="open_loop",duration=96,lev=lev)
             for lev in (16.,4.,1.,.25)]
POLICIES += [dict(name=f"openloop16_T{t}",kind="open_loop",duration=t,lev=16.) for t in (1,4,16,48)]
ALIASES={"openloop16_T96":"constant16_T96"}
DELTA=.01/len(POLICIES)
CPU_DEADLINE=float("inf")
ACTIVE_TRANSITIONS=0


class CpuBudgetExceeded(Exception):
    pass


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def actions(policy,ledger,t,kernel,frozen):
    if policy["kind"]=="adaptive":
        return original.action(frozen,ledger,t,HORIZON,INITIAL,kernel,GAMMA).astype(float)
    return np.full(ledger.shape,policy["lev"] if t>=HORIZON-policy["duration"] else 0.)


def rollout(policy,kernel,indices,signs,frozen):
    """Paired same-action accounts; only active transitions need evaluation."""
    global ACTIVE_TRANSITIONS
    n=len(indices)
    ledger=np.full(n,INITIAL)
    gross_sum=np.zeros(n);cost_sum=np.zeros(n)
    for t in range(HORIZON-policy["duration"],HORIZON):
        if time.process_time()>CPU_DEADLINE:
            raise CpuBudgetExceeded()
        lev=actions(policy,ledger,t,kernel,frozen)
        q=np.minimum(.25*lev,kernel.cap[indices[:,t]])
        gross=q*kernel.spread[indices[:,t]]*signs[:,t]
        cost=FEE*q
        ledger+=gross-cost
        gross_sum+=GAMMA**t*gross;cost_sum+=GAMMA**t*cost
        ACTIVE_TRANSITIONS+=n
    forgiveness=GAMMA**(HORIZON-1)*np.maximum(-ledger,0)
    booked=gross_sum-cost_sum
    raw=booked+forgiveness;residual=forgiveness-cost_sum
    return dict(raw=raw,residual=residual,booked=booked,gross=gross_sum,cost=cost_sum,
                forgiveness=forgiveness,
                identity_error=float(np.max(np.abs(raw-gross_sum-residual))))


def cost_and_booked_bounds(policy,kernel):
    start=HORIZON-policy["duration"]
    weights=float(np.sum(GAMMA**np.arange(start,HORIZON)))
    maximum=policy["lev"]*.25*FEE*weights
    if policy["kind"]=="adaptive":
        return maximum,-kernel.stats[16]["mean_cost"],"upper bound from mandatory first trade"
    return maximum,-kernel.stats[policy["lev"]]["mean_cost"]*weights,"exact expectation"


def check_source(kernel,frozen):
    sys.path[:0]=[str(ROOT/"LIABILITY_STUDY_2026-09-29"),
                   str(ROOT/"ACCOUNTING_OPTIONS_2026-10-14/study_n")]
    from envs_b import LEV
    from envs_n import LagEnv
    rng=np.random.default_rng(CHECK_SEED)
    indices,signs=original.draws(rng,kernel,2,HORIZON)
    error=0.
    for policy in POLICIES:
        paired=rollout(policy,kernel,indices,signs,frozen)
        for j in range(2):
            market=SimpleNamespace(n=1000,spread=np.zeros(1000),cap_mwh=np.zeros(1000))
            market.spread[200:200+HORIZON]=kernel.spread[indices[j]]*signs[j]
            market.cap_mwh[200:200+HORIZON]=kernel.cap[indices[j]]
            for arm,key in (("TFG","raw"),("TBK","booked")):
                env=LagEnv(market,arm,train=False)
                env.reset(options=dict(e0=INITIAL/original.K0,tau0=HORIZON,t0=200))
                value=0.
                for t in range(HORIZON):
                    lev=float(actions(policy,np.array([env.L]),t,kernel,frozen)[0])
                    _,reward,_,_,_=env.step(int(np.flatnonzero(LEV==lev)[0]))
                    value+=GAMMA**t*reward*original.K0/100
                discrepancy=abs(value-paired[key][j]);error=max(error,discrepancy)
                assert discrepancy<1e-7,(policy["name"],arm,discrepancy)
    return dict(policies=len(POLICIES),arm_path_checks=len(POLICIES)*4,
                max_abs_error_eur=error,validation_seed=CHECK_SEED,
                frozen_environment_step_calls=len(POLICIES)*4*HORIZON)


def projected_score(values,lower,upper):
    log=np.log(2/DELTA)
    return float(values.mean()-np.sqrt(2*values.var(ddof=1)*log/NCONFIRM)
                 -7*(upper-lower)*log/(3*(NCONFIRM-1)))


def main():
    global CPU_DEADLINE
    if (HERE/"results.json").exists() or (HERE/"policy_summary.csv").exists():
        raise RuntimeError("Completed or partial outputs already exist; preserve them")
    cpu0,wall0=time.process_time(),time.perf_counter()
    CPU_DEADLINE=cpu0+116
    old_path=OLD/"selected_case_followup.json"
    old=json.loads(old_path.read_text())
    frozen=old["frozen_policy"]
    expected=dict(kind="two_stage",early_lev=16,late_lev=16,early_barrier=1.,late_barrier=2.,switch=.5)
    assert frozen==expected and old["zone"]=="DK1" and old["duration"]==HORIZON and old["initial_ledger_eur"]==INITIAL
    assert sha(OLD/"benchmark_adaptive_witness.py")==old["kernel_script_sha256"]
    panel=ROOT/"GENERALIZATION_STUDY_2026-09-29/data_panel/panel_DK1.pkl"
    assert sha(panel)==old["input_sha256"][str(panel.relative_to(ROOT))]
    data=pd.read_pickle(panel);data=data.loc[data.split=="train"].iloc[200:]
    kernel=original.EmpiricalKernel(np.abs(data.imb.to_numpy(float)-data.spot.to_numpy(float)),
                                   .25*.25*data.bal_mw.to_numpy(float),FEE)
    source=check_source(kernel,frozen)
    validation_transitions=ACTIVE_TRANSITIONS
    search_indices,search_signs=original.draws(np.random.default_rng(SEARCH_SEED),kernel,NSEARCH,HORIZON)
    caps={};scores=[]
    for policy in POLICIES:
        pair=rollout(policy,kernel,search_indices,search_signs,frozen)
        maximum,booked,kind=cost_and_booked_bounds(policy,kernel)
        candidates=[]
        for cap in CAPS:
            values=np.minimum(pair["residual"],cap)
            row=dict(policy=policy["name"],cap=cap,search_mean=float(values.mean()),
                     search_variance=float(values.var(ddof=1)),
                     projected_confirmation_lower=projected_score(values,-maximum,cap))
            candidates.append(row);scores.append(row)
        caps[policy["name"]]=max(candidates,key=lambda r:r["projected_confirmation_lower"])["cap"]
    search_transitions=ACTIVE_TRANSITIONS-validation_transitions
    buffers={p["name"]:np.empty(NCONFIRM) for p in POLICIES}
    completed={p["name"]:0 for p in POLICIES}
    sums={p["name"]:{k:0. for k in ("raw","residual","booked","gross","cost","forgiveness")} for p in POLICIES}
    max_identity=0.;status="COMPLETE"
    rng=np.random.default_rng(CONFIRM_SEED)
    confirmation_start=ACTIVE_TRANSITIONS
    for batch in range(NCONFIRM//BATCH):
        indices,signs=original.draws(rng,kernel,BATCH,HORIZON)
        try:
            for policy in POLICIES:
                name=policy["name"]
                paired=rollout(policy,kernel,indices,signs,frozen)
                n=completed[name]
                buffers[name][n:n+BATCH]=np.minimum(paired["residual"],caps[name])
                for field in sums[name]:sums[name][field]+=float(paired[field].sum())
                completed[name]+=BATCH
                max_identity=max(max_identity,paired["identity_error"])
        except CpuBudgetExceeded:
            status="INCOMPLETE_cpu_guard";break
    complete=all(n==NCONFIRM for n in completed.values())
    if not complete:status="INCOMPLETE_cpu_guard"
    rows=[];details=[]
    for policy in POLICIES:
        name=policy["name"];n=completed[name]
        maximum,booked,kind=cost_and_booked_bounds(policy,kernel)
        bound=original.empirical_bernstein(buffers[name],-maximum,caps[name],DELTA) if complete else None
        row=dict(policy=name,duration=policy["duration"],completed_confirmation_paths=n,
                 cap=caps[name],expected_booked_return_upper_or_exact=booked,booked_statement=kind,
                 cost_max=maximum,positive_witness_confirmed=(bound["lower"]>0) if bound else None,
                 search_active_transitions=NSEARCH*policy["duration"],
                 confirmation_active_transitions=n*policy["duration"],
                 aliases=";".join(alias for alias,target in ALIASES.items() if target==name))
        row.update({f"mean_{k}":v/n for k,v in sums[name].items()} if n else {})
        if bound:row.update({"capped_mean":bound["mean"],"capped_variance":bound["variance"],
                             "radius":bound["radius"],"lower":bound["lower"],"delta":DELTA})
        rows.append(row);details.append(dict(policy=policy,cap=caps[name],certificate=bound,summary=row))
    exact={str(lev):kernel.one_action(INITIAL,HORIZON,lev=lev) for lev in (.25,1,4,16)}
    frame=pd.DataFrame(rows);frame.to_csv(HERE/"policy_summary.csv",index=False)
    pd.DataFrame(scores).to_csv(HERE/"cap_search_scores.csv",index=False)
    inputs=[panel,old_path,OLD/"benchmark_adaptive_witness.py",
            ROOT/"LIABILITY_STUDY_2026-09-29/envs_b.py",ROOT/"ACCOUNTING_OPTIONS_2026-10-14/study_n/envs_n.py"]
    result=dict(status="EXPLORATORY_"+status,unique_policies=len(POLICIES),aliases=ALIASES,
        initial_ledger_eur=INITIAL,horizon=HORIZON,gamma=GAMMA,fee=FEE,frozen_adaptive_controller=frozen,
        planned_search_paths_per_policy=NSEARCH,planned_confirmation_paths_per_policy=NCONFIRM,
        all_fixed_confirmation_budgets_complete=complete,search_seed=SEARCH_SEED,confirmation_seed=CONFIRM_SEED,
        familywise_error_budget=.01,delta_per_bound=DELTA,allocated_one_sided_lower_bounds=len(POLICIES),
        policy_results=details,exact_isolated_action_values=exact,flat_inclusive_exact_isolated_reference=max(0,*exact.values()),
        source_check=source,max_accounting_identity_error=max_identity,
        validation_vectorized_transitions=validation_transitions,search_active_transitions=search_transitions,
        confirmation_active_transitions_including_partial=ACTIVE_TRANSITIONS-confirmation_start,
        actual_vectorized_transitions_all_phases=ACTIVE_TRANSITIONS,
        outer_confirmation_slots_if_all_policies_complete=len(POLICIES)*NCONFIRM*HORIZON,
        cpu_seconds=time.process_time()-cpu0,wall_seconds=time.perf_counter()-wall0,numeric_threads=1,
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs},script_sha256=sha(__file__),
        plan_sha256=sha(HERE/"EXPLORATORY_PLAN.md"),R_files_read=False,old_files_modified=False,
        dp_solved=False,learners_trained=0,
        interpretation="Matched fixed-policy certificates under the selected empirical IID law. Positive simple baseline suffices; nonpositive lower bound is abstention, not necessity or safety. No acquisition/search-superiority/generalization claim.")
    (HERE/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="policy_results"},indent=2),flush=True)
    print(frame.to_string(index=False),flush=True)


if __name__=="__main__":main()
