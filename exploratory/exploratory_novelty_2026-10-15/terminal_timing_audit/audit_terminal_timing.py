"""EXPLORATORY Study N terminal bonus-timing comparison; no frozen writes.

Recreate frozen 300-bin/1801-point model. Sparse matrices implement exactly
the frozen linear interpolation expectation, with floating summation-order
differences checked against stored probes. Correct V0=D/gamma puts bonus in
the last environmental reward, rather than one discount factor later.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[name]="1"

import numpy as np
import pandas as pd
from scipy import sparse

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
N=ROOT/"ACCOUNTING_OPTIONS_2026-10-14/study_n"
GAMMA=.99
K0=20000.
LEVELS=np.array([0,.25,1,4,16.])
GRID=np.linspace(-3*K0,6*K0,1801)
LP=(-.25,-.1,.02,.05,.1,.25,.5,1.,2.)
TP=(1,4,16,48,96,192)
START=time.perf_counter()
CAP=240.
INPUTS=set()


def check_time():
    if time.perf_counter()-START>CAP:
        raise TimeoutError("240-second single-thread audit cap reached")


def binned(zone):
    p=ROOT/f"GENERALIZATION_STUDY_2026-09-29/data_panel/panel_{zone}.pkl"
    INPUTS.add(p)
    d=pd.read_pickle(p)
    d=d.loc[d.split=="train"].iloc[200:]
    s=np.abs(d.imb.to_numpy(float)-d.spot.to_numpy(float))
    cap=.25*.25*d.bal_mw.to_numpy(float)
    rows=[]
    for lev in LEVELS:
        q=np.minimum(lev*.25,cap)
        z,f=q*s,1.*q
        parts=np.array_split(np.argsort(z),300)
        zb=np.array([z[i].mean() for i in parts])
        fb=np.array([f[i].mean() for i in parts])
        rows.append(np.r_[zb-fb,-zb-fb])
    return np.array(rows)


def matrices(X):
    n=len(GRID)
    matrices=[]
    for x in X:
        check_time()
        pts=np.clip(GRID[:,None]+x[None,:],GRID[0],GRID[-1])
        i1=np.clip(np.searchsorted(GRID,pts,side="right"),1,n-1)
        i0=i1-1
        w=(pts-GRID[i0])/(GRID[i1]-GRID[i0])
        rows=np.repeat(np.arange(n),len(x))
        p=sparse.coo_matrix((np.r_[(1-w).ravel(),w.ravel()]/len(x),
              (np.r_[rows,rows],np.r_[i0.ravel(),i1.ravel()])),shape=(n,n)).tocsr()
        p.eliminate_zeros()
        matrices.append(p)
    return matrices


def argbest(Q):
    return np.argmax(Q-1e-9*K0*np.arange(5)[None,:],axis=1)


def solve(zone):
    stored_path=N/f"dp_n2_{zone}.json"
    INPUTS.add(stored_path)
    stored=json.loads(stored_path.read_text())
    X=binned(zone)
    P=matrices(X)
    mean=X.mean(axis=1)[None,:]
    v_old=np.maximum(-GRID,0.)
    v_new=v_old/GAMMA
    probes=[]
    full=[]
    frozen_errors=[]
    frozen_action_mismatches=[]
    for tau in range(1,193):
        check_time()
        q_old=mean+GAMMA*np.column_stack([p@v_old for p in P])
        q_new=mean+GAMMA*np.column_stack([p@v_new for p in P])
        b_old,b_new=argbest(q_old),argbest(q_new)
        old_gain=q_old.max(1)-q_old[:,0]
        new_gain=q_new.max(1)-q_new[:,0]
        full.append({"zone":zone,"tau":tau,"grid_states":len(GRID),
            "changed_actions":int((b_old!=b_new).sum()),
            "largest_abs_optimal_gap_change_eur":float(abs(new_gain-old_gain).max()),
            "largest_abs_max16_gap_change_eur":float(abs((q_new[:,-1]-q_new[:,0])-(q_old[:,-1]-q_old[:,0])).max())})
        v_old=q_old[np.arange(len(GRID)),b_old]
        v_new=q_new[np.arange(len(GRID)),b_new]
        if tau not in TP:
            continue
        for equity in LP:
            i=int(np.argmin(abs(GRID-equity*K0)))
            ident=f"L{equity:g}_tau{tau}"
            saved_gain=stored["TFG"]["gain_eur"][ident]
            saved_action=stored["TFG"]["lev"][ident]
            frozen_errors.append(abs(float(old_gain[i])-saved_gain))
            if LEVELS[b_old[i]]!=saved_action:
                frozen_action_mismatches.append(ident)
            probes.append({"zone":zone,"arm":"TFG","probe":ident,"equity_allocations":equity,"tau":tau,
                "stored_action":saved_action,"old_recomputed_action":float(LEVELS[b_old[i]]),
                "corrected_action":float(LEVELS[b_new[i]]),"action_changed":bool(b_old[i]!=b_new[i]),
                "stored_optimal_minus_flat_eur":saved_gain,
                "old_optimal_minus_flat_eur":float(old_gain[i]),"corrected_optimal_minus_flat_eur":float(new_gain[i]),
                "optimal_gap_change_eur":float(new_gain[i]-old_gain[i]),
                "old_max16_minus_flat_eur":float(q_old[i,-1]-q_old[i,0]),
                "corrected_max16_minus_flat_eur":float(q_new[i,-1]-q_new[i,0]),
                "old_Q_all_actions_eur":q_old[i].tolist(),"corrected_Q_all_actions_eur":q_new[i].tolist()})
            # With no closing bonus, V=0 and flat is optimal analytically.
            saved_book=stored["TBK"]["lev"][ident]
            assert saved_book==0 and stored["TBK"]["gain_eur"][ident]==0
            probes.append({"zone":zone,"arm":"TBK","probe":ident,"equity_allocations":equity,"tau":tau,
                "stored_action":saved_book,"old_recomputed_action":0.,"corrected_action":0.,"action_changed":False,
                "stored_optimal_minus_flat_eur":0.,"old_optimal_minus_flat_eur":0.,"corrected_optimal_minus_flat_eur":0.,
                "optimal_gap_change_eur":0.,"old_max16_minus_flat_eur":float(mean[0,-1]),
                "corrected_max16_minus_flat_eur":float(mean[0,-1]),
                "old_Q_all_actions_eur":mean[0].tolist(),"corrected_Q_all_actions_eur":mean[0].tolist()})
    return probes,full,{"zone":zone,"sparse_nonzero_entries":sum(p.nnz for p in P),
        "max_recomputed_vs_stored_gap_error_eur":max(frozen_errors),
        "recomputed_vs_stored_action_mismatches":frozen_action_mismatches}


def main():
    inputs=[N/"dp_n2.py",N/"dp_n_fee.py",N/"envs_n.py"]
    INPUTS.update(inputs)
    probes,full,checks=[],[],[]
    status="complete"
    try:
        for zone in ("DK1","DK2"):
            p,f,c=solve(zone)
            probes+=p;full+=f;checks.append(c)
            print(json.dumps(c),flush=True)
    except TimeoutError as error:
        status="time_cap_incomplete"
        print(str(error),flush=True)
    result={"status":status,"label":"EXPLORATORY isolated terminal-bonus timing correction",
        "model":{"bins":300,"grid_points":1801,"grid_allocations":[-3,6],"gamma":GAMMA,"horizon":192,"fee":1.},
        "correction":"V0=(-ledger)+/gamma, making last-step bonus discount gamma^(tau-1), not gamma^tau",
        "stored_reproduction_checks":checks,"probes":probes,
        "changed_TFG_registered_probes":[p for p in probes if p["arm"]=="TFG" and p["action_changed"]],
        "max_abs_registered_optimal_gap_change_eur":max((abs(p["optimal_gap_change_eur"]) for p in probes),default=0.),
        "full_grid_changed_state_times":sum(f["changed_actions"] for f in full),
        "full_grid_total_state_times":sum(f["grid_states"] for f in full),
        "seconds":time.perf_counter()-START,"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(INPUTS)},
        "scope":"Only reward timing changed; same binned mixture, clipping and grid. No statistical verdicts or raw files changed."}
    (HERE/"exploratory_terminal_timing.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    pd.DataFrame(probes).to_csv(HERE/"exploratory_terminal_probe_comparison.csv",index=False)
    pd.DataFrame(full).to_csv(HERE/"exploratory_terminal_grid_change_by_tau.csv",index=False)
    print(json.dumps({k:result[k] for k in ("status","changed_TFG_registered_probes","max_abs_registered_optimal_gap_change_eur",
           "full_grid_changed_state_times","full_grid_total_state_times","seconds")},indent=2),flush=True)
    return 0 if status=="complete" else 2


if __name__=="__main__":
    raise SystemExit(main())
