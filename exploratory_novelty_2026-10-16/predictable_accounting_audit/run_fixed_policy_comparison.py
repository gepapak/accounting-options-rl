"""Exploratory frozen development-case estimator comparison, full fixed n."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import json,time,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import predictable_audit as a
import run_heldout_attempt2 as h

HERE,ROOT=a.HERE,a.ROOT
OUT=HERE/'fixed_policy_outputs'
DELTA=.05/45

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    cpu,wall=time.process_time(),time.perf_counter()
    if OUT.exists():raise FileExistsError('Previous output is immutable.')
    OUT.mkdir()
    paths=[Path(__file__),HERE/'FIXED_POLICY_PLAN.md',HERE/'predictable_audit.py',HERE/'run_heldout_attempt2.py',
        ROOT/'exploratory_novelty_2026-10-16/matched_multistep_baselines/results.json',
        ROOT/'GENERALIZATION_STUDY_2026-09-29/data_panel/panel_DK1.pkl']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    (OUT/'frozen_inputs.json').write_text(json.dumps(dict(input_sha256=hashes,nsearch=8192,nconfirm=32768,
        delta_per_bound=DELTA,planned_bounds=45),indent=2),encoding='utf-8')
    d=pd.read_pickle(paths[-1]);d=d.loc[d.split=='train'].iloc[200:]
    k=a.Kernel(abs(d.imb.to_numpy(float)-d.spot.to_numpy(float)),.25*.25*d.bal_mw.to_numpy(float))
    definitions=[('adaptive',96,dict(kind='two_stage',early_lev=16,late_lev=16,early_barrier=1.,late_barrier=2.,switch=.5))]
    definitions += [(f'constant{l}_T96',96,dict(kind='constant',lev=l)) for l in (16,4,1,.25)]
    definitions += [(f'final16_T{t}',t,dict(kind='constant',lev=16)) for t in (1,4,16,48)]
    search=a.old.draws(np.random.default_rng(5130000),k,8192,96)
    confirm=a.old.draws(np.random.default_rng(5140000),k,32768,96)
    records=[];transitions=0;executed=0
    olddelta=h.DELTA;h.DELTA=DELTA
    for name,t,p in definitions:
        factor=a.GAMMA**(96-t)
        b=a.bounds(k,p,10000.,t,'terminal')
        for key in b:b[key]*=factor
        outputs=[]
        for draw in (search,confirm):
            r=a.rollout(k,p,draw[0][:,-t:],draw[1][:,-t:],10000.,'terminal')
            transitions+=r['transitions'];executed+=r['executed_actions']
            for key in ('raw','residual','predictable','cost','forgiveness','booked','mean_cost_residual','expected_cost'):
                r[key]*=factor
            outputs.append(r)
        choices=h.views(outputs[0],b,32768)
        for method in h.METHODS:
            v=h.confirm(outputs[1],b,method,choices[method]['cap'])
            records.append(dict(policy_name=name,policy=p,active_duration=t,factor=factor,method=method,
                cap=choices[method]['cap'],bound=v,booked_upper=b['booked_upper'],
                certified=v['lower']>0 and b['booked_upper']<0))
    h.DELTA=olddelta
    result=dict(status='EXPLORATORY_complete',records=records,nsearch=8192,nconfirm=32768,delta_per_bound=DELTA,
        family_error=.05,planned_bounds=45,shared_unique_active_transitions=transitions,
        represented_full_clock_transitions=9*(8192+32768)*96,executed_actions=executed,
        conditional_oracle_points=k.oracle_points,preprocessing_rows=k.n,repair=k.repair(),
        cpu_seconds=time.process_time()-cpu,wall_seconds=time.perf_counter()-wall,numeric_threads=1,input_sha256=hashes)
    result['summary']=[dict(method=m,certified=sum(r['method']==m and r['certified'] for r in records)) for m in h.METHODS]
    (OUT/'summary.json').write_text(h.encode(result,indent=2),encoding='utf-8')
    flat=[{**{key:value for key,value in r.items() if key not in ('policy','bound')},**r['bound']} for r in records]
    pd.DataFrame(flat).to_csv(OUT/'certificates.csv',index=False)
    print(h.encode({k:result[k] for k in ('status','summary','shared_unique_active_transitions','conditional_oracle_points','cpu_seconds','wall_seconds')},indent=2))

if __name__=='__main__':main()
