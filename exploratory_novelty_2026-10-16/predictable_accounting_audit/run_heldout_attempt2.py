"""Fixed exploratory TEST-kernel evaluation; never overwrites an earlier attempt."""
from __future__ import annotations
import hashlib,itertools,json,time,os,sys
from pathlib import Path
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import numpy as np
import pandas as pd
import predictable_audit as a

HERE,ROOT=a.HERE,a.ROOT
OUTPUT=HERE/'heldout_outputs_attempt2'
METHODS=('raw','clipped_raw','clipped_residual','clipped_mean_cost_residual','predictable')
CASES=tuple(itertools.product(('DK1','DK2'),('floor','terminal'),(.1,.5,1.),(16,96)))
REPEATS=3
CAPS=(32.,128.,512.,2048.,8192.,32768.,131072.)
BSEARCH,BCONFIRM=262144,2097152
DELTA=.05/(len(CASES)*REPEATS*10)

def encode(value,**kwargs):
    def default(obj):
        if isinstance(obj,np.generic):return obj.item()
        raise TypeError(type(obj).__name__)
    return json.dumps(value,default=default,**kwargs)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def bound(values,lo,hi,n,delta=DELTA):
    mean=float(values.mean());variance=float(values.var(ddof=1));lg=np.log(2/delta)
    radius=float(np.sqrt(2*variance*lg/n)+7*(hi-lo)*lg/(3*(n-1)))
    return dict(mean=mean,variance=variance,radius=radius,lower=mean-radius,
                range_lower=float(lo),range_upper=float(hi),n=n,delta=delta)

def views(r,b,n):
    rows={}
    rows['raw']={**bound(r['raw'],b['raw_lower'],b['raw_upper'],n), 'cap':None}
    pad=b['predictable_numerical_allowance']
    rows['predictable']={**bound(r['predictable'],b['comp_lower']-pad-1e-7,b['comp_upper']+pad+1e-7,n), 'cap':None}
    rows['predictable']['lower']-=b['predictable_numerical_allowance']
    for method,key,lo in [('clipped_raw','raw',b['raw_lower']),('clipped_residual','residual',-b['cost_max']),
                          ('clipped_mean_cost_residual','mean_cost_residual',-b['mean_cost_max'])]:
        candidates=[]
        for cap in CAPS:
            hi=min(cap,b['raw_upper']) if key=='raw' else cap
            padding=pad if method=='clipped_mean_cost_residual' else 0.
            candidates.append({**bound(np.minimum(r[key],cap),lo-padding,hi+padding,n),'cap':cap})
        rows[method]=max(candidates,key=lambda x:x['lower'])
    return rows

def confirm(r,b,method,cap):
    if method=='predictable':
        x=r['predictable'];lo=b['comp_lower'];hi=b['comp_upper']
    elif method=='raw':x=r['raw'];lo=b['raw_lower'];hi=b['raw_upper']
    elif method=='clipped_raw':x=np.minimum(r['raw'],cap);lo=b['raw_lower'];hi=min(cap,b['raw_upper'])
    elif method=='clipped_mean_cost_residual':x=np.minimum(r['mean_cost_residual'],cap);lo=-b['mean_cost_max'];hi=cap
    else:x=np.minimum(r['residual'],cap);lo=-b['cost_max'];hi=cap
    if method in ('predictable','clipped_mean_cost_residual'):
        lo-=b['predictable_numerical_allowance'];hi+=b['predictable_numerical_allowance']
    result=a.certificate(x,lo,hi,DELTA)
    result['uncorrected_lower']=result['lower']
    result['numerical_allowance']=b['predictable_numerical_allowance'] if method in ('predictable','clipped_mean_cost_residual') else 1e-7
    result['lower']-=result['numerical_allowance']
    return result

def main():
    cpu,wall=time.process_time(),time.perf_counter()
    if OUTPUT.exists():raise FileExistsError('Earlier outputs retained; do not rerun in place.')
    OUTPUT.mkdir()
    inputs={}
    for z in ('DK1','DK2'):
        p=ROOT/f'GENERALIZATION_STUDY_2026-09-29/data_panel/panel_{z}.pkl'
        inputs[str(p.relative_to(ROOT))]=sha(p)
    for p in (Path(__file__),HERE/'predictable_audit.py',HERE/'EXPLORATORY_PLAN.md',a.OLD/'benchmark_adaptive_witness.py',
              ROOT/'LIABILITY_STUDY_2026-09-29/envs_b.py',ROOT/'GENERALIZATION_STUDY_2026-09-29/envs.py'):
        inputs[str(p.relative_to(ROOT))]=sha(p)
    frozen=dict(status='EXPLORATORY_FROZEN_before_evaluation',input_sha256=inputs,
        cases=len(CASES),repeats=REPEATS,bounds_per_case_repeat=10,delta_per_bound=DELTA,
        family_error=.05,search_transition_budget=BSEARCH,confirmation_transition_budget=BCONFIRM,
        versions=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__))
    (OUTPUT/'frozen_inputs.json').write_text(encode(frozen,indent=2),encoding='utf-8')
    # Software checks use development TRAIN only before touching held-out payoff curves.
    train=pd.read_pickle(ROOT/'GENERALIZATION_STUDY_2026-09-29/data_panel/panel_DK1.pkl')
    train=train.loc[train.split=='train'].iloc[200:]
    kt=a.Kernel(abs(train.imb.to_numpy(float)-train.spot.to_numpy(float)),.25*.25*train.bal_mw.to_numpy(float))
    validation=a.validate(kt)
    kernels={};preprocess={}
    for z in ('DK1','DK2'):
        begin=time.perf_counter()
        d=pd.read_pickle(ROOT/f'GENERALIZATION_STUDY_2026-09-29/data_panel/panel_{z}.pkl')
        d=d.loc[d.split=='test'].iloc[200:]
        kernels[z]=a.Kernel(abs(d.imb.to_numpy(float)-d.spot.to_numpy(float)),.25*.25*d.bal_mw.to_numpy(float))
        preprocess[z]=dict(rows=len(d),seconds=time.perf_counter()-begin,
                           repair=kernels[z].repair(),oracle_roundoff_allowance=kernels[z].oracle_error)
    preprocessing_oracle_points=sum(k.oracle_points for k in kernels.values())
    policies=a.policies();assert len(policies)==13
    rows=[];neutral_rows=[];trace=[];unique_calls=0;executed=0;oracle_points=0;status='complete'
    for case_id,(zone,account,equity,horizon) in enumerate(CASES):
        k=kernels[zone];initial=equity*a.K0
        nsearch=BSEARCH//(len(policies)*horizon);nconfirm=BCONFIRM//horizon
        exact=a.exact_one_action(k,initial,account,horizon)
        for rep in range(REPEATS):
            if time.process_time()-cpu>900:
                status='partial_CPU_guard';break
            offset=100*case_id+rep
            search=a.old.draws(np.random.default_rng(4100000+offset),k,nsearch,horizon)
            search_views=[]
            for candidate,p in enumerate(policies):
                r=a.rollout(k,p,*search,initial,account);b=a.bounds(k,p,initial,horizon,account)
                unique_calls+=r['transitions'];executed+=r['executed_actions'];oracle_points+=r['total_oracle_points']
                v=views(r,b,nconfirm);search_views.append(v)
                for method in METHODS:trace.append(dict(case_id=case_id,repetition=rep,candidate=candidate,
                    method=method,projected_score=v[method]['lower'],cap=v[method]['cap'],nsearch=nsearch))
            selected={m:max(range(len(policies)),key=lambda i:search_views[i][m]['lower']) for m in METHODS}
            neutral=(4,6,8)[rep]
            draw=a.old.draws(np.random.default_rng(4200000+offset),k,nconfirm,horizon)
            caches={}
            for candidate in sorted(set(selected.values())|{neutral}):
                r=a.rollout(k,policies[candidate],*draw,initial,account)
                caches[candidate]=r
                unique_calls+=r['transitions'];executed+=r['executed_actions'];oracle_points+=r['total_oracle_points']
            common=dict(case_id=case_id,zone=zone,account=account,equity_multiple=equity,horizon=horizon,repetition=rep,
                nsearch_per_candidate=nsearch,nconfirm=nconfirm,exact_one_action=exact['value'],
                exact_one_action_leverage=exact['leverage'],exact_one_action_positive=exact['value']>1e-10,
                logical_search_transitions=nsearch*horizon*len(policies),logical_confirmation_transitions=nconfirm*horizon)
            for method in METHODS:
                candidate=selected[method];p=policies[candidate];b=a.bounds(k,p,initial,horizon,account)
                c=confirm(caches[candidate],b,method,search_views[candidate][method]['cap'])
                row={**common,'method':method,'candidate':candidate,'policy':p,
                    'cap':search_views[candidate][method]['cap'],'bound':c,'booked_upper':b['booked_upper'],
                    'certified':c['lower']>0 and b['booked_upper']<0,
                    'full_charge_same_actions_positive':False,
                    'raw_mean_descriptive':float(caches[candidate]['raw'].mean()),
                    'predictable_mean_descriptive':float(caches[candidate]['predictable'].mean()),
                    'identity_error':caches[candidate]['identity_error']}
                rows.append(row)
            p=policies[neutral];b=a.bounds(k,p,initial,horizon,account)
            for method in METHODS:
                c=confirm(caches[neutral],b,method,search_views[neutral][method]['cap'])
                neutral_rows.append({**common,'method':method,'candidate':neutral,'policy':p,
                    'cap':search_views[neutral][method]['cap'],'bound':c,'booked_upper':b['booked_upper'],
                    'certified':c['lower']>0 and b['booked_upper']<0})
            (OUTPUT/f'case_{case_id:02d}_rep_{rep}.json').write_text(encode(
                dict(selected=rows[-5:],neutral=neutral_rows[-5:]),indent=2),encoding='utf-8')
            print(f'case {case_id+1}/24 rep {rep+1}/3 {zone} {account} e{equity} H{horizon} '+
                  ' '.join(f'{r["method"]}:{int(r["certified"])}' for r in rows[-5:])+f' CPU{time.process_time()-cpu:.1f}',flush=True)
        if status!='complete':break
    result=dict(status='EXPLORATORY_'+status,planned_case_repetitions=72,completed_case_repetitions=len(rows)//5,
        selected_rows=rows,neutral_rows=neutral_rows,preprocessing=preprocess,validation=validation,
        delta_per_bound=DELTA,family_error=.05,planned_lower_bounds=720,actual_unique_transitions=unique_calls,
        executed_actions=executed,rollout_conditional_oracle_points=oracle_points,
        conditional_oracle_points=sum(k.oracle_points for k in kernels.values()),
        preprocessing_oracle_points=preprocessing_oracle_points,cpu_seconds=time.process_time()-cpu,
        wall_seconds=time.perf_counter()-wall,numeric_threads=1,input_sha256=inputs,
        R_inputs_used=False,learners_trained=0,dp_solved=False)
    for label,records in [('selected_summary',rows),('neutral_summary',neutral_rows)]:
        result[label]=[dict(method=m,pairs=sum(r['method']==m for r in records),
            certified=sum(r['method']==m and r['certified'] for r in records),
            new_beyond_one_action=sum(r['method']==m and r['certified'] and not r['exact_one_action_positive'] for r in records))
            for m in METHODS]
    (OUTPUT/'summary.json').write_text(encode(result,indent=2),encoding='utf-8')
    pd.DataFrame(trace).to_csv(OUTPUT/'search_trace.csv',index=False)
    print(encode({k:result[k] for k in ('status','completed_case_repetitions','selected_summary','neutral_summary','actual_unique_transitions','conditional_oracle_points','cpu_seconds','wall_seconds')},indent=2),flush=True)

if __name__=='__main__':main()
