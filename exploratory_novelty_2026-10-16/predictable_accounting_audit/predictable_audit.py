"""Exploratory accounting-specific Doob-compensator audit, no DP or learner.

Conditional expectations are integrated over a declared finite empirical law.
This is not an estimator of an unknown future market law from black-box samples.
"""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import sys
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
sys.dont_write_bytecode=True
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=ROOT/'exploratory_novelty_2026-10-15/adaptive_witness_empirical'
sys.path.insert(0,str(OLD))
import benchmark_adaptive_witness as old

GAMMA,K0=old.GAMMA,old.K0
LEVELS=(0.,.25,1.,4.,16.)

class Kernel(old.EmpiricalKernel):
    def __init__(self,*args,**kwargs):
        self.oracle_points=0
        super().__init__(*args,**kwargs)
        for lev,s in self.stats.items():
            s['prefix']=np.r_[0.,np.cumsum(s['losses'])]
            s['h0']=float(self.h(lev,0.))
        size=max(len(s['losses']) for s in self.stats.values())
        eps=np.finfo(float).eps
        g=size*eps/(1-size*eps)
        # Conservative accumulation/multiply/subtract allowance for finite rows.
        self.oracle_error=64*(g+eps)*(1+max(np.abs(s['losses']).max() for s in self.stats.values()))

    def h(self,lev,ledger):
        v=np.asarray(ledger,float)
        self.oracle_points+=v.size
        s=self.stats[lev];loss=s['losses']
        i=np.searchsorted(loss,v,side='right')
        value=(s['suffix'][i]-(len(loss)-i)*v)/len(loss)
        return np.where(v>=loss[-1],0.,np.where(v<=loss[0],s['mean_cost']-v,value))

    def drift(self,lev,ledger):
        """Stable Delta=h(n)-(-n)+; avoid subtracting two large debt values."""
        n=np.asarray(ledger,float)
        out=np.empty(n.shape,float)
        positive=n>=0
        if positive.any(): out[positive]=self.h(lev,n[positive])
        if (~positive).any():
            s=self.stats[lev];v=n[~positive]
            self.oracle_points+=v.size
            j=np.searchsorted(s['losses'],v,side='left')
            out[~positive]=s['mean_cost']+(v*j-s['prefix'][j])/len(s['losses'])
        return out

    def repair(self):
        ratios={str(l):(s['h0']/s['mean_cost'] if s['mean_cost']>0 else
                       (float('inf') if s['h0']>0 else 0.)) for l,s in self.stats.items()}
        rho=max(ratios.values())
        return dict(ratio_by_action=ratios,rho=rho,alpha_all_state_safe=max(0.,1-1/rho) if rho else 0.,
                    scope='fixed empirical law, declared actions, no-edge, all adaptive policies and finite horizons; charged advantage over same-account flat (equals charged return for solvent starts)')

def policies():
    return ([dict(kind='one_action')]+[dict(kind='constant',lev=l) for l in (.25,1,4,16)]
       +[dict(kind='barrier',barrier=b) for b in (.5,1.,2.)]
       +[dict(kind='two_stage',early_barrier=b,late_barrier=c,switch=.5,
              early_lev=16,late_lev=16) for b,c in ((1.,2.),(1.,.5),(2.,.5),(2.,2.))]
       +[dict(kind='myopic')])

def action(policy,ledger,t,horizon,initial,kernel,account):
    if policy['kind']=='constant':return np.full(ledger.shape,policy['lev'],float)
    if policy['kind']=='one_action':
        when=0 if account=='floor' else horizon-1
        return np.full(ledger.shape,16. if t==when else 0.,float)
    if t==0:return np.full(ledger.shape,16.,float)
    obs=np.clip(ledger,-3*K0,5*K0)
    if policy['kind']=='myopic':
        best=np.zeros(ledger.shape);out=np.zeros(ledger.shape)
        for lev in (.25,1,4,16):
            value=(kernel.h(lev,obs) if account=='floor' else
                   GAMMA**(horizon-1-t)*kernel.drift(lev,obs))-kernel.stats[lev]['mean_cost']
            better=value>best;best[better]=value[better];out[better]=lev
        return out
    return old.action(policy,ledger,t,horizon,initial,kernel).astype(float)

def bounds(kernel,policy,initial,horizon,account,alpha=0.):
    if alpha!=0.:raise ValueError('This empirical rollout/bounds interface evaluates alpha0 only; use repair() for the analytic charge family.')
    if initial<0:raise ValueError('This empirical interface uses solvent starts; negative starts require flat-default subtraction.')
    w=GAMMA**np.arange(horizon);d=w[-1]
    if policy['kind']=='constant':admissible=(policy['lev'],)
    elif policy['kind'] in ('barrier','one_action'):admissible=(0.,16.)
    elif policy['kind']=='two_stage':admissible=tuple(set((0.,16.,policy['early_lev'],policy['late_lev'])))
    else:admissible=LEVELS
    costs=np.array([0. if l==0 else kernel.stats[l]['mean_cost'] for l in admissible])
    h0=np.array([0. if l==0 else kernel.stats[l]['h0'] for l in admissible])
    comp_lower=-float(w.sum())*float(costs.max())
    weights=w if account=='floor' else np.full(horizon,d)
    comp_upper=float(np.maximum(0.,((1-alpha)*weights[:,None]*h0-w[:,None]*costs).max(axis=1)).sum())
    active=[l for l in admissible if l]
    maxloss=max(-kernel.stats[l]['min_x'] for l in active)
    maxgain=max(0.,max(kernel.stats[l]['max_x'] for l in active))
    costmax=max(active)*.25*kernel.fee*float(w.sum())
    forgmax=(maxloss*float(w.sum()) if account=='floor' else
             d*max(0.,horizon*maxloss-initial))
    if account=='floor':rawlo=max(-initial,-maxloss*float(w.sum()));rawhi=maxgain*float(w.sum())
    else:
        wt=float(np.sum(np.arange(1,horizon)*GAMMA**np.arange(horizon-1)))
        rawlo=max(-maxloss*float(w.sum()),-d*initial-(1-GAMMA)*maxloss*wt)
        rawhi=maxgain*float(w.sum())
    if policy['kind']=='one_action':
        lev=16;discount=1. if account=='floor' else d
        mean=discount*(kernel.h(lev,initial)-kernel.stats[lev]['mean_cost'])
        comp_lower=comp_upper=mean
        costmax=discount*4*kernel.fee
        forgmax=discount*max(0.,kernel.max_loss-initial)
        rawlo=discount*(max(0.,initial+kernel.stats[16]['min_x'])-initial)
        rawhi=discount*(max(0.,initial+kernel.stats[16]['max_x'])-initial)
        bookupper=-discount*kernel.stats[16]['mean_cost']
    elif policy['kind']=='constant':
        bookupper=-float(w.sum())*kernel.stats[policy['lev']]['mean_cost']
    else:bookupper=-kernel.stats[16]['mean_cost']
    return dict(comp_lower=comp_lower,comp_upper=comp_upper,cost_max=costmax,
        mean_cost_max=(float(w.sum())*float(costs.max()) if policy['kind']!='one_action' else -bookupper),
        forgiveness_max=forgmax,raw_lower=rawlo,raw_upper=rawhi,booked_upper=bookupper,
        predictable_numerical_allowance=2*horizon*kernel.oracle_error)

def rollout(kernel,policy,indices,signs,initial,account):
    n,horizon=indices.shape
    start_queries=kernel.oracle_points
    ledger=np.full(n,initial,float);gross_sum=np.zeros(n);cost_sum=np.zeros(n)
    forgiveness=np.zeros(n);comp=np.zeros(n);expected_cost=np.zeros(n);query_count=0;executed=0
    for t in range(horizon):
        lev=action(policy,ledger,t,horizon,initial,kernel,account)
        for l in (.25,1,4,16):
            mask=lev==l
            if not mask.any():continue
            s=kernel.stats[l]
            drift=(kernel.h(l,ledger[mask]) if account=='floor' else kernel.drift(l,ledger[mask]))
            coeff=GAMMA**t if account=='floor' else GAMMA**(horizon-1)
            comp[mask]+=coeff*drift-GAMMA**t*s['mean_cost']
            expected_cost[mask]+=GAMMA**t*s['mean_cost']
            query_count+=int(mask.sum())
        q=np.minimum(lev*.25,kernel.cap[indices[:,t]])
        gross=q*kernel.spread[indices[:,t]]*signs[:,t]
        cost=q*kernel.fee
        before=ledger+gross-cost
        if account=='floor':
            forgiveness+=GAMMA**t*np.maximum(-before,0.)
            ledger=np.maximum(before,0.)
        else:ledger=before
        gross_sum+=GAMMA**t*gross;cost_sum+=GAMMA**t*cost
        executed+=int(np.count_nonzero(lev))
    if account=='terminal':forgiveness=GAMMA**(horizon-1)*np.maximum(-ledger,0.)
    raw=gross_sum-cost_sum+forgiveness;residual=forgiveness-cost_sum
    b=bounds(kernel,policy,initial,horizon,account)
    assert np.all(comp>=b['comp_lower']-1e-6) and np.all(comp<=b['comp_upper']+1e-6)
    return dict(raw=raw,residual=residual,predictable=comp,cost=cost_sum,
                mean_cost_residual=forgiveness-expected_cost,expected_cost=expected_cost,
                forgiveness=forgiveness,booked=gross_sum-cost_sum,
                transitions=n*horizon,executed_actions=executed,drift_queries=query_count,
                total_oracle_points=kernel.oracle_points-start_queries,
                identity_error=float(np.max(np.abs(raw-gross_sum-residual))))

def certificate(x,lower,upper,delta):
    # A small outward padding protects displayed numeric support checks.
    return old.empirical_bernstein(x,lower-1e-7,upper+1e-7,delta)

def representations(result,b,cap,rawcap,delta):
    return dict(raw=certificate(result['raw'],b['raw_lower'],b['raw_upper'],delta),
        clipped_raw=certificate(np.minimum(result['raw'],rawcap),b['raw_lower'],min(rawcap,b['raw_upper']),delta),
        clipped_residual=certificate(np.minimum(result['residual'],cap),-b['cost_max'],cap,delta),
        predictable=certificate(result['predictable'],b['comp_lower'],b['comp_upper'],delta))

def exact_one_action(kernel,initial,account,horizon):
    discount=1. if account=='floor' else GAMMA**(horizon-1)
    values={0.:0.,**{l:discount*(kernel.h(l,initial)-s['mean_cost']) for l,s in kernel.stats.items()}}
    l=max(values,key=values.get)
    return dict(value=values[l],leverage=l,best_nonflat=max(v for a,v in values.items() if a))

def validate(kernel):
    errors=[]
    for l,s in kernel.stats.items():
        losses=s['losses']
        for n in (-1e8,-100000.,-10000.,-1.,0.,1.,10000.,100000.,1e8):
            direct=float(np.maximum(losses-n,0).mean())-max(-n,0.)
            actual=float(kernel.drift(l,np.array([n]))[0])
            errors.append(abs(actual-direct))
            assert -1e-6<=actual<=s['h0']+1e-6
    assert max(errors)<1e-6,max(errors)
    # Compare terminal state updates/returns against the unchanged implementation.
    dr=old.draws(np.random.default_rng(4010000),kernel,16,16)
    error=0.
    for p in (dict(kind='barrier',barrier=1.),policies()[8]):
        a=rollout(kernel,p,*dr,.5*K0,'terminal');b=old.rollout(kernel,p,*dr,.5*K0)
        error=max(error,float(np.max(np.abs(a['raw']-b['raw']))))
    assert error<1e-8
    # Independent frozen floor replay at its declared .10 fee (no monkey patch).
    sys.path.insert(0,str(ROOT/'LIABILITY_STUDY_2026-09-29'))
    from envs_b import LiabilityEnv,LEV
    from types import SimpleNamespace
    floor_kernel=Kernel(kernel.spread,kernel.cap,fee=.1)
    floor_error=0.;floor_checks=0
    for p in (dict(kind='barrier',barrier=1.),policies()[8]):
        predicted=rollout(floor_kernel,p,*dr,.5*K0,'floor')
        for j in range(2):
            size=216
            market=SimpleNamespace(n=size,spot=np.zeros(size),spr_lags=np.zeros((size,5)),
                time=np.zeros((size,4)),spread=np.zeros(size),cap_mwh=np.zeros(size))
            market.spread[200:216]=kernel.spread[dr[0][j]]*dr[1][j]
            market.cap_mwh[200:216]=kernel.cap[dr[0][j]]
            env=LiabilityEnv(market,'ZF',train=False);env.reset(options={'e0':.5})
            total=0.
            for t in range(16):
                lev=action(p,np.array([env.reported]),t,16,.5*K0,floor_kernel,'floor')[0]
                _,r,_,_,_=env.step(int(np.flatnonzero(LEV==lev)[0]))
                total+=GAMMA**t*r*K0/100
            floor_error=max(floor_error,abs(total-predicted['raw'][j]));floor_checks+=1
    assert floor_error<1e-8
    return dict(direct_query_checks=len(errors),max_direct_query_error=max(errors),
                immutable_terminal_rollout_error=error,immutable_floor_fee=.1,
                immutable_floor_path_checks=floor_checks,immutable_floor_rollout_error=floor_error)
