"""Sample-only financial-event certificates; no conditional-moment integration.

Numerical functions only. Importing this module never simulates or fits a model.
The audited floor account is an event-level financial component of the native
engine; other cashflows may change the next event's starting cash.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import numpy as np

METHODS=('raw','capped_raw','capped_residual','paired','ridge_control','mlp_control','cash_capped_raw','cash_ridge_control','cash_mlp_control','cash_local_control')
ALPHAS=(0.,1.)
GAMMA=.995


def require(condition,message):
    if not bool(condition):raise AssertionError(message)


def validate_method(method,alpha):
    require(method in METHODS,'Declared audit method')
    require(np.isfinite(alpha) and 0<=alpha<=1,'Charge fraction in [0,1]')
    require(not method.startswith('cash_') or alpha==0.,'Cash-bound methods are alpha0 only')


def numerical_pad(bounds,choice=None,horizon=96):
    """Predeclared engineering allowance, not a formal floating-point proof."""
    budget=float((choice or {}).get('budget',0.))
    scale=1+float(bounds['discounted_gross_bound'])+float(bounds['discounted_cost_bound'])+budget
    return max(1e-6,1e-10*scale*int(horizon))


def summarize(trace,gamma=GAMMA):
    """Local pair evaluated on ACTUAL pre-event states, without a row-law oracle."""
    before=np.asarray(trace['event_cash_before'],float)
    gross=np.asarray(trace['event_gross'],float)
    costs=np.asarray(trace['event_cost'],float)
    injections=np.asarray(trace['event_injection'],float)
    base=np.asarray(trace['event_unsigned_base_gross'],float)
    coins=np.asarray(trace['event_sign_epsilon'],int)
    steps=np.asarray(trace['event_step'],int)
    require(np.all(before>=0),'V2 requires exactly nonnegative pre-event cash')
    require(np.all(costs>=0),'Financial cost must be nonnegative')
    require(np.all(np.isfinite(np.r_[before,gross,costs,injections,base])),'Finite native trace')
    require(np.all(np.isin(coins,[-1,0,1])),'Known hidden fair sign or deterministic event')
    require(np.allclose(gross,np.where(coins!=0,coins*base,0.),atol=1e-6,rtol=1e-10),'Local sign contract')
    actual=np.maximum(-before-gross+costs,0.)
    require(np.allclose(actual,injections,atol=1e-6,rtol=1e-10),'Separate financial-event injection')
    plus=np.maximum(-before-base+costs,0.)
    minus=np.maximum(-before+base+costs,0.)
    paired=(plus+minus)/2
    deterministic=coins==0
    require(np.allclose(paired[deterministic],injections[deterministic],atol=1e-6),'Deterministic fee events paired exactly')
    weights=float(gamma)**steps
    old=(coins!=0)&(steps<6)
    # Retain every cost and coin; remove only direct inherited maturity G/I/A.
    target_paired=np.where(old,0.,paired)
    target_gross=np.where(old,0.,gross)
    target_injections=np.where(old,0.,injections)
    pair_contribution=weights*(target_paired-costs)
    future=pair_contribution.sum()-np.cumsum(pair_contribution)
    raw_contribution=weights*(target_gross-costs+target_injections)
    raw_future=raw_contribution.sum()-np.cumsum(raw_contribution)
    known_current=np.where(old,0.,weights*(base+(plus-minus)/2))
    return dict(gross=float(weights@target_gross),cost=float(weights@costs),forgiveness=float(weights@target_injections),
                paired_forgiveness=float(weights@target_paired),
                excluded_old_gross=float(weights@(gross-target_gross)),
                excluded_old_forgiveness=float(weights@(injections-target_injections)),
                excluded_old_paired_forgiveness=float(weights@(paired-target_paired)),
                excluded_old_maturity_events=int(np.count_nonzero(old)),events=len(steps),maturity_events=int(np.count_nonzero(coins)),
                identity_error=float(np.max(np.abs(actual-injections),initial=0.)),
                pair_event=paired,weights=weights,future_target=future,raw_future_target=raw_future,
                known_current_raw_delta=known_current,coin=coins,
                active_indices=np.flatnonzero(coins!=0))


def branch_features(trace,summary,*,money_scale,horizon,policy_index,profile_index,branch):
    """The SAME predictor receives two synthetic sign branches of common data.

    These are regression features, not claimed exact next native observations.
    Structural centering works even if this is a poor state/transition model.
    """
    ids=summary['active_indices']
    if len(ids)==0:return np.empty((0,32),float)
    obs=np.asarray(trace['event_actor_observation'],float)[ids]
    require(obs.ndim==2 and obs.shape[1] in (12,20),'Authentic learned actor observation dimension')
    padded=np.zeros((len(ids),20));padded[:,:obs.shape[1]]=obs
    before=np.asarray(trace['event_cash_before'],float)[ids]/money_scale
    base=np.asarray(trace['event_unsigned_base_gross'],float)[ids]/money_scale
    costs=np.asarray(trace['event_cost'],float)[ids]/money_scale
    branch=np.broadcast_to(np.asarray(branch,float),(len(ids),))
    after=np.maximum(before+branch*base-costs,0.)
    injection=np.maximum(-before-branch*base+costs,0.)
    action=np.asarray(trace['event_actor_action'],float)[ids]
    step=np.asarray(trace['event_step'],float)[ids]/max(horizon-1,1)
    context=np.column_stack([before,base,costs,branch,after,injection,action,step,
                             np.full(len(ids),policy_index/19.),np.full(len(ids),float(profile_index)),
                             np.full(len(ids),obs.shape[1]/20.),summary['pair_event'][ids]/money_scale])
    result=np.column_stack([padded,context])
    require(result.shape[1]==32 and np.all(np.isfinite(result)),'Fixed finite nuisance feature contract')
    return result


def fit_rows(trace,summary,**feature_options):
    ids=summary['active_indices']
    return (branch_features(trace,summary,branch=summary['coin'][ids],**feature_options),
            summary['future_target'][ids])


def predictor_deltas(trace,summary,models,**feature_options):
    plus=branch_features(trace,summary,branch=1,**feature_options)
    minus=branch_features(trace,summary,branch=-1,**feature_options)
    if not len(plus):return {name:np.empty(0) for name in models}
    output={name:(np.asarray(model.predict(plus))-np.asarray(model.predict(minus)))/2
            for name,model in models.items()}
    for name in output:
        if name.startswith('cash_'):output[name]+=summary['known_current_raw_delta'][summary['active_indices']]
    require(all(np.all(np.isfinite(x)) for x in output.values()),'Frozen predictor must return finite branches')
    return output


def budgeted_control(deltas,coins,beta,budget):
    """Exactly centered predictable spending; budget never depends on this coin."""
    require(np.isfinite(beta) and np.isfinite(budget) and beta>=0 and budget>=0,'Finite nonnegative preselected control settings')
    delta=np.asarray(deltas,float);coins=np.asarray(coins,float)
    require(delta.shape==coins.shape and np.all(np.isfinite(delta)) and np.all(np.isin(coins,[-1,1])),'Finite matching maturity innovation vector')
    proposed=np.abs(beta*delta)
    previous=np.r_[0.,np.cumsum(proposed[:-1])] if len(proposed) else np.empty(0)
    amplitude=np.minimum(proposed,np.maximum(budget-previous,0.))
    control=float(np.sum(coins*np.sign(delta)*amplitude))
    # Source-only large-budget stress test needs a scale-aware floating guard.
    # This changes no statistic; the existing G/C/B numerical CI pad is retained.
    guard=max(1e-7,128*np.finfo(float).eps*(1+budget))
    require(np.sum(amplitude)<=budget+guard and abs(control)<=budget+guard,'Pathwise absolute-amplitude budget with disclosed engineering allowance')
    return control


def empirical_bernstein(values,lower,upper,delta,numerical_allowance=1e-6):
    x=np.asarray(values,float)
    require(len(x)>=2 and np.all(np.isfinite(x)),'Independent whole-episode sample')
    require(0<delta<1 and upper>=lower,'Known deterministic support and error budget')
    pad=float(numerical_allowance)
    require(np.isfinite(pad) and pad>=0,'Finite nonnegative numerical allowance')
    require(np.min(x)>=lower-pad and np.max(x)<=upper+pad,'Samples within declared support')
    lo,hi=float(lower-pad),float(upper+pad)
    mean=float(x.mean());variance=float(x.var(ddof=1));log=float(np.log(2/delta))
    radius=float(np.sqrt(2*variance*log/len(x))+7*(hi-lo)*log/(3*(len(x)-1)))
    return dict(n=len(x),mean=mean,variance=variance,radius=radius,lower=mean-radius-pad,
                support_lower=lo,support_upper=hi,delta=float(delta),numerical_allowance=pad)


def sample_values(rows,method,alpha,choice):
    """Rows are per-episode summaries plus the two frozen-predictor deltas."""
    validate_method(method,alpha)
    output=[]
    for row in rows:
        if method in ('raw','capped_raw','cash_capped_raw','cash_ridge_control','cash_mlp_control','cash_local_control'):
            value=row['gross']-row['cost']+(1-alpha)*row['forgiveness']
            if method.endswith('_control'):
                model=method.removesuffix('_control')
                value-=budgeted_control(row['delta_'+model],row['maturity_coins'],choice['beta'],choice['budget'])
        elif method=='capped_residual':value=(1-alpha)*row['forgiveness']-row['cost']
        else:
            value=(1-alpha)*row['paired_forgiveness']-row['cost']
            if method.endswith('_control'):
                model=method.removesuffix('_control')
                value-=budgeted_control(row['delta_'+model],row['maturity_coins'],choice['beta'],choice['budget'])
        if method!='raw':value=min(value,choice['cap'])
        output.append(value)
    return np.asarray(output,float)


def deterministic_support(bounds,method,alpha,choice):
    validate_method(method,alpha)
    costs=float(bounds['discounted_cost_bound']);gross=float(bounds['discounted_gross_bound'])
    require(np.isfinite(costs+gross) and costs>=0 and gross>=0,'Finite deterministic path supports')
    # A financial component cannot use the complete-account -initial_cash bound:
    # excluded physical transfers may replenish cash before later financial loss.
    rawlo=-(gross+costs)
    rawhi=gross
    if method.startswith('cash_'):
        K=float(bounds['cash_inflow_bound'])
        require(np.isfinite(K) and K>=0,'Source-derived excluded-inflow bound')
        extra=float(choice['budget']) if method.endswith('_control') else 0.
        return -K-extra,min(float(choice['cap']),rawhi+extra)
    if method=='raw':return rawlo,rawhi
    if method=='capped_raw':return rawlo,min(float(choice['cap']),rawhi)
    extra=float(choice.get('budget',0.)) if method.endswith('_control') else 0.
    known_upper=((1-alpha)*gross if method=='capped_residual' else (1-alpha)*gross/2)+extra
    return -costs-extra,min(float(choice['cap']),known_upper)


def select_choice(rows,method,alpha,bounds,delta,n_confirmation,caps,betas,budget_ratios):
    if method=='raw':return dict(cap=None,beta=0.,budget=0.),[]
    candidates=[]
    for cap in caps:
        settings=[(0.,0.)]
        if method.endswith('_control'):
            settings += [(float(beta),float(cap*ratio)) for beta in betas if beta>0
                         for ratio in budget_ratios if ratio>0]
        for beta,budget in settings:
            choice=dict(cap=float(cap),beta=beta,budget=budget)
            lo,hi=deterministic_support(bounds,method,alpha,choice)
            values=sample_values(rows,method,alpha,choice)
            # Search score is a projected fixed-budget heuristic, not a certificate.
            mean=float(values.mean());variance=float(values.var(ddof=1));log=np.log(2/delta)
            score=float(mean-np.sqrt(2*variance*log/n_confirmation)-7*(hi-lo)*log/(3*(n_confirmation-1)))
            candidates.append(dict(**choice,mean=mean,variance=variance,projected_score=score))
    best=max(candidates,key=lambda r:(r['projected_score'],-r['budget'],-r['beta'],-r['cap']))
    return {key:best[key] for key in ('cap','beta','budget')},candidates


def valid_pairs():
    return tuple((alpha,method) for alpha in ALPHAS for method in METHODS if not method.startswith('cash_') or alpha==0.)
