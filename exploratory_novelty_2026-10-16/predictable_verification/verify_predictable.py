"""Independent read-only arithmetic/model audit; no simulation, DP, or training.

The recorded path moments can be checked, but paths were not shipped, so this
does not claim to independently reconstruct Monte Carlo means or variances.
"""
from __future__ import annotations
import ast
from collections import Counter
from fractions import Fraction
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import runpy
import sys
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCE=ROOT/'exploratory_novelty_2026-10-16/predictable_accounting_audit'
METHODS=('raw','clipped_raw','clipped_residual','clipped_mean_cost_residual','predictable')
CASES=tuple(itertools.product(('DK1','DK2'),('floor','terminal'),(.1,.5,1.),(16,96)))
POLICIES=([dict(kind='one_action')]+[dict(kind='constant',lev=l) for l in (.25,1,4,16)]
    +[dict(kind='barrier',barrier=b) for b in (.5,1.,2.)]
    +[dict(kind='two_stage',early_barrier=b,late_barrier=c,switch=.5,early_lev=16,late_lev=16)
      for b,c in ((1.,2.),(1.,.5),(2.,.5),(2.,2.))]+[dict(kind='myopic')])


class Audit:
    def __init__(self):
        self.assertions=0;self.hash_checks=0;self.max_errors={}

    def require(self,condition,label):
        self.assertions+=1
        if not condition:raise AssertionError(label)

    def near(self,actual,expected,label,tol=1e-8,category='arithmetic'):
        error=abs(float(actual)-float(expected))
        self.max_errors[category]=max(error,self.max_errors.get(category,0.))
        self.require(math.isfinite(error) and error<=tol,f'{label}: {actual!r} != {expected!r}')

    def hash(self,path,expected):
        self.hash_checks+=1
        self.require(hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,f'hash {path}')


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def check_eb(a,b,n,delta,mean='mean',variance='variance',lo='range_lower',hi='range_upper',correction=0.):
    a.require(b['n' if 'n' in b else 'n_in_bound']==n,'fixed bound n')
    a.near(b['delta'],delta,'delta',0.)
    radius=math.sqrt(2*b[variance]*math.log(2/delta)/n)+7*(b[hi]-b[lo])*math.log(2/delta)/(3*(n-1))
    a.near(b['radius'],radius,'EB radius',1e-9,'EB_radius')
    a.near(b['lower'],b[mean]-radius-correction,'EB lower',1e-9,'EB_lower')
    a.require(b[variance]>=0 and b[lo]<=b[mean]<=b[hi],'moments within deterministic support')
    if 'observed_min' in b:
        a.require(b[lo]-1e-9<=b['observed_min']<=b['observed_max']<=b[hi]+1e-9,'observed support')
    if 'uncorrected_lower' in b:a.near(b['uncorrected_lower'],b[mean]-radius,'uncorrected lower',1e-9)
    if 'upper' in b:a.near(b['upper'],b[mean]+radius,'unallocated upper arithmetic',1e-9)


def normalize_retry(source,unbounded=False):
    tree=ast.parse(source)
    class Normalize(ast.NodeTransformer):
        def visit_FunctionDef(self,node):
            if not unbounded and node.name=='encode':return None
            return self.generic_visit(node)
        def visit_Constant(self,node):
            if not unbounded and node.value=='heldout_outputs_attempt2':return ast.copy_location(ast.Constant('heldout_outputs'),node)
            return node
        def visit_Call(self,node):
            node=self.generic_visit(node)
            if not unbounded and isinstance(node.func,ast.Name) and node.func.id=='encode':
                node.func=ast.Attribute(ast.Name('json',ast.Load()),'dumps',ast.Load())
            return node
        def visit_Assign(self,node):
            node=self.generic_visit(node)
            if unbounded and any(isinstance(t,ast.Attribute) and t.attr=='h0' for t in node.targets):
                if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='float':node.value=node.value.args[0]
            return node
    return ast.dump(Normalize().visit(tree),include_attributes=False)


def panel_statistics(zone,split='test'):
    data=pd.read_pickle(ROOT/f'GENERALIZATION_STUDY_2026-09-29/data_panel/panel_{zone}.pkl')
    data=data.loc[data.split==split].iloc[200:]
    price=np.abs(data.imb.to_numpy(float)-data.spot.to_numpy(float));cap=.25*.25*data.bal_mw.to_numpy(float)
    stats={}
    for lev in (.25,1,4,16):
        q=np.minimum(.25*lev,cap);cost=q;magnitude=q*price
        stats[lev]=dict(c=float(cost.mean()),h0=float(.5*np.mean(np.maximum(cost-magnitude,0)+np.maximum(cost+magnitude,0))),
                       min_x=float(np.min(-magnitude-cost)),max_x=float(np.max(magnitude-cost)),cost=cost,mag=magnitude)
    size=2*len(data);eps=np.finfo(float).eps;gamma=size*eps/(1-size*eps)
    maximum=max(max(abs(s['min_x']),abs(s['max_x'])) for s in stats.values())
    error=64*(gamma+eps)*(1+maximum)
    return dict(n=len(data),stats=stats,error=error)


def curve(s,cash):return float(.5*np.mean(np.maximum(s['cost']-s['mag']-cash,0)+np.maximum(s['cost']+s['mag']-cash,0)))


def independent_ranges(kernel,p,initial,h,account):
    stats=kernel['stats'];w=.99**np.arange(h);d=w[-1];sw=float(w.sum())
    if p['kind']=='constant':levels=(p['lev'],)
    elif p['kind'] in ('barrier','one_action'):levels=(0,16)
    elif p['kind']=='two_stage':levels=tuple(set((0,16,p['early_lev'],p['late_lev'])))
    else:levels=(0,.25,1,4,16)
    active=[l for l in levels if l];costs=np.array([stats[l]['c'] if l else 0. for l in levels]);h0=np.array([stats[l]['h0'] if l else 0. for l in levels])
    comp_lo=-sw*float(max(costs));weights=w if account=='floor' else np.full(h,d)
    comp_hi=float(np.maximum(0,(weights[:,None]*h0-w[:,None]*costs).max(axis=1)).sum())
    maxloss=max(-stats[l]['min_x'] for l in active);maxgain=max(0,max(stats[l]['max_x'] for l in active))
    costmax=max(active)*.25*sw;meancostmax=sw*float(max(costs))
    if account=='floor':rawlo=max(-initial,-maxloss*sw)
    else:rawlo=max(-maxloss*sw,-d*initial-(1-.99)*maxloss*sum(t*.99**(t-1) for t in range(1,h)))
    rawhi=maxgain*sw
    if p['kind']=='one_action':
        discount=1. if account=='floor' else d;s=stats[16]
        comp_lo=comp_hi=discount*(curve(s,initial)-s['c'])
        costmax=discount*4;meancostmax=discount*s['c'];booked=-meancostmax
        rawlo=discount*(max(0,initial+s['min_x'])-initial);rawhi=discount*(max(0,initial+s['max_x'])-initial)
    elif p['kind']=='constant':booked=-sw*stats[p['lev']]['c']
    else:booked=-stats[16]['c']
    return dict(comp_lo=comp_lo,comp_hi=comp_hi,costmax=costmax,meancostmax=meancostmax,
                rawlo=rawlo,rawhi=rawhi,booked=booked,pad=2*h*kernel['error'])


def paired_results(records):
    idx={(r['case_id'],r['repetition'],r['method']):r for r in records}
    pairs={}
    for m in METHODS[:-1]:
        counts=Counter((idx[c,j,'predictable']['certified'],idx[c,j,m]['certified']) for c in range(24) for j in range(3))
        pairs[m]=dict(both_positive=counts[True,True],predictable_only=counts[True,False],comparator_only=counts[False,True],neither=counts[False,False])
    return pairs


def audit_heldout(a):
    folder=SOURCE/'heldout_outputs_attempt2';s=load(folder/'summary.json');f=load(folder/'frozen_inputs.json');first=load(SOURCE/'heldout_outputs/frozen_inputs.json')
    a.require(s['status']=='EXPLORATORY_complete' and s['completed_case_repetitions']==72,'complete heldout family')
    a.require(s['input_sha256']==f['input_sha256'],'result/frozen manifest agree')
    for name,digest in s['input_sha256'].items():a.hash(ROOT/name,digest)
    for name,digest in first['input_sha256'].items():a.hash(ROOT/name,digest)
    old=(SOURCE/'run_heldout_benchmark.py').read_text();new=(SOURCE/'run_heldout_attempt2.py').read_text()
    a.require(normalize_retry(old)==normalize_retry(new),'retry only output path and scalar JSON encoding')
    shared1={k:v for k,v in first['input_sha256'].items() if not k.endswith('run_heldout_benchmark.py')}
    shared2={k:v for k,v in f['input_sha256'].items() if not k.endswith('run_heldout_attempt2.py')}
    a.require(shared1==shared2,'same numerical core, plan, panels and dependencies across retry')
    a.require(set(x.name for x in (SOURCE/'heldout_outputs').iterdir())=={'frozen_inputs.json'},'failed attempt retained only pre-evaluation manifest')
    delta=.05/720;a.near(s['delta_per_bound'],delta,'heldout delta',0.);a.require(s['planned_lower_bounds']==720,'720 family bounds')
    trace=pd.read_csv(folder/'search_trace.csv');a.require(len(trace)==72*13*5,'complete search trace')
    kernels={z:panel_statistics(z) for z in ('DK1','DK2')}
    for zone,k in kernels.items():
        pp=s['preprocessing'][zone];a.require(pp['rows']==k['n']==20934,'all declared TEST rows')
        a.near(pp['oracle_roundoff_allowance'],k['error'],'oracle error bound',1e-14)
        ratios={str(l):st['h0']/st['c'] for l,st in k['stats'].items()}
        for lev,ratio in ratios.items():a.near(pp['repair']['ratio_by_action'][lev],ratio,'repair ratio',1e-9)
        a.near(pp['repair']['alpha_all_state_safe'],1-1/max(ratios.values()),'analytic repair threshold',1e-12)
    expected_unique=0;positive_cases=set();row_count=0;range_max=0.;cohorts={}
    selected={(r['case_id'],r['repetition'],r['method']):r for r in s['selected_rows']}
    neutral={(r['case_id'],r['repetition'],r['method']):r for r in s['neutral_rows']}
    a.require(len(selected)==len(neutral)==360,'complete unique row grids')
    for case_id,(zone,account,equity,h) in enumerate(CASES):
        k=kernels[zone];cash=equity*20000;discount=1. if account=='floor' else .99**(h-1)
        exact={0.:0.,**{l:discount*(curve(st,cash)-st['c']) for l,st in k['stats'].items()}}
        best=max(exact,key=exact.get);reference=exact[best]
        if reference>1e-10:positive_cases.add(case_id)
        for rep in range(3):
            local=trace.loc[(trace.case_id==case_id)&(trace.repetition==rep)]
            a.require(len(local)==65,'five method scores for each candidate')
            chosen={}
            for method in METHODS:
                sub=local.loc[local.method==method].sort_values('candidate');a.require(sub.candidate.tolist()==list(range(13)),'all 13 fixed candidates')
                winner=sub.loc[sub.projected_score.idxmax()];chosen[method]=int(winner.candidate)
            expected_unique+=262080+(len(set(chosen.values())|{(4,6,8)[rep]}))*(2097152//h*h)
            detail=load(folder/f'case_{case_id:02d}_rep_{rep}.json')
            for label,index in [('selected',selected),('neutral',neutral)]:
                a.require(detail[label]==[index[case_id,rep,m] for m in METHODS],'per-repetition output agrees with summary')
                for method in METHODS:
                    r=index[case_id,rep,method];row_count+=1
                    a.require((r['zone'],r['account'],r['equity_multiple'],r['horizon'])==(zone,account,equity,h),'fixed heldout case definition')
                    candidate=chosen[method] if label=='selected' else (4,6,8)[rep]
                    a.require(r['candidate']==candidate and r['policy']==POLICIES[candidate],'selection or fixed neutral controller')
                    nsearch=262144//(13*h);nconfirm=2097152//h
                    a.require(r['nsearch_per_candidate']==nsearch and r['nconfirm']==nconfirm,'fixed trajectory budgets')
                    a.require(r['logical_search_transitions']==nsearch*13*h==262080,'logical search budget')
                    a.require(r['logical_confirmation_transitions']==nconfirm*h<=2097152,'logical confirmation budget')
                    a.near(r['exact_one_action'],reference,'direct isolated reference',1e-9,'exact_reference')
                    a.require(r['exact_one_action_leverage']==best and r['exact_one_action_positive']==(reference>1e-10),'exact reference decision')
                    b=independent_ranges(k,r['policy'],cash,h,account);v=r['bound'];cap=r['cap'];pad=b['pad']
                    if method=='raw':lo,hi,correction=b['rawlo'],b['rawhi'],1e-7
                    elif method=='clipped_raw':lo,hi,correction=b['rawlo'],min(cap,b['rawhi']),1e-7
                    elif method=='clipped_residual':lo,hi,correction=-b['costmax'],cap,1e-7
                    elif method=='clipped_mean_cost_residual':lo,hi,correction=-b['meancostmax']-pad,cap+pad,pad
                    else:lo,hi,correction=b['comp_lo']-pad,b['comp_hi']+pad,pad
                    a.near(v['range_lower'],lo-1e-7,'independent lower support',1e-7,'range');a.near(v['range_upper'],hi+1e-7,'independent upper support',1e-7,'range')
                    a.near(v['numerical_allowance'],correction,'numerical bias correction',1e-14)
                    check_eb(a,v,nconfirm,delta,correction=correction)
                    a.near(r['booked_upper'],b['booked'],'booked harm',1e-10)
                    a.require(r['certified']==(v['lower']>0 and b['booked']<0),'certificate decision')
                    a.require(not r['certified'] or reference>1e-10,'no new one-action-missed state')
                    row_score=local.loc[(local.method==method)&(local.candidate==candidate)].iloc[0]
                    a.require((cap is None and pd.isna(row_score.cap)) or cap==row_score.cap,'SEARCH-only selected cap')
                    if label=='selected':
                        a.require(not r['full_charge_same_actions_positive'],'same-action fully charged counterfactual')
                        if method=='raw':a.near(v['mean'],r['raw_mean_descriptive'],'raw recorded mean identity',1e-9)
                        if method=='predictable':a.near(v['mean'],r['predictable_mean_descriptive'],'predictable recorded mean identity',1e-9)
                    elif candidate==selected[case_id,rep,method]['candidate']:
                        a.require(v==selected[case_id,rep,method]['bound'],'same controller/path/cap gives identical paired moments')
    a.near(s['actual_unique_transitions'],expected_unique,'shared unique simulation calls',0.)
    a.require(s['conditional_oracle_points']>=s['rollout_conditional_oracle_points']+s['preprocessing_oracle_points'],'oracle totals include outside-rollout queries')
    a.require(s['preprocessing_oracle_points']==8,'four initial moment queries per kernel')
    expected_selected=[7,17,24,24,36];expected_neutral=[5,18,24,24,33]
    for label,records,expected in [('selected',s['selected_rows'],expected_selected),('neutral',s['neutral_rows'],expected_neutral)]:
        totals={m:sum(r['certified'] for r in records if r['method']==m) for m in METHODS}
        a.require(list(totals.values())==expected,f'{label} method counts')
        a.require(s[label+'_summary']==[dict(method=m,pairs=72,certified=totals[m],new_beyond_one_action=0) for m in METHODS],f'{label} stored summary')
        cohorts[label]=dict(counts=totals,paired_vs_predictable=paired_results(records))
    stability={m:dict(Counter(sum(selected[c,j,m]['certified'] for j in range(3)) for c in range(24))) for m in METHODS}
    neutral_by_controller={str(p):{m:sum(r['certified'] for r in s['neutral_rows'] if r['method']==m and r['candidate']==p) for m in METHODS} for p in (4,6,8)}
    return dict(complete_bounds=row_count,distinct_cases=24,positive_exact_one_action_cases=len(positive_cases),zero_reference_cases=24-len(positive_cases),
                cohorts=cohorts,selected_repetition_success_distribution=stability,neutral_counts_by_fixed_controller=neutral_by_controller,
                actual_unique_transitions=expected_unique,conditional_oracle_points=s['conditional_oracle_points'],outside_rollout_oracle_points=s['conditional_oracle_points']-s['rollout_conditional_oracle_points'],
                oracle_repair_thresholds={z:s['preprocessing'][z]['repair']['alpha_all_state_safe'] for z in kernels},
                numerical_kernel_and_plan_identical_across_retry=True,
                no_independent_sample_moment_reconstruction=True)


def pareto_call(p,k):return p/(p-1)-k if k<=1 else k**(1-p)/(p-1)


def audit_unbounded(a):
    u=load(SOURCE/'unbounded_outputs/results.json');a.hash(SOURCE/'run_unbounded_validation.py',u['script_sha256']);a.hash(SOURCE/'UNBOUNDED_PLAN.md',u['plan_sha256'])
    a.hash(SOURCE/'run_unbounded_attempt1_snapshot.py','f3c8c75ae75a6617b1ef2b45ebc8d2c870f66970eff566a4c5696e8467d91ab7')
    a.require(normalize_retry((SOURCE/'run_unbounded_attempt1_snapshot.py').read_text(),True)==normalize_retry((SOURCE/'run_unbounded_validation.py').read_text(),True),'Pareto retry only float scalar conversion')
    a.require(u['status']=='EXPLORATORY_COMPLETE' and u['full_fixed_budgets_complete'] and u['completed_cases']==16,'complete Pareto family')
    delta=.05/32;a.near(u['delta_per_bound'],delta,'Pareto family delta',0.)
    a.require(u['allocated_one_sided_events']==32,'32 allocated Pareto lower bounds')
    w=.99**np.arange(32);d=w[-1];sw=float(sum(w));positive=[];variance_worse=[];negative_controls=0;counts=Counter()
    grid=tuple(itertools.product((1.5,2.5),('fixed','proportional'),('floor','terminal'),(.8,.95)))
    for cid,(p,law,account,alpha) in enumerate(grid):
        r=u['cases'][cid];a.require((r['case_id'],r['pareto_shape'],r['cost_law'],r['accounting'],r['alpha'])==(cid,p,law,account,alpha),'all16 fixed Pareto cases')
        m=p/(p-1);fixed=.1 if law=='fixed' else 0.;rate=.05 if law=='proportional' else 0.;c=fixed+rate*m;h0=(m+c)/2 if law=='fixed' else (1+rate)*m/2;star=1-c/h0
        h_initial=.5*(1+rate)**p*(2-fixed)**(1-p)/(p-1)
        first=(1-alpha)*(1. if account=='floor' else d)*h_initial-c
        lo=first-c*float(sum(w[1:]));hi=first+(float(sum(w[1:]))*max(0,(1-alpha)*h0-c) if account=='floor' else float(sum(max(0,(1-alpha)*d*h0-wt*c) for wt in w[1:])))
        for key,value in [('mean_cost',c),('h0',h0),('uniform_repair_alpha',star),('fixed_first_compensator_term',first),('expected_booked_upper',-c)]:a.near(r[key],value,'Pareto '+key,1e-12,'Pareto_oracle')
        a.require(r['n_search']==4096 and r['completed_confirmation_paths']==r['planned_confirmation_paths']==65536 and r['confirmation_batches']==16,'Pareto whole-path budgets')
        a.require((r['search_seed'],r['confirmation_seed'])==(202610161100+cid,202610161200+cid),'Pareto independent fixed streams')
        a.require(r['gross_second_moment_finite']==(p>2) and r['actual_cost_second_moment_finite']==(law=='fixed' or p>2),'variance classifications')
        a.require(r['actual_cost_support_unbounded']==(law=='proportional') and r['unit_trade_gross_support_unbounded'],'unbounded support declared')
        a.require(r['analytically_all_policy_nonpositive']==(alpha>=star),'analytic repair control')
        a.require(len(r['all_cap_search_scores'])==5 and [v['cap'] for v in r['all_cap_search_scores']]==[1,4,16,64,256],'all Pareto cap variants retained')
        for entry in r['all_cap_search_scores']:
            b=entry['score'];a.require(b['n_observed']==4096 and b['projected_search_score_only'],'cap uses search moments only')
            check_eb(a,b,65536,delta,variance='sample_variance',lo='support_lower',hi='support_upper')
            a.near(b['support_lower'],-c*sw,'mean-cost residual lower support',1e-12);a.near(b['support_upper'],entry['cap'],'residual upper cap',0.)
        cap=max(r['all_cap_search_scores'],key=lambda z:z['score']['lower'])['cap'];a.require(r['selected_residual_cap']==cap,'cap maximizes independent search heuristic')
        comp=r['uncapped_compensator_bound'];base=r['capped_mean_cost_residual_bound']
        for b in (comp,base):
            a.require(b['n_observed']==65536 and not b['projected_search_score_only'],'fresh fixed confirmation')
            check_eb(a,b,65536,delta,variance='sample_variance',lo='support_lower',hi='support_upper')
        a.near(comp['support_lower'],lo,'first-term predictable lower range',1e-12,'Pareto_range');a.near(comp['support_upper'],hi,'first-term predictable upper range',1e-12,'Pareto_range')
        a.near(base['support_lower'],-c*sw,'baseline known expected-cost lower bound',1e-12);a.near(base['support_upper'],cap,'baseline search-frozen cap',0.)
        means=r['descriptive_confirmation_means']
        a.near(means['compensator'],(1-alpha)*means['predictable_forgiveness']-means['predictable_cost'],'predictable mean identity',1e-12)
        a.near(means['mean_cost_residual'],(1-alpha)*means['actual_forgiveness']-means['predictable_cost'],'mean-cost residual mean identity',1e-12)
        a.near(means['raw'],means['gross']+(1-alpha)*means['actual_forgiveness']-means['actual_cost'],'raw accounting identity',1e-11)
        a.near(means['booked'],means['gross']-means['actual_cost'],'booked identity',1e-11)
        a.near(comp['mean'],means['compensator'],'recorded compensator mean',1e-12)
        a.require(base['mean']<=means['mean_cost_residual']+1e-12,'upper clipping is a lower estimator')
        a.require(r['uncapped_compensator_positive']==(comp['lower']>0) and r['capped_mean_cost_residual_positive']==(base['lower']>0),'Pareto positive decisions')
        if comp['lower']>0:positive.append(cid)
        if comp['sample_variance']>base['sample_variance']:variance_worse.append(cid)
        a.near(r['paired_sample_variance_ratio_comp_over_capped'],comp['sample_variance']/base['sample_variance'],'recorded variance ratio',1e-12)
        if alpha==.95:
            negative_controls+=1;a.require(hi<=1e-12 and comp['observed_max']<=1e-12 and not r['uncapped_compensator_positive'] and not r['capped_mean_cost_residual_positive'],'analytic .95 controls')
        for phase,n in [('search_counts',4096),('confirmation_counts',65536)]:
            ct=r[phase];a.require(ct['logical_transitions']==ct['global_magnitude_draws']==n*32,'Pareto logical calls')
            a.require(n<=ct['active_transitions']<=n*32,'mandatory trade and active calls')
            a.require(ct['conditional_curve_scalar_evaluations']==ct['expected_cost_scalar_lookups']==ct['active_transitions'],'oracle query counts')
            for key,value in ct.items():counts[key]+=value
    for key,value in counts.items():a.near(u['total_computation_counts'][key],value,'Pareto total '+key,1e-9)
    a.require(positive==[0,4,12],'three fixed Pareto positive cases')
    a.require([r['case_id'] for r in u['cases'] if r['capped_mean_cost_residual_positive']]==positive,'same three sign discoveries')
    a.require(len(variance_worse)==4,'four counterexamples to universal variance improvement')
    for record in u['deterministic_oracle_checks']:
        p=record['shape'];law=record['cost_law'];m=Fraction(str(p))/(Fraction(str(p))-1);c=Fraction(1,10) if law=='fixed' else m/20;h0=(m+c)/2 if law=='fixed' else m*Fraction(21,40);star=1-c/h0
        for key,value in [('mean_magnitude_exact',m),('mean_cost_exact',c),('h0_exact',h0),('uniform_repair_alpha_exact',star)]:a.require(Fraction(record[key])==value,'exact Fraction '+key)
        rate=.05 if law=='proportional' else 0.;fixed=.1 if law=='fixed' else 0.;h2=.5*(1+rate)**p*(2-fixed)**(1-p)/(p-1)
        a.near(record['one_step_h_initial'],h2,'independent Pareto stop-loss at initial cash',1e-13,'Pareto_oracle')
        a.near(record['one_step_expected_charged_alpha0p8'],.2*h2-float(c),'independent one-step charged mean',1e-13)
        extreme=[]
        for n in (-1e300,-1e100,0,1e100,1e300):
            extreme.append(.5*(1+rate)*pareto_call(p,(n-fixed)/(1+rate)) if n>=0 else float(c)+.5*(1-rate)*pareto_call(p,(fixed-n)/(1-rate)))
        for x,y in zip(record['extreme_ledger_values'],extreme):a.near(x,y,'independent stable extreme drift',1e-13)
    return dict(complete_bounds=32,cases=16,positive_case_ids=positive,methods_same_discoveries=True,negative_controls=negative_controls,
                compensator_variance_worse_case_ids=variance_worse,logical_transitions=int(counts['logical_transitions']),active_transitions=int(counts['active_transitions']),
                exact_analytic_oracles_checked=4,identical_stream_scalar_only_replay=True,no_bounded_raw_assumption=True)


def audit_fixed_policy(a):
    folder=SOURCE/'fixed_policy_outputs';s=load(folder/'summary.json');f=load(folder/'frozen_inputs.json')
    a.require(s['status']=='EXPLORATORY_complete' and len(s['records'])==45,'complete fixed-policy45 family')
    a.require(s['input_sha256']==f['input_sha256'],'fixed-policy manifest/result agreement')
    for name,digest in s['input_sha256'].items():a.hash(ROOT/name,digest)
    a.require(s['nsearch']==f['nsearch']==8192 and s['nconfirm']==f['nconfirm']==32768,'fixed-policy planned budgets')
    delta=.05/45;a.near(s['delta_per_bound'],delta,'fixed-policy confirmation delta',0.)
    a.require(s['planned_bounds']==f['planned_bounds']==45,'45 fixed-policy bounds')
    k=panel_statistics('DK1','train');a.require(k['n']==s['preprocessing_rows']==28059,'same TRAIN model as selected development case')
    old=load(ROOT/'exploratory_novelty_2026-10-16/matched_multistep_baselines/results.json')
    names=['adaptive','constant16_T96','constant4_T96','constant1_T96','constant0.25_T96','final16_T1','final16_T4','final16_T16','final16_T48']
    definitions=[(96,dict(kind='two_stage',early_lev=16,late_lev=16,early_barrier=1.,late_barrier=2.,switch=.5))]
    definitions.extend((96,dict(kind='constant',lev=l)) for l in (16,4,1,.25))
    definitions.extend((t,dict(kind='constant',lev=16)) for t in (1,4,16,48))
    records={(r['policy_name'],r['method']):r for r in s['records']};a.require(len(records)==45,'unique policy-method rows')
    positives=[];table=[]
    for index,(name,(duration,p)) in enumerate(zip(names,definitions)):
        matching=old['policy_results'][index]['policy'];a.require(matching['duration']==duration,'unchanged matched-family duration')
        if index==0:a.require(old['frozen_adaptive_controller']==p,'unchanged old adaptive controller')
        else:a.require(matching['lev']==p['lev'],'unchanged fixed leverage')
        factor=.99**(96-duration);b=independent_ranges(k,p,10000.,duration,'terminal');b={key:value*factor for key,value in b.items()}
        for method in METHODS:
            r=records[name,method];v=r['bound'];pad=b['pad'];cap=r['cap']
            a.require(r['policy']==p and r['active_duration']==duration,'same fixed controller across five representations')
            a.near(r['factor'],factor,'deadline compiler factor',0.)
            if method=='raw':lo,hi,correction=b['rawlo'],b['rawhi'],1e-7
            elif method=='clipped_raw':lo,hi,correction=b['rawlo'],min(cap,b['rawhi']),1e-7
            elif method=='clipped_residual':lo,hi,correction=-b['costmax'],cap,1e-7
            elif method=='clipped_mean_cost_residual':lo,hi,correction=-b['meancostmax']-pad,cap+pad,pad
            else:lo,hi,correction=b['comp_lo']-pad,b['comp_hi']+pad,pad
            a.near(v['range_lower'],lo-1e-7,'fixed-policy lower support',1e-7,'fixed_range');a.near(v['range_upper'],hi+1e-7,'fixed-policy upper support',1e-7,'fixed_range')
            a.near(v['numerical_allowance'],correction,'fixed-policy numerical allowance',1e-14)
            check_eb(a,v,32768,delta,correction=correction)
            a.near(r['booked_upper'],b['booked'],'fixed-policy booked harm',1e-10)
            a.require(r['certified']==(v['lower']>0 and b['booked']<0),'fixed-policy certificate decision')
            a.require(cap is None if method in ('raw','predictable') else cap in (32,128,512,2048,8192,32768,131072),'unchanged cap grid')
            if r['certified']:positives.append(dict(policy=name,method=method,lower=v['lower'],booked_upper=r['booked_upper']))
        table.append(dict(policy=name,lower_by_method={m:records[name,m]['bound']['lower'] for m in METHODS}))
    a.require([(x['policy'],x['method']) for x in positives]==[('adaptive','predictable'),('final16_T16','predictable'),('final16_T48','predictable')],'fixed-policy positive identities')
    a.require(s['summary']==[dict(method=m,certified=3 if m=='predictable' else 0) for m in METHODS],'fixed-policy method counts')
    a.require(s['shared_unique_active_transitions']==(8192+32768)*549==22487040,'fixed-policy active calls')
    a.require(s['represented_full_clock_transitions']==9*(8192+32768)*96,'fixed-policy outer slots')
    a.require(s['conditional_oracle_points']==s['executed_actions']+4==20829763,'fixed-policy conditional query count')
    # Read-only provenance: default arguments bind at import, while confirm uses
    # the mutable module DELTA. The search heuristic is valid, but retains .05/720.
    runner=(SOURCE/'run_fixed_policy_comparison.py').read_text()
    a.require('olddelta=h.DELTA;h.DELTA=DELTA' in runner and 'h.DELTA=olddelta' in runner,'in-memory delta override and restoration')
    parsed=ast.parse((SOURCE/'run_heldout_attempt2.py').read_text())
    bound_def=next(node for node in parsed.body if isinstance(node,ast.FunctionDef) and node.name=='bound')
    a.require(isinstance(bound_def.args.defaults[-1],ast.Name) and bound_def.args.defaults[-1].id=='DELTA','search default captures original heldout delta')
    return dict(complete_bounds=45,n_search=8192,n_confirmation=32768,positive_certificates=positives,all_policy_lower_bounds=table,
                counts={m:3 if m=='predictable' else 0 for m in METHODS},search_heuristic_delta=.05/720,confirmation_delta=delta,
                search_default_delta_disclosure='Only the SEARCH heuristic retains the heldout definition-time delta .05/720; all45 reported confirmation bounds correctly use .05/45. No retuning or rerun.',
                actual_unique_active_transitions=s['shared_unique_active_transitions'],conditional_oracle_points=s['conditional_oracle_points'],
                original_case_selected_development=True)


def audit_all():
    audit=Audit();heldout=audit_heldout(audit);unbounded=audit_unbounded(audit);fixed=audit_fixed_policy(audit)
    matched_folder=ROOT/'exploratory_novelty_2026-10-16/matched_multistep_baselines'
    # The matched checker independently integrates all isolated actions and costs.
    module=runpy.run_path(str(matched_folder/'verify_results.py'),run_name='read_only_matched_verification')
    import contextlib,io
    capture=io.StringIO()
    with contextlib.redirect_stdout(capture):module['main']()
    matched=json.loads(capture.getvalue())
    return dict(status='PASS',assertions=audit.assertions,source_hash_checks=audit.hash_checks,maximum_recomputation_errors=audit.max_errors,
                heldout=heldout,unbounded=unbounded,fixed_policy=fixed,matched_multistep=matched,
                no_new_simulation=True,no_new_training=True,no_registered_artifact_modified=True,
                limitation='Recorded path means/variances are checked for arithmetic/support/paired consistency; independent path-level reconstruction requires resimulation and was not performed.')


if __name__=='__main__':
    result=audit_all()
    if '--write' in sys.argv:(HERE/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
