"""Read-only core validation on existing software traces and exact artifacts.

No engine/checkpoint is imported, no draw is made, no predictor is fitted and
no statistical discovery or new learned-policy endpoint is emitted.
"""
from __future__ import annotations
from fractions import Fraction as F
import hashlib
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

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
REPO=PARENT.parents[1]
PAPER1=REPO.parents[1]
ENGINE=PAPER1/'Prototype5'
ATTEMPT=PARENT/'native_validation/attempt_1'


class Checks:
    def __init__(self):self.n=0;self.max_errors={};self.inputs={}
    def yes(self,value,message):
        self.n+=1
        if not bool(value):raise AssertionError(message)
    def near(self,x,y,message,tol=1e-6,category='arithmetic'):
        error=float(np.max(np.abs(np.asarray(x)-np.asarray(y)),initial=0.))
        self.max_errors[category]=max(error,self.max_errors.get(category,0.))
        self.yes(np.isfinite(error) and error<=tol,f'{message}: error {error}')
    def bind(self,path,expected=None):
        path=Path(path);digest=hashlib.sha256(path.read_bytes()).hexdigest()
        try:key=str(path.relative_to(REPO))
        except ValueError:key=str(path)
        self.inputs[key]=digest
        if expected is not None:self.yes(digest==expected,'source/snapshot hash '+key)
        return digest


def sequential_control(delta,coin,beta,budget):
    remaining=budget;value=0.;spent=0.
    for d,e in zip(delta,coin):
        amount=min(abs(beta*d),remaining)
        value+=e*np.sign(d)*amount;remaining-=amount;spent+=amount
    return float(value),float(spent)


class FixedFunction:
    """Deterministic test function, never fitted; not an empirical critic."""
    def __init__(self,kind):self.kind=kind
    def predict(self,x):
        if self.kind=='constant':return np.full(len(x),7.)
        return (3*x[:,23]-2*x[:,24]+.1*x[:,28]+.25*x[:,0])


def exact_checks(check):
    artifact=PARENT/'exact_sample_only_checks.json';d=json.loads(artifact.read_text())
    check.bind(artifact);check.bind(PARENT/'exact_sample_only_checks.py',d['script_sha256']);check.bind(PARENT/'SAMPLE_ONLY_THEORY.md',d['theory_sha256'])
    check.yes(d['status']=='EXACT_CHECKS_PASS' and d['every_exact_assertion_passed'],'exact suite declared complete')
    check.yes(d['floor_path_families']==8 and len(d['floor_families'])==8,'eight exact floor families')
    check.yes(d['floor_certificate_cases']==len(d['floor_cases'])==864,'864 exact floor parameter cases')
    check.yes(d['terminal_certificate_cases']==len(d['terminal_cases'])==81,'81 exact terminal parameter cases')
    costmax=F(d['floor_discounted_financial_cost_upper'])
    for family in d['floor_families']:
        check.yes(F(family['probability_sum'])==1,'floor exact probabilities')
        check.yes(F(family['expected_booked_financial_return'])==-F(family['expected_financial_cost']),'exact booked expected-cost identity')
        check.yes(family['exact_pairing_identity'] and family['native_floor_tanaka_is_existing_injection'],'pairing and Tanaka equivalence')
    for row in d['floor_cases']:
        charged,paired,corrected,capped,control=(F(row[k]) for k in ('charged_financial_mean','uncapped_paired_mean','uncapped_corrected_mean','capped_corrected_mean','control_mean'))
        check.yes(charged==paired==corrected and control==0,'exact arbitrary-model unbiasedness')
        check.yes(capped<=charged and F(row['cap_lower_bias'])==charged-capped,'upper cap lower-mean property')
        lo,hi=F(row['certified_support_lower']),F(row['certified_support_upper']);budget=F(row['control_budget'])
        check.yes(lo==-costmax-budget and hi==F(row['upper_cap']),'exact cost plus control support')
        check.yes(lo<=F(row['observed_capped_min'])<=F(row['observed_capped_max'])<=hi,'exact complete enumerated support')
        check.yes(0<=F(row['maximum_spent_absolute_amplitude'])<=budget,'exact pathwise amplitude spending')
        check.yes(row['range_and_mean_checks_pass'],'exact case completion')
    for row in d['terminal_cases']:
        check.yes(F(row['expected_charged_relative_flat'])==F(row['sample_tanaka_mean']),'asymmetric no-edge terminal Tanaka identity')
        check.yes(F(row['capped_mean'])<=F(row['sample_tanaka_mean']),'terminal capping lower bias')
        check.yes(F(row['lower_support'])<=F(row['capped_mean'])<=F(row['upper_support']),'terminal support on exact mean')
        check.yes(row['all_exact_checks_pass'],'terminal exact completion')
    bad=d['invalid_constructions']
    check.yes(F(bad['actual_model_mixed']['invalid_control_mean'])==-1 and F(bad['actual_model_mixed']['invalid_corrected_capped_mean'])==1,'invalid true/model mixture counterexample')
    check.yes(F(bad['actual_model_mixed']['correctly_centered_model_control_mean'])==0,'same-model centering repair')
    check.yes(F(bad['aggregate_net_clamp']['ordered_financial_injection_mean'])==F(4,5) and F(bad['aggregate_net_clamp']['incorrectly_aggregated_mean'])==F(13,20),'financial event ordering counterexample')
    check.yes(F(bad['unrelated_floor_injection']['flat_financial_target'])==0 and F(bad['unrelated_floor_injection']['false_target_if_unrelated_injection_included'])==F(3,10),'unrelated injection attribution counterexample')
    check.yes(F(bad['fair_mean_without_fair_sign']['expected_gross'])==0 and F(bad['fair_mean_without_fair_sign']['naive_sign_control_mean'])==F(-1,3),'no-edge alone does not supply fair sign')
    check.yes(d['monte_carlo_draws']==0 and d['learners_trained']==0,'exact evidence is non-Monte-Carlo/non-training')
    return dict(floor_families=8,floor_cases=864,terminal_cases=81,all_recorded_Fraction_identities_checked=True,
                exact_sources_hash_matched=True,not_an_independent_reenumeration=True)


def verify():
    check=Checks();core_path=PARENT/'sample_estimators.py';check.bind(core_path)
    core=runpy.run_path(str(core_path),run_name='read_only_sample_core')
    report=json.loads((ATTEMPT/'validation_report.json').read_text());check.bind(ATTEMPT/'validation_report.json')
    check.yes(report['status']=='PASS' and len(report['cases'])==2,'only the two completed software paths')
    for name,digest in report['source_sha256'].items():check.bind(ATTEMPT/name,digest)
    cases=[]
    for case in report['cases']:
        name=case['case'];path=ATTEMPT/(name+'_trace.npz');check.bind(path)
        with np.load(path,allow_pickle=False) as loaded:trace={key:loaded[key].copy() for key in loaded.files}
        for source,digest in case['metadata']['source_sha256'].items():check.bind(ENGINE/source,digest)
        summary=core['summarize'](trace);steps=trace['event_step'];weights=.995**steps
        before,gross,cost,injection,after=(trace['event_'+k] for k in ('cash_before','gross','cost','injection','cash_after'))
        coin=trace['event_sign_epsilon'];base=trace['event_unsigned_base_gross'];active=coin!=0
        check.yes(len(trace['path_step'])==96 and np.all(np.diff(steps)>=0),'complete ordered native clock')
        check.near(after,np.maximum(before+gross-cost,0),'native event cash map',2e-6,'event_cash')
        check.near(after-before,gross-cost+injection,'native financial event identity',2e-6,'event_identity')
        check.near(gross,np.where(active,coin*base,0),'actual maturity sign contract',1e-7)
        check.yes(np.all(base[~active]==0) and np.all(cost[active]==0),'separate cost-only and gross-only events')
        check.near(summary['pair_event'][active],.5*np.maximum(np.abs(base[active])-before[active],0),'exact fair pair magnitude identity',1e-6)
        check.near(summary['pair_event'][~active],injection[~active],'deterministic cost injection',1e-6)
        check.yes(len(np.unique(steps[active]))==int(active.sum()),'one aggregated maturity coin per calendar step')
        for field,event in [('gross',gross),('cost',cost),('forgiveness',injection)]:check.near(summary[field],float(weights@event),'discounted '+field,1e-6)
        pair_events=weights*(summary['pair_event']-cost)
        direct_future=np.r_[np.cumsum(pair_events[:0:-1])[::-1],0.] if len(pair_events) else np.empty(0)
        check.near(summary['future_target'],direct_future,'globally discounted post-event future target',2e-6,'future_target')
        for field,event in [('gross',gross),('cost',cost),('injection',injection)]:
            check.near(trace['path_financial_'+field],np.bincount(steps,weights=event,minlength=96),'event/path '+field,2e-6)
        check.near(trace['path_native_cash_after']-trace['path_native_cash_before'],trace['path_financial_gross']-trace['path_financial_cost']+trace['path_financial_injection']+trace['path_excluded_native_cash_change'],'complete cash/excluded transfer attribution',2e-6)
        check.yes(case['summary']['terminal_open_contracts']==0 and trace['path_open_contracts'][-1]==0,'native terminal queue empty')
        dim=trace['event_actor_observation'].shape[1];check.yes(dim==(12 if name=='plain7' else 20),'authentic actor input dimension')
        options=dict(money_scale=1_000_000.,horizon=96,policy_index=0 if dim==12 else 10,profile_index=0)
        plus=core['branch_features'](trace,summary,branch=1,**options);minus=core['branch_features'](trace,summary,branch=-1,**options)
        check.yes(plus.shape==minus.shape==(int(active.sum()),32),'fixed nuisance feature shape including empty cases')
        actual_features,target=core['fit_rows'](trace,summary,**options)
        check.near(target,summary['future_target'][active],'fit target provenance',0.)
        check.near(actual_features,np.where(coin[active,None]>0,plus,minus),'actual synthetic branch labels',0.)
        flipped={key:value.copy() for key,value in trace.items()}
        flipped['event_sign_epsilon'][active]*=-1;flipped['event_gross'][active]*=-1
        flipped['event_injection']=np.maximum(-before-flipped['event_gross']+cost,0.)
        flipped_summary=core['summarize'](flipped)
        check.near(core['branch_features'](flipped,flipped_summary,branch=1,**options),plus,'hypothetical plus features independent of current actual coin',0.)
        check.near(core['branch_features'](flipped,flipped_summary,branch=-1,**options),minus,'hypothetical minus features independent of current actual coin',0.)
        fixed={'ridge':FixedFunction('constant'),'mlp':FixedFunction('linear')}
        deltas=core['predictor_deltas'](trace,summary,fixed,**options)
        check.near(deltas['ridge'],0.,'zero symmetric predictor control',0.)
        check.near(deltas['mlp'],.5*(fixed['mlp'].predict(plus)-fixed['mlp'].predict(minus)),'same-model delta identity',0.)
        for delta in deltas.values():
            for beta in (0,.25,.5,1,2):
                for budget in (0,1,4,1000):
                    expected,spent=sequential_control(delta,coin[active],beta,budget)
                    actual=core['budgeted_control'](delta,coin[active],beta,budget)
                    check.near(actual,expected,'vectorized/sequential spending equivalence',1e-9,'budget')
                    check.yes(spent<=budget+1e-9 and abs(actual)<=budget+1e-9,'absolute-amplitude path support')
                    check.near(core['budgeted_control'](delta,-coin[active],beta,budget),-actual,'coin-linear control for fixed common deltas',1e-9)
        sw=float(np.sum(.995**np.arange(96)));declared=case['summary']['support']
        bounds=dict(discounted_cost_bound=sw*declared['per_step_cost_bound'],discounted_gross_bound=sw*declared['per_step_gross_bound'])
        # These are the historical SOFTWARE path's declared model envelopes,
        # not a claim that current final-experiment support code is validated.
        row={**summary,'delta_ridge':deltas['ridge'],'delta_mlp':deltas['mlp'],'maturity_coins':coin[active]}
        for alpha in (0.,1.):
            for method in core['METHODS']:
                for cap in (1.,100.,1_000_000.):
                    choice=dict(cap=cap,beta=.5,budget=4.)
                    value=core['sample_values']([row],method,alpha,choice)[0]
                    lo,hi=core['deterministic_support'](bounds,method,alpha,choice)
                    check.yes(lo-2e-6<=value<=hi+2e-6,'computed value in supplied declared support')
                    if method=='raw':direct=summary['gross']-summary['cost']+(1-alpha)*summary['forgiveness']
                    elif method=='capped_raw':direct=min(summary['gross']-summary['cost']+(1-alpha)*summary['forgiveness'],cap)
                    elif method=='capped_residual':direct=min((1-alpha)*summary['forgiveness']-summary['cost'],cap)
                    else:
                        direct=(1-alpha)*summary['paired_forgiveness']-summary['cost']
                        if method.endswith('_control'):direct-=sequential_control(deltas[method.removesuffix('_control')],coin[active],.5,4.)[0]
                        direct=min(direct,cap)
                    check.near(value,direct,'representation construction',1e-6)
        cases.append(dict(case=name,source='existing deterministic software trace',calendar_steps=96,actor_input_dimension=dim,
                          event_count=len(steps),local_pairing_and_feature_contract_checked=True,no_financial_endpoint_released=True))
    # Deterministic arithmetic unit check, NOT an episode CI or a data draw.
    x=np.array([-1.,0.,1.,0.]);delta=.05/520.;pad=1e-6
    b=core['empirical_bernstein'](x,-1,1,delta,pad)
    radius=math.sqrt(2*(2/3)*math.log(2/delta)/4)+7*(2+2*pad)*math.log(2/delta)/(3*3)
    check.near(b['variance'],2/3,'ddof1 variance arithmetic',1e-15)
    check.near(b['radius'],radius,'padded empirical Bernstein arithmetic',1e-12)
    check.near(b['lower'],-radius-pad,'expectation correction after support expansion',1e-12)
    exact=exact_checks(check)
    return dict(status='PASS',purpose='software and exact mathematical validation only',checks=check.n,input_sha256=check.inputs,
                maximum_recomputation_errors=check.max_errors,software_traces=cases,exact_artifacts=exact,
                new_simulator_steps=0,new_random_draws=0,new_checkpoint_inference=0,new_fitted_predictors=0,
                financial_endpoints_emitted=False,
                caveat='Stored software traces validate core operations, not experiment power, positivity, or final adapter support for every future profile/context.')


if __name__=='__main__':
    result=verify()
    if '--write' in sys.argv:(HERE/'SAMPLE_CORE_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
