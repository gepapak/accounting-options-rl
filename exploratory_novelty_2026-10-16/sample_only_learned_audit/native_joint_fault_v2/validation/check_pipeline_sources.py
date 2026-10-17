"""Source and literal-array v2 checks; no engine, actor, fitting or random draw."""
from datetime import datetime,timezone
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import numpy as np
import joint_fault_estimators as core
import run_joint_fault_audit as runner


def main():
    checks=[]
    def yes(value,name):
        if not bool(value):raise AssertionError(name)
        checks.append(name)
    def close(a,b,name):yes(np.allclose(a,b,atol=1e-9,rtol=1e-12),name)
    sources=sorted(HERE.glob('*.py'))+[HERE/'validation/verify_joint_fault_results.py']
    for path in sources:ast.parse(path.read_text(encoding='utf-8'));yes(True,'AST:'+path.name)
    spec=importlib.util.spec_from_file_location('independent_v2_arithmetic',HERE/'validation/verify_joint_fault_results.py')
    independent=importlib.util.module_from_spec(spec);spec.loader.exec_module(independent)
    pspec=importlib.util.spec_from_file_location('independent_public_v2',HERE/'reproduce_public_joint_fault.py')
    public=importlib.util.module_from_spec(pspec);pspec.loader.exec_module(public)
    cases=runner.cases()
    yes(len(cases)==20 and [r['case_id'] for r in cases]==list(range(20)),'all20fixedactors')
    yes(runner.CAPS==tuple(1000.*4**j for j in range(11)),'fixed11capgrid')
    yes(len(core.valid_pairs())==16 and len(core.valid_pairs())*20+20==340,'alpha0tenalpha1sixfamily340')
    yes(runner.DELTA==independent.DELTA==.05/340,'uniform340allocation')
    yes({36300000,37300000,38300000}==set(independent.PHASE_BAND.values()),'freshphasebands')
    fields={key:[] for key in ('step','cash_before','cash_after','gross','cost','injection','unsigned_base_gross','sign_epsilon','actor_observation','actor_action')}
    cash=2.
    def event(step,base,cost,coin):
        nonlocal cash
        before=cash;gross=coin*base;injection=max(-before-gross+cost,0.);cash=max(before+gross-cost,0.)
        values=dict(step=step,cash_before=before,cash_after=cash,gross=gross,cost=cost,injection=injection,
            unsigned_base_gross=base,sign_epsilon=coin,actor_observation=np.zeros(12),actor_action=.25)
        for key,value in values.items():fields[key].append(value)
    for step in range(96):
        event(step,0.,1.,0)
        if step==1:event(step,10.,0.,-1)
        if step==6:event(step,5.,0.,1)
    trace={'event_'+key:np.asarray(value) for key,value in fields.items()}
    summary=core.summarize(trace)
    steps=trace['event_step'];coins=trace['event_sign_epsilon'];old=(coins!=0)&(steps<6);w=.995**steps
    local=(np.maximum(-trace['event_cash_before']-trace['event_unsigned_base_gross']+trace['event_cost'],0)+
        np.maximum(-trace['event_cash_before']+trace['event_unsigned_base_gross']+trace['event_cost'],0))/2
    close(summary['gross'],np.sum(w[~old]*trace['event_gross'][~old]),'removeonlyoldG')
    close(summary['cost'],np.sum(w*trace['event_cost']),'retainallforwardC')
    close(summary['forgiveness'],np.sum(w[~old]*trace['event_injection'][~old]),'removeonlyoldIkeepfeeI')
    close(summary['paired_forgiveness'],np.sum(w[~old]*local[~old]),'removeonlyoldA')
    yes(summary['excluded_old_maturity_events']==1 and len(summary['active_indices'])==2,'keepalloldnewcoins')
    contribution=w*(np.where(old,0.,local)-trace['event_cost'])
    expected=np.array([np.sum(contribution[i+1:]) for i in range(len(contribution))])
    close(summary['future_target'],expected,'newtargetGLOBALaftereventfuturelabels')
    X,y=core.fit_rows(trace,summary,money_scale=1000.,horizon=96,policy_index=0,profile_index=1)
    yes(X.shape==(2,32),'oldcoinandnewcoinFITrowsretained')
    close(y,expected[coins!=0],'FITlabelusesnewtarget')
    raw_contribution=w*(np.where(old,0.,trace['event_gross'])-trace['event_cost']+np.where(old,0.,trace['event_injection']))
    expected_raw=np.array([np.sum(raw_contribution[i+1:]) for i in range(len(raw_contribution))])
    close(summary['raw_future_target'],expected_raw,'RAWfutureFITlabelsGLOBALaftereventnewtarget')
    class LiteralPredictor:
        def predict(self,X):return X[:,23]*2.
    delta=core.predictor_deltas(trace,summary,{name:LiteralPredictor() for name in ('ridge','mlp','cash_ridge','cash_mlp')},
        money_scale=1000.,horizon=96,policy_index=0,profile_index=1)
    close(delta['ridge'],np.full(2,2.),'pairedfuturehalfdiffunchanged')
    close(delta['cash_ridge'],2.+summary['known_current_raw_delta'][summary['active_indices']],'RAWfuturehalfdiffplusknowncurrent')
    close(delta['cash_mlp'][0],2.,'oldcurrentzerobutoldfuturecontrolretained')
    base_row={k:summary[k] for k in ('gross','cost','forgiveness','paired_forgiveness')}
    base_row.update(maturity_coins=[-1,1],delta_ridge=[3.,-4.],delta_mlp=[0.,8.],
        delta_cash_ridge=[4.,-3.],delta_cash_mlp=[0.,8.],delta_cash_local=summary['known_current_raw_delta'][summary['active_indices']].tolist())
    close(summary['known_current_raw_delta'][summary['active_indices']][0],0.,'oldcurrentRAWlocalzero')
    expected_local=w*(trace['event_unsigned_base_gross']+(np.maximum(-trace['event_cash_before']-trace['event_unsigned_base_gross']+trace['event_cost'],0)-np.maximum(-trace['event_cash_before']+trace['event_unsigned_base_gross']+trace['event_cost'],0))/2)
    close(summary['known_current_raw_delta'][~old],expected_local[~old],'currentRAWdeltaGLOBALdiscountedatnonzerostep')
    full_local=sum(c*d for c,d in zip(base_row['maturity_coins'],base_row['delta_cash_local']))
    close(base_row['gross']-base_row['cost']+base_row['forgiveness']-full_local,base_row['paired_forgiveness']-base_row['cost'],'RAWminusfulllocalequalsPAIRnewtarget')
    proposed=np.array([1e8*(1+i%11)+.1 for i in range(16)])
    control=core.budgeted_control(proposed,np.ones(16),1.,4194304000.)
    yes(abs(control)<=4194304000.+1e-4,'largepredeclaredbudgetfloatguardregression')
    rows=[dict(base_row) for _ in range(256)]
    bounds=dict(discounted_cost_bound=100.,discounted_gross_bound=20.,cash_inflow_bound=12.)
    for tiny in (0.,1.,1e12):
        b=dict(discounted_cost_bound=tiny,discounted_gross_bound=tiny,cash_inflow_bound=tiny)
        close(core.numerical_pad(b),public.numerical_pad(b),'publicpadmatchescoreatfloorandsourcelarge'+str(tiny))
    for alpha,method in core.valid_pairs():
        choice=dict(cap=None if method=='raw' else 1000.,beta=.5 if method.endswith('_control') else 0.,budget=4. if method.endswith('_control') else 0.)
        x=core.sample_values(rows,method,alpha,choice)
        close(x,independent.values(rows,method,alpha,choice),'independentvalue:'+method+str(alpha))
        close(x,[public.path_value(row,method,alpha,choice) for row in rows],'publicvalue:'+method+str(alpha))
        lo,hi=core.deterministic_support(bounds,method,alpha,choice)
        close([lo,hi],independent.support(bounds,method,alpha,choice),'independentsupport:'+method+str(alpha))
        close([lo,hi],public.deterministic_support(bounds,method,alpha,choice),'publicsupport:'+method+str(alpha))
        a=core.empirical_bernstein(x,lo,hi,runner.DELTA,core.numerical_pad(bounds,choice))
        b=independent.eb(x,lo,hi,independent.pad(bounds,choice))
        for key in a:close(a[key],b[key],'independentEB:'+method+str(alpha)+key)
        c=public.empirical_bernstein(x.tolist(),lo,hi,public.numerical_pad(bounds,choice))
        for key in a:close(a[key],c[key],'publicEB:'+method+str(alpha)+key)
    try:core.deterministic_support(bounds,'cash_capped_raw',1.,dict(cap=1000.))
    except AssertionError:yes(True,'cashalpha1explicitlyrejected')
    else:raise AssertionError('Cash method improperlyenabledalpha1')
    choice,candidates=core.select_choice(rows[:32],'ridge_control',0.,bounds,runner.DELTA,256,runner.CAPS,runner.BETAS,runner.BUDGET_RATIOS)
    yes(len(candidates)==11*17,'all187frozenlearnedsettingslogged')
    yes(choice['cap'] in runner.CAPS,'capchosenonlyfixedgrid')
    yes(not any(name in sys.modules for name in ('joint_fault_adapter','joint_fault_support','environment','policy','stable_baselines3')),'noengineactoradapterimports')
    yes(not (HERE/'execution').exists(),'noexperimentfreezeorlaunch')
    original=json.loads((HERE.parent/'execution/run_001/SOURCE_FREEZE.json').read_text())
    for name,digest in original['source_sha256'].items():
        yes(hashlib.sha256((HERE.parent/name).read_bytes()).hexdigest()==digest,'originalfrozenunchanged:'+name)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    report=dict(status='PASS SOURCE AND LITERAL ARITHMETIC ONLY',created_utc=datetime.now(timezone.utc).isoformat(),checks=len(checks),
        check_names=checks,source_sha256={p.relative_to(HERE).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        new_native_queries=0,new_actor_inference=0,new_regressor_fits=0,new_random_draws=0,source_freeze_created=False)
    target=HERE/'validation'/('PIPELINE_SOURCE_CHECK_'+stamp+'.json')
    with target.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,sort_keys=True,allow_nan=False)
    print(json.dumps(dict(status=report['status'],checks=len(checks),report=target.name,new_native_queries=0)))


if __name__=='__main__':main()
