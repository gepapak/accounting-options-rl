"""Source preparation or ROOT-authorized original-body hand gate.

--prepare reads/AST-checks/hashes only. --run needs exact matching ROOT_GO;
it constructs no Env, reads no market data, and calls no model. It compiles
the named ORIGINAL public-engine methods verbatim into a hand proxy. Only
the shadow settlement-price ARRAY values change; original methods do not.
This is not a full Env rollout or evidence of market-state reachability.
"""
from __future__ import annotations
import argparse
import ast
import copy
from datetime import datetime,timezone
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict,Tuple,Optional,Any
import floor_inverse as I

HERE=Path(__file__).resolve().parent
CODE=HERE.parents[1]
ENGINE=CODE/'engine'
SOURCE=ENGINE/'environment.py'
FEES=ENGINE/'market_fee_protocol.py'
PROTOCOL=HERE/'FROZEN_HAND_PROTOCOL_V1.json'
AUTH=HERE/'ROOT_HAND_AUTHORIZATION_V1.json'
METHODS=('_is_horizon_settlement_mtm','_horizon_settlement_price','_horizon_settlement_basis',
 '_horizon_payoff_denominator','_is_mwh_volume_payoff','_mwh_volume_conversion_price',
 '_horizon_contract_volume_mwh','_horizon_unit_payoff','_horizon_entry_price',
 '_open_horizon_settlement_contract','_charge_horizon_roll_cost','_estimate_horizon_roll_cost',
 '_liquidity_tail_impact_multiplier','_settle_horizon_contracts_for_step',
 '_open_horizon_contract_totals','_collateral_required_dkk','_apply_collateral_cash_drag')
TOPS=('_open_horizon_settlement_contract','_settle_horizon_contracts_for_step','_apply_collateral_cash_drag')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as out:json.dump(v,out,indent=2,sort_keys=True,allow_nan=False);out.write('\n')
def utc():return datetime.now(timezone.utc).isoformat()


def cases():
    return [dict(name='NO_COST',cash=100.,fee=0.,funding=0.,entry=500.,settlement=501.,mixed=False),
     dict(name='FEE_FLOOR_NATIVE_TIR_COUNTEREXAMPLE',cash=0.,fee=10.,funding=0.,entry=500.,settlement=501.,mixed=False),
     dict(name='LOSING_REAL_BRANCH',cash=5.,fee=1.,funding=0.,entry=500.,settlement=496.,mixed=False),
     dict(name='POST_FUNDING_BOTH_CENSORED',cash=0.,fee=0.,funding=10.,entry=500.,settlement=501.,mixed=False),
     dict(name='PRE_FEE_POST_FUNDING',cash=0.,fee=10.,funding=2.,entry=500.,settlement=501.,mixed=False),
     dict(name='TWO_DELIVERIES',cash=0.,fee=10.,funding=0.,entry=500.,settlement=501.,mixed=True)]


def source_nodes():
    tree=ast.parse(SOURCE.read_text(encoding='utf-8-sig'))
    found={n.name:n for cls in tree.body if isinstance(cls,ast.ClassDef) for n in cls.body
           if isinstance(n,ast.FunctionDef) and n.name in METHODS}
    if set(found)!=set(METHODS):raise ValueError('Named source methods not unique/complete')
    # Fail closed on this gate's actual source integration features.
    terms={name:ast.unparse(n) for name,n in found.items()}
    assert 'self.budget = max(0.0,' in terms['_charge_horizon_roll_cost']
    assert 'self.budget = max(0.0,' in terms['_settle_horizon_contracts_for_step']
    assert 'self.budget = max(0.0,' in terms['_apply_collateral_cash_drag']
    assert 'total_pnl += pnl' in terms['_settle_horizon_contracts_for_step']
    assert 'self._charge_horizon_roll_cost' in terms['_open_horizon_settlement_contract']
    return found


def prepare():
    if AUTH.exists() or (HERE/'runs').exists():raise ValueError('Freeze must precede GO and actual outputs')
    nodes=source_nodes()
    sources=[SOURCE,FEES,ENGINE/'financial_engine.py',ENGINE/'trading_engine.py']
    own=[HERE/n for n in ('floor_inverse.py','check_pure.py','run_source_hand_gate.py',
                         'SOURCE_FEASIBILITY_V1.md','PURE_CHECKS_V1.json','PURE_ATTEMPT_1_FAILURE.md')]
    bindings={str(p.relative_to(CODE)).replace('\\','/'):sha(p) for p in sources+own}
    freeze=dict(status='FROZEN_BEFORE_ANY_ORIGINAL_BODY_EXECUTION',created_utc=utc(),source_sha256=bindings,
      method_ast_sha256={n:hashlib.sha256(ast.dump(v).encode()).hexdigest() for n,v in nodes.items()},
      cases=cases(),fixed_order=['fresh_open_and_known_fee','delayed_due_settlement','remaining_claim_funding'],
      budget=dict(original_top_level_calls=54,original_helper_body_calls_max=2000,hand_proxy_instances=18,
                  original_Env_instances=0,Env_resets=0,Env_steps=0,model_calls=0,market_reads=0,PPO_fits=0),
      cash_operand_domain=2**50,per_cash_write_RN_error_DKK='1/8',
      original_methods_edited=False,
      shadow_intervention='deep copy of hand state; replace only due delivery settlement array values with verified RN-antisymmetric in-support values',
      known_receipts=['initial reported cash','fresh own executed fee','post-settlement own funding charge'],
      estimator_inputs=['known receipts','real ending reported cash','shadow ending reported cash'],
      gold_evaluator_only=['original settler returned gross PnL','raw source diagnostics'],
      required_checks=['unqueried original vs real branch exact economic parity','no mutation of real inputs by shadow',
       'same due claims and own cost','original source gross cash payoff exact antisymmetry',
       'all cash arithmetic operands and exact pre-floor sums inside domain','booked truth contained in RN interval',
       'fee-floor case canonical TIR=+1 vs source booked=-9; ordered RN interval contains -9 and excludes +1',
       'fully post-cost-censored case returns a non-singleton interval rather than false exactness'],
      scope='isolated original source kernel hand states; no full dynamics, reachability, policy performance, shaping inversion or new AI novelty claim',
      approval_schema=dict(status='ROOT_GO_NATIVE_REWARD_IDENTIFICATION_HAND_V1',protocol_sha256='EXACT_THIS_FILE_SHA'))
    write(PROTOCOL,freeze)
    print(json.dumps(dict(status=freeze['status'],protocol_sha256=sha(PROTOCOL),files=len(bindings))))


def verify():
    freeze=json.loads(PROTOCOL.read_text(encoding='utf-8'))
    for rel,digest in freeze['source_sha256'].items():
        if sha(CODE/rel)!=digest:raise ValueError('Frozen bytes changed: '+rel)
    if freeze['cases']!=cases():raise ValueError('Frozen case roster changed')
    return freeze


def make_source_class(counter):
    """Call only after exact protocol authorization; source definitions unchanged."""
    import numpy as np
    globals_=dict(np=np,Dict=Dict,Tuple=Tuple,Optional=Optional,Any=Any)
    # Original fee module only defines constants/functions/imports, no data or Env.
    exec(compile(FEES.read_text(encoding='utf-8-sig'),str(FEES),'exec'),globals_)
    for n in ('normalize_market_fee_model','base_execution_fee_components'):
        original=globals_[n]
        def wrapped(*args,__fn=original,__name=n,**kwargs):
            counter['helper_attempted']+=1
            if counter['helper_attempted']>2000:raise RuntimeError('Original helper budget exhausted')
            result=__fn(*args,**kwargs);counter['helper_returned']+=1;return result
        globals_[n]=wrapped
    methods=[copy.deepcopy(n) for n in source_nodes().values()]
    cls=ast.ClassDef(name='OriginalHandProxy',bases=[],keywords=[],body=methods,decorator_list=[])
    module=ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[]))
    exec(compile(module,str(SOURCE),'exec'),globals_)
    result=globals_['OriginalHandProxy']
    for n in METHODS:
        original=getattr(result,n)
        def wrapped(self,*args,__fn=original,__name=n,**kwargs):
            counter['helper_attempted']+=1
            if counter['helper_attempted']>2000:raise RuntimeError('Original helper budget exhausted')
            value=__fn(self,*args,**kwargs);counter['helper_returned']+=1;return value
        setattr(result,n,wrapped)
    return result,np


def old_claim(delivery,entry,q):
    return dict(open_step=delivery,delivery_step=delivery,settle_step=12,entry_price=entry,
      exposures={'wind_instrument_value':q*500.},volumes_mwh={'wind_instrument_value':q})


def make_proxy(cls,np,case):
    env=cls();env.budget=case['cash']
    env.config=SimpleNamespace(mtm_return_model='horizon_settlement',mtm_settlement_horizon_steps=6,
      mtm_horizon_payoff_denominator_mode='mwh_volume',mtm_reference_price_dkk_per_mwh=500.,
      mtm_entry_price_mode='current_price',time_step_hours=1.,market_fee_model='legacy_notional_fixed',
      friction_cost_multiplier=1.,transaction_cost_bps=0.,transaction_fixed_cost=case['fee'],
      transaction_fee_dkk_per_mwh=0.,half_spread_bp=0.,impact_coef_bp=0.,liquidity_tail_impact_multiplier=1.,
      mtm_external_settlement_min_price_dkk_per_mwh=-111750.,mtm_external_settlement_max_price_dkk_per_mwh=111750.,
      enable_collateral_cash_drag=case['funding']>0.,collateral_notional_margin_fraction=1.,
      collateral_stress_loss_fraction=0.,collateral_stress_price_dkk_per_mwh=0.,
      collateral_funding_rate_annual=case['funding']*8766./500.)
    env._price_raw=np.full(40,500.,dtype=np.float64)
    env._horizon_settlement_price_raw=np.full(40,500.,dtype=np.float64)
    env._horizon_settlement_price_raw[6]=case['settlement']
    env._horizon_settlement_basis_component=np.zeros(40,dtype=np.float64)
    env._horizon_contracts=[old_claim(6,case['entry'],1.)]
    if case['mixed']:
        env._horizon_contracts.append(old_claim(0,600.,-.5))
        env._horizon_settlement_price_raw[0]=598.
    env._last_horizon_open_count=0
    return env


def economic(env):
    return dict(cash=float(env.budget),queue=copy.deepcopy(env._horizon_contracts),
      roll_cost=float(env._last_horizon_roll_cost),funding=float(env._last_collateral_funding_cost_dkk),
      settled=int(env._last_horizon_settlement_count))


def state_digest(env):
    """All hand-instance data, including untouched private arrays, bitwise."""
    def encode(v):
        if type(v) is float:return {'float64_hex':v.hex()}
        if type(v) in (str,int,bool,type(None)):return v
        if isinstance(v,dict):return {str(k):encode(w) for k,w in sorted(v.items())}
        if isinstance(v,(list,tuple)):return [encode(w) for w in v]
        if isinstance(v,SimpleNamespace):return encode(vars(v))
        if hasattr(v,'tobytes'):return dict(dtype=str(v.dtype),shape=list(v.shape),bytes_sha256=hashlib.sha256(v.tobytes()).hexdigest())
        raise TypeError(type(v).__name__)
    return hashlib.sha256(json.dumps(encode(vars(env)),sort_keys=True).encode()).hexdigest()


def run():
    freeze=verify();auth=json.loads(AUTH.read_text(encoding='utf-8'))
    if auth.get('status')!='ROOT_GO_NATIVE_REWARD_IDENTIFICATION_HAND_V1' or auth.get('protocol_sha256')!=sha(PROTOCOL):
        raise ValueError('Exact root GO required before original functions are compiled or called')
    output=HERE/'runs'/'hand_v1';output.mkdir(parents=True,exist_ok=False)
    counter=dict(top_attempted=0,top_returned=0,helper_attempted=0,helper_returned=0)
    rows=[]
    write(output/'STARTED.json',dict(protocol_sha256=sha(PROTOCOL),authorization_sha256=sha(AUTH),started_utc=utc()))
    try:
        cls,np=make_source_class(counter)
        def call(env,name,*args):
            counter['top_attempted']+=1
            if counter['top_attempted']>54:raise RuntimeError('Top-level source budget exhausted')
            value=getattr(env,name)(*args);counter['top_returned']+=1;return value
        for case in freeze['cases']:
            original=make_proxy(cls,np,case)
            original_digest=state_digest(original)
            # The only private-price read is inside the trusted shadow builder;
            # these prices/payoffs are never an input to the cash inverse.
            due={int(c['delivery_step']):float(original._horizon_settlement_price_raw[int(c['delivery_step'])])
                 for c in original._horizon_contracts if c['settle_step']<=12}
            shadows=I.due_reflection_map(original._horizon_contracts,12,due)
            real=copy.deepcopy(original);shadow=copy.deepcopy(original);unqueried=copy.deepcopy(original)
            for d,v in shadows.items():shadow._horizon_settlement_price_raw[d]=v
            cash_obs={};gold={};after_open={}
            for name,env in (('unqueried',unqueried),('shadow',shadow),('real',real)):
                call(env,'_open_horizon_settlement_contract',12,{'wind_instrument_value':500.})
                cost=float(env._last_horizon_roll_cost);before=float(env.budget);after_open[name]=before
                gross,_=call(env,'_settle_horizon_contracts_for_step',12)
                pre_funding=float(env.budget)
                funding=float(call(env,'_apply_collateral_cash_drag',12))
                # Gold checks the arithmetic premise; the estimator never sees it.
                for a,w in ((case['cash'],-cost),(before,gross),(pre_funding,-funding)):
                    if max(abs(F.from_float(a)),abs(F.from_float(float(w))),abs(F.from_float(a)+F.from_float(float(w))))>2**50:
                        raise ValueError('RN cash operand-domain violation')
                cash_obs[name]=float(env.budget)
                gold[name]=dict(gross=float(gross),cost=cost,funding=funding,
                    booked=str(F.from_float(float(gross))-F.from_float(cost)-F.from_float(funding)))
            assert economic(real)==economic(unqueried)
            assert gold['real']==gold['unqueried']
            assert state_digest(real)==state_digest(unqueried)
            assert state_digest(original)==original_digest
            assert np.array_equal(original._horizon_settlement_price_raw,unqueried._horizon_settlement_price_raw)
            assert gold['shadow']['gross']==-gold['real']['gross']
            assert gold['shadow']['cost']==gold['real']['cost']==case['fee']
            assert gold['shadow']['funding']==gold['real']['funding']
            assert after_open['real']==I.prefix_cash(case['cash'],(-gold['real']['cost'],),binary64=True)
            receipts=dict(initial_cash=case['cash'],pre_known_writes=[-gold['real']['cost']],
                          post_known_writes=[-gold['real']['funding']])
            estimate=I.identify(receipts['initial_cash'],tuple(receipts['pre_known_writes']),tuple(receipts['post_known_writes']),
                                cash_obs['real'],cash_obs['shadow'],binary64=True)
            booked=F(gold['real']['booked']);assert estimate['booked'].contains(booked)
            total_cost=gold['real']['cost']+gold['real']['funding']
            r=cash_obs['real']-case['cash'];rt=cash_obs['shadow']-case['cash']
            naive=(1. if r>rt else -1.)*(max(r,rt)+total_cost)-total_cost if abs(r-rt)>1e-12 else -total_cost
            if case['name']=='FEE_FLOOR_NATIVE_TIR_COUNTEREXAMPLE':
                assert gold['real']['booked']=='-9' and naive==1.
                assert not estimate['booked'].contains(1.)
            if case['name']=='POST_FUNDING_BOTH_CENSORED':
                assert estimate['gross'].lower<estimate['gross'].upper
            rows.append(dict(case=case['name'],known_receipts=receipts,reported_ending_cash=cash_obs,
               estimated_gross=estimate['gross'].record(),estimated_booked=estimate['booked'].record(),
               RN_cash_error_radius=str(estimate['cash_error_radius']),naive_TIR_on_cash_increment=naive,
               evaluator_only_gold=gold,source_parity=True))
            write(output/(case['name']+'.json'),rows[-1])
        assert counter['top_attempted']==counter['top_returned']==54
        assert counter['helper_attempted']==counter['helper_returned']
        write(output/'RESULTS.json',dict(status='COMPLETED_SOURCE_KERNEL_HAND_GATE',rows=rows,calls=counter,
          protocol_sha256=sha(PROTOCOL),scope=freeze['scope'],completed_utc=utc()))
        print(json.dumps(dict(status='COMPLETED_SOURCE_KERNEL_HAND_GATE',cases=len(rows),calls=counter)))
    except BaseException as exc:
        write(output/'FAILED.json',dict(exception=repr(exc),calls=counter,completed_cases=len(rows),failed_utc=utc()))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.prepare:prepare()
    elif args.run:run()
    else:print('Source-only preparation or exact root-authorized original-function hand gate; use --prepare first.')
