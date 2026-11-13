"""Prepared observational adapter; importing this module calls no native code.

Actual original-source calls happen ONLY when installed around an authorized
native run. The estimator API is a separate pure function with an allowlist.
The original environment, functions and policy observations are not replaced.
"""
from __future__ import annotations
import copy
from fractions import Fraction as F
import math
import sys
from types import MethodType, SimpleNamespace
import floor_inverse as V1
import known_sum_v3 as V2

KEYS=('wind_instrument_value','solar_instrument_value','hydro_instrument_value')
SETTLER='_settle_horizon_contracts_for_step'
HELPERS=('_is_horizon_settlement_mtm','_horizon_settlement_price',
 '_horizon_settlement_basis','_horizon_payoff_denominator','_horizon_unit_payoff',
 '_is_mwh_volume_payoff','_horizon_contract_volume_mwh','_mwh_volume_conversion_price')
COSTS=('_apply_market_access_fee','_charge_horizon_roll_cost','_apply_collateral_cash_drag')
CONFIG=('mtm_return_model','mtm_horizon_payoff_denominator_mode',
 'mtm_reference_price_dkk_per_mwh','mtm_external_settlement_min_price_dkk_per_mwh',
 'mtm_external_settlement_max_price_dkk_per_mwh')
API_FIELDS={'cash_before','cash_real_after','cash_shadow_after','own_entries',
 'public_centers','public_price_support'}
DOMAIN=2**50


class Unsupported(ValueError):
    pass


def finite(value):
    v=float(value)
    if not math.isfinite(v):raise Unsupported('Finite represented value required')
    return v


def restricted_estimator(api):
    """No env, market, price, source gross, policy state or moments accepted."""
    if set(api)!=API_FIELDS:raise ValueError('Restricted audit API fields differ')
    contract=V2.public_pair_contract(api['own_entries'],api['public_centers'],api['public_price_support'])
    inv=V2.identify_cash(api['cash_before'],api['cash_real_after'],api['cash_shadow_after'],contract)
    support=V2.own_support(api['own_entries'],api['public_price_support'])
    real=inv['real_only'].meet(support);paired=inv['paired'].meet(support)
    return dict(real_only=real,paired=paired,K=contract['K'],
                pair_error_radius=contract['pair_error_radius'],cash_error_radius=inv['cash_error'])


def due_entries(queue,step):
    """Own accepted source quantities in exactly the source traversal order."""
    entries=[]
    for c in queue:
        if int(c.get('settle_step',10**18))>step:continue
        if 'entry_price' not in c:raise Unsupported('Missing accepted own entry price')
        d=int(c.get('delivery_step',c.get('open_step',step)));p=finite(c['entry_price'])
        exposures=c.get('exposures',{}) or {};volumes=c.get('volumes_mwh',{}) or {}
        for key in exposures:
            if key not in KEYS:continue
            # Accepted MWh is required. Do not ask an ex-post notional oracle.
            if key not in volumes:raise Unsupported('Missing accepted own executed MWh')
            entries.append((d,finite(volumes[key]),p))
    return entries


def source_shape_gold(entries,prices,cash_before):
    """Independent evaluator only: source-order floating operations and bounds."""
    z=0.
    for d,q,p in entries:
        unit=float(prices[d]-p);term=float(q*unit)
        for v in (V1.frac(z),V1.frac(term),V1.frac(z)+V1.frac(term)):
            if abs(v)>DOMAIN:raise Unsupported('Gross summation operand domain unsupported')
        z+=term
    if max(abs(V1.frac(cash_before)),abs(V1.frac(z)),abs(V1.frac(cash_before)+V1.frac(z)))>DOMAIN:
        raise Unsupported('Cash addition operand domain unsupported')
    return z


class SettlementObserver:
    """Uses CPython call/return observation, never installs native method hooks.

    Gold prices/G live here and are passed only to the evaluator and shadow
    service. restricted_estimator sees an explicit allowlisted value object.
    The estimator withholding is a research interface restriction, not native
    data inaccessibility. Original actors still see their original observations.
    """
    def __init__(self,env,*,center=500.,max_due_events=166,max_proxy_calls=10000):
        self.env=env;self.center=finite(center);self.max_due_events=max_due_events
        self.max_proxy_calls=max_proxy_calls;self.proxy_calls=0;self.proxy_counts={}
        self.proxy_settler_calls=0;self.events=[];self.steps=[];self.active=None
        self.pending=None;self.previous_profile=None;self.profile_active=False
        cls=type(env)
        self.functions={name:getattr(cls,name) for name in ('step',SETTLER,*HELPERS,*COSTS)}
        for name,fn in self.functions.items():
            if name in env.__dict__ or getattr(env,name).__func__ is not fn:
                raise AssertionError('Original unhooked economic method required: '+name)
        self.code_names={fn.__code__:name for name,fn in self.functions.items()}
        self.cfg=SimpleNamespace(**{name:copy.deepcopy(getattr(env.config,name))
                                    for name in CONFIG if hasattr(env.config,name)})
        mode=str(getattr(self.cfg,'mtm_horizon_payoff_denominator_mode','reference_price')).strip().lower().replace('-','_')
        if mode not in ('mwh','mwh_volume'):raise ValueError('Only original linear MWh source mode supported')
        self.support=(finite(getattr(self.cfg,CONFIG[3],-111750.)),finite(getattr(self.cfg,CONFIG[4],111750.)))

    def __enter__(self):
        if self.profile_active or sys.getprofile() is not None:
            raise RuntimeError('Require single-thread CPU and no pre-existing profiler')
        self.previous_profile=sys.getprofile();self.profile_active=True
        sys.setprofile(self._profile);return self

    def __exit__(self,*exc):
        sys.setprofile(self.previous_profile);self.profile_active=False
        if self.active is not None or self.pending is not None:
            if exc[0] is None:raise AssertionError('Unclosed original source event')

    def _proxy(self,snapshot,prices,bases):
        """No native constructor, actor, market reader, full clone or Env.step."""
        length=1008
        arr=[float('nan')]*length;bas=[float('nan')]*length
        for d,v in prices.items():
            if not 0<=d<length:raise ValueError('Delivery outside pinned episode')
            arr[d]=finite(v);bas[d]=finite(bases[d])
        proxy=SimpleNamespace(config=copy.deepcopy(self.cfg),budget=snapshot['cash_before'],
            _horizon_contracts=copy.deepcopy(snapshot['queue']),
            _horizon_settlement_price_raw=arr,_horizon_settlement_basis_component=bas)
        for name in (SETTLER,*HELPERS):
            fn=self.functions[name]
            def counted(obj,*args,_fn=fn,_name=name,**kw):
                if self.proxy_calls>=self.max_proxy_calls:raise RuntimeError('Frozen extra original-body budget exhausted')
                self.proxy_calls+=1;self.proxy_counts[_name]=self.proxy_counts.get(_name,0)+1
                if _name==SETTLER:self.proxy_settler_calls+=1
                return _fn(obj,*args,**kw)
            setattr(proxy,name,MethodType(counted,proxy))
        answer=getattr(proxy,SETTLER)(snapshot['step'])
        return proxy,answer

    def _profile(self,frame,event,arg):
        if event not in ('call','return') or frame.f_locals.get('self') is not self.env:return
        name=self.code_names.get(frame.f_code)
        if name is None:return
        if name=='step':
            if event=='call':
                if self.active is not None:raise AssertionError('Nested original step')
                self.active=dict(step=int(self.env.t),cost_receipts=[],events=[],settler_calls=0)
            else:self._finish_step()
            return
        if self.active is None:return
        if name in COSTS and event=='return':
            cost=finite(arg)
            if cost<0:raise AssertionError('Unexpected signed source cost receipt')
            self.active['cost_receipts'].append((name,cost));return
        if name==SETTLER:
            if event=='call':
                self.active['settler_calls']+=1
                if self.active['settler_calls']>1:raise AssertionError('Unexpected multiple settlement writes')
                step=int(frame.f_locals['i']);queue=copy.deepcopy(self.env._horizon_contracts)
                due=[c for c in queue if int(c.get('settle_step',10**18))<=step]
                self.pending=dict(step=step,queue=queue,cash_before=finite(self.env.budget),
                                  due_count=len(due),paid_prices={},paid_bases={})
            else:self._finish_settlement(arg)
        elif self.pending is not None and event=='return' and name in ('_horizon_settlement_price','_horizon_settlement_basis'):
            d=int(frame.f_locals['t']);target='paid_prices' if name.endswith('_price') else 'paid_bases'
            value=finite(arg);prior=self.pending[target].get(d,value)
            if prior!=value:raise AssertionError('One delivery had inconsistent paid source values')
            self.pending[target][d]=value

    def _finish_settlement(self,answer):
        snap=self.pending
        if snap is None:raise AssertionError('Original source return without call')
        self.pending=None
        if not snap['due_count']:
            if float(answer[0])!=0.:raise AssertionError('Gross without accepted due claim')
            return
        if len(self.events)>=self.max_due_events:raise RuntimeError('Frozen due-event budget exhausted')
        gold=finite(answer[0]);after=finite(self.env.budget)
        original_state=(after,copy.deepcopy(self.env._horizon_contracts))
        real_proxy,real_answer=self._proxy(snap,snap['paid_prices'],snap['paid_bases'])
        if (real_proxy.budget,real_proxy._horizon_contracts,real_answer)!=(after,self.env._horizon_contracts,answer):
            raise AssertionError('Unflipped original settlement-body parity failed')
        row=dict(step=snap['step'],due_contracts=snap['due_count'],gold_gross=gold,
                 cash_before=snap['cash_before'],cash_real_after=after,unflipped_source_parity=True)
        try:
            entries=due_entries(snap['queue'],snap['step'])
            row['own_entries']=entries
            if source_shape_gold(entries,snap['paid_prices'],snap['cash_before'])!=gold:
                raise AssertionError('Full-information source-price reconstruction failed')
            support=V2.own_support(entries,self.support)
            row['real_only_gross']=V1.inverse_branch(after,snap['cash_before'],0,1,F(1,8)).meet(support)
            row['direct_price_gross']=gold
            centers={d:self.center for d,q,p in entries}
            try:
                reflected={d:V2.reflect_public(s,centers[d],self.support) for d,s in snap['paid_prices'].items()}
            except ValueError as unsupported:
                raise Unsupported(str(unsupported)) from unsupported
            expected_shadow=source_shape_gold(entries,reflected,snap['cash_before'])
            shadow,shadow_answer=self._proxy(snap,reflected,snap['paid_bases'])
            if float(shadow_answer[0])!=expected_shadow:raise AssertionError('Shadow arithmetic source parity failed')
            api=dict(cash_before=snap['cash_before'],cash_real_after=after,
                     cash_shadow_after=finite(shadow.budget),own_entries=entries,
                     public_centers=centers,public_price_support=self.support)
            inv=restricted_estimator(api)
            if inv['real_only']!=row['real_only_gross']:
                raise AssertionError('Real-only comparator did not get identical common information')
            if abs(V1.frac(gold)+V1.frac(expected_shadow)-inv['K'])>inv['pair_error_radius']:
                raise AssertionError('Declared known-sum discrepancy bound failed')
            if not inv['paired'].contains(gold) or not inv['real_only'].contains(gold):
                raise AssertionError('Source-gross interval containment failed')
            row.update(status='AUDITED',api=api,estimator=inv,
                gold_shadow_gross=expected_shadow,direct_price_gross=gold,
                cash_shadow_after=finite(shadow.budget))
        except Unsupported as unsupported:
            # Semantic/support exclusions are retained, never redefined post hoc.
            row.update(status='ABSTAIN_UNSUPPORTED',reason=str(unsupported))
        if original_state!=(finite(self.env.budget),self.env._horizon_contracts):
            raise AssertionError('Read-only shadow service changed actual cash/queue')
        self.events.append(row);self.active['events'].append(len(self.events)-1)

    def _finish_step(self):
        step=self.active
        if step is None or self.pending is not None:raise AssertionError('Original step exit incoherent')
        if len(step['events'])>1:raise AssertionError('Multiple due events in original step')
        cost=sum((V1.frac(v) for name,v in step['cost_receipts']),F(0))
        gross=F(0)
        if step['events']:
            row=self.events[step['events'][0]];gross=V1.frac(row['gold_gross'])
            row.update(own_step_cost=cost,gold_step_booked=gross-cost)
            if 'real_only_gross' in row:
                row['real_only_step_booked']=row['real_only_gross'].shift(-cost)
            if row['status']=='AUDITED':
                row['paired_step_booked']=row['estimator']['paired'].shift(-cost)
        step.update(own_step_cost=cost,gold_gross=gross,gold_step_booked=gross-cost)
        self.steps.append(step);self.active=None

    def summary(self):
        if self.active is not None or self.pending is not None:raise AssertionError('Partial event')
        audited=[r for r in self.events if r['status']=='AUDITED'];narrow=0;widths=[];real_widths=[];errors=[]
        for r in audited:
            real=r['estimator']['real_only'];pair=r['estimator']['paired']
            rw=real.upper-real.lower;pw=pair.upper-pair.lower
            if pw<rw:narrow+=1
            widths.append(float(pw));real_widths.append(float(rw))
            errors.append(float((pair.lower+pair.upper)/2-V1.frac(r['gold_gross'])))
        return dict(native_steps_observed=len(self.steps),due_events=len(self.events),audited_events=len(audited),
            unsupported_events=len(self.events)-len(audited),strictly_narrower_than_real_only=narrow,
            real_only_widths_dkk=real_widths,paired_widths_dkk=widths,
            midpoint_signed_errors_dkk=errors,max_abs_midpoint_error_dkk=max(map(abs,errors),default=None),
            real_floor_events=sum(r['cash_real_after']==0 for r in self.events),
            both_floor_audited_events=sum(r['cash_real_after']==0 and r['cash_shadow_after']==0 for r in audited),
            proxy_original_body_calls=self.proxy_calls,proxy_original_body_counts=self.proxy_counts,
            proxy_settler_calls=self.proxy_settler_calls,
            interpretation='Conservative source-arithmetic intervals; no exact IEEE uniqueness, policy improvement or independent generalization claim.')

    def released_price_baseline(self,reader,*,now=1007,max_queries=166):
        """Optional AFTER-run public-reader reference; never enters the estimator.

        Query only released rows through the existing source getter. The +7
        delay is that simulator reader's declared convention, not a verified
        real-world publication time. No extra original Env.step is called.
        """
        prices={};rows=[]
        for row in self.events:
            entries=row.get('own_entries')
            if entries is None:
                rows.append(dict(step=row['step'],status='NO_ACCEPTED_RECEIPT'));continue
            deliveries={d for d,q,p in entries}
            release=max(d+7 for d in deliveries)
            if release>now:
                rows.append(dict(step=row['step'],status='NOT_YET_RELEASED',release=release));continue
            for d in deliveries:
                if d not in prices:
                    if len(prices)>=max_queries:raise RuntimeError('Frozen additional public-price query budget exhausted')
                    raw=finite(reader(d,now)['settlement_price'])
                    prices[d]=min(max(raw,self.support[0]),self.support[1])
            gross=source_shape_gold(entries,prices,row['cash_before'])
            if gross!=row['gold_gross']:raise AssertionError('Released-price/paid-price reconstruction mismatch')
            rows.append(dict(step=row['step'],release=release,relative_native_step_latency=release-row['step'],
                gross=gross,step_booked=V1.frac(gross)-row['own_step_cost'],status='EXACT_SOURCE_PRICE_BASELINE'))
        return dict(rows=rows,additional_public_reader_queries=len(prices),
            interpretation='Reader release convention only. Original engine ledger is already immediate; no natural secrecy claim.')
