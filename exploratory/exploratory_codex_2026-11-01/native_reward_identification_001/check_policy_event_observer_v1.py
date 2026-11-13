"""Own tiny stand-in checks only; no imported/executed external source bodies."""
import copy
from fractions import Fraction as F
import json
from pathlib import Path
from types import SimpleNamespace
from policy_event_observer_v1 import SettlementObserver,restricted_estimator,KEYS


class OwnStandIn:
    """All bodies in this class are authored for this pure adapter check."""
    def __init__(self,price=490.,cash=0.,entry=500.,q=1.):
        self.config=SimpleNamespace(mtm_return_model='horizon_settlement',
            mtm_horizon_payoff_denominator_mode='mwh',mtm_reference_price_dkk_per_mwh=500.,
            mtm_external_settlement_min_price_dkk_per_mwh=-111750.,
            mtm_external_settlement_max_price_dkk_per_mwh=111750.)
        self.t=6;self.budget=cash;self._horizon_settlement_price_raw=[price]*1008
        self._horizon_settlement_basis_component=[0.]*1008
        self._horizon_contracts=[dict(open_step=0,delivery_step=0,settle_step=6,entry_price=entry,
            exposures={KEYS[0]:500.*q},volumes_mwh={KEYS[0]:q})]
    def _is_horizon_settlement_mtm(self):return True
    def _horizon_settlement_price(self,t):return float(self._horizon_settlement_price_raw[t])
    def _horizon_settlement_basis(self,t):return float(self._horizon_settlement_basis_component[t])
    def _horizon_payoff_denominator(self,p):return 500.
    def _horizon_unit_payoff(self,p,s):return float(s-p)
    def _is_mwh_volume_payoff(self):return True
    def _horizon_contract_volume_mwh(self,p,q):return float(q/500.)
    def _mwh_volume_conversion_price(self,p):return 500.
    def _apply_market_access_fee(self,t):return 0.
    def _charge_horizon_roll_cost(self,n,t,total_volume_mwh=None):return 0.
    def _apply_collateral_cash_drag(self,t):return 0.
    def _settle_horizon_contracts_for_step(self,i):
        total=0.;per={k:0. for k in KEYS};remaining=[]
        for c in self._horizon_contracts:
            if c['settle_step']>i:remaining.append(c);continue
            d=c['delivery_step'];s=self._horizon_settlement_price(d);self._horizon_settlement_basis(d)
            p=c['entry_price'];u=self._horizon_unit_payoff(p,s)
            for k in c['exposures']:
                v=float(c['volumes_mwh'][k]*u);total+=v;per[k]+=v
        self._horizon_contracts=remaining;self.budget=max(0.,self.budget+total)
        return total,per
    def step(self):
        self._apply_market_access_fee(self.t)
        answer=self._settle_horizon_contracts_for_step(self.t)
        self._apply_collateral_cash_drag(self.t);self.t+=1
        return answer


def main():
    rows=[]
    for price,cash in ((490.,0.),(510.,0.),(500.,0.),(490.,100.),(-111750.,0.)):
        reference=OwnStandIn(price,cash);instrumented=OwnStandIn(price,cash)
        answer=reference.step();observer=SettlementObserver(instrumented)
        with observer:actual=instrumented.step()
        assert actual==answer and reference.__dict__==instrumented.__dict__
        assert len(observer.events)==1 and len(observer.steps)==1
        row=observer.events[0]
        if price==-111750.:
            assert row['status']=='ABSTAIN_UNSUPPORTED' and 'real_only_gross' in row
        else:
            assert row['status']=='AUDITED' and row['paired_step_booked'].contains(answer[0])
            leaking=dict(row['api'],gold_gross=answer[0])
            try:restricted_estimator(leaking)
            except ValueError:pass
            else:raise AssertionError('Estimator accepted a forbidden gross field')
        released=observer.released_price_baseline(lambda delivery,now:dict(settlement_price=price))
        assert released['additional_public_reader_queries']==1
        assert released['rows'][0]['gross']==answer[0]
        assert released['rows'][0]['relative_native_step_latency']==1
        rows.append(observer.summary())
    # Known K may allow both branches to floor; this is retained as an interval.
    env=OwnStandIn(price=-10.,cash=0.,entry=10.)
    observer=SettlementObserver(env,center=0.)
    with observer:env.step()
    event=observer.events[0];interval=event['estimator']['paired']
    assert event['cash_real_after']==event['cash_shadow_after']==0
    assert interval.lower<interval.upper and interval.contains(-20.)
    out=dict(status='PASS_OWN_STANDIN_OBSERVER_CHECKS',standin_cases=6,
        cases=rows,both_floor_interval=interval.record(),external_original_body_calls=0,
        native_Env_instances=0,model_calls=0,market_reads=0,
        scope='Only authored stand-in bodies executed. No native observer/runtime parity claim.')
    path=Path(__file__).with_name('POLICY_EVENT_OBSERVER_PURE_CHECKS_V2.json')
    with path.open('x',encoding='utf-8') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in out.items() if k!='cases'},indent=2))


if __name__=='__main__':main()
