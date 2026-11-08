"""Pure own-code generalized-pair checks; no native/module/data execution."""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import json
import floor_inverse as V1
import known_sum_v2 as V2

HERE=Path(__file__).resolve().parent


def source_shape(entries,prices):
    total=0.;partial=[]
    for d,q,p in entries:
        u=float(prices[d]-p);term=float(q*u);before=total;total+=term
        assert max(abs(F.from_float(before)),abs(F.from_float(term)),abs(F.from_float(before)+F.from_float(term)))<=2**50
        partial.append(total)
    return total


def main():
    checks=0;gains=0
    # Repeated delivery with conflicting entries, signed quantities and netQ0
    # are included; no common-entry assumption is made by the inverse.
    portfolios=[[(0,1.,500.)],[(0,1.,500.),(0,-1.,600.)],
                [(0,.1,100.1),(0,.3,500.)],[(0,1.,500.),(1,-.5,600.)]]
    for entries,r,e,s in product(portfolios,(0.,500.),(0.,10.,1e6),(-1000.,.1,500.,10000.)):
        prices={d:float(s) for d,q,p in entries};centers={d:r for d,q,p in entries}
        reflected={d:V2.reflect_public(v,centers[d]) for d,v in prices.items()}
        z=source_shape(entries,prices);zs=source_shape(entries,reflected)
        contract=V2.public_pair_contract(entries,centers)
        assert abs(F.from_float(z)+F.from_float(zs)-contract['K'])<=contract['pair_error_radius']
        yr=max(0.,e+z);ys=max(0.,e+zs)
        inverse=V2.identify_cash(e,yr,ys,contract)
        support=V2.own_support(entries)
        real=inverse['real_only'].meet(support);paired=inverse['paired'].meet(support)
        assert paired.contains(z) and real.contains(z)
        assert paired.lower>=real.lower and paired.upper<=real.upper
        gains+=int(paired.upper-paired.lower<real.upper-real.lower);checks+=1
    # Pure full-cash example: a negative financial settlement is floored, then
    # a funding charge and a positive independent battery receipt are written.
    # Real-only cash is compatible with a positive booked total; pair excludes it.
    k0=dict(K=F(0),pair_error_radius=F(0))
    result=V2.identify_cash(0.,12.,22.,k0,post_writes=(-20.,12.))
    real_booked=result['real_only'].shift(-8.)
    paired_booked=result['paired'].shift(-8.)
    assert real_booked.upper>0 and paired_booked.upper<0
    out=dict(status='PASS_PURE_GENERALIZED_PAIR',cases=checks,strict_interval_narrowing_cases=gains,
       original_body_calls=0,Env_instances=0,model_calls=0,market_reads=0,
       positive_reported_negative_booked_pure_example=dict(real_cash=12.,shadow_cash=22.,
         known_post_writes=[-20.,12.],source_booked_total=-38.,
         real_only_booked=real_booked.record(),paired_booked=paired_booked.record()),
       scope='own-code arithmetic only; no native reachability or performance claim')
    with (HERE/'KNOWN_SUM_PURE_CHECKS_V2.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
