"""Fresh pure comparator check; does not overwrite the V2 96-case receipt."""
from fractions import Fraction as F
from itertools import product
import json
from pathlib import Path
import floor_inverse as V1
import known_sum_v2 as V2
import known_sum_v3 as V3
from check_known_sum_v2 import source_shape


def main():
    n=narrow_loose=narrow_tight=improved_support=0
    portfolios=[[(0,1.,500.)],[(0,1.,500.),(0,-1.,600.)],
                [(0,.1,100.1),(0,.3,500.)],[(0,1.,500.),(1,-.5,600.)]]
    for entries,r,e,s in product(portfolios,(0.,500.),(0.,10.,1e6),(-1000.,.1,500.,10000.)):
        prices={d:float(s) for d,q,p in entries};centers={d:r for d,q,p in entries}
        reflected={d:V3.reflect_public(v,centers[d]) for d,v in prices.items()}
        z=source_shape(entries,prices);zs=source_shape(entries,reflected)
        inv=V3.identify_cash(e,max(0.,e+z),max(0.,e+zs),V3.public_pair_contract(entries,centers))
        loose=V2.own_support(entries);tight=V3.own_support(entries)
        assert loose.contains(z) and tight.contains(z)
        assert tight.lower>=loose.lower and tight.upper<=loose.upper
        rl=inv['real_only'].meet(loose);pl=inv['paired'].meet(loose)
        rt=inv['real_only'].meet(tight);pt=inv['paired'].meet(tight)
        assert rt.contains(z) and pt.contains(z)
        narrow_loose+=int(pl.upper-pl.lower<rl.upper-rl.lower)
        narrow_tight+=int(pt.upper-pt.lower<rt.upper-rt.lower)
        improved_support+=int(tight.upper-tight.lower<loose.upper-loose.lower);n+=1
    # Exact-real ambiguity and equality boundary, outside any source execution.
    exact=0
    for a,t,k in product((F(-2),F(0),F(2)),(F(0),F(3)),(F(-10),F(0),F(10))):
        h=t-a
        if k<=2*h:
            identified=V1.inverse_branch(t,a,t,1).meet(V1.inverse_branch(t,a+k,t,-1))
            assert identified==V1.Interval(k-h,h)
            assert identified.lower==identified.upper if k==2*h else identified.lower<identified.upper
            exact+=1
    # Same delivery, opposite quantities: the ideal gross is the known entry
    # difference. The new common-price support captures this cancellation.
    netzero=V3.own_support([(0,1.,500.),(0,-1.,600.)])
    assert netzero.contains(100.) and netzero.upper-netzero.lower<1.
    out=dict(status='PASS_PURE_TIGHT_COMMON_DELIVERY_SUPPORT',cases=n,
        strict_pair_gains_v2_loose_support=narrow_loose,
        strict_pair_gains_v3_common_delivery_support=narrow_tight,
        tightened_support_cases=improved_support,exact_floor_set_cases=exact,
        net_zero_gross_support=netzero.record(),external_original_body_calls=0,
        Env_instances=0,model_calls=0,market_reads=0,
        scope='Arithmetic feasibility only; no event-frequency, new AI algorithm, policy or native outcome claim.')
    path=Path(__file__).with_name('KNOWN_SUM_PURE_CHECKS_V3.json')
    with path.open('x',encoding='utf-8') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
