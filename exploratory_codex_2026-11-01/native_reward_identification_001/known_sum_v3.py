"""Pure V3 baseline correction; all V1/V2 source and results are preserved.

The paired inverse is unchanged. The public-support comparator now respects
that all accepted contracts for one delivery share the same settlement price.
"""
from fractions import Fraction as F
import floor_inverse as V1
from known_sum_v2 import public_pair_contract,reflect_public,identify_cash,ulp_bound


def own_support(own_entries,price_support=(-111750.,111750.)):
    lo,hi=map(V1.frac,price_support)
    if lo>=hi:raise ValueError('Ordered public support required')
    grouped={};entry_cash=F(0);error=F(0)
    for delivery,q,p in own_entries:
        q,p=V1.frac(q),V1.frac(p)
        grouped[delivery]=grouped.get(delivery,F(0))+q
        entry_cash+=q*p
        du=max(abs(lo-p),abs(hi-p));eu=ulp_bound(du)
        error+=abs(q)*eu+ulp_bound(abs(q)*(du+eu))+F(1,8)
    low=-entry_cash;high=-entry_cash
    for q in grouped.values():
        low+=min(q*lo,q*hi);high+=max(q*lo,q*hi)
    return V1.Interval(low-error,high+error)
