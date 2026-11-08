"""Pure successor: interval inversion with a PUBLIC known paired-gross sum.

Original hand V1 files/results remain unchanged. This module calls no source.
"""
from fractions import Fraction as F
import math
import floor_inverse as V1


def ulp_bound(bound):
    """Conservative absolute RN error using one full ulp of an upper bound."""
    b=abs(V1.frac(bound));v=float(b)
    if not math.isfinite(v):raise ValueError('Finite arithmetic domain required')
    if F.from_float(v)<b:v=math.nextafter(v,math.inf)
    return F.from_float(math.ulp(v))


def public_pair_contract(own_entries,centers,price_support=(-111750.,111750.)):
    """own_entries: (delivery,q_MWh,own_entry_price), all observed own fills.

    K=2 sum q*(R-entry) is exact real algebra on represented own inputs.
    eta bounds source-G_real+source-G_shadow-K for original float subtraction,
    products and source-order accumulation. Summation operands/exact sums must
    independently stay inside +-2**50, as checked in the native evaluator.
    Reflection is RN(2R-S); returned shadow prices MUST pass source support.
    No private S, private volume, true G or conditional moment is an input.
    """
    lo,hi=map(V1.frac,price_support)
    if lo>=hi:raise ValueError('Ordered public support required')
    k=F(0);branch=F(0);reflect_error=F(0);n=0
    for delivery,q,p in own_entries:
        q,p=V1.frac(q),V1.frac(p);r=V1.frac(centers[delivery]);n+=1
        diff=max(abs(lo-p),abs(hi-p));eu=ulp_bound(diff)
        ep=ulp_bound(abs(q)*(diff+eu))
        branch+=abs(q)*eu+ep+F(1,8)
        er=F(0) if r==0 else ulp_bound(max(abs(2*r-lo),abs(2*r-hi)))
        reflect_error+=abs(q)*er
        k+=2*q*(r-p)
    return dict(K=k,pair_error_radius=2*branch+reflect_error,term_count=n,
                roundoff_domain=2**50,price_support=(lo,hi))


def reflect_public(settlement,center,price_support=(-111750.,111750.)):
    s,r=float(settlement),float(center);lo,hi=map(float,price_support)
    if not all(math.isfinite(v) for v in (s,r,lo,hi)):raise ValueError('Finite represented prices')
    if not lo<=s<=hi:raise ValueError('Real source getter outside public support')
    shadow=float(2.*r-s)
    if not math.isfinite(shadow) or not lo<=shadow<=hi:
        raise ValueError('Public-center reflected source price outside support')
    return shadow


def identify_cash(cash_before,real_after,shadow_after,contract,post_writes=()):
    a,t=V1.affine_after_settlement(cash_before,post_writes)
    cash_error=F(1+len(post_writes),8)
    real=V1.inverse_branch(real_after,a,t,1,cash_error)
    shadow=V1.inverse_branch(shadow_after,a+contract['K'],t,-1,
                             cash_error+contract['pair_error_radius'])
    paired=real.meet(shadow)
    return dict(real_only=real,paired=paired,cash_error=cash_error,
                pair_error=contract['pair_error_radius'],K=contract['K'])


def own_support(own_entries,price_support=(-111750.,111750.)):
    """Same public S support and own fills; no twin or hidden magnitudes."""
    lo,hi=map(V1.frac,price_support);low=high=error=F(0)
    for delivery,q,p in own_entries:
        q,p=V1.frac(q),V1.frac(p)
        low+=min(q*(lo-p),q*(hi-p));high+=max(q*(lo-p),q*(hi-p))
        du=max(abs(lo-p),abs(hi-p));eu=ulp_bound(du)
        error+=abs(q)*eu+ulp_bound(abs(q)*(du+eu))+F(1,8)
    return V1.Interval(low-error,high+error)
