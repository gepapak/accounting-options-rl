"""Exploratory ordered-cash inverse; pure code, no native imports.

This is a floor-affine inverse, not a new generic inversion algorithm.
Input receipts are starting public cash and known own signed cash writes.
Only one aggregate settlement write is unknown; its shadow flips its sign.
Returned intervals refer to source booked flow, not the shaped agent reward.
"""
from dataclasses import dataclass
from fractions import Fraction as F
import math


def frac(x):
    if isinstance(x,F): return x
    if isinstance(x,int): return F(x)
    if type(x) is float and math.isfinite(x): return F.from_float(x)
    raise ValueError("Finite represented input required")


@dataclass(frozen=True)
class Interval:
    lower: F | None
    upper: F | None

    def meet(self,other):
        ls=[v for v in (self.lower,other.lower) if v is not None]
        us=[v for v in (self.upper,other.upper) if v is not None]
        out=Interval(max(ls) if ls else None,min(us) if us else None)
        if out.lower is not None and out.upper is not None and out.lower>out.upper:
            raise ValueError("Inconsistent cash/sign/cost contract")
        return out

    def shift(self,k):
        k=frac(k)
        return Interval(None if self.lower is None else self.lower+k,
                        None if self.upper is None else self.upper+k)

    def contains(self,z):
        z=frac(z)
        return (self.lower is None or self.lower<=z) and (self.upper is None or z<=self.upper)

    def record(self):
        return {"lower":None if self.lower is None else str(self.lower),
                "upper":None if self.upper is None else str(self.upper),
                "point":self.lower is not None and self.lower==self.upper}


def prefix_cash(initial,known_writes,binary64=False):
    """Reproduce known pre-settlement writes in the declared order."""
    value=float(initial) if binary64 else frac(initial)
    for w in known_writes:
        value=max(0.,value+float(w)) if binary64 else max(F(0),value+frac(w))
    return value


def affine_after_settlement(cash_before_settlement,post_writes):
    """Real arithmetic: final cash is max(z+A,T), with known A,T."""
    a=frac(cash_before_settlement); threshold=F(0)
    if a<0: raise ValueError("Public source cash is nonnegative")
    for w in post_writes:
        w=frac(w);a+=w;threshold=max(F(0),threshold+w)
    return a,threshold


def inverse_branch(cash,a,threshold,sign,epsilon=0):
    """Interval inverse with a declared additive final observation error."""
    if sign not in (-1,1): raise ValueError("Fixed real/shadow signs required")
    y=frac(cash); e=frac(epsilon)
    if e<0 or y<0: raise ValueError("Nonnegative error and observed cash required")
    lo=max(F(0),y-e);hi=y+e
    if hi<threshold: raise ValueError("Cash below the map's possible floor")
    u=hi-a
    l=lo-a if lo>threshold else None
    return Interval(l,u) if sign==1 else Interval(-u,None if l is None else -l)


def identify(initial,pre_writes,post_writes,real_cash,shadow_cash,*,binary64=False,domain_bound=2**50):
    """Identify aggregate represented settlement z, then booked z+sum(w).

    Exact real arithmetic gives the sharp set for the specified cash maps.
    Binary64 mode reproduces the known pre-prefix exactly and relaxes each
    remaining cash-write error by 1/8 DKK, conditional on every intermediate
    operand and pre-floor result staying in [-2**50,2**50]. This is a closed
    conservative interval, NOT a sharp discrete binary64 identified set.
    The caller must independently enforce/audit that operand-domain premise.
    No statistical coverage or conditional market-fairness claim is made.
    """
    if binary64 and domain_bound!=2**50: raise ValueError("Only declared RN domain supported")
    pre=prefix_cash(initial,pre_writes,binary64=binary64)
    a,t=affine_after_settlement(pre,post_writes)
    eps=F(1+len(post_writes),8) if binary64 else F(0)
    z=inverse_branch(real_cash,a,t,1,eps).meet(inverse_branch(shadow_cash,a,t,-1,eps))
    booked=z.shift(sum((frac(w) for w in (*pre_writes,*post_writes)),F(0)))
    return {"gross":z,"booked":booked,"cash_before_settlement":frac(pre),
            "A":a,"T":t,"cash_error_radius":eps,
            "scope":"RN conservative source-cash interval" if binary64 else "sharp real-valued floor-map set"}


def reflected_settlement(entry,settlement,floor=-111750.,ceiling=111750.):
    """Choose an in-support binary64 shadow preserving RN(S-P) antisymmetry.

    The trusted shadow intervention can read its private price; the estimator
    does not receive this price, payoff, or gold ledger. Reject if no tested
    adjacent representative has the exact opposite source unit payoff.
    """
    p,s=float(entry),float(settlement)
    if not all(math.isfinite(x) for x in (p,s,floor,ceiling)): raise ValueError("Finite price required")
    u=s-p; center=float(2.*p-s)
    candidates=[center]
    for direction in (-math.inf,math.inf):
        v=center
        for _ in range(4):
            v=math.nextafter(v,direction);candidates.append(v)
    for v in candidates:
        if floor<=v<=ceiling and float(v-p)==-u: return v
    raise ValueError("No verified representable in-support sign reflection")


def due_reflection_map(contracts,step,settlements):
    entries={}
    for c in contracts:
        if int(c['settle_step'])>step: continue
        delivery=int(c.get('delivery_step',c.get('open_step',step)))
        entry=float(c['entry_price'])
        if delivery in entries and entries[delivery]!=entry:
            raise ValueError("One exogenous delivery price cannot reflect about two entry prices")
        entries[delivery]=entry
    return {d:reflected_settlement(p,settlements[d]) for d,p in entries.items()}
