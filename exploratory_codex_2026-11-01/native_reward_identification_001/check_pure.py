"""Own rational checks only; does not compile/import/call native source."""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import json
import floor_inverse as I

HERE=Path(__file__).resolve().parent


def forward(initial,pre,z,post,binary64=False):
    c=I.prefix_cash(initial,pre,binary64=binary64)
    c=max(0.,c+float(z)) if binary64 else max(F(0),c+F(z))
    for w in post: c=max(0.,c+float(w)) if binary64 else max(F(0),c+F(w))
    return c


def main():
    count=0;nonpoints=0
    for e,c,z,f,b in product(range(4),range(5),range(-5,6),range(5),range(3)):
        pre=(-c,);post=(-f,b)
        yr=forward(e,pre,z,post);ys=forward(e,pre,-z,post)
        result=I.identify(e,pre,post,yr,ys)
        assert result['gross'].contains(z)
        assert result['booked'].contains(z-c-f+b)
        for endpoint in (result['gross'].lower,result['gross'].upper):
            assert endpoint is not None
            assert forward(e,pre,endpoint,post)==yr and forward(e,pre,-endpoint,post)==ys
        nonpoints+=int(result['gross'].lower!=result['gross'].upper);count+=1
    # Concrete native ordering, not C: fee write floor happens before payoff.
    e=0;c=10;z=1;yr=forward(e,(-c,),z,());ys=forward(e,(-c,),-z,())
    r,rt=yr-e,ys-e
    canonical_tir=(max(r,rt)+c)-c
    broader=I.identify(e,(-c,),(),yr,ys)
    assert (yr,ys,canonical_tir)==(1,0,1)
    assert broader['booked'].lower==broader['booked'].upper==z-c==-9
    # Current forward cost and past matured liability are separately retained.
    ambiguous=I.identify(0,(0,),(-10,),0,0)
    assert ambiguous['gross']==I.Interval(F(-10),F(10))
    assert ambiguous['booked']==I.Interval(F(-20),F(0))
    rn_checks=0
    for e,c,z,f in product((0.,1e-9,1.,1e6),(0.,.1,10.),(-13.7,-.1,0.,.1,13.7),(0.,.3,10.)):
        yr=forward(e,(-c,),z,(-f,),True);ys=forward(e,(-c,),-z,(-f,),True)
        result=I.identify(e,(-c,),(-f,),yr,ys,binary64=True)
        assert result['gross'].contains(z)
        assert result['booked'].contains(I.frac(z)-I.frac(c)-I.frac(f));rn_checks+=1
    reflections=0;unavailable_reflections=[]
    for p,s in product((10.,100.1,500.,2000.),(-1000.,.1,100.,501.,10000.)):
        try:v=I.reflected_settlement(p,s)
        except ValueError:
            unavailable_reflections.append([p,s]);continue
        assert float(v-p)==-float(s-p);reflections+=1
    rejected=0
    try:I.reflected_settlement(2000.,-111750.)
    except ValueError:rejected+=1
    try:I.due_reflection_map([dict(settle_step=12,delivery_step=6,entry_price=500.),dict(settle_step=12,delivery_step=6,entry_price=501.)],12,{6:502.})
    except ValueError:rejected+=1
    assert rejected==2
    out=dict(status='PURE_CHECKS_PASS_NO_NATIVE_EXECUTION',exact_rational_cases=count,
      nonpoint_realized_cases=nonpoints,binary64_containment_cases=rn_checks,reflection_cases=reflections,
      fail_closed_cases=rejected,unavailable_reflections=unavailable_reflections,
      counterexample=dict(initial_cash=0,cost=10,
       real_cash=1,shadow_cash=0,canonical_TIR=1,booked=-9,ordered_interval=broader['booked'].record()),
      nonidentified_example=ambiguous['booked'].record(),native_Env_instances=0,native_body_calls=0,
      model_calls=0,market_reads=0)
    p=HERE/'PURE_CHECKS_V1.json'
    with p.open('x',encoding='utf-8') as out_file:json.dump(out,out_file,indent=2);out_file.write('\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
