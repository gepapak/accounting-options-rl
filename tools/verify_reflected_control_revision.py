"""Independent numerical check of the explicit reflected-control candidate.

Checks the boundary conditions and HJB inequality for intermediate controls,
including partial charges. Does not alter registered numerical benchmarks.
"""
import numpy as np
from scipy.optimize import brentq


def candidate(rho, beta):
    r1=(rho+np.sqrt(rho*rho+4))/2
    r2=-1/r1
    def coeff(h): return -1/(r2*(-np.expm1((r2-r1)*h)))
    def equation(h): return coeff(h)*np.exp(r2*h)*(1-r2/r1)-rho/beta
    hi=1.
    while equation(hi)>0: hi*=2
    h=brentq(equation,1e-10,hi,xtol=1e-12)
    b=coeff(h)
    def values(x):
        x=np.asarray(x)
        first=b*(-r2/r1)*np.exp(r2*h-r1*(h-x))
        second=b*np.exp(r2*x)
        w=-rho/beta+first+second
        dw=r1*first+r2*second
        ddw=r1*r1*first+r2*r2*second
        return w,dw,ddw
    return h,values


def main():
    count=0
    for rho in np.logspace(-4,2,13):
        for beta in (1.,.5,.1,.01):
            h,values=candidate(rho,beta)
            w,dw,ddw=values(np.linspace(0,h,101))
            scale=max(1.,rho/beta)
            assert abs(dw[0]+1)<1e-8*scale
            assert abs(w[-1])<1e-8*scale and abs(dw[-1])<1e-8*scale
            assert w.min()>-1e-8*scale and ddw.min()>0
            assert dw.max()<1e-8*scale
            for u in np.linspace(0,1,21):
                hamiltonian=beta*u*u*ddw-rho*u*(beta*dw+1)
                assert np.max(hamiltonian-beta*w)<1e-8*scale
            assert np.max(np.abs(beta*ddw-rho*(beta*dw+1)-beta*w))<1e-8*scale
            count+=1
    for beta in (1.,.5,.1,.01):
        rho=1e-8
        h,_=candidate(rho,beta)
        assert abs(np.exp(-h)/(rho/(2*beta))-1)<2e-5
    print(f'PASS: {count} fee/charge candidates; reflection, smooth fit, positivity, convexity,')
    print('      HJB inequalities for21 controls at101 states, and small-fee asymptotics.')
    print('Numerical verification of the explicit candidate; not a substitute for the proof.')


if __name__=='__main__': main()
