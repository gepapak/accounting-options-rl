"""Independent mathematical and direct-network checks of the new audit."""
import audit_frontier as evaluator
from pathlib import Path
import json,itertools,sys
import numpy as np
from summarize_frontier import frontier

def main():
    # Exact common-sign counterexample: the same spread and fee for both trades.
    a,c=2.,.5
    for n in range(1,11):
        wrong=0
        for signs in itertools.product([-1,1],repeat=n):
            r=np.array([0.,1.,4.])*(a-c)*np.mean(np.array(signs)>0)
            wrong+=int(np.argmax(r)==2)
        assert wrong==2**n-1
    r=np.array([0.,.75,3.]);f=np.array([0.,1.25,5.])
    segments=frontier(r,f)
    assert segments[0]['winner']==2 and abs(segments[0]['end']-.6)<1e-12
    assert segments[1]['winner']==0
    # Random finite sets check every interval's maximizer and monotonicity.
    rng=np.random.default_rng(61421)
    for _ in range(200):
        f=rng.uniform(0,10,12);b=rng.uniform(-5,5,12)
        frontier(b+f,f)
    # Fresh-cohort bounds agree with original data and fixed protocol.
    summary=json.loads((Path(__file__).parent/'frontier_certification/summary.json').read_text())
    assert summary['draws']==2000 and summary['total_error']==.05
    assert all(x['all_worst_booked_certified'] and x['positive_expected_reported_certified'] for x in summary['zones'].values())
    # Re-evaluate three stored paths with the direct network, bypassing the cache.
    # Consume all 2000 RNG signs each step to preserve original path indices.
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from run_b import network_layers,greedy
    from envs_b import load_market,LEV,K0,Q1,FEE
    for label in ['DK1_s0_a0','DK2_s2_a0','DK1_s2_a1','DK2_s4_a1']:
        z=label.split('_')[0];m=load_market(z,'test','RM')
        model=PPO.load(evaluator.OLD/label/'policy.zip',device='cpu');layers=network_layers(model,'PPO')
        rng=np.random.default_rng(2026101011 if z=='DK1' else 2026101012)
        e=np.full(3,K0);book=e.copy();f=np.zeros(3);cost=np.zeros(3)
        for t in range(200,m.n):
            actions=greedy(layers,np.clip(e/K0,0,5)[:,None]);q=np.clip(LEV[actions]*Q1,-m.cap_mwh[t],m.cap_mwh[t])
            x=q*abs(m.spread[t])*(2*rng.integers(0,2,2000)[:3]-1)-FEE*np.abs(q)
            f+=np.maximum(-e-x,0);e=np.maximum(e+x,0);book+=x;cost+=FEE*np.abs(q)
        result=json.loads((Path(__file__).parent/'frontier_certification/jobs'/label/'result.json').read_text())
        for key,value in [('reported_returns',100*(e/K0-1)),('booked_returns',100*(book/K0-1)),('injections_pct',100*f/K0),('costs_pct',100*cost/K0)]:
            assert np.allclose(value,result[key][:3],atol=1e-8), (label,key,value,result[key][:3])
        print('Direct-network replay agrees:',label,flush=True)
    print('PASS: exact selection probabilities, 200 finite-policy frontiers, both joint certificates, four direct-network path replays.')
if __name__=='__main__':main()
