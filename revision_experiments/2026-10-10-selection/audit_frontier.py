"""Frozen-policy fresh-sign validation of subsidy-dependent model selection.

Run freeze first. No training, checkpoint selection or tuning on fresh results.
Lookup actions are certified by a global Lipschitz bound; uncertain inputs use
the original exported network. Evaluation therefore does not approximate actions.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import sys,json,hashlib,argparse,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
ROOT=Path(os.environ.get('RCT_ROOT',r'C:\Users\gepap\Documents\GitHub\rl-trading-reality-check'))
OLD=ROOT/'revision_experiments/2026-10-10/continuation_experiment/jobs'
OUT=Path(__file__).parent/'frontier_experiment'
sys.path.insert(0,str(ROOT/'LIABILITY_STUDY_2026-09-29'))
N=1000
SEEDS={'DK1':2026101001,'DK2':2026101002}
WINNERS={'DK1':'DK1_s0_a0','DK2':'DK2_s2_a0'}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def freeze():
    from datetime import datetime,timezone
    OUT.mkdir(exist_ok=True)
    manifest=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),draws=N,evaluation_seeds=SEEDS,
        discovery_winners=WINNERS,runner_sha256=sha(Path(__file__)),
        protocol='All 20 frozen L policies; fresh 1000 sign draws per zone shared across policies; full chronological test panel from index 200; starting equity K; identical reflected continuation. Discovery winners fixed from old 100 draws. Primary: fresh reported/booked means of fixed discovery winner. Secondary: re-scored leaderboard frontier S(beta)=R-beta*F, beta in [0,1], paired ranks, breakpoints, Monte Carlo errors. No training or policy re-selection for primary. No external preregistration; conditional evaluation only.',
        hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(OLD.glob('*/policy.zip'))},
        panel_hashes={z:sha(ROOT/'GENERALIZATION_STUDY_2026-09-29/data_panel'/f'panel_{z}.pkl') for z in SEEDS})
    dest=OUT/'prospective_manifest.json'
    if dest.exists():raise RuntimeError('Protocol already frozen; refusing overwrite')
    dest.write_text(json.dumps(manifest,indent=2));print('Frozen',manifest['frozen_utc'])

class CertifiedLookup:
    def __init__(self,layers):
        from run_b import greedy
        self.layers=layers;self.h=0.0001
        grid=np.arange(50001)*self.h
        # Absolute difference between final logits k and j has global Lipschitz
        # constant ||W_k-W_j||_1 times the infinity-norm prefix constant.
        prefix=1.
        for kind,w,b in layers[:-1]:
            if kind=='lin':prefix*=np.max(np.abs(w).sum(axis=1))
            elif kind not in ('tanh','relu'):raise TypeError(kind)
        kind,w,b=layers[-1];assert kind=='lin'
        lip=np.abs(w[:,None,:]-w[None,:,:]).sum(axis=2)*prefix
        logits=[]
        for start in range(0,len(grid),2048):
            x=grid[start:start+2048,None]
            for kind,w,b in layers:x=x@w.T+b if kind=='lin' else (np.tanh(x) if kind=='tanh' else np.maximum(x,0))
            logits.append(x)
        logits=np.concatenate(logits);best=logits.argmax(axis=1)
        gaps=logits[np.arange(len(grid)),best,None]-logits
        safe=gaps>lip[best]*self.h/2+1e-9
        safe[np.arange(len(grid)),best]=True
        self.actions=best;self.certified=safe.all(axis=1)
        # Independent deterministic stress grid; does not inspect evaluation signs.
        rng=np.random.default_rng(32145);test=rng.uniform(0,5,10000)
        assert np.array_equal(self(test)[0],greedy(layers,test[:,None]))

    def __call__(self,x):
        from run_b import greedy
        x=np.clip(np.asarray(x),0,5);index=np.rint(x/self.h).astype(int)
        actions=self.actions[index].copy();bad=~self.certified[index]
        if bad.any():actions[bad]=greedy(self.layers,x[bad,None])
        return actions,int(bad.sum())

def evaluate(label):
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from run_b import network_layers
    from envs_b import load_market,LEV,Q1,FEE,K0
    zone=label.split('_')[0];dest=OUT/'jobs'/label;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'result.json').exists():return label+' cached'
    start=time.time();policy=PPO.load(OLD/label/'policy.zip',device='cpu')
    lookup=CertifiedLookup(network_layers(policy,'PPO'))
    m=load_market(zone,'test','RM');rng=np.random.default_rng(SEEDS[zone])
    e=np.full(N,K0);book=e.copy();f=np.zeros(N);cost=np.zeros(N);exposure=np.zeros(N);fallback=0
    for t in range(200,m.n):
        actions,bad=lookup(e/K0);fallback+=bad
        q=np.clip(LEV[actions]*Q1,-m.cap_mwh[t],m.cap_mwh[t])
        gross=q*abs(m.spread[t])*(2*rng.integers(0,2,N)-1)
        c=FEE*np.abs(q);x=gross-c;f+=np.maximum(-e-x,0)
        e=np.maximum(e+x,0);book+=x;cost+=c;exposure+=np.abs(LEV[actions])
    assert np.allclose(e-book,f,atol=1e-7)
    result=dict(label=label,region=zone,draws=N,sign_seed=SEEDS[zone],seconds=time.time()-start,
        reported_returns=(100*(e/K0-1)).tolist(),booked_returns=(100*(book/K0-1)).tolist(),
        injections_pct=(100*f/K0).tolist(),costs_pct=(100*cost/K0).tolist(),
        mean_leverage=(exposure/(m.n-200)).tolist(),certified_grid_fraction=float(lookup.certified.mean()),
        exact_network_fallback_fraction=fallback/(N*(m.n-200)))
    (dest/'result.json').write_text(json.dumps(result,indent=2))
    return label+f' {result["seconds"]:.1f}s R={np.mean(result["reported_returns"]):.2f} B={np.mean(result["booked_returns"]):.2f}'

def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    if args.freeze:return freeze()
    manifest=json.loads((OUT/'prospective_manifest.json').read_text())
    assert manifest['runner_sha256']==sha(Path(__file__))
    for name,digest in manifest['hashes'].items():assert sha(ROOT/name)==digest
    for z,digest in manifest['panel_hashes'].items():assert sha(ROOT/'GENERALIZATION_STUDY_2026-09-29/data_panel'/f'panel_{z}.pkl')==digest
    labels=[p.parent.name for p in sorted(OLD.glob('*/policy.zip'))];assert len(labels)==20
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs=[pool.submit(evaluate,label) for label in labels]
        for job in as_completed(jobs):print(job.result(),flush=True)
if __name__=='__main__':main()
