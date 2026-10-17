"""Prospective revision experiment: identical reflected state/continuation, two rewards.
No registered study is modified. Outputs include policies and normalization statistics.
"""
from pathlib import Path
import os, sys, json, time, hashlib, argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
ROOT=Path(r'C:\Users\gepap\Documents\GitHub\rl-trading-reality-check')
sys.path.insert(0,str(ROOT/'LIABILITY_STUDY_2026-09-29'))
OUT=Path(__file__).parent/'continuation_experiment'
import numpy as np

def job(region,seed,alpha,smoke=False):
    import torch, gymnasium as gym
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecEnv,VecNormalize
    from envs_b import load_market,LEV,K0,Q1,FEE,E0_TRAIN
    from run_b import network_layers,greedy
    from envs import EPISODE
    label=f'{region}_s{seed}_a{alpha}'
    folder=OUT/('smoke' if smoke else 'jobs')/label
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists(): return label+' cached'
    start=time.time()
    train=load_market(region,'train','RM')
    class Batch(VecEnv):
        def __init__(self):
            self.rng=np.random.default_rng(seed*100+71234)
            self.t=np.zeros(8,dtype=int); self.age=self.t.copy(); self.e=np.zeros(8)
            super().__init__(8,gym.spaces.Box(0,5,(1,),np.float32),gym.spaces.Discrete(9))
        def obs(self): return np.clip(self.e/K0,0,5).astype(np.float32)[:,None]
        def restart(self,indices):
            self.t[indices]=self.rng.integers(200,train.n-EPISODE,size=len(indices))
            self.age[indices]=0
            self.e[indices]=K0*np.exp(self.rng.uniform(np.log(.05),np.log(2),len(indices)))
        def reset(self): self.restart(np.arange(8)); return self.obs()
        def step_async(self,actions): self.actions=actions
        def step_wait(self):
            q=np.clip(LEV[self.actions]*Q1,-train.cap_mwh[self.t],train.cap_mwh[self.t])
            eps=2*self.rng.integers(0,2,8)-1
            x=q*np.abs(train.spread[self.t])*eps-FEE*np.abs(q)
            f=np.maximum(-self.e-x,0)
            reward=100*(x+(1-alpha)*f)/K0
            self.e=np.maximum(self.e+x,0); self.age+=1; self.t+=1
            done=self.age>=EPISODE
            infos=[{} for _ in range(8)]
            for i in np.flatnonzero(done): infos[i]['terminal_observation']=self.obs()[i].copy()
            self.restart(np.flatnonzero(done))
            return self.obs(),reward.astype(np.float32),done,infos
        def close(self): pass
        def get_attr(self,name,indices=None): return [getattr(self,name,None)]*8
        def set_attr(self,name,value,indices=None): setattr(self,name,value)
        def env_method(self,name,*args,indices=None,**kwargs): return [getattr(self,name)(*args,**kwargs)]*8
        def env_is_wrapped(self,wrapper_class,indices=None): return [False]*8
    v=VecNormalize(Batch(),norm_obs=False,norm_reward=True,gamma=.99)
    model=PPO('MlpPolicy',v,seed=seed,n_steps=1024,batch_size=256,policy_kwargs={'net_arch':[64,64]},device='cpu',verbose=0)
    model.learn(total_timesteps=16384 if smoke else 1000000)
    model.save(folder/'policy'); v.save(folder/'normalization.pkl')
    layers=network_layers(model,'PPO')
    grid=np.array([0,.02,.05,.1,.25,.5,1,2,3])
    predictions=greedy(layers,grid[:,None])
    assert np.array_equal(predictions,model.predict(grid.astype(np.float32)[:,None],deterministic=True)[0])
    split='train' if smoke else 'test'
    market=load_market(region,split,'RM')
    n=3 if smoke else 100
    rng=np.random.default_rng(810010 if region=='DK1' else 810020)
    equity=np.full(n,K0); ledger=equity.copy(); injections=np.zeros(n); cost=np.zeros(n); exposure=np.zeros(n)
    for t in range(200,market.n):
        action=greedy(layers,np.clip(equity/K0,0,5)[:,None])
        q=np.clip(LEV[action]*Q1,-market.cap_mwh[t],market.cap_mwh[t])
        x=q*abs(market.spread[t])*(2*rng.integers(0,2,n)-1)-FEE*np.abs(q)
        injections+=np.maximum(-equity-x,0)
        ledger+=x; equity=np.maximum(equity+x,0)
        cost+=FEE*np.abs(q); exposure+=np.abs(LEV[action])
    assert np.allclose(equity-ledger,injections,atol=1e-7)
    result=dict(region=region,seed=seed,alpha=alpha,split=split,steps=model.num_timesteps,
                seconds=time.time()-start,grid=grid.tolist(),leverage=np.abs(LEV[predictions]).tolist(),
                reported_returns=(100*(equity/K0-1)).tolist(),booked_returns=(100*(ledger/K0-1)).tolist(),
                injections_pct=(100*injections/K0).tolist(),costs_pct=(100*cost/K0).tolist(),
                mean_leverage=(exposure/(market.n-200)).tolist())
    (folder/'result.json').write_text(json.dumps(result,indent=2))
    return label+f' {result["seconds"]:.1f}s low={result["leverage"][3]} high={result["leverage"][-2]}'

def main():
    p=argparse.ArgumentParser(); p.add_argument('--smoke',action='store_true'); p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    jobs=[(r,s,a,args.smoke) for s in range(1 if args.smoke else 5) for r in (['DK1'] if args.smoke else ['DK1','DK2']) for a in [0,1]]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending=[pool.submit(job,*j) for j in jobs]
        for f in as_completed(pending): print(f.result(),flush=True)
if __name__=='__main__': main()
