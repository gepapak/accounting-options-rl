"""What the engine's accounting pays, with no learning: constant positions in the placebo market. Exploratory (development of Study EXT)."""
import sys, time
import numpy as np
from envs_ext import ExtEnv, POSITIONS, W0, split_moves, daily_returns
close, ret, dates = daily_returns()
print("days", len(ret), dates[0], dates[-1], "train", len(split_moves('train')), "test", len(split_moves('test')))
for s in ("train","test"):
    m = np.abs(split_moves(s)); print(s, "mean|r| %.4f  q90 %.4f  q99 %.4f  max %.4f  P(|r|>0.1) %.4f  P(|r|>0.2) %.4f" % (m.mean(), np.quantile(m,.9), np.quantile(m,.99), m.max(), (m>.1).mean(), (m>.2).mean()))
for split in ("train","test"):
    env = ExtEnv(split, "placebo", "REP", seed=1)
    print(f"--- {split}: constant requested position, 3000 placebo episodes: reported, booked (percent of W0), bust share, mean steps")
    for a,pos in enumerate(POSITIONS):
        rep=[];boo=[];bust=[];steps=[]
        t0=time.time()
        for ep in range(3000):
            env.reset()
            R=B=0.0;d=tr=False;n=0
            while not(d or tr):
                _,_,d,tr,i=env.step(a); R+=i['reported'];B+=i['booked'];n+=1
            rep.append(R);boo.append(B);bust.append(i['bust']);steps.append(n)
        print(f"  {pos:>4d}: reported {np.mean(rep):+8.2f}  booked {np.mean(boo):+8.2f} (se {np.std(boo)/np.sqrt(3000):.2f})  bust {np.mean(bust):.3f}  steps {np.mean(steps):.1f}  {3000*30/(time.time()-t0)/1000:.1f}k steps/s")
