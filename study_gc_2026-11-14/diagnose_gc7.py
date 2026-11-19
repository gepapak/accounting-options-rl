"""Study GC, after the run (post hoc, not registered): why GC7 failed. Replays the saved policy of crypto LINE seed 10 on the shared test
bank in the published simulator and prints every step on which the reading misses the booked step profit by more than 0.001 USD.
Read-only. Run from this folder with the study environment; output kept in gc7_diagnosis.txt."""
import copy
import os
import sys
import warnings

warnings.filterwarnings("ignore")
STUDY = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, STUDY)
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import envs_gc as G

name = sys.argv[1] if len(sys.argv) > 1 else "crypto__LINE__PPO__s10"
market = name.split("__")[0]
model = PPO.load(os.path.join(STUDY, "results_gc", "policies", name + ".zip"), device="cpu")
vn = VecNormalize.load(os.path.join(STUDY, "results_gc", "policies", name + "__vecnormalize.pkl"), DummyVecEnv([lambda: G.GCEnv(market, "ORIG", False, 0)]))
vn.training = False
env = G.GCEnv(market, "ORIG", train=False, seed=0, audit=True)
bank = G.test_bank(market)
sym = G.MARKETS[market]["symbol"]
bad = 0
for k, close in enumerate(bank):
    o, _ = env.reset(options={"path": close, "balance": G.BALANCE})
    done = False
    while not done:
        e = env.env.unwrapped
        tick = e._current_tick
        pre = e.simulator
        pre_orders = [(od.type.name, od.volume, od.entry_price, od.profit) for od in pre.orders]
        pre_state = dict(balance=pre.balance, equity=pre.equity, margin=pre.margin, level=pre.margin_level)
        a = model.predict(vn.normalize_obs(o[None, :]), deterministic=True)[0][0]
        line = env.line_reward(np.asarray(a, dtype=np.float64))
        prev = env.prev_ledger
        o, r, te, tr, info = env.step(a)
        sim = e.simulator
        booked = G.ledger(sim, env.e0) - prev
        if abs(line - booked) > 1e-3:
            bad += 1
            p0, p1 = float(e.prices[sym][tick, 0]), float(e.prices[sym][tick + 1, 0])
            print(f"path {k} tick {tick}: line {line:.4f} booked {booked:.4f} published {100 * r:.4f}  p0 {p0:.2f} p1 {p1:.2f} move {100 * (p1 / p0 - 1):+.3f}%")
            print(f"   action {np.round(a, 3)}  pre {pre_state}  pre orders {pre_orders}")
            print(f"   post orders {[(od.type.name, od.volume, od.entry_price, round(od.profit, 4)) for od in sim.orders]}  closed this path "
                  f"{[(od.type.name, od.volume, od.entry_price, od.exit_price, round(od.profit, 4)) for od in sim.closed_orders][-2:]}")
            print(f"   post balance {sim.balance:.4f} equity {sim.equity:.4f} margin {sim.margin:.4f}")
        done = te or tr
print("steps with an error above 0.001 USD:", bad)
