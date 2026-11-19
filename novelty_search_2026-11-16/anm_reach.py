"""Exploratory: how often does gym-anm's default reward clipping bind in ANM6Easy, and how much cost does it forgive?
Records the simulator's raw energy-loss cost and constraint penalty before the env clips them (costs_clipping = (1, 100)).
Policies: PPO's initial Gaussian policy (N(0, 1) per action dimension, clipped to the action box, as SB3 does), a policy uniform
on the box, and the zero action. Unmodified gym-anm 2.0.1; the env is only observed."""
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import gymnasium as gym
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 600


def run(policy, seed):
    env = gym.make("gym_anm:ANM6Easy-v0")
    u = env.unwrapped
    rec = []
    orig = u.simulator.transition

    def spy(*a, **k):
        out = orig(*a, **k)
        rec.append((float(out[2]), float(out[3]), bool(out[4])))
        return out
    u.simulator.transition = spy
    rng = np.random.default_rng(seed)
    o, _ = env.reset(seed=seed)
    lo, hi = env.action_space.low, env.action_space.high
    rewards, t0 = [], time.time()
    for _ in range(N):
        if policy == "initial":
            a = np.clip(rng.standard_normal(lo.shape), lo, hi).astype(np.float32)
        elif policy == "uniform":
            a = rng.uniform(lo, hi).astype(np.float32)
        else:
            a = np.clip(np.zeros(lo.shape), lo, hi).astype(np.float32)
        o, r, te, tr, _ = env.step(a)
        rewards.append(float(r))
        if te or tr:
            o, _ = env.reset()
    e = np.array([x[0] for x in rec]); p = np.array([x[1] for x in rec]); conv = np.array([x[2] for x in rec])
    c1, c2 = u.costs_clipping
    ok = conv
    el, pe = np.abs(e[ok]), p[ok]
    print(f"{policy:<8} steps {N}, {1000 * (time.time() - t0) / N:.0f} ms/step, collapsed {int((~conv).sum())}; "
          f"energy loss |e|: median {np.median(el):.3f}, clipped (>{c1}) {np.mean(el > c1):.1%}, forgiven mean {np.mean(np.maximum(el - c1, 0)):.3f}; "
          f"penalty: median {np.median(pe):.3f}, >0 {np.mean(pe > 0):.1%}, clipped (>{c2}) {np.mean(pe > c2):.1%}, forgiven mean {np.mean(np.maximum(pe - c2, 0)):.2f}; "
          f"mean reward {np.mean(rewards):.2f}")


for pol in ("initial", "zero", "uniform"):
    run(pol, 0)
