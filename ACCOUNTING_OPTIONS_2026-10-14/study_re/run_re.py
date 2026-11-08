"""Study RE runner (PREREGISTRATION_RE.md): does learned exploitation of the floor vanish where the market cannot reach it?

Study F's design (placebo DK1 market, fixed sizing, equity as the only input, PPO with 8 environments, reward normalization,
10^6 steps) under two rules, FL and ZF, with the market's reach scaled by c in SCALES. Same seed and scale across the two rules
form a pair.

    python run_re.py --workers 8
    python run_re.py --smoke        # development check: tiny budget, one seed, two scales; excluded
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LIAB = HERE.parents[1] / "LIABILITY_STUDY_2026-09-29"
OUT = HERE / "results_re"
RULES = ("FL", "ZF")
SEEDS = (0, 1, 2, 3, 4)
STEPS = 1_000_000
N_ENVS = 8
N_DRAWS = 20
EGRID = (0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0)


def job_path(rule, scale, seed, smoke=False) -> Path:
    return OUT / ("jobs_smoke" if smoke else "jobs") / f"{rule}__c{scale:g}__PPO__s{seed}.json"


def re_job(rule: str, scale: float, seed: int, steps: int, smoke: bool) -> str:
    out = job_path(rule, scale, seed, smoke)
    if out.exists():
        return f"skip {out.name}"
    sys.path[:0] = [str(LIAB), str(HERE)]
    import gymnasium as gym
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from envs_b import LEV, outcome
    from envs_re import ScaledLiabilityEnv, load_market
    from run_b import greedy, network_layers
    t0 = time.time()
    mtr = load_market("DK1", "train", "NE")
    split = "train" if smoke else "test"

    class EquityOnly(gym.ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            self.observation_space = gym.spaces.Box(0, 5, shape=(1,), dtype=np.float32)

        def observation(self, obs):
            return np.array([obs[6]], dtype=np.float32)   # reported equity / K0, clipped to [0, 5]

    class EpisodeCounter(gym.Wrapper):
        """Counts training episodes, and those in which the floor was reached (ZF) or the account closed (FL)."""
        def __init__(self, env):
            super().__init__(env)
            self.episodes = 0
            self.floor_episodes = 0

        def step(self, action):
            obs, r, term, trunc, info = self.env.step(action)
            if term or trunc:
                self.episodes += 1
                self.floor_episodes += int(self.env.unwrapped.floor_bound or self.env.unwrapped.closed)
            return obs, r, term, trunc, info

    bases = []

    def make(i):
        def f():
            b = ScaledLiabilityEnv(mtr, rule, scale, train=True, seed=seed * 100 + i)
            bases.append(b)
            return EpisodeCounter(EquityOnly(b))
        return f

    venv = DummyVecEnv([make(i) for i in range(N_ENVS)])
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=0.99)
    model = PPO("MlpPolicy", venv, seed=seed, n_steps=1024, batch_size=256, policy_kwargs=dict(net_arch=[64, 64]),
                device="cpu", verbose=0)
    model.learn(total_timesteps=steps)
    layers = network_layers(model, "PPO")
    counters = [e for e in venv.venv.envs]
    floor_steps = int(sum(b.floor_steps for b in bases))
    floor_episodes = int(sum(c.floor_episodes for c in counters))
    episodes = int(sum(c.episodes for c in counters))
    # exactness check of the NumPy forward pass
    chk = EquityOnly(ScaledLiabilityEnv(mtr, rule, scale, train=True, seed=seed + 1000))
    o, _ = chk.reset()
    for _ in range(300):
        assert int(greedy(layers, np.asarray(o)[None, :])[0]) == int(model.predict(o, deterministic=True)[0]), "fast policy mismatch"
        o, _, d, _, _ = chk.step(chk.action_space.sample())
        if d:
            o, _ = chk.reset()
    lev = {f"{e:g}": float(abs(LEV[greedy(layers, np.array([[min(e, 5.0)]]))[0]])) for e in EGRID}
    # a digest of every policy parameter, to test identity between the FL and ZF agents of a pair
    digest = None
    try:
        import hashlib
        flat = np.concatenate([p.detach().cpu().numpy().ravel() for p in model.policy.parameters()])
        digest = hashlib.sha256(flat.tobytes()).hexdigest()
    except Exception:  # noqa: BLE001
        pass
    # placebo-market evaluation from equity K0 over sign draws of the evaluation period, at the same scale
    draws = []
    for k in range(3 if smoke else N_DRAWS):
        env = ScaledLiabilityEnv(load_market("DK1", split, "NE", draw=k), rule, scale, train=False)
        o, _ = env.reset(options={"e0": 1.0})
        done = False
        while not done:
            o, _, done, _, _ = env.step(int(greedy(layers, np.asarray(o)[6:7][None, :])[0]))
        draws.append(outcome(env))
    res = dict(rule=rule, scale=scale, algo="PPO", seed=seed, steps=steps, eval_split=split, lev=lev,
               X=lev["0.02"] - lev["2"], policy_sha256=digest,
               train_episodes=episodes, train_floor_episodes=floor_episodes, train_floor_steps=floor_steps,
               drawmean_reported_return_pct=float(np.mean([d["reported_return_pct"] for d in draws])),
               drawmean_ledger_return_pct=float(np.mean([d["ledger_return_pct"] for d in draws])),
               drawmean_mean_abs_lev_open=float(np.mean([d["mean_abs_lev_open"] for d in draws])),
               drawshare_floor_bound=float(np.mean([d["floor_bound"] for d in draws])),
               drawshare_closed=float(np.mean([d["closed"] for d in draws])),
               seconds=round(time.time() - t0, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    return (f"done {out.name} ({res['seconds']}s): X={res['X']:g} floor_episodes={floor_episodes}/{episodes} "
            f"floor_steps={floor_steps}")


def _init_worker():
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"


def main() -> int:
    from envs_re import SCALES
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    jobs = [(r, c, s) for s in SEEDS for c in SCALES for r in RULES]
    if a.smoke:
        jobs = [j for j in jobs if j[2] == 0 and j[1] in (1.0, 0.01)]
    steps = 16_384 if a.smoke else STEPS
    todo = [j for j in jobs if not job_path(*j, smoke=a.smoke).exists()]
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} complete, running {len(todo)} with {a.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init_worker) as ex:
        futs = {ex.submit(re_job, *j, steps, a.smoke): j for j in todo}
        for f in as_completed(futs):
            try:
                print(time.strftime("%H:%M:%S"), f.result(), flush=True)
            except Exception as e:  # noqa: BLE001
                print(time.strftime("%H:%M:%S"), "FAILED", futs[f], repr(e)[:300], flush=True)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    raise SystemExit(main())
