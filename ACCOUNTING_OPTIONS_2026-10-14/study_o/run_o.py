"""Study O runner (PREREGISTRATION_O.md): does a learnability index computed before training predict whether PPO agents
learn to exploit partial loss forgiveness?

Study F's well-posed PPO set-up (equity as the only input, 8 parallel environments, reward normalization, 1M steps,
gamma 0.99), placebo market, DK1/DK2, with a share alpha of every forgiven loss charged in the reward (envs_o.py).

    python run_o.py --workers 7
    python run_o.py --smoke        # development check: tiny budget, TRAIN split only
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
OUT = HERE / "results_o"
ALPHAS = (0.0, 0.25, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0)   # 1.0 = matched control (every loss charged)
SEEDS = [30, 31, 32, 33, 34]
REGIONS = ["DK1", "DK2"]
STEPS = 1_000_000
N_ENVS = 8
N_DRAWS = 20
GAMMA = 0.99
EGRID = (0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)


def job_path(alpha, seed, region, smoke=False) -> Path:
    return OUT / ("jobs_smoke" if smoke else "jobs") / f"a{alpha:g}__PPO__s{seed}__{region}.json"


def o_job(alpha: float, seed: int, region: str, steps: int, smoke: bool) -> str:
    out = job_path(alpha, seed, region, smoke)
    if out.exists():
        return f"skip {out.name}"
    sys.path[:0] = [str(LIAB), str(HERE)]
    import gymnasium as gym
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from envs_b import LEV, load_market, outcome
    from envs_o import PartialEnv
    from run_b import greedy, network_layers
    t0 = time.time()
    mtr = load_market(region, "train", "NE")
    split = "train" if smoke else "test"

    class EquityOnly(gym.ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            self.observation_space = gym.spaces.Box(0, 5, shape=(1,), dtype=np.float32)

        def observation(self, obs):
            return np.array([obs[6]], dtype=np.float32)   # reported equity / K0, clipped to [0, 5]

    venv = DummyVecEnv([(lambda i: (lambda: EquityOnly(PartialEnv(mtr, alpha, train=True, seed=seed * 100 + i))))(i)
                        for i in range(N_ENVS)])
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=GAMMA)
    model = PPO("MlpPolicy", venv, seed=seed, n_steps=1024, batch_size=256, gamma=GAMMA, policy_kwargs=dict(net_arch=[64, 64]),
                device="cpu", verbose=0)
    model.learn(total_timesteps=steps)
    layers = network_layers(model, "PPO")
    act = lambda o: int(greedy(layers, np.asarray(o)[6:7][None, :])[0])  # noqa: E731
    # exactness check of the NumPy forward pass
    chk = EquityOnly(PartialEnv(mtr, alpha, train=True, seed=seed + 1000))
    o, _ = chk.reset()
    for _ in range(300):
        assert int(greedy(layers, np.asarray(o)[None, :])[0]) == int(model.predict(o, deterministic=True)[0]), "fast policy mismatch"
        o, _, d, _, _ = chk.step(chk.action_space.sample())
        if d:
            o, _ = chk.reset()
    lev = {f"{e:g}": float(abs(LEV[greedy(layers, np.array([[min(e, 5.0)]]))[0]])) for e in EGRID}
    # placebo-market evaluation from equity K0 over sign draws of the evaluation period
    draws = []
    for k in range(3 if smoke else N_DRAWS):
        env = PartialEnv(load_market(region, split, "NE", draw=k), alpha, train=False)
        o, _ = env.reset(options={"e0": 1.0})
        done = False
        while not done:
            o, _, done, _, _ = env.step(act(o))
        draws.append(outcome(env))
    res = dict(alpha=alpha, gamma=GAMMA, obsset="EQ", algo="PPO", seed=seed, region=region, steps=steps, eval_split=split, lev=lev,
               drawmean_reported_return_pct=float(np.mean([d["reported_return_pct"] for d in draws])),
               drawmean_ledger_return_pct=float(np.mean([d["ledger_return_pct"] for d in draws])),
               drawmean_mean_abs_lev_open=float(np.mean([d["mean_abs_lev_open"] for d in draws])),
               drawshare_floor_bound=float(np.mean([d["floor_bound"] for d in draws])),
               draws_reported_return_pct=[d["reported_return_pct"] for d in draws],
               seconds=round(time.time() - t0, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    return f"done {out.name} ({res['seconds']}s): L(0.02)={lev['0.02']:g} L(0.1)={lev['0.1']:g} L(2)={lev['2']:g}"


def _init_worker():
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    jobs =[(al, s, g) for s in SEEDS for al in ALPHAS for g in REGIONS]
    if a.smoke:
        jobs = [j for j in jobs if j[1] == SEEDS[0] and j[2] == "DK1" and j[0] in (0.0, 1.0)]
    steps = 16_384 if a.smoke else STEPS
    todo = [j for j in jobs if not job_path(*j, smoke=a.smoke).exists()]
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} complete, running {len(todo)} with {a.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init_worker) as ex:
        futs = {ex.submit(o_job, *j, steps, a.smoke): j for j in todo}
        for f in as_completed(futs):
            try:
                print(time.strftime("%H:%M:%S"), f.result(), flush=True)
            except Exception as e:  # noqa: BLE001
                print(time.strftime("%H:%M:%S"), "FAILED", futs[f], repr(e)[:300], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
