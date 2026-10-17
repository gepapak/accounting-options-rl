"""Study Q runner (PREREGISTRATION_Q.md): does the learnability index transfer from PPO to other learners?

Study O's environment (study_o/envs_o.PartialEnv: a share alpha of every forgiven loss charged in the reward, continued
trading; placebo market; DK1/DK2; equity as the only input) trained with Study I's other learners: A2C with a rollout of
64 steps, generalized advantage estimation and advantage normalization, and DQN at its defaults; 8 parallel environments,
reward normalization, gamma 0.99, 10^6 steps.

    python run_q.py --workers 7
    python run_q.py --smoke        # development check: tiny budget, TRAIN split only
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
STUDY_O = HERE.parent / "study_o"
OUT = HERE / "results_q"
LEARNERS = {"A2C-n64": ("A2C", dict(n_steps=64, gae_lambda=0.95, normalize_advantage=True)), "DQN-default": ("DQN", {})}
ALPHAS = (0.0, 0.5, 0.8, 0.9, 1.0)        # 1.0 = matched control (every loss charged)
SEEDS = [50, 51, 52, 53, 54]
REGIONS = ["DK1", "DK2"]
STEPS = 1_000_000
N_ENVS = 8
N_DRAWS = 20
GAMMA = 0.99
EGRID = (0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)


def job_path(learner, alpha, seed, region, smoke=False) -> Path:
    return OUT / ("jobs_smoke" if smoke else "jobs") / f"{learner}__a{alpha:g}__s{seed}__{region}.json"


def q_job(learner: str, alpha: float, seed: int, region: str, steps: int, smoke: bool) -> str:
    out = job_path(learner, alpha, seed, region, smoke)
    if out.exists():
        return f"skip {out.name}"
    sys.path[:0] = [str(LIAB), str(STUDY_O)]
    import gymnasium as gym
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import A2C, DQN
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from envs_b import LEV, load_market, outcome
    from envs_o import PartialEnv
    from run_b import greedy
    from run_i import network_layers_any
    algo, kw = LEARNERS[learner]
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
    Algo = A2C if algo == "A2C" else DQN
    model = Algo("MlpPolicy", venv, seed=seed, gamma=GAMMA, policy_kwargs=dict(net_arch=[64, 64]), device="cpu", verbose=0, **kw)
    model.learn(total_timesteps=steps)
    layers = network_layers_any(model, algo)
    act = lambda o: int(greedy(layers, np.asarray(o)[6:7][None, :])[0])  # noqa: E731
    chk = EquityOnly(PartialEnv(mtr, alpha, train=True, seed=seed + 1000))   # exactness check of the NumPy forward pass
    o, _ = chk.reset()
    for _ in range(300):
        assert int(greedy(layers, np.asarray(o)[None, :])[0]) == int(model.predict(o, deterministic=True)[0]), "fast policy mismatch"
        o, _, d, _, _ = chk.step(chk.action_space.sample())
        if d:
            o, _ = chk.reset()
    lev = {f"{e:g}": float(abs(LEV[greedy(layers, np.array([[min(e, 5.0)]]))[0]])) for e in EGRID}
    draws = []
    for k in range(3 if smoke else N_DRAWS):
        env = PartialEnv(load_market(region, split, "NE", draw=k), alpha, train=False)
        o, _ = env.reset(options={"e0": 1.0})
        done = False
        while not done:
            o, _, done, _, _ = env.step(act(o))
        draws.append(outcome(env))
    res = dict(learner=learner, algo=algo, kwargs=kw, alpha=alpha, gamma=GAMMA, obsset="EQ", seed=seed, region=region, steps=steps,
               eval_split=split, lev=lev,
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
    jobs = [(ln, al, s, g) for s in SEEDS for ln in LEARNERS for al in ALPHAS for g in REGIONS]
    if a.smoke:
        jobs = [j for j in jobs if j[2] == SEEDS[0] and j[3] == "DK1" and j[1] == 1.0]
    todo = [j for j in jobs if not job_path(*j, smoke=a.smoke).exists()]
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} complete, running {len(todo)} with {a.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init_worker) as ex:
        futs = {ex.submit(q_job, *j, 16_384 if a.smoke else STEPS, a.smoke): j for j in todo}
        for f in as_completed(futs):
            try:
                print(time.strftime("%H:%M:%S"), f.result(), flush=True)
            except Exception as e:  # noqa: BLE001
                print(time.strftime("%H:%M:%S"), "FAILED", futs[f], repr(e)[:300], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
