"""No-free-lunch (NFL) training pilot runner (EXPLORATORY, see PLAN.md).

    python run_nfl.py --workers 4                      # pilot: PLAIN, NFL, ORACLE on DK1, seeds 80-82
    python run_nfl.py --smoke                          # development check: tiny budget
    python run_nfl.py --arms PLAIN NFL --seeds 83 84 --regions DK2 --accounting LL
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
N_ENVS = 8
GAMMA = 0.99
EGRID = (0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
N_TEST = 40
# arm -> (reward mode of real envs, number of real envs, number of placebo envs)
ARMS = {"PLAIN": ("reported", 8, 0), "NFL": ("reported", 4, 4), "ORACLE": ("booked", 8, 0)}


def job_path(arm, seed, region, accounting, smoke):
    return OUT / ("smoke" if smoke else "jobs") / f"{arm}__{accounting}__s{seed}__{region}.json"


def evaluate(model, region, accounting, kind, n_ep, seed):
    """Greedy policy on the test split; returns mean booked and reported P&L in % of K0, and forgiven loss."""
    from envs_nfl import K0, NFLEnv
    env = NFLEnv(region, "test", kind, "reported", accounting, seed=seed)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_ep):
        o, _ = env.reset(options={"e0": float(np.exp(rng.uniform(np.log(0.05), np.log(2.0))))})
        done = False
        while not done:
            a = int(model.predict(o, deterministic=True)[0])
            o, _, done, _, _ = env.step(a)
        rows.append(((env.ledger - env.e0) / K0 * 100.0, (env.reported - env.e0) / K0 * 100.0, env.forgiven / K0 * 100.0,
                     env.floor_hits))
    r = np.array(rows)
    return {"booked_pct_K0": float(r[:, 0].mean()), "booked_se": float(r[:, 0].std(ddof=1) / np.sqrt(len(r))),
            "reported_pct_K0": float(r[:, 1].mean()), "forgiven_pct_K0": float(r[:, 2].mean()),
            "floor_hits": float(r[:, 3].mean())}


def nfl_job(arm, seed, region, accounting, steps, smoke):
    out = job_path(arm, seed, region, accounting, smoke)
    if out.exists():
        return f"skip {out.name}"
    sys.path.insert(0, str(HERE))
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from envs_nfl import LEV, NFLEnv
    t0 = time.time()
    mode, n_real, n_pl = ARMS[arm]
    fns = [(lambda i: (lambda: NFLEnv(region, "train", "edge", mode, accounting, seed=seed * 100 + i)))(i) for i in range(n_real)]
    fns += [(lambda i: (lambda: NFLEnv(region, "train", "placebo", "nfl_placebo", accounting, seed=seed * 100 + 50 + i)))(i)
            for i in range(n_pl)]
    venv = VecNormalize(DummyVecEnv(fns), norm_obs=False, norm_reward=True, gamma=GAMMA)
    model = PPO("MlpPolicy", venv, seed=seed, n_steps=1024, batch_size=256, gamma=GAMMA, policy_kwargs=dict(net_arch=[64, 64]),
                device="cpu", verbose=0)
    model.learn(total_timesteps=steps)
    lev = {f"{e:g}|{z:+d}": float(LEV[int(model.predict(np.array([e, z], np.float32), deterministic=True)[0])])
           for e in EGRID for z in (-1, 1)}
    n_ep = 4 if smoke else N_TEST
    res = dict(arm=arm, seed=seed, region=region, accounting=accounting, steps=steps, lev=lev,
               test_edge=evaluate(model, region, accounting, "edge", n_ep, 9000 + seed),
               test_placebo=evaluate(model, region, accounting, "placebo", n_ep, 9500 + seed),
               seconds=round(time.time() - t0, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    L = lambda e: (lev[f"{e:g}|-1"], lev[f"{e:g}|+1"])  # noqa: E731
    return (f"done {out.name} ({res['seconds']}s): L(0)={L(0)} L(0.02)={L(0.02)} L(1)={L(1)} "
            f"booked {res['test_edge']['booked_pct_K0']:+.1f}% placebo-reported {res['test_placebo']['reported_pct_K0']:+.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[80, 81, 82])
    ap.add_argument("--regions", nargs="+", default=["DK1"])
    ap.add_argument("--accounting", default="ZF")
    ap.add_argument("--steps", type=int, default=1_000_000)
    a = ap.parse_args()
    steps = 20_000 if a.smoke else a.steps
    jobs = [(arm, s, r) for arm in a.arms for s in a.seeds for r in a.regions]
    print(f"{len(jobs)} jobs, {a.workers} workers, {steps} steps, accounting {a.accounting}", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(nfl_job, arm, s, r, a.accounting, steps, a.smoke): (arm, s, r) for arm, s, r in jobs}
        for f in as_completed(futs):
            try:
                print(time.strftime("%H:%M:%S"), f.result(), flush=True)
            except Exception as e:  # noqa: BLE001
                print(time.strftime("%H:%M:%S"), "FAILED", futs[f], repr(e)[:400], flush=True)


if __name__ == "__main__":
    main()
