"""Study S runner (PREREGISTRATION_S.md): does a sign-paired reward prevent the loss-forgiveness exploit without the ledger?

    python run_s.py --workers 8            # both parts, resume-safe (completed runs are skipped)
    python run_s.py --part A --workers 8
    python run_s.py --smoke                # development check: 20,000 steps, one seed per arm; excluded
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
OUT = HERE / "results_s"
STEPS = 1_000_000
N_ENVS = 8
GAMMA = 0.99
ARMS_A = ("PLAIN", "ORACLE", "SPR", "KL0.003", "KL0.03", "KL0.3")
SEEDS_A = list(range(100, 110))
ARMS_B = ("ZF", "SPR", "LEDGER")
LEARNERS_B = {"PPO": ("PPO", dict(n_steps=1024, batch_size=256)),
              "A2C-n64": ("A2C", dict(n_steps=64, gae_lambda=0.95, normalize_advantage=True))}
SEEDS_B = list(range(110, 120))
REGIONS_B = ("DK1", "DK2")
PROBE_E_A = (0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
EGRID_B = (0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
N_TEST_A = 40
N_DRAWS_B = 20


def path_a(arm, seed, smoke=False):
    return OUT / ("smoke" if smoke else "A") / f"{arm}__s{seed}__DK1.json"


def path_b(learner, arm, seed, region, smoke=False):
    return OUT / ("smoke" if smoke else "B") / f"{learner}__{arm}__s{seed}__{region}.json"


def _paths():
    sys.path[:0] = [str(LIAB), str(STUDY_O), str(HERE)]


def _check_greedy(layers, model, env, greedy, obs_fn=lambda o: o):
    o, _ = env.reset()
    for _ in range(300):
        x = np.asarray(obs_fn(o), dtype=np.float32)
        assert int(greedy(layers, x[None, :])[0]) == int(model.predict(x, deterministic=True)[0]), "fast policy mismatch"
        o, _, d, _, _ = env.step(env.action_space.sample())
        if d:
            o, _ = env.reset()


def job_a(arm, seed, steps, smoke):
    out = path_a(arm, seed, smoke)
    if out.exists():
        return f"skip {out.name}"
    _paths()
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from envs_s import K0, LEV, EdgeEnv
    from run_b import greedy, network_layers
    from spr_s import make_kl_wrapper, make_spr_edge
    t0 = time.time()
    mode = "booked" if arm == "ORACLE" else "reported"
    if arm == "SPR":
        fns = [(lambda s: (lambda: make_spr_edge(EdgeEnv("DK1", "train", "edge", "reported", seed=s))))(seed * 100 + i) for i in range(N_ENVS)]
    else:
        fns = [(lambda s: (lambda: EdgeEnv("DK1", "train", "edge", mode, seed=s)))(seed * 100 + i) for i in range(N_ENVS)]
    venv = DummyVecEnv(fns)
    kl = None
    if arm.startswith("KL"):
        kl = make_kl_wrapper()(venv, float(arm[2:]))
        venv = kl
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=GAMMA)
    model = PPO("MlpPolicy", venv, seed=seed, n_steps=1024, batch_size=256, gamma=GAMMA, policy_kwargs=dict(net_arch=[64, 64]),
                device="cpu", verbose=0)
    if kl is not None:
        kl.model = model
    model.learn(total_timesteps=steps)
    layers = network_layers(model, "PPO")
    _check_greedy(layers, model, EdgeEnv("DK1", "train", "edge", "reported", seed=seed + 1000), greedy)
    lev = {}
    for e in PROBE_E_A:
        for z in (-1, 1):
            for rem in (0.05, 0.5):
                lev[f"{e:g}|{z:+d}|{rem:g}"] = float(LEV[int(greedy(layers, np.array([[e, z, rem, 0.0]]))[0])])

    def test(kind, n_ep, s):
        env = EdgeEnv("DK1", "test", kind, "reported", seed=s)
        rng = np.random.default_rng(s)
        rows = []
        for _ in range(n_ep):
            o, _ = env.reset(options={"e0": float(np.exp(rng.uniform(np.log(0.05), np.log(2.0))))})
            done, rep = False, 0.0
            while not done:
                o, r, done, _, _ = env.step(int(greedy(layers, o[None, :])[0]))
                rep += r
            rows.append(((env.ledger - env.e0) / K0 * 100.0, rep, env.forgiven / K0 * 100.0))
        r = np.array(rows)
        return {"booked_pct_K0": float(r[:, 0].mean()), "booked_se": float(r[:, 0].std(ddof=1) / np.sqrt(len(r))),
                "reported_pct_K0": float(r[:, 1].mean()), "forgiven_pct_K0": float(r[:, 2].mean())}

    n_ep = 4 if smoke else N_TEST_A
    res = dict(part="A", arm=arm, seed=seed, region="DK1", steps=steps, lev=lev,
               test_edge=test("edge", n_ep, 9000 + seed), test_placebo=test("placebo", n_ep, 9500 + seed),
               seconds=round(time.time() - t0, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    return f"done A {out.name} ({res['seconds']}s): booked {res['test_edge']['booked_pct_K0']:+.1f}%"


def job_b(learner, arm, seed, region, steps, smoke):
    out = path_b(learner, arm, seed, region, smoke)
    if out.exists():
        return f"skip {out.name}"
    _paths()
    import gymnasium as gym
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import A2C, PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from envs_b import LEV, load_market, outcome
    from envs_o import PartialEnv
    from run_b import greedy, network_layers
    from spr_s import make_spr_fair
    algo, kw = LEARNERS_B[learner]
    t0 = time.time()
    mtr = load_market(region, "train", "NE")
    split = "train" if smoke else "test"

    class EquityOnly(gym.ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            self.observation_space = gym.spaces.Box(0, 5, shape=(1,), dtype=np.float32)

        def observation(self, obs):
            return np.array([obs[6]], dtype=np.float32)

    alpha = 1.0 if arm == "LEDGER" else 0.0
    wrap = make_spr_fair if arm == "SPR" else (lambda e: e)
    venv = DummyVecEnv([(lambda i: (lambda: EquityOnly(wrap(PartialEnv(mtr, alpha, train=True, seed=seed * 100 + i)))))(i)
                        for i in range(N_ENVS)])
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=GAMMA)
    Algo = PPO if algo == "PPO" else A2C
    model = Algo("MlpPolicy", venv, seed=seed, gamma=GAMMA, policy_kwargs=dict(net_arch=[64, 64]), device="cpu", verbose=0, **kw)
    model.learn(total_timesteps=steps)
    layers = network_layers(model, "PPO")            # A2C has the same policy head (study I/Q)
    _check_greedy(layers, model, PartialEnv(mtr, alpha, train=True, seed=seed + 1000), greedy, obs_fn=lambda o: np.asarray(o)[6:7])
    lev = {f"{e:g}": float(abs(LEV[greedy(layers, np.array([[min(e, 5.0)]]))[0]])) for e in EGRID_B}
    draws = []
    for k in range(3 if smoke else N_DRAWS_B):
        env = PartialEnv(load_market(region, split, "NE", draw=k), 0.0, train=False)
        o, _ = env.reset(options={"e0": 1.0})
        done = False
        while not done:
            o, _, done, _, _ = env.step(int(greedy(layers, np.asarray(o)[6:7][None, :])[0]))
        draws.append(outcome(env))
    res = dict(part="B", learner=learner, algo=algo, kwargs=kw, arm=arm, alpha=alpha, seed=seed, region=region, steps=steps,
               eval_split=split, lev=lev,
               drawmean_reported_return_pct=float(np.mean([d["reported_return_pct"] for d in draws])),
               drawmean_ledger_return_pct=float(np.mean([d["ledger_return_pct"] for d in draws])),
               seconds=round(time.time() - t0, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    return f"done B {out.name} ({res['seconds']}s): L(0.02)={lev['0.02']:g} L(2)={lev['2']:g}"


def _init_worker():
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--part", choices=["A", "B", "AB"], default="AB")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    steps = 20_000 if a.smoke else STEPS
    jobs = []
    if "A" in a.part:
        jobs += [("A", arm, s) for s in (SEEDS_A[:1] if a.smoke else SEEDS_A) for arm in ARMS_A]
    if "B" in a.part:
        jobs += [("B", ln, arm, s, r) for s in (SEEDS_B[:1] if a.smoke else SEEDS_B) for ln in LEARNERS_B for arm in ARMS_B
                 for r in (REGIONS_B[:1] if a.smoke else REGIONS_B)]
    todo = [j for j in jobs if not (path_a(j[1], j[2], a.smoke) if j[0] == "A" else path_b(j[1], j[2], j[3], j[4], a.smoke)).exists()]
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} complete, running {len(todo)} with {a.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init_worker) as ex:
        futs = {}
        for j in todo:
            f = ex.submit(job_a, j[1], j[2], steps, a.smoke) if j[0] == "A" else ex.submit(job_b, j[1], j[2], j[3], j[4], steps, a.smoke)
            futs[f] = j
        for f in as_completed(futs):
            try:
                print(time.strftime("%H:%M:%S"), f.result(), flush=True)
            except Exception as e:  # noqa: BLE001
                print(time.strftime("%H:%M:%S"), "FAILED", futs[f], repr(e)[:300], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
