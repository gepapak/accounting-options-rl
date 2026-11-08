"""Study X runner: do the twin rewards still prevent the exploit when the reward is NOT a single-floor censor (a two-write floor)?

    python run_x.py --workers 6                  # all registered jobs, resume-safe (completed runs are skipped)
    python run_x.py --smoke --out DIR            # development check: 20,000 steps, one seed per arm; excluded

Training, probes and evaluation are those of Study V's run_v.py (PPO, 8 environments, 1,024-step rollouts, batch 256, gamma 0.99, 64x64, reward
normalization, 10^6 steps; the same probes and the same 100 fixed test windows and start equities), with the sleeve replaced by the two-write
variant of envs_x.py in training, in the probes and in the evaluation. Arms: PLAIN (reported reward), ORACLE (booked reward), SPR and TIR.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LIAB = HERE.parents[1] / "LIABILITY_STUDY_2026-09-29"
STEPS = 1_000_000
N_ENVS = 8
GAMMA = 0.99
ARMS = ("PLAIN", "ORACLE", "SPR", "TIR")
SEEDS = list(range(190, 198))
SETTING = "S_DK1"
LABEL = "X_DK1+Z"
PROBE_E = (0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
N_TEST = 100


def job_path(out: Path, arm: str, seed: int) -> Path:
    return out / f"{LABEL}__{arm}__s{seed}.json"


def _paths():
    sys.path[:0] = [str(LIAB), str(HERE), str(HERE.parent / "study_t"), str(HERE.parent / "study_s")]


def x_job(out_dir: str, arm: str, seed: int, steps: int, smoke: bool) -> str:
    out = job_path(Path(out_dir), arm, seed)
    if out.exists():
        return f"skip {out.name}"
    _paths()
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from envs_t import K0, LEV, Q1
    from envs_x import EdgeEnvMF, EdgeEnvMFZ
    from run_b import greedy, network_layers
    from spr_t import make_spr, make_tir
    t0 = time.time()

    def build(s):
        if arm == "ORACLE":
            return EdgeEnvMFZ(SETTING, "train", "edge", "booked", seed=s)
        base = EdgeEnvMFZ(SETTING, "train", "edge", "reported", seed=s)
        if arm == "PLAIN":
            return base
        if arm == "SPR":
            return make_spr(base)
        if arm == "TIR":
            return make_tir(base)
        raise ValueError(arm)

    fns = [(lambda s: (lambda: build(s)))(seed * 100 + i) for i in range(N_ENVS)]
    venv = VecNormalize(DummyVecEnv(fns), norm_obs=False, norm_reward=True, gamma=GAMMA)
    kw = dict(seed=seed, n_steps=1024, batch_size=256, gamma=GAMMA, policy_kwargs=dict(net_arch=[64, 64]), device="cpu", verbose=0)
    model = PPO("MlpPolicy", venv, **kw)
    model.learn(total_timesteps=steps)
    layers = network_layers(model, "PPO")
    chk = EdgeEnvMF(SETTING, "train", "edge", "reported", seed=seed + 1000)
    o, _ = chk.reset()
    for _ in range(300):                                   # exactness check of the NumPy forward pass
        assert int(greedy(layers, np.asarray(o, dtype=np.float32)[None, :])[0]) == int(model.predict(o, deterministic=True)[0]), "fast policy mismatch"
        o, _, d, _, _ = chk.step(chk.action_space.sample())
        if d:
            o, _ = chk.reset()
    lev = {}
    for e in PROBE_E:
        for z in (-1, 1):
            for rem in (0.05, 0.5):
                lev[f"{e:g}|{z:+d}|{rem:g}"] = float(LEV[int(greedy(layers, np.array([[e, z, rem, 0.0]]))[0])])

    def test(kind, n_ep, s, e0_fixed=None):
        env = EdgeEnvMF(SETTING, "test", kind, "reported", seed=s)
        rng = np.random.default_rng(s)
        rows = []
        for _ in range(n_ep):
            e0 = e0_fixed if e0_fixed is not None else float(np.exp(rng.uniform(np.log(0.05), np.log(2.0))))
            o, _ = env.reset(options={"e0": e0})
            done, rep = False, 0.0
            while not done:
                o, r, done, _, _ = env.step(int(greedy(layers, o[None, :])[0]))
                rep += r
            rows.append(((env.ledger - env.e0) / K0 * 100.0, rep, env.forgiven / K0 * 100.0))
        r = np.array(rows)
        return {"booked_pct_K0": float(r[:, 0].mean()), "booked_se": float(r[:, 0].std(ddof=1) / np.sqrt(len(r))),
                "reported_pct_K0": float(r[:, 1].mean()), "forgiven_pct_K0": float(r[:, 2].mean())}

    def audit(kind, n_ep, s):
        """Ledger-free estimates of the trained agent's booked return on the same test episodes as test(): per episode the sum of the reported,
        SPR and TIR step rewards (percent of K0) against the ledger's booked return; twin queries are made at evaluation time only."""
        env = EdgeEnvMF(SETTING, "test", kind, "reported", seed=s)
        rng = np.random.default_rng(s)
        rows = []
        for _ in range(n_ep):
            env.reset(options={"e0": float(np.exp(rng.uniform(np.log(0.05), np.log(2.0))))})
            done = False
            acc = dict(rep=0.0, spr=0.0, tir=0.0)
            while not done:
                a = int(greedy(layers, env._obs()[None, :])[0])
                q = float(np.clip(float(LEV[a]) * Q1, -env.cap[env.t], env.cap[env.t]))
                sigma = float(np.sign(q)) * float(env.eps_ep[env.t - env.t0])
                twin = copy.copy(env)
                twin.flip_next = True
                r_twin = twin.step(a)[1]
                _, r, done, _, info = env.step(a)
                c = info["cost"] / K0 * 100.0
                g_hat = max(r, r_twin) + c if abs(r - r_twin) > 1e-12 else 0.0
                acc["rep"] += r
                acc["spr"] += 0.5 * (r - r_twin) - c
                acc["tir"] += sigma * g_hat - c
            rows.append((acc["rep"], acc["spr"], acc["tir"], (env.ledger - env.e0) / K0 * 100.0))
        r = np.array(rows)
        booked = r[:, 3]
        est = {"reported": r[:, 0], "spr": r[:, 1], "tir": r[:, 2]}
        res = {"booked": float(booked.mean())}
        for k, v in est.items():
            res[k] = float(v.mean())
            res[f"mae_{k}"] = float(np.abs(v - booked).mean())
        res["episodes"] = {"booked": booked.tolist(), **{k: v.tolist() for k, v in est.items()}}
        return res

    n_ep = 4 if smoke else N_TEST
    res = dict(setting=LABEL, arm=arm, seed=seed, steps=steps, lev=lev, test_edge=test("edge", n_ep, 9000),
               test_placebo=test("placebo", n_ep, 9500), test_zero=test("edge", n_ep, 9000, e0_fixed=0.0),
               audit_edge=audit("edge", n_ep, 9000), audit_placebo=audit("placebo", n_ep, 9500))
    for k in ("edge", "placebo"):                          # the audit replays the same episodes as the test
        assert abs(res[f"test_{k}"]["booked_pct_K0"] - res[f"audit_{k}"]["booked"]) < 1e-6, "audit episodes differ from test episodes"
        assert abs(res[f"test_{k}"]["reported_pct_K0"] - res[f"audit_{k}"]["reported"]) < 1e-6, "audit reported return differs"
    res["seconds"] = round(time.time() - t0, 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    return f"done {out.name} ({res['seconds']}s): booked {res['test_edge']['booked_pct_K0']:+.1f}%"


def _init_worker():
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None, help="output folder (default: results_x/ next to this file)")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    a = ap.parse_args()
    steps = 20_000 if a.smoke else a.steps
    out = Path(a.out) if a.out else HERE / "results_x"
    seeds = a.seeds or (SEEDS[:1] if a.smoke else SEEDS)
    jobs = [(arm, s) for s in seeds for arm in (a.arms or ARMS)]
    todo = [j for j in jobs if not job_path(out, *j).exists()]
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} complete, running {len(todo)} with {a.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init_worker) as ex:
        futs = {ex.submit(x_job, str(out), arm, s, steps, a.smoke): (arm, s) for arm, s in todo}
        for f in as_completed(futs):
            try:
                print(time.strftime("%H:%M:%S"), f.result(), flush=True)
            except Exception as e:  # noqa: BLE001
                print(time.strftime("%H:%M:%S"), "FAILED", futs[f], repr(e)[:300], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
