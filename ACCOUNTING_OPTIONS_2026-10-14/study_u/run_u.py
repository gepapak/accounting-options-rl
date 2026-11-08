"""Study U runner (PREREGISTRATION_U.md): do twin-based rewards prevent the settlement-lag and terminal-liability exploits of Study N?

Study N's registered environments and training protocol (zombie window and terminal limited liability, placebo market, observation of the
ledger and the time left, PPO with Study F's configuration, 10^6 steps; the copy `study_t/envs_tn.py` reproduces `study_n/envs_n.py` step for
step), with the forgiven arms additionally trained on the sign-paired reward (SPR) and the twin-identified reward (TIR).

    python run_u.py --workers 8                     # all registered jobs, resume-safe
    python run_u.py --smoke                         # development check: 16,384 steps, one seed per arm; excluded
Development pilots (exploratory) use --out, --arms and --seeds.
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
STUDY_T = HERE.parent / "study_t"
STEPS = 1_000_000
N_ENVS = 8
GAMMA = 0.99
N_DRAWS = 20
ARMS = ("ZFG", "ZBK", "ZSPR", "ZTIR", "TFG", "TBK", "TSPR", "TTIR")   # Z/T: setting; FG forgiven, BK booked, SPR/TIR trained on the twin reward
SEEDS = [140, 141, 142, 143, 144]
REGIONS = ["DK1", "DK2"]
Z_PROBE_N = (-0.5, -0.25, -0.1, -0.02)
Z_PROBE_K = (96, 48, 12)
Z_PROBE_E = (0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0)
T_PROBE_L = (-0.25, -0.1, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
T_PROBE_TAU = (1, 4, 16, 48, 96, 192)


def job_path(out: Path, arm: str, seed: int, region: str) -> Path:
    return out / f"{arm}__PPO__s{seed}__{region}.json"


def base_arm(arm: str) -> str:
    """The environment arm: SPR/TIR arms are trained in the forgiven environment."""
    return arm[0] + "FG" if arm[1:] in ("SPR", "TIR") else arm


def u_job(out_dir: str, arm: str, seed: int, region: str, steps: int, split: str, smoke: bool) -> str:
    out = job_path(Path(out_dir), arm, seed, region)
    if out.exists():
        return f"skip {out.name}"
    sys.path[:0] = [str(LIAB), str(STUDY_T)]
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from envs_tn import H_TERM, K0, LEV, W_LAG, LagEnvT, load_market
    from run_b import greedy, network_layers
    from spr_t import make_spr_n, make_tir_n
    t0 = time.time()
    env_arm = base_arm(arm)
    wrap = make_spr_n if arm[1:] == "SPR" else (make_tir_n if arm[1:] == "TIR" else (lambda e: e))
    mtr = load_market(region, "train", "NE")
    venv = DummyVecEnv([(lambda i: (lambda: wrap(LagEnvT(mtr, env_arm, train=True, seed=seed * 100 + i))))(i) for i in range(N_ENVS)])
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=GAMMA)
    model = PPO("MlpPolicy", venv, seed=seed, n_steps=1024, batch_size=256, gamma=GAMMA, policy_kwargs=dict(net_arch=[64, 64]),
                device="cpu", verbose=0)
    model.learn(total_timesteps=steps)
    layers = network_layers(model, "PPO")
    chk = LagEnvT(mtr, env_arm, train=True, seed=seed + 1000)
    o, _ = chk.reset()
    for _ in range(300):
        assert int(greedy(layers, np.asarray(o)[None, :])[0]) == int(model.predict(o, deterministic=True)[0]), "fast policy mismatch"
        o, _, te, tr, _ = chk.step(chk.action_space.sample())
        if te or tr:
            o, _ = chk.reset()
    lev = lambda X: np.abs(LEV[greedy(layers, np.asarray(X, dtype=np.float64))])  # noqa: E731
    probes = {}
    if arm[0] == "Z":
        for n in Z_PROBE_N:
            for k in Z_PROBE_K:
                probes[f"zombie_n{n:g}_k{k}"] = float(lev([[n, k / W_LAG]])[0])
        for e in Z_PROBE_E:
            probes[f"solvent_e{e:g}"] = float(lev([[min(e, 5.0), 0.0]])[0])
    else:
        for L in T_PROBE_L:
            for tau in T_PROBE_TAU:
                probes[f"L{L:g}_tau{tau}"] = float(lev([[L, tau / H_TERM]])[0])
    draws = []
    for k in range(3 if smoke else N_DRAWS):
        meval = load_market(region, split, "NE", draw=k)
        if arm[0] == "Z":
            env = LagEnvT(meval, env_arm, train=False)
            o, _ = env.reset(options={"e0": 1.0, "t0": 200})
            done = False
            while not done:
                o, _, te, tr, _ = env.step(int(greedy(layers, np.asarray(o)[None, :].astype(np.float64))[0]))
                done = te or tr
            draws.append(dict(reported_pct=100 * (env.reported() / K0 - 1), booked_pct=100 * (env.L / K0 - 1), closed=env.closed,
                              forgiven_pct=100 * env.forgiven_amt / K0, zombie_steps=env.zombie_steps,
                              zombie_maxlev_share=env.zombie_max / max(env.zombie_steps, 1)))
        else:
            reps, books = [], []
            for t0b in range(200, meval.n - H_TERM - 1, H_TERM):
                env = LagEnvT(meval, env_arm, train=False)
                o, _ = env.reset(options={"e0": 1.0, "t0": t0b, "tau0": H_TERM})
                done = False
                while not done:
                    o, _, te, tr, _ = env.step(int(greedy(layers, np.asarray(o)[None, :].astype(np.float64))[0]))
                    done = te or tr
                reps.append(env.reported() / K0 - 1)
                books.append(env.L / K0 - 1)
            draws.append(dict(reported_pct=100 * float(np.mean(reps)), booked_pct=100 * float(np.mean(books))))
    res = dict(arm=arm, env_arm=env_arm, algo="PPO", seed=seed, region=region, steps=steps, eval_split=split, probes=probes,
               drawmean_reported_pct=float(np.mean([d["reported_pct"] for d in draws])),
               drawmean_booked_pct=float(np.mean([d["booked_pct"] for d in draws])), draws=draws, seconds=round(time.time() - t0, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    key = "zombie_n-0.1_k48" if arm[0] == "Z" else "L0.1_tau16"
    return f"done {out.name} ({res['seconds']}s): {key}={probes[key]:g} booked={res['drawmean_booked_pct']:.1f}"


def _init_worker():
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--regions", nargs="+", default=None)
    ap.add_argument("--steps", type=int, default=STEPS)
    a = ap.parse_args()
    steps = 16_384 if a.smoke else a.steps
    split = "train" if (a.smoke or a.out) else "test"
    out = Path(a.out) if a.out else HERE / "results_u" / ("smoke" if a.smoke else "")
    if a.arms or a.seeds or a.regions or a.smoke:
        jobs = [(arm, s, r) for arm in (a.arms or ARMS) for s in (a.seeds or ([SEEDS[0]] if a.smoke else [90])) for r in (a.regions or ["DK1"])]
    else:
        jobs = [(arm, s, r) for s in SEEDS for arm in ARMS for r in REGIONS]
    todo = [j for j in jobs if not job_path(out, *j).exists()]
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} complete, running {len(todo)} with {a.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init_worker) as ex:
        futs = {ex.submit(u_job, str(out), arm, s, r, steps, split, a.smoke): (arm, s, r) for arm, s, r in todo}
        for f in as_completed(futs):
            try:
                print(time.strftime("%H:%M:%S"), f.result(), flush=True)
            except Exception as e:  # noqa: BLE001
                print(time.strftime("%H:%M:%S"), "FAILED", futs[f], repr(e)[:300], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
