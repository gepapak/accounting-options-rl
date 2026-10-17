"""Study N runner (PREREGISTRATION_N.md): matched forgiven-versus-booked accounting in two settings (zombie window,
terminal limited liability). PPO with Study F's well-posed configuration (8 parallel environments, reward
normalization, 1M steps), seeds 0-4, DK1/DK2, placebo market.

    python run_n.py --workers 7                  # full study (TEST split evaluation)
    python run_n.py --pilot --steps 300000       # disclosed pilot: seed 0, DK1, TRAIN split only
    python run_n.py --smoke                      # plumbing check
"""
from __future__ import annotations
import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
LIAB = Path(r"C:/Users/gepap/Desktop/Thesis/Paper1/rl-trading-reality-check-main/LIABILITY_STUDY_2026-09-29")
sys.path.insert(0, str(LIAB))
ARMS = ["ZFG", "ZBK", "TFG", "TBK"]
SEEDS = [0, 1, 2, 3, 4]
REGIONS = ["DK1", "DK2"]
STEPS = 1_000_000
N_ENVS = 8
N_DRAWS = 20
Z_PROBE_N = (-0.5, -0.25, -0.1, -0.02)
Z_PROBE_K = (96, 48, 12)
Z_PROBE_E = (0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0)
T_PROBE_L = (-0.25, -0.1, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
T_PROBE_TAU = (1, 4, 16, 48, 96, 192)


def job_path(arm, seed, region, tag):
    return HERE / f"results_n{tag}" / "jobs" / f"{arm}__PPO__s{seed}__{region}.json"


def n_job(arm: str, seed: int, region: str, steps: int, split: str, tag: str) -> str:
    out = job_path(arm, seed, region, tag)
    if out.exists():
        return f"skip {out.name}"
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from envs_n import H_TERM, LEV, W_LAG, K0, LagEnv, load_market
    from run_b import greedy, network_layers
    t0 = time.time()
    mtr = load_market(region, "train", "NE")
    venv = DummyVecEnv([(lambda i: (lambda: LagEnv(mtr, arm, train=True, seed=seed * 100 + i)))(i) for i in range(N_ENVS)])
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=0.99)
    model = PPO("MlpPolicy", venv, seed=seed, n_steps=1024, batch_size=256, policy_kwargs=dict(net_arch=[64, 64]),
                device="cpu", verbose=0)
    model.learn(total_timesteps=steps)
    layers = network_layers(model, "PPO")
    chk = LagEnv(mtr, arm, train=True, seed=seed + 1000)
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
    # placebo-market evaluation over independent sign draws (exploratory outcomes)
    draws = []
    for k in range(3 if tag == "_smoke" else N_DRAWS):
        meval = load_market(region, split, "NE", draw=k)
        if arm[0] == "Z":
            env = LagEnv(meval, arm, train=False)
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
                env = LagEnv(meval, arm, train=False)
                o, _ = env.reset(options={"e0": 1.0, "t0": t0b, "tau0": H_TERM})
                done = False
                while not done:
                    o, _, te, tr, _ = env.step(int(greedy(layers, np.asarray(o)[None, :].astype(np.float64))[0]))
                    done = te or tr
                reps.append(env.reported() / K0 - 1); books.append(env.L / K0 - 1)
            draws.append(dict(reported_pct=100 * float(np.mean(reps)), booked_pct=100 * float(np.mean(books))))
    res = dict(arm=arm, algo="PPO", seed=seed, region=region, steps=steps, eval_split=split, probes=probes,
               drawmean_reported_pct=float(np.mean([d["reported_pct"] for d in draws])),
               drawmean_booked_pct=float(np.mean([d["booked_pct"] for d in draws])), draws=draws,
               seconds=round(time.time() - t0, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    key = "zombie_n-0.1_k48" if arm[0] == "Z" else "L0.1_tau16"
    return f"done {out.name} ({res['seconds']}s): {key}={probes[key]:g} rep={res['drawmean_reported_pct']:.1f} book={res['drawmean_booked_pct']:.1f}"


def _init_worker():
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--steps", type=int, default=STEPS)
    a = ap.parse_args()
    if a.smoke:
        jobs, steps, split, tag = [(r, 0, "DK1") for r in ARMS], 16_384, "train", "_smoke"
    elif a.pilot:
        jobs, steps, split, tag = [(r, 0, "DK1") for r in ARMS], a.steps, "train", "_pilot"
    else:
        jobs, steps, split, tag = [(r, s, g) for s in SEEDS for r in ARMS for g in REGIONS], STEPS, "test", ""
    todo = [j for j in jobs if not job_path(*j, tag).exists()]
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} complete, running {len(todo)} with {a.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init_worker) as ex:
        futs = {ex.submit(n_job, *j, steps, split, tag): j for j in todo}
        for f in as_completed(futs):
            try:
                print(time.strftime("%H:%M:%S"), f.result(), flush=True)
            except Exception as e:  # noqa: BLE001
                print(time.strftime("%H:%M:%S"), "FAILED", futs[f], repr(e)[:400], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
