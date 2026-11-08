"""Study FT runner (PREREGISTRATION_FT.md): accountings that a sign query cannot read, and whether the line through the realized move can.

    python run_ft.py --workers 7                                        # all registered jobs, resume-safe (completed runs are skipped)
    python run_ft.py --smoke --out DIR                                  # development check: 20,000 steps, one seed per arm; excluded
    python run_ft.py --out DIR --seeds 98 --parts A --arms REP LEDGER   # pilot (exploratory)

Training, probes and evaluation follow Studies T to Z: PPO, 8 environments, 1,024-step rollouts, batch 256, gamma 0.99, 64x64, reward normalization, 10^6 steps, the S_DK1 sleeve on Study T's market draw,
100 fixed test windows. Part A: the unrealized bucket. Part B: the carried deficit, half of the training episodes in deficit. Part C: the carried deficit, all of them in deficit. Values are percent of K0 per episode.
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
STEPS = 1_000_000
N_ENVS = 8
GAMMA = 0.99
ARMS = ("REP", "LEDGER", "SPR", "TIR", "TSR")
PARTS = ("A", "B", "C")
SEEDS = list(range(320, 326))
N_TEST = 100
LOW_E = (0.0, 0.02, 0.05, 0.1)
BUCKET_DEFICITS = (-0.05, -0.5, -2.0)


def job_path(out: Path, part: str, arm: str, seed: int) -> Path:
    return out / f"FT_{part}__{arm}__s{seed}.json"


def _paths():
    sys.path[:0] = [str(LIAB), str(HERE), str(HERE.parent / "study_t")]


def build_env(part: str, arm: str, seed: int):
    from envs_ft import BucketEnv, DeficitEnv
    from rewards_ft import make_spr, make_tir, make_tsr
    mode = "booked" if arm == "LEDGER" else "reported"
    if part == "A":
        base = BucketEnv("train", "edge", mode, seed=seed, zero_starts=True)
    else:
        base = DeficitEnv("train", "edge", mode, seed=seed, deficit_starts=True if part == "B" else 1.0)
    if arm in ("REP", "LEDGER"):
        return base
    return {"SPR": make_spr, "TIR": make_tir, "TSR": make_tsr}[arm](base)


def forward(layers, X):
    x = np.asarray(X, dtype=np.float64)
    for kind, w, b in layers:
        x = x @ w.T + b if kind == "lin" else (np.tanh(x) if kind == "tanh" else np.maximum(x, 0.0))
    return x


def ft_job(out_dir: str, part: str, arm: str, seed: int, steps: int, smoke: bool) -> str:
    out = job_path(Path(out_dir), part, arm, seed)
    if out.exists():
        return f"skip {out.name}"
    _paths()
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from envs_ft import K0, LEV, NLEV, REALIZE, BucketEnv, DeficitEnv
    from rewards_ft import T_FAR, line_reward, tail_estimate, twin_reward
    from run_b import network_layers
    t0 = time.time()

    fns = [(lambda s: (lambda: build_env(part, arm, s)))(seed * 100 + i) for i in range(N_ENVS)]
    venv = VecNormalize(DummyVecEnv(fns), norm_obs=False, norm_reward=True, gamma=GAMMA)
    model = PPO("MlpPolicy", venv, seed=seed, n_steps=1024, batch_size=256, gamma=GAMMA, policy_kwargs=dict(net_arch=[64, 64]), device="cpu", verbose=0)
    model.learn(total_timesteps=steps)
    layers = network_layers(model, "PPO")

    def act_many(obs_rows):
        logits = forward(layers, obs_rows)
        lev = np.argmax(logits[:, :NLEV], axis=1)
        if part == "A":
            return lev, np.argmax(logits[:, NLEV:], axis=1)
        return lev, None

    make = (lambda split, kind, mode, s: BucketEnv(split, kind, mode, seed=s)) if part == "A" else (lambda split, kind, mode, s: DeficitEnv(split, kind, mode, seed=s))
    chk = make("train", "edge", "reported", seed + 1000)
    o, _ = chk.reset()
    for _ in range(300):                                   # exactness check of the NumPy forward pass
        lev, f = act_many(np.asarray(o, dtype=np.float32)[None, :])
        slow = np.asarray(model.predict(o, deterministic=True)[0]).ravel()
        assert int(lev[0]) == int(slow[0]) and (f is None or int(f[0]) == int(slow[1])), "fast policy mismatch"
        o, _, d, _, _ = chk.step(chk.action_space.sample())
        if d:
            o, _ = chk.reset()

    def action_of(obs):
        lev, f = act_many(obs[None, :])
        return np.array([lev[0], f[0]]) if part == "A" else int(lev[0])

    # probes: greedy leverage (and realized fraction) at low reported equity and, in Part A, at an empty account with a negative bucket
    probes = {}
    for z in (-1, 1):
        for rem in (0.05, 0.5):
            for e in LOW_E + (0.25, 0.5, 1.0, 2.0):
                obs = np.array([[e, z, rem, 0.0]], dtype=np.float32)
                lev, f = act_many(obs)
                probes[f"low|{e:g}|{z:+d}|{rem:g}"] = [float(LEV[lev[0]]), float(REALIZE[f[0]]) if f is not None else 0.0]
            if part == "A":
                for b in BUCKET_DEFICITS:
                    obs = np.array([[np.clip(b, -5, 5), z, rem, np.clip(b, -5, 5)]], dtype=np.float32)
                    lev, f = act_many(obs)
                    probes[f"def|{b:g}|{z:+d}|{rem:g}"] = [float(LEV[lev[0]]), float(REALIZE[f[0]])]

    def run_episode(env, options, audit: bool):
        o, _ = env.reset(options=options)
        done = False
        acc = dict(rep=0.0, spr=0.0, tir=0.0, tsr=0.0, lower=0.0, gross=0.0, inside=1, lev=0.0, steps=0, credit_steps=0)
        while not done:
            a = action_of(env._obs())
            if audit:
                sigma = env.realized_sign(a)
                r_twin = twin_reward(env, a)
                g_tail = tail_estimate(env, a)
            lev = env.leverage(a)
            _, r, done, _, info = env.step(a)
            c = info["cost"] / K0 * 100.0
            acc["rep"] += r
            acc["gross"] += abs(info["gross"])
            acc["lev"] += abs(lev)
            acc["steps"] += 1
            if audit:
                g_hat = max(r, r_twin) + c if abs(r - r_twin) > 1e-12 else 0.0
                acc["spr"] += 0.5 * (r - r_twin) - c
                acc["tir"] += sigma * g_hat - c
                acc["tsr"] += g_tail - c
                lo, x = -r_twin - 2.0 * c, info["pnl"] / K0 * 100.0
                acc["lower"] += lo
                acc["inside"] &= int(lo <= x + 1e-9 and x <= r + 1e-9)
        booked = (env.ledger - env.e0) / K0 * 100.0
        return booked, acc, env.forgiven / K0 * 100.0

    def starts(rng, kind_start):
        if kind_start == "funded":
            return {"e0": float(np.exp(rng.uniform(np.log(0.05), np.log(2.0))))}
        if kind_start == "zero":
            return {"e0": 0.0}
        return ({"cash0": 0.0, "bucket0": -0.5} if part == "A" else {"e0": -0.5})     # "deficit"

    def test(kind, n_ep, s, kind_start):
        env = make("test", kind, "reported", s)
        rng = np.random.default_rng(s)
        rows = []
        for _ in range(n_ep):
            booked, acc, forgiven = run_episode(env, starts(rng, kind_start), audit=False)
            rows.append((booked, acc["rep"], forgiven, acc["lev"] / acc["steps"]))
        r = np.array(rows)
        return {"booked_pct_K0": float(r[:, 0].mean()), "booked_se": float(r[:, 0].std(ddof=1) / np.sqrt(len(r))), "reported_pct_K0": float(r[:, 1].mean()), "forgiven_pct_K0": float(r[:, 2].mean()),
                "mean_abs_leverage": float(r[:, 3].mean()), "booked_episodes": r[:, 0].tolist(), "reported_episodes": r[:, 1].tolist()}

    def audit(kind, n_ep, s, kind_start):
        env = make("test", kind, "reported", s)
        rng = np.random.default_rng(s)
        rows, inside = [], []
        for _ in range(n_ep):
            booked, acc, _ = run_episode(env, starts(rng, kind_start), audit=True)
            rows.append((acc["rep"], acc["spr"], acc["tir"], acc["tsr"], acc["lower"], booked))
            inside.append(acc["inside"])
        r = np.array(rows)
        booked = r[:, 5]
        res = {"booked": float(booked.mean()), "steps_all_inside_bracket": int(np.sum(inside)), "episodes_lower_le_booked": int(np.sum(r[:, 4] <= booked + 1e-6)), "episodes_booked_le_reported": int(np.sum(booked <= r[:, 0] + 1e-6))}
        for k, j in (("reported", 0), ("spr", 1), ("tir", 2), ("tsr", 3), ("lower", 4)):
            res[k] = float(r[:, j].mean())
            res[f"mae_{k}"] = float(np.abs(r[:, j] - booked).mean())
            res[f"maxerr_{k}"] = float(np.abs(r[:, j] - booked).max())
        return res

    n_ep = 4 if smoke else N_TEST
    res = dict(part=part, arm=arm, seed=seed, steps=steps, probes=probes,
               test_edge=test("edge", n_ep, 9000, "funded"), test_zero=test("edge", n_ep, 9000, "zero"), test_deficit=test("edge", n_ep, 9000, "deficit"), test_placebo=test("placebo", n_ep, 9500, "funded"),
               audit_edge=audit("edge", n_ep, 9000, "funded"), audit_deficit=audit("edge", n_ep, 9000, "deficit"), audit_placebo=audit("placebo", n_ep, 9500, "funded"))
    res["seconds"] = round(time.time() - t0, 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    low = [abs(v[0]) for k, v in probes.items() if k.startswith("low|") and float(k.split("|")[1]) <= 0.1]
    return (f"done {out.name} ({res['seconds']}s): X_low {np.mean(np.array(low) == 16):.2f}; E funded {res['test_edge']['booked_pct_K0']:+.1f} zero {res['test_zero']['booked_pct_K0']:+.1f} deficit {res['test_deficit']['booked_pct_K0']:+.1f}; "
            f"placebo rep {res['test_placebo']['reported_pct_K0']:+.1f} booked {res['test_placebo']['booked_pct_K0']:+.1f}; audit MAE tsr {res['audit_edge']['mae_tsr']:.2e} tir {res['audit_edge']['mae_tir']:.2f} spr {res['audit_edge']['mae_spr']:.2f}")


def _init_worker():
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--parts", nargs="+", default=None)
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    a = ap.parse_args()
    steps = 20_000 if a.smoke else a.steps
    out = Path(a.out) if a.out else HERE / "results_ft"
    seeds = a.seeds or (SEEDS[:1] if a.smoke else SEEDS)
    jobs = [(p, arm, s) for s in seeds for p in (a.parts or PARTS) for arm in (a.arms or ARMS) if a.seeds or a.smoke or arm != "TSR" or s in SEEDS[:3]]
    todo = [j for j in jobs if not job_path(out, *j).exists()]
    print(time.strftime("%Y-%m-%d %H:%M:%S"), f"{len(jobs)} jobs, {len(jobs) - len(todo)} complete, running {len(todo)} with {a.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init_worker) as ex:
        futs = {ex.submit(ft_job, str(out), p, arm, s, steps, a.smoke): (p, arm, s) for p, arm, s in todo}
        for f in as_completed(futs):
            try:
                print(time.strftime("%H:%M:%S"), f.result(), flush=True)
            except Exception as e:  # noqa: BLE001
                print(time.strftime("%H:%M:%S"), "FAILED", futs[f], repr(e)[:400], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
