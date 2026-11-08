"""Study Z runner (PREREGISTRATION_Z.md): does splitting the account defeat the cheap remedy, and do the twin rewards still work?

    python run_z.py --workers 7                  # all registered jobs, resume-safe (completed runs are skipped)
    python run_z.py --smoke --out DIR            # development check: 20,000 steps, one seed per arm; excluded
    python run_z.py --out DIR --seeds 99 --arms REP RUINP5   # pilot (exploratory)

Training, probes and evaluation follow Study X's run_x.py (PPO, 8 environments, 1,024-step rollouts, batch 256, gamma 0.99, 64x64, reward
normalization, 10^6 steps, half of the training episodes from zero equity; 100 fixed test windows), with the one-account sleeve replaced by the
two-account sleeve of envs_z.py and the action by a leverage per account.
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
SPLIT_ARMS = ("REP", "LEDGER", "RUINP5", "RUINP25", "RUINA5", "SPR", "TIR")
POOLED_ARMS = ("P_REP", "P_RUIN5")
ARMS = SPLIT_ARMS + POOLED_ARMS
SEEDS = list(range(210, 218))
LABEL = "Z_DK1+Z"
PROBE_E = (0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)      # symmetric probes: total equity e (multiples of K0), split equally
PROBE_H = (0.25, 0.5, 1.0, 2.0)                             # asymmetric probes: one account empty, the other at h
N_TEST = 100


def job_path(out: Path, arm: str, seed: int) -> Path:
    return out / f"{LABEL}__{arm}__s{seed}.json"


def _paths():
    sys.path[:0] = [str(LIAB), str(HERE), str(HERE.parent / "study_t")]


def greedy_md(layers, X: np.ndarray, nlev: int) -> np.ndarray:
    """Deterministic greedy (leverage index of A, of B) for a batch of observations; identical to model.predict(deterministic=True)."""
    x = np.asarray(X, dtype=np.float64)
    for kind, w, b in layers:
        x = x @ w.T + b if kind == "lin" else (np.tanh(x) if kind == "tanh" else np.maximum(x, 0.0))
    return np.stack([np.argmax(x[:, :nlev], axis=1), np.argmax(x[:, nlev:], axis=1)], axis=1)


def build_env(arm: str, seed: int):
    from envs_z import make_env
    from rewards_z import make_ruin_accounts, make_ruin_portfolio, make_spr, make_tir
    structure = "pooled" if arm.startswith("P_") else "split"
    mode = "booked" if arm == "LEDGER" else "reported"
    base = make_env(structure, "train", "edge", mode, seed=seed, zero_starts=True)
    if arm in ("REP", "LEDGER", "P_REP"):
        return base
    if arm.startswith("RUINP") or arm == "P_RUIN5":
        return make_ruin_portfolio(base, float(arm.replace("P_RUIN", "").replace("RUINP", "")))
    if arm.startswith("RUINA"):
        return make_ruin_accounts(base, float(arm.replace("RUINA", "")))
    if arm == "SPR":
        return make_spr(base)
    if arm == "TIR":
        return make_tir(base)
    raise ValueError(arm)


def z_job(out_dir: str, arm: str, seed: int, steps: int, smoke: bool) -> str:
    out = job_path(Path(out_dir), arm, seed)
    if out.exists():
        return f"skip {out.name}"
    _paths()
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from envs_z import K0, LEV, NLEV, make_env
    from run_b import network_layers
    t0 = time.time()
    structure = "pooled" if arm.startswith("P_") else "split"

    fns = [(lambda s: (lambda: build_env(arm, s)))(seed * 100 + i) for i in range(N_ENVS)]
    venv = VecNormalize(DummyVecEnv(fns), norm_obs=False, norm_reward=True, gamma=GAMMA)
    kw = dict(seed=seed, n_steps=1024, batch_size=256, gamma=GAMMA, policy_kwargs=dict(net_arch=[64, 64]), device="cpu", verbose=0)
    model = PPO("MlpPolicy", venv, **kw)
    model.learn(total_timesteps=steps)
    layers = network_layers(model, "PPO")
    chk = make_env(structure, "train", "edge", "reported", seed=seed + 1000)
    o, _ = chk.reset()
    for _ in range(300):                                   # exactness check of the NumPy forward pass
        fast = greedy_md(layers, np.asarray(o, dtype=np.float32)[None, :], NLEV)[0]
        slow = np.asarray(model.predict(o, deterministic=True)[0]).ravel()
        assert int(fast[0]) == int(slow[0]) and int(fast[1]) == int(slow[1]), "fast policy mismatch"
        o, _, d, _, _ = chk.step(chk.action_space.sample())
        if d:
            o, _ = chk.reset()

    def act(obs_rows):
        g = greedy_md(layers, np.asarray(obs_rows, dtype=np.float64), NLEV)
        return LEV[g[:, 0]], LEV[g[:, 1]]

    probes = {}
    for z in (-1, 1):
        for rem in (0.05, 0.5):
            for e in PROBE_E:
                la, lb = act([[e / 2, e / 2, z, rem]])
                probes[f"sym|{e:g}|{z:+d}|{rem:g}"] = [float(la[0]), float(lb[0])]
            for h in PROBE_H:
                la, lb = act([[0.0, h, z, rem]])
                probes[f"emptyA|{h:g}|{z:+d}|{rem:g}"] = [float(la[0]), float(lb[0])]
                la, lb = act([[h, 0.0, z, rem]])
                probes[f"emptyB|{h:g}|{z:+d}|{rem:g}"] = [float(la[0]), float(lb[0])]

    def run_episode(env, e0, twins: bool):
        o, _ = env.reset(options={"e0": e0})
        done = False
        acc = dict(rep=0.0, spr=0.0, tir=0.0, lower=0.0, gross=0.0, net=0.0, inside=1)
        steps_ = 0
        while not done:
            a = greedy_md(layers, env._obs()[None, :], NLEV)[0]
            if twins:
                qa, qb, _, _ = env.legs(a)
                sigma = float(np.sign(qa + qb)) * float(env.eps_ep[env.t - env.t0])
                twin = copy.copy(env)
                twin.flip_next = True
                r_twin = twin.step(a)[1]
            _, r, done, _, info = env.step(a)
            c = info["cost"] / K0 * 100.0
            acc["rep"] += r
            acc["gross"] += info["gross"]
            acc["net"] += abs(info["net"])
            steps_ += 1
            if twins:
                g_hat = max(r, r_twin) + c if abs(r - r_twin) > 1e-12 else 0.0
                acc["spr"] += 0.5 * (r - r_twin) - c
                acc["tir"] += sigma * g_hat - c
                lo, x = -r_twin - 2.0 * c, info["pnl"] / K0 * 100.0
                acc["lower"] += lo
                acc["inside"] &= int(lo <= x + 1e-9 and x <= r + 1e-9)       # the step bracket of Proposition 10
        booked = (env.ledger - env.e0) / K0 * 100.0
        return booked, acc, env.forgiven / K0 * 100.0, env.n_ruin_portfolio, env.n_ruin_accounts, steps_

    def test(kind, n_ep, s, e0_fixed=None):
        env = make_env(structure, "test", kind, "reported", seed=s)
        rng = np.random.default_rng(s)
        rows = []
        for _ in range(n_ep):
            e0 = e0_fixed if e0_fixed is not None else float(np.exp(rng.uniform(np.log(0.05), np.log(2.0))))
            booked, acc, forgiven, rp, ra, n = run_episode(env, e0, twins=False)
            rows.append((booked, acc["rep"], forgiven, rp, ra, acc["gross"] / n, acc["net"] / n))
        r = np.array(rows)
        return {"booked_pct_K0": float(r[:, 0].mean()), "booked_se": float(r[:, 0].std(ddof=1) / np.sqrt(len(r))),
                "reported_pct_K0": float(r[:, 1].mean()), "forgiven_pct_K0": float(r[:, 2].mean()),
                "ruin_portfolio": float(r[:, 3].mean()), "ruin_accounts": float(r[:, 4].mean()),
                "gross_mwh": float(r[:, 5].mean()), "net_mwh": float(r[:, 6].mean()),
                "reported_episodes": r[:, 1].tolist(), "booked_episodes": r[:, 0].tolist()}

    def audit(kind, n_ep, s):
        """Ledger-free estimates on the test episodes of test(): sums of the reported, SPR and TIR rewards and of the bracket's lower end."""
        env = make_env(structure, "test", kind, "reported", seed=s)
        rng = np.random.default_rng(s)
        rows, inside = [], []
        for _ in range(n_ep):
            booked, acc, *_ = run_episode(env, float(np.exp(rng.uniform(np.log(0.05), np.log(2.0)))), twins=True)
            rows.append((acc["rep"], acc["spr"], acc["tir"], acc["lower"], booked))
            inside.append(acc["inside"])
        r = np.array(rows)
        booked = r[:, 4]
        res = {"booked": float(booked.mean()), "steps_all_inside_bracket": int(np.sum(inside)),
               "episodes_lower_le_booked": int(np.sum(r[:, 3] <= booked + 1e-6)),
               "episodes_booked_le_reported": int(np.sum(booked <= r[:, 0] + 1e-6))}
        for k, j in (("reported", 0), ("spr", 1), ("tir", 2), ("lower", 3)):
            res[k] = float(r[:, j].mean())
            res[f"mae_{k}"] = float(np.abs(r[:, j] - booked).mean())
        res["episodes"] = {"booked": booked.tolist(), "reported": r[:, 0].tolist(), "spr": r[:, 1].tolist(), "tir": r[:, 2].tolist(),
                           "lower": r[:, 3].tolist()}
        return res

    n_ep = 4 if smoke else N_TEST
    res = dict(setting=LABEL, arm=arm, structure=structure, seed=seed, steps=steps, probes=probes,
               test_edge=test("edge", n_ep, 9000), test_placebo=test("placebo", n_ep, 9500),
               test_zero=test("edge", n_ep, 9000, e0_fixed=0.0),
               audit_edge=audit("edge", n_ep, 9000), audit_placebo=audit("placebo", n_ep, 9500))
    for k in ("edge", "placebo"):                          # the audit replays the same episodes as the test
        assert abs(res[f"test_{k}"]["booked_pct_K0"] - res[f"audit_{k}"]["booked"]) < 1e-6, "audit episodes differ from test episodes"
        assert abs(res[f"test_{k}"]["reported_pct_K0"] - res[f"audit_{k}"]["reported"]) < 1e-6, "audit reported return differs"
    res["seconds"] = round(time.time() - t0, 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    return (f"done {out.name} ({res['seconds']}s): booked edge {res['test_edge']['booked_pct_K0']:+.1f}%, placebo reported "
            f"{res['test_placebo']['reported_pct_K0']:+.1f}% booked {res['test_placebo']['booked_pct_K0']:+.1f}%")


def _init_worker():
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None, help="output folder (default: results_z/ next to this file)")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    a = ap.parse_args()
    steps = 20_000 if a.smoke else a.steps
    out = Path(a.out) if a.out else HERE / "results_z"
    seeds = a.seeds or (SEEDS[:1] if a.smoke else SEEDS)
    jobs = [(arm, s) for s in seeds for arm in (a.arms or ARMS)]
    todo = [j for j in jobs if not job_path(out, *j).exists()]
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} complete, running {len(todo)} with {a.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init_worker) as ex:
        futs = {ex.submit(z_job, str(out), arm, s, steps, a.smoke): (arm, s) for arm, s in todo}
        for f in as_completed(futs):
            try:
                print(time.strftime("%H:%M:%S"), f.result(), flush=True)
            except Exception as e:  # noqa: BLE001
                print(time.strftime("%H:%M:%S"), "FAILED", futs[f], repr(e)[:300], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
