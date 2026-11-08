"""Study T runner (PREREGISTRATION_T.md): replication of Study S's edge-market test on a new market draw, the occupancy-measure (ORPO)
baseline, and two further market settings.

    python run_t.py --workers 8                 # all registered jobs, resume-safe (completed runs are skipped)
    python run_t.py --part A --workers 8
    python run_t.py --smoke                     # development check: 20,000 steps, one seed per arm; excluded
Development pilots (exploratory) use --out, --settings, --arms and --seeds.
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
ARMS_A = ("PLAIN", "ORACLE", "SPR", "TIR", "OM1", "OM0.1", "OM0.05", "OM0.02", "OM0.01")      # OM<c>: ORPO sqrt-chi2 with scale-independent coefficient c
SEEDS_A = list(range(120, 130))
SETTING_A = "S_DK1"
ARMS_B = ("PLAIN", "ORACLE", "SPR", "TIR")
SETTINGS_B = ("T_DK2", "T_DK1M")
SEEDS_B = list(range(130, 140))
PROBE_E = (0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
N_TEST = 100            # the same 100 test windows and start equities for every run (fixed evaluation seeds)


def job_path(out: Path, setting: str, arm: str, seed: int) -> Path:
    return out / f"{setting}__{arm}__s{seed}.json"


def _paths():
    sys.path[:0] = [str(LIAB), str(HERE), str(HERE.parent / "study_s")]


def t_job(out_dir: str, setting: str, arm: str, seed: int, steps: int, smoke: bool) -> str:
    out = job_path(Path(out_dir), setting, arm, seed)
    if out.exists():
        return f"skip {out.name}"
    _paths()
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from envs_t import K0, LEV, EdgeEnvT
    from orpo_t import ORPOPPO
    from run_b import greedy, network_layers
    from spr_t import make_spr, make_tir, make_tmr
    t0 = time.time()
    mode = "booked" if arm == "ORACLE" else "reported"
    if arm in ("SPR", "TMR", "TIR"):
        wrap = {"SPR": make_spr, "TMR": make_tmr, "TIR": make_tir}[arm]
        fns = [(lambda s: (lambda: wrap(EdgeEnvT(setting, "train", "edge", "reported", seed=s))))(seed * 100 + i) for i in range(N_ENVS)]
    else:
        fns = [(lambda s: (lambda: EdgeEnvT(setting, "train", "edge", mode, seed=s)))(seed * 100 + i) for i in range(N_ENVS)]
    venv = VecNormalize(DummyVecEnv(fns), norm_obs=False, norm_reward=True, gamma=GAMMA)
    kw = dict(seed=seed, n_steps=1024, batch_size=256, gamma=GAMMA, policy_kwargs=dict(net_arch=[64, 64]), device="cpu", verbose=0)
    if arm.startswith("OM"):
        model = ORPOPPO("MlpPolicy", venv, **kw)
        ref_fns = [(lambda s: (lambda: EdgeEnvT(setting, "train", "edge", "reported", seed=s)))(seed * 100 + 50 + i) for i in range(N_ENVS)]
        model.configure_orpo(ref_fns, coeff_c=float(arm[2:]), seed=seed)
    else:
        model = PPO("MlpPolicy", venv, **kw)
    model.learn(total_timesteps=steps)
    layers = network_layers(model, "PPO")
    chk = EdgeEnvT(setting, "train", "edge", "reported", seed=seed + 1000)
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

    def test(kind, n_ep, s):
        env = EdgeEnvT(setting, "test", kind, "reported", seed=s)
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

    n_ep = 4 if smoke else N_TEST
    res = dict(setting=setting, arm=arm, seed=seed, steps=steps, lev=lev, test_edge=test("edge", n_ep, 9000),
               test_placebo=test("placebo", n_ep, 9500), seconds=round(time.time() - t0, 1))
    if arm.startswith("OM"):
        res["om_sigma"] = model.om_sigma
        res["om_log_tail"] = model.om_log[-5:]            # (trimmed mean chi2, mean raw penalty, discriminator accuracy)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    return f"done {out.name} ({res['seconds']}s): booked {res['test_edge']['booked_pct_K0']:+.1f}%"


def _init_worker():
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--part", choices=["A", "B", "AB"], default="AB")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None, help="output folder (default: results_t/ next to this file; smoke: results_t/smoke)")
    ap.add_argument("--settings", nargs="+", default=None)
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--steps", type=int, default=STEPS)
    a = ap.parse_args()
    steps = 20_000 if a.smoke else a.steps
    out = Path(a.out) if a.out else HERE / "results_t" / ("smoke" if a.smoke else "")
    jobs = []
    if a.settings or a.arms or a.seeds:                               # exploratory pilots
        for st in a.settings or [SETTING_A]:
            for arm in a.arms or ARMS_B:
                for s in a.seeds or [90]:
                    jobs.append((st, arm, s))
    else:
        if "A" in a.part:
            jobs += [(SETTING_A, arm, s) for s in (SEEDS_A[:1] if a.smoke else SEEDS_A) for arm in ARMS_A]
        if "B" in a.part:
            jobs += [(st, arm, s) for s in (SEEDS_B[:1] if a.smoke else SEEDS_B) for st in SETTINGS_B for arm in ARMS_B]
    todo = [j for j in jobs if not job_path(out, *j).exists()]
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} complete, running {len(todo)} with {a.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init_worker) as ex:
        futs = {ex.submit(t_job, str(out), st, arm, s, steps, a.smoke): (st, arm, s) for st, arm, s in todo}
        for f in as_completed(futs):
            try:
                print(time.strftime("%H:%M:%S"), f.result(), flush=True)
            except Exception as e:  # noqa: BLE001
                print(time.strftime("%H:%M:%S"), "FAILED", futs[f], repr(e)[:300], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
