"""Study RS runner (PREREGISTRATION_RS.md): Study T's edge-market test with the real sign of every price move.

    python run_rs.py --workers 8 --finish     # all registered jobs, resume-safe; when all are done, runs aggregate_rs.py once
    python run_rs.py --smoke                  # development check: 20,000 steps, every arm, one seed, 4 test episodes; excluded
    python run_rs.py --pilot                  # testbed pilot 2 (the registered testbed): PLAIN and ORACLE, seeds 98-99 -> pilot2_results/
    python run_rs.py --pilot1                 # testbed pilot 1 (no zero-equity training starts): same jobs -> pilot_results/
Training environments start half of their episodes at zero equity (envs_rs.EdgeEnvRS zero_starts, as Study V Part C); --pilot1 omits this.
Progress lines name the finished run and its duration only (no outcome), so the run can be monitored blind.
"""
from __future__ import annotations

import argparse
import copy
import ctypes
import datetime
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LIAB = HERE.parent / "github" / "accounting-options-rl" / "LIABILITY_STUDY_2026-09-29"
STEPS = 1_000_000
N_ENVS = 8
GAMMA = 0.99
ARMS = ("PLAIN", "ORACLE", "SPR", "TIR", "RUIN5")
SETTINGS = ("RS_DK1", "RS_DK2")         # smoke and pilots
REGISTERED_SETTINGS = ("RS_DK2",)       # the registered run (DK1 dropped after the pilots; see PREREGISTRATION_RS.md)
SEEDS = list(range(300, 316))
PILOT_ARMS, PILOT_SEEDS = ("PLAIN", "ORACLE"), (98, 99)
PROBE_E = (0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
PROBE_Z = (-1, 0, 1)
N_TEST = 100            # the same 100 test windows and start equities for every run (fixed evaluation seeds)
ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001


def job_path(out: Path, setting: str, arm: str, seed: int) -> Path:
    return out / f"{setting}__{arm}__s{seed}.json"


def rs_job(out_dir: str, setting: str, arm: str, seed: int, steps: int, smoke: bool, zero_starts: bool = True) -> str:
    out = job_path(Path(out_dir), setting, arm, seed)
    if out.exists():
        return f"skip {out.name}"
    sys.path[:0] = [str(HERE), str(LIAB)]
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from envs_rs import K0, LEV, EdgeEnvRS
    from rewards_v import make_ruin
    from run_b import greedy, network_layers
    from spr_t import make_spr, make_tir
    t0 = time.time()
    mode = "booked" if arm == "ORACLE" else "reported"
    wrap = {"SPR": make_spr, "TIR": make_tir, "RUIN5": lambda e: make_ruin(e, 5.0)}.get(arm, lambda e: e)
    fns = [(lambda s: (lambda: wrap(EdgeEnvRS(setting, "train", "edge", mode, seed=s, zero_starts=zero_starts))))(seed * 100 + i)
           for i in range(N_ENVS)]
    venv = VecNormalize(DummyVecEnv(fns), norm_obs=False, norm_reward=True, gamma=GAMMA)
    model = PPO("MlpPolicy", venv, seed=seed, n_steps=1024, batch_size=256, gamma=GAMMA, policy_kwargs=dict(net_arch=[64, 64]),
                device="cpu", verbose=0)
    model.learn(total_timesteps=steps)
    layers = network_layers(model, "PPO")
    chk = EdgeEnvRS(setting, "train", "edge", "reported", seed=seed + 1000)
    o, _ = chk.reset()
    for _ in range(300):                                   # exactness check of the NumPy forward pass
        assert int(greedy(layers, np.asarray(o, dtype=np.float32)[None, :])[0]) == int(model.predict(o, deterministic=True)[0]), "fast policy mismatch"
        o, _, d, _, _ = chk.step(chk.action_space.sample())
        if d:
            o, _ = chk.reset()
    lev = {}
    for e in PROBE_E:
        for z in PROBE_Z:
            for rem in (0.05, 0.5):
                lev[f"{e:g}|{z:+d}|{rem:g}"] = float(LEV[int(greedy(layers, np.array([[e, z, rem, 0.0]]))[0])])

    def run_episode(env, e0, twins):
        o, _ = env.reset(options={"e0": e0})
        done = False
        acc = dict(rep=0.0, spr=0.0, tir=0.0, lower=0.0, steps=0, inside=0, ident=0, ident_inexact=0, max_unident_err=0.0, n_lev16=0)
        while not done:
            a = int(greedy(layers, o[None, :])[0])
            if twins:
                sigma = float(np.sign(LEV[a])) * float(env.eps_ep[env.t - env.t0])
                twin = copy.copy(env)
                twin.flip_next = True
                r_twin = twin.step(a)[1]
            o, r, done, _, info = env.step(a)
            acc["rep"] += r
            acc["steps"] += 1
            acc["n_lev16"] += int(abs(LEV[a]) == 16.0)
            if twins:
                c, x = info["cost"] / K0 * 100.0, info["pnl"] / K0 * 100.0
                ident = abs(r - r_twin) > 1e-12
                tir = sigma * (max(r, r_twin) + c if ident else 0.0) - c
                acc["spr"] += 0.5 * (r - r_twin) - c
                acc["tir"] += tir
                lo = -r_twin - 2.0 * c
                acc["lower"] += lo
                acc["inside"] += int(lo <= x + 1e-9 and x <= r + 1e-9)          # the step bracket
                if ident:
                    acc["ident"] += 1
                    acc["ident_inexact"] += int(abs(tir - x) > 1e-9)
                else:
                    acc["max_unident_err"] = max(acc["max_unident_err"], abs(tir - x))
        booked = (env.ledger - env.e0) / K0 * 100.0
        return booked, acc, env.forgiven / K0 * 100.0

    def episodes(kind, n_ep, s, twins, e0_fixed=None):
        env = EdgeEnvRS(setting, "test", kind, "reported", seed=s)
        rng = np.random.default_rng(s)
        out_ = []
        for _ in range(n_ep):
            e0 = e0_fixed if e0_fixed is not None else float(np.exp(rng.uniform(np.log(0.05), np.log(2.0))))
            out_.append(run_episode(env, e0, twins))
        return out_

    def test(kind, n_ep, s, e0_fixed=None):
        rows = [(b, acc["rep"], f, acc["n_lev16"] / acc["steps"]) for b, acc, f in episodes(kind, n_ep, s, False, e0_fixed)]
        r = np.array(rows)
        return {"booked_pct_K0": float(r[:, 0].mean()), "booked_se": float(r[:, 0].std(ddof=1) / np.sqrt(len(r))),
                "reported_pct_K0": float(r[:, 1].mean()), "forgiven_pct_K0": float(r[:, 2].mean()), "share_lev16": float(r[:, 3].mean()),
                "booked_episodes": r[:, 0].tolist()}

    def audit(kind, n_ep, s):
        """Ledger-free estimates on the test episodes of test(): sums of the reported, SPR and TIR rewards and of the bracket's lower end."""
        eps_ = episodes(kind, n_ep, s, True)
        booked = np.array([b for b, _, _ in eps_])
        res = {"booked": float(booked.mean()), "steps": int(sum(a["steps"] for _, a, _ in eps_)),
               "steps_inside_bracket": int(sum(a["inside"] for _, a, _ in eps_)),
               "steps_identified": int(sum(a["ident"] for _, a, _ in eps_)),
               "identified_steps_tir_inexact": int(sum(a["ident_inexact"] for _, a, _ in eps_)),
               "max_unidentified_step_error": float(max(a["max_unident_err"] for _, a, _ in eps_))}
        for k in ("rep", "spr", "tir", "lower"):
            v = np.array([a[k] for _, a, _ in eps_])
            name = "reported" if k == "rep" else k
            res[name] = float(v.mean())
            res[f"err_{name}"] = float(abs(v.mean() - booked.mean()))          # error of the mean over the test episodes
            res[f"mae_{name}"] = float(np.abs(v - booked).mean())            # mean absolute per-episode error
            res[f"episodes_{name}"] = v.tolist()
        res["episodes_lower_le_booked"] = int(sum(a["lower"] <= b + 1e-6 for b, a, _ in eps_))
        res["episodes_booked_le_reported"] = int(sum(b <= a["rep"] + 1e-6 for b, a, _ in eps_))
        return res

    n_ep = 4 if smoke else N_TEST
    res = dict(setting=setting, arm=arm, seed=seed, steps=steps, zero_starts=zero_starts, lev=lev, test_edge=test("edge", n_ep, 9000),
               test_placebo=test("placebo", n_ep, 9500), test_zero=test("edge", n_ep, 9000, e0_fixed=0.0),
               audit_edge=audit("edge", n_ep, 9000), audit_placebo=audit("placebo", n_ep, 9500))
    for k in ("edge", "placebo"):                          # the audit replays the same episodes as the test
        assert abs(res[f"test_{k}"]["booked_pct_K0"] - res[f"audit_{k}"]["booked"]) < 1e-6, "audit episodes differ from test episodes"
        assert abs(res[f"test_{k}"]["reported_pct_K0"] - res[f"audit_{k}"]["reported"]) < 1e-6, "audit reported return differs"
    res["seconds"] = round(time.time() - t0, 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    return f"done {out.name} ({res['seconds']}s)"


def _init_worker():
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"


def run_log(text: str) -> None:
    with open(HERE / "RUN_LOG_RS.md", "a", encoding="utf-8") as f:
        f.write(text + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--pilot1", action="store_true")
    ap.add_argument("--finish", action="store_true", help="after a complete registered run, run aggregate_rs.py once")
    a = ap.parse_args()
    if os.name == "nt":                                    # hold idle sleep while this process runs (per-process request; ends with it)
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    zero_starts = not a.pilot1
    if a.smoke:
        steps, out = 20_000, HERE / "results_rs" / "smoke2"
        jobs = [(st, arm, SEEDS[0]) for st in SETTINGS for arm in ARMS]
    elif a.pilot or a.pilot1:
        steps, out = STEPS, HERE / ("pilot_results" if a.pilot1 else "pilot2_results")
        jobs = [(st, arm, s) for s in PILOT_SEEDS for st in SETTINGS for arm in PILOT_ARMS]
    else:
        steps, out = STEPS, HERE / "results_rs"
        jobs = [(st, arm, s) for s in SEEDS for st in REGISTERED_SETTINGS for arm in ARMS]
    todo = [j for j in jobs if not job_path(out, *j).exists()]
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} complete, running {len(todo)} with {a.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init_worker) as ex:
        futs = {ex.submit(rs_job, str(out), st, arm, s, steps, a.smoke, zero_starts): (st, arm, s) for st, arm, s in todo}
        for f in as_completed(futs):
            try:
                print(time.strftime("%H:%M:%S"), f.result(), flush=True)
            except Exception as e:  # noqa: BLE001
                print(time.strftime("%H:%M:%S"), "FAILED", futs[f], repr(e)[:300], flush=True)
    n_done = sum(job_path(out, *j).exists() for j in jobs)
    print(time.strftime("%H:%M:%S"), f"finished: {n_done} of {len(jobs)} result files", flush=True)
    if a.finish and not (a.smoke or a.pilot or a.pilot1):
        hm = datetime.datetime.now().strftime("%H:%M")
        if n_done < len(jobs):
            run_log(f"- **{hm}.** `run_rs.py` exited with {n_done} of {len(jobs)} runs finished; the registered analysis was not run. "
                    f"Resume with `python run_rs.py --workers 8 --finish` (finished runs are skipped).")
            return 1
        if (HERE / "VERDICTS_RS.md").exists():
            print("VERDICTS_RS.md exists; the registered analysis is not run again", flush=True)
            return 0
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        r = subprocess.run([sys.executable, "aggregate_rs.py"], cwd=HERE, capture_output=True, text=True, encoding="utf-8", env=env)
        (HERE / "finish_rs_aggregate.out").write_text(r.stdout + "\n" + r.stderr, encoding="utf-8")
        run_log(f"- **{hm}.** All {len(jobs)} runs finished. The registered analysis (`python aggregate_rs.py`) was run once, automatically "
                f"by `run_rs.py --finish` (exit code {r.returncode}), and wrote `runs_rs.csv` and `VERDICTS_RS.md`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
