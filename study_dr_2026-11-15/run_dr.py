"""Study DR runner (PREREGISTRATION_DR.md): PPO trained in gym-mtsim 2.0.0 as published (ORIG) in six arms and without the zero floor
(FLPATCH) in the strongest arm of each lever. Learner, wrapper and measures are Study GC's (run_gc.py), with the arm's training
distribution and leverage (envs_dr.py).

    <venv>\\Scripts\\python run_dr.py --workers 8
    <venv>\\Scripts\\python run_dr.py --smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results_dr"
STEPS = 1_000_000                   # PPO rounds up to whole rollouts of 8 x 1,024 steps: 1,007,616
N_ENVS = 8
SEEDS = {("V2", "ORIG"): range(600, 606), ("V4", "ORIG"): range(600, 610), ("V8", "ORIG"): range(600, 610),
         ("V16", "ORIG"): range(600, 610), ("L30", "ORIG"): range(600, 606), ("L500", "ORIG"): range(600, 610),
         ("V16", "FLPATCH"): range(600, 606), ("L500", "FLPATCH"): range(600, 606)}
ORDER = [("V16", "ORIG"), ("L500", "ORIG"), ("V8", "ORIG"), ("V4", "ORIG"), ("V16", "FLPATCH"), ("L500", "FLPATCH"), ("V2", "ORIG"),
         ("L30", "ORIG")]           # runs that are expected to trade (slower steps) start first
SMOKE_SEED = 699


def job_path(arm: str, cond: str, seed: int, smoke: bool = False) -> Path:
    return OUT / ("jobs_smoke" if smoke else "jobs") / f"{arm}__{cond}__PPO__s{seed}.json"


def evaluate(model, vecnorm, arm: str, bank: np.ndarray) -> dict:
    """Deterministic policy on a bank of test paths from 10,000 USD, in gym-mtsim as published and at the arm's leverage: reported
    return (change in equity) and booked return (the simulator's order records)."""
    import envs_dr as D
    env = D.DREnv(arm, "ORIG", train=False, seed=0)
    rows = []
    for close in bank:
        o, _ = env.reset(options={"path": close, "balance": D.BALANCE})
        done = False
        while not done:
            a = model.predict(vecnorm.normalize_obs(o[None, :]), deterministic=True)[0][0]
            o, _, te, tr, info = env.step(a)
            done = te or tr
        rows.append(info["stats"])
    rep = [100.0 * r["reported"] / D.BALANCE for r in rows]
    led = [100.0 * r["booked"] / D.BALANCE for r in rows]
    return dict(reported_return_pct=float(np.mean(rep)), booked_return_pct=float(np.mean(led)), gap=float(np.mean(rep) - np.mean(led)),
                draws_reported=rep, draws_booked=led, floored_share=float(np.mean([r["floored"] for r in rows])),
                wiped_share=float(np.mean([r["wiped"] for r in rows])), orders_per_episode=float(np.mean([r["orders"] for r in rows])),
                max_leverage=float(np.mean([r["lev_max"] for r in rows])), test_paths=len(rows))


def dr_job(arm: str, cond: str, seed: int, steps: int, smoke: bool) -> str:
    out = job_path(arm, cond, seed, smoke)
    if out.exists():
        return f"skip {out.name}"
    import importlib.metadata

    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    import envs_dr as D
    t0 = time.time()
    base = DummyVecEnv([(lambda i: (lambda: D.DREnv(arm, cond, train=True, seed=seed * 100 + i)))(i) for i in range(N_ENVS)])
    venv = VecNormalize(base, norm_obs=True, norm_reward=True, gamma=0.99)
    model = PPO("MlpPolicy", venv, seed=seed, n_steps=1024, batch_size=256, policy_kwargs=dict(net_arch=[64, 64]), device="cpu",
                verbose=0)
    model.learn(total_timesteps=steps)
    venv.training = False

    def predict(x):                                     # probe through the frozen observation normalization
        return model.predict(venv.normalize_obs(x), deterministic=True)[0][0]
    expo = {f"{e:g}": D.exposure(predict, arm, e) for e in D.EGRID}
    counts = base.get_attr("count")
    train = {k: (max(c[k] for c in counts) if k == "line_max_abs_err" else int(sum(c[k] for c in counts))) for k in counts[0]}
    nominal, stressed = D.nominal_bank(), D.stressed_bank()
    if smoke:
        nominal, stressed = nominal[:5], stressed[:5]
    digest = hashlib.sha256()
    for name, p in sorted(model.policy.state_dict().items()):
        digest.update(name.encode())
        digest.update(p.detach().cpu().numpy().tobytes())
    res = dict(arm=arm, k_hi=D.ARMS[arm]["k_hi"], leverage=D.ARMS[arm]["leverage"], condition=cond, algo="PPO", seed=seed,
               steps=int(model.num_timesteps), exposure=expo, xlow=float(np.mean([expo[f"{e:g}"] for e in D.EGRID if e <= 0.25])),
               test_nominal=evaluate(model, venv, arm, nominal), test_stressed_k4=evaluate(model, venv, arm, stressed), train=train,
               policy_sha256=digest.hexdigest(),
               versions={p: importlib.metadata.version(p) for p in ("gym-mtsim", "pandas", "numpy", "gymnasium", "stable-baselines3", "torch")},
               seconds=round(time.time() - t0, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    pol = out.parent.parent / ("policies_smoke" if smoke else "policies")
    pol.mkdir(parents=True, exist_ok=True)
    model.save(str(pol / out.stem))
    venv.save(str(pol / (out.stem + "__vecnormalize.pkl")))
    out.write_text(json.dumps(res, indent=1))
    n, s = res["test_nominal"], res["test_stressed_k4"]
    return (f"done {out.name} ({res['seconds']}s): x(0.1)={expo['0.1']:.2f} xlow={res['xlow']:.3f} x(1)={expo['1']:.2f} | nominal rep "
            f"{n['reported_return_pct']:+.2f} booked {n['booked_return_pct']:+.2f} | k4 rep {s['reported_return_pct']:+.2f} booked "
            f"{s['booked_return_pct']:+.2f} wiped {s['wiped_share']:.2f} | train floored eps {train['floored_episodes']}/{train['episodes']}")


def _init_worker():
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        jobs = [(arm, c, SMOKE_SEED) for arm, c in ORDER]
    else:
        jobs = [(arm, c, s) for arm, c in ORDER for s in SEEDS[(arm, c)]]
    steps = 16_384 if a.smoke else STEPS
    todo = [j for j in jobs if not job_path(*j, smoke=a.smoke).exists()]
    print(time.strftime("%H:%M:%S"), f"{len(jobs)} jobs, {len(jobs) - len(todo)} complete, running {len(todo)} with {a.workers} workers",
          flush=True)
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init_worker) as ex:
        futs = {ex.submit(dr_job, *j, steps, a.smoke): j for j in todo}
        for f in as_completed(futs):
            try:
                print(time.strftime("%H:%M:%S"), f.result(), flush=True)
            except Exception as e:  # noqa: BLE001
                print(time.strftime("%H:%M:%S"), "FAILED", futs[f], repr(e)[:400], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
