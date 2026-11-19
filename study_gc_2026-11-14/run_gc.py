"""Study GC runner (PREREGISTRATION_GC.md): PPO trained in gym-mtsim 2.0.0 as published (ORIG), without the zero floor on the balance
(FLPATCH) and, in the crypto market, on the reading of the published reward (LINE). Learner, wrapper and measures are Study G's.

    <venv>\\Scripts\\python run_gc.py --workers 8
    <venv>\\Scripts\\python run_gc.py --smoke
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
OUT = HERE / "results_gc"
STEPS = 1_000_000
N_ENVS = 8
CELLS = [("crypto", "ORIG"), ("crypto", "FLPATCH"), ("crypto", "LINE"), ("forex", "ORIG"), ("forex", "FLPATCH")]
SEEDS = {"crypto": list(range(10, 20)), "forex": list(range(10, 15))}


def job_path(market: str, cond: str, seed: int, smoke: bool = False) -> Path:
    return OUT / ("jobs_smoke" if smoke else "jobs") / f"{market}__{cond}__PPO__s{seed}.json"


def evaluate(model, vecnorm, market: str, smoke: bool) -> dict:
    """Deterministic policy on the shared placebo test paths from a balance of 10,000 USD, in gym-mtsim as published: reported return
    (change in equity) and booked return (the simulator's order records), with the reading audited on every step."""
    from envs_gc import BALANCE, GCEnv, test_bank
    bank = test_bank(market)
    if smoke:
        bank = bank[:5]
    env = GCEnv(market, "ORIG", train=False, seed=0, audit=True)
    rows = []
    for close in bank:
        o, _ = env.reset(options={"path": close, "balance": BALANCE})
        done = False
        while not done:
            a = model.predict(vecnorm.normalize_obs(o[None, :]), deterministic=True)[0][0]
            o, _, te, tr, info = env.step(a)
            done = te or tr
        rows.append(info["stats"])
    rep = [100.0 * r["reported"] / BALANCE for r in rows]
    led = [100.0 * r["booked"] / BALANCE for r in rows]
    return dict(reported_return_pct=float(np.mean(rep)), booked_return_pct=float(np.mean(led)), draws_reported=rep, draws_booked=led,
                floored_share=float(np.mean([r["floored"] for r in rows])), wiped_share=float(np.mean([r["wiped"] for r in rows])),
                orders_per_episode=float(np.mean([r["orders"] for r in rows])),
                max_leverage=float(np.mean([r["lev_max"] for r in rows])),
                reading_max_abs_err_usd=float(max(r["line_max_abs_err"] for r in rows)), test_paths=len(rows))


def gc_job(market: str, cond: str, seed: int, steps: int, smoke: bool) -> str:
    out = job_path(market, cond, seed, smoke)
    if out.exists():
        return f"skip {out.name}"
    import importlib.metadata

    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from envs_gc import EGRID, GCEnv, exposure
    t0 = time.time()
    base = DummyVecEnv([(lambda i: (lambda: GCEnv(market, cond, train=True, seed=seed * 100 + i)))(i) for i in range(N_ENVS)])
    venv = VecNormalize(base, norm_obs=True, norm_reward=True, gamma=0.99)
    model = PPO("MlpPolicy", venv, seed=seed, n_steps=1024, batch_size=256, policy_kwargs=dict(net_arch=[64, 64]), device="cpu",
                verbose=0)
    model.learn(total_timesteps=steps)
    venv.training = False

    def predict(x):                                     # probe through the frozen observation normalization
        return model.predict(venv.normalize_obs(x), deterministic=True)[0][0]
    expo = {f"{e:g}": exposure(predict, market, e) for e in EGRID}
    counts = base.get_attr("count")
    train = {k: (max(c[k] for c in counts) if k == "line_max_abs_err" else int(sum(c[k] for c in counts))) for k in counts[0]}
    digest = hashlib.sha256()
    for name, p in sorted(model.policy.state_dict().items()):
        digest.update(name.encode())
        digest.update(p.detach().cpu().numpy().tobytes())
    res = dict(market=market, condition=cond, algo="PPO", seed=seed, steps=int(model.num_timesteps), exposure=expo,
               **evaluate(model, venv, market, smoke), train=train, policy_sha256=digest.hexdigest(),
               versions={p: importlib.metadata.version(p) for p in ("gym-mtsim", "pandas", "numpy", "gymnasium", "stable-baselines3", "torch")},
               seconds=round(time.time() - t0, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    pol = out.parent.parent / ("policies_smoke" if smoke else "policies")
    pol.mkdir(parents=True, exist_ok=True)
    model.save(str(pol / out.stem))
    venv.save(str(pol / (out.stem + "__vecnormalize.pkl")))
    out.write_text(json.dumps(res, indent=1))
    return (f"done {out.name} ({res['seconds']}s): x(0.1)={expo['0.1']:.2f} x(1)={expo['1']:.2f} rep={res['reported_return_pct']:+.1f} "
            f"booked={res['booked_return_pct']:+.1f} floored={res['floored_share']:.2f} train_floor_steps={train['floor_steps']}")


def _init_worker():
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        jobs = [(m, c, 99) for m, c in CELLS]
    else:
        jobs = [(m, c, s) for m in ("crypto", "forex") for s in SEEDS[m] for mm, c in CELLS if mm == m]
        jobs.sort(key=lambda j: j[1] != "LINE")          # the slowest runs (the reading costs about two extra steps) start first
    steps = 16_384 if a.smoke else STEPS
    todo = [j for j in jobs if not job_path(*j, smoke=a.smoke).exists()]
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} complete, running {len(todo)} with {a.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init_worker) as ex:
        futs = {ex.submit(gc_job, *j, steps, a.smoke): j for j in todo}
        for f in as_completed(futs):
            try:
                print(time.strftime("%H:%M:%S"), f.result(), flush=True)
            except Exception as e:  # noqa: BLE001
                print(time.strftime("%H:%M:%S"), "FAILED", futs[f], repr(e)[:400], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
