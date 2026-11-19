"""Exploratory pilot (16 Nov 2026, not registered): PPO trained THROUGH an unmodified mainstream backtesting engine, on a fair placebo
market where no policy has an edge. Same agent, same market, two engines that agree on every solvent trade: backtesting.py (equity
floored at 0 and the run stopped at ruin) and backtrader (every loss booked).

Learner: the paper's standard PPO (SB3 2.7, 8 environments, 1,024-step rollouts, batch 256, 64-64, discount 0.99, reward
normalization). After training:
  probe  the deterministic action at a flat account with equity e x 10,000 USD (e in EQ) and 25/50/75% of the episode left;
  tests  200 fair test paths (fixed seed) from 10,000 USD and from 1,000 USD, in the training engine: the engine's reported final
         equity and the booked equity from its own trade records.

    <venv>\\Scripts\\python -B rl_engine_pilot.py --steps 500000 --seeds 1 2 3 4 --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "rl_engine_results"
EQ = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
LEFT = (0.25, 0.5, 0.75)
TEST_SEED = 20261117


def job(engine: str, seed: int, steps: int, levels=None, tag: str = "", l_max: float = 200.0, mags_file: str = "eurusd_mags.npy",
        mag_scale: float = 1.0, test_seed: int = TEST_SEED, out_dir: str = "rl_engine_results") -> str:
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"
    os.environ["TQDM_DISABLE"] = "1"
    sys.path.insert(0, str(HERE))
    import warnings
    warnings.filterwarnings("ignore")
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    import rl_engine_env as R
    levels = tuple(levels) if levels else R.LEVELS
    outdir = HERE / out_dir
    out = outdir / f"{engine}__s{seed}__{steps}{tag}.json"
    if out.exists():
        return f"skip {out.name}"
    t0 = time.time()
    mags = np.load(HERE / mags_file)
    cls = {"btpy": R.BtpyEnv, "btpyfl": R.BtpyBookedEnv, "bt": R.BacktraderEnv}[engine]     # btpyfl: floor removed (control)
    kw = dict(levels=levels, l_max=l_max, mag_scale=mag_scale)
    venv = DummyVecEnv([(lambda i: (lambda: cls(mags, seed=seed * 100 + i, **kw)))(i) for i in range(8)])
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=0.99)
    model = PPO("MlpPolicy", venv, seed=seed, n_steps=1024, batch_size=256, policy_kwargs=dict(net_arch=[64, 64]), device="cpu", verbose=0)
    model.learn(total_timesteps=steps)
    venv.close()
    probe = {}
    for e in EQ:
        for left in LEFT:
            a, _ = model.predict(np.array([[e, 0.0, left]], dtype=np.float32), deterministic=True)
            probe[f"{e:g}|{left:g}"] = levels[int(np.asarray(a).ravel()[0])]
    tests = {}
    rng = np.random.default_rng(test_seed)
    bank = [R.placebo(rng, mags, mag_scale) for _ in range(200)]
    for cash in (10_000.0, 1_000.0):
        env = cls(mags, seed=0, train=False, **kw)
        rep, book, zero = [], [], []
        for close in bank:
            o, _ = env.reset(options={"path": close, "cash": cash})
            done = False
            while not done:
                a, _ = model.predict(o[None, :], deterministic=True)
                o, r, te, tr, info = env.step(int(np.asarray(a).ravel()[0]))
                done = te or tr
            s = info["stats"]
            rep.append(100 * (s["reported_equity"] / cash - 1))
            book.append(100 * (s["booked_equity"] / cash - 1))
            zero.append(bool(s["hit_zero"]))
        env.close()
        tests[f"{cash:g}"] = dict(reported_pct=float(np.mean(rep)), booked_pct=float(np.mean(book)), hit_zero=float(np.mean(zero)),
                                  draws_reported=rep, draws_booked=book)
    low = [probe[f"{e:g}|{l:g}"] for e in EQ if e <= 0.25 for l in LEFT]
    res = dict(engine=engine, seed=seed, steps=int(model.num_timesteps), levels=list(levels), tag=tag, l_max=l_max, mags=mags_file,
               mag_scale=mag_scale, probe=probe,
               low_equity_mean_leverage=float(np.mean(low)),
               low_equity_share_max=float(np.mean([x >= 64 for x in low])), tests=tests, seconds=round(time.time() - t0, 1))
    res["test_seed"] = test_seed
    outdir.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    t10, t1 = tests["10000"], tests["1000"]
    return (f"done {out.name} ({res['seconds']} s): low-equity mean leverage {res['low_equity_mean_leverage']:.1f}, share >= 64 "
            f"{res['low_equity_share_max']:.2f} | from 10k: reported {t10['reported_pct']:+.2f} booked {t10['booked_pct']:+.2f} zero {t10['hit_zero']:.2f}"
            f" | from 1k: reported {t1['reported_pct']:+.2f} booked {t1['booked_pct']:+.2f} zero {t1['hit_zero']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--engines", nargs="+", default=["btpy", "bt"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--levels", type=float, nargs="+", default=None, help="target leverages (default: rl_engine_env.LEVELS)")
    ap.add_argument("--tag", default="", help="suffix for the result files of a variant")
    ap.add_argument("--lmax", type=float, default=200.0, help="engine leverage limit (backtesting.py margin = 1/lmax)")
    ap.add_argument("--mags", default="eurusd_mags.npy", help="file of placebo magnitudes")
    ap.add_argument("--magscale", type=float, default=1.0)
    ap.add_argument("--test-seed", type=int, default=TEST_SEED, help="seed of the 200 test paths (the pilots used 20261117)")
    ap.add_argument("--out", default="rl_engine_results", help="results folder, relative to this script")
    a = ap.parse_args()
    jobs = [(e, s, a.steps, a.levels, a.tag, a.lmax, a.mags, a.magscale, a.test_seed, a.out) for s in a.seeds for e in a.engines]
    with ProcessPoolExecutor(a.workers) as ex:
        for msg in ex.map(job, *zip(*jobs)):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
