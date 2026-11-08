"""Exploratory, not registered, written after the registered analysis of Study EXT: does the exploit survive the simulator's DEFAULT reward, the log return of the portfolio's valuation?

    python explore_log_reward.py --workers 8

The engine's default reward is log(V_t / V_{t-1}), with the same omission: on a step that takes the valuation to zero or below it ends the episode and does not call the reward function.
This reward is not linear in the market input, so the reward diagram and the twin rewards do not apply as stated. Everything else is as in Study EXT: same positions, market, learner, seeds
300-307, probes and test windows, and the hashed files are imported unchanged. Writes results_log/*.json and results_log/SUMMARY.md. The reference is the LEDGER and REP arms of Study EXT.
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
OUT = HERE / "results_log"
SEEDS = list(range(300, 308))
STEPS = 1_000_000
N_ENVS = 8
GAMMA = 0.99
N_TEST = 100
TEST_SEED = 9500                # as in run_ext.py


def make_env(split: str, seed: int):
    sys.path.insert(0, str(HERE))
    from envs_ext import ExtEnv
    from gym_trading_env.environments import basic_reward_function
    env = ExtEnv(split, "placebo", "REP", seed=seed)
    env.core.reward_function = basic_reward_function          # the engine's default, set on the same public attribute its constructor sets
    return env


def job(seed: int, steps: int) -> str:
    out = OUT / f"LOG__s{seed}.json"
    if out.exists():
        return f"skip {out.name}"
    sys.path.insert(0, str(HERE))
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from envs_ext import POSITIONS
    from run_ext import PROBE_POS, PROBE_REM, PROBE_W
    t0 = time.time()
    np.random.seed(seed)
    fns = [(lambda s: (lambda: make_env("train", s)))(seed * 100 + i) for i in range(N_ENVS)]
    venv = VecNormalize(DummyVecEnv(fns), norm_obs=False, norm_reward=True, gamma=GAMMA)
    model = PPO("MlpPolicy", venv, seed=seed, n_steps=1024, batch_size=256, gamma=GAMMA, policy_kwargs=dict(net_arch=[64, 64]), device="cpu", verbose=0)
    model.learn(total_timesteps=steps)
    pos = np.array(POSITIONS, dtype=float)

    def greedy(o):
        return int(model.predict(np.asarray(o, dtype=np.float32), deterministic=True)[0])

    probes = [float(pos[greedy([z, w, p, rem])]) for w in PROBE_W for z in (-1.0, 1.0) for rem in PROBE_REM for p in PROBE_POS]
    vals = np.abs(np.array(probes))
    env = make_env("test", TEST_SEED)
    starts = np.random.default_rng(TEST_SEED).integers(0, 2**31 - 1, size=N_TEST)
    rows = []
    for st in starts:
        o, _ = env.reset(options={"start": int(st)})
        b = r_sum = 0.0
        lev = []
        done = trunc = False
        while not (done or trunc):
            a = greedy(o)
            lev.append(abs(pos[a]))
            o, _, done, trunc, info = env.step(a)
            b += info["booked"]
            r_sum += info["reported"]
        rows.append((b, r_sum, float(info["bust"]), float(np.mean(lev))))
    r = np.array(rows)
    res = dict(seed=seed, steps=steps, X=float((vals == 10).mean()), mean_abs_probe=float(vals.mean()), probe_share_at_5_or_more=float((vals >= 5).mean()), booked=float(r[:, 0].mean()), booked_se=float(r[:, 0].std(ddof=1) / np.sqrt(len(r))),
               sum_log_reward=float(r[:, 1].mean()), bust=float(r[:, 2].mean()), mean_abs_position=float(r[:, 3].mean()), seconds=round(time.time() - t0, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    return f"done {out.name} ({res['seconds']}s): X {res['X']:.2f} |pos| {res['mean_abs_probe']:.2f} bust {res['bust']:.2f} booked {res['booked']:+.1f}"


def _init():
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"


def summary() -> None:
    d = [json.loads(p.read_text()) for p in sorted(OUT.glob("LOG__s*.json"))]
    if not d:
        return
    lines = ["# Study EXT, exploratory: the simulator's default logarithmic reward", "", "Not registered; written after the registered analysis. Eight PPO seeds; same positions, market, probes and test windows as Study EXT.", "",
             "| seed | X | mean probed position | share of probes at 5 or more | bust share | mean abs position in test | booked, % per episode |", "|---|---|---|---|---|---|---|"]
    for x in d:
        lines.append(f"| {x['seed']} | {x['X']:.3f} | {x['mean_abs_probe']:.2f} | {x['probe_share_at_5_or_more']:.3f} | {x['bust']:.3f} | {x['mean_abs_position']:.2f} | {x['booked']:+.2f} |")

    def m(k):
        return float(np.mean([x[k] for x in d]))

    lines += ["", f"Means over {len(d)} seeds: X {m('X'):.3f}, mean probed position {m('mean_abs_probe'):.2f}, bust share {m('bust'):.3f}, booked {m('booked'):+.2f}."]
    (OUT / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    a = ap.parse_args()
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init) as ex:
        futs = {ex.submit(job, s, a.steps): s for s in a.seeds}
        for f in as_completed(futs):
            try:
                print(time.strftime("%H:%M:%S"), f.result(), flush=True)
            except Exception as e:  # noqa: BLE001
                print(time.strftime("%H:%M:%S"), "FAILED", futs[f], repr(e)[:400], flush=True)
    summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
