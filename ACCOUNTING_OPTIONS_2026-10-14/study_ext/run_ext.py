"""Study EXT runner (PREREGISTRATION_EXT.md): does PPO learn the bankruptcy-step omission of the official, unmodified Gym-Trading-Env in a market without edge, and what removes it?

    python run_ext.py --workers 3                 # all registered jobs, resume-safe
    python run_ext.py --out DIR --seeds 99 --arms REP LEDGER   # pilot (exploratory)
    python run_ext.py --smoke --out DIR           # 20,000 steps, one seed per arm; excluded

PPO as in Studies T to Z: 8 environments, 1,024-step rollouts, batch 256, gamma 0.99, 64x64, reward normalization, 10^6 steps. Training market: placebo, magnitudes of the train split, fresh signs
every episode. Test: 100 fixed windows of the test split, placebo market. All values are percent of the initial wealth per 30-day episode.
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
STEPS = 1_000_000
N_ENVS = 8
GAMMA = 0.99
ARMS = ("REP", "LEDGER", "SPR", "TIR", "TSR", "RUIN5", "RUIN25", "RUIN100", "FINDER", "FINDER_DEF")
SEEDS = list(range(300, 308))
SEEDS_OF = {"TSR": SEEDS[:4], "FINDER_DEF": SEEDS[:2]}      # TSR's reward equals the ledger's wherever it is exact; the finder's control has a reward that is zero everywhere
STEPS_OF = {"FINDER_DEF": 100_000}
DEFAULT_POSITIONS = (0, 1)
N_TEST = 100
PROBE_W = (0.25, 0.5, 1.0, 2.0, 4.0)
PROBE_REM = (0.2, 0.5, 0.9)
PROBE_POS = (-1.0, 0.0, 1.0)


def job_path(out: Path, arm: str, seed: int) -> Path:
    return out / f"EXT__{arm}__s{seed}.json"


def build_env(arm: str, seed: int):
    import gymnasium as gym
    from envs_ext import ExtEnv
    if arm.startswith("RUIN"):
        strength = float(arm.replace("RUIN", ""))
        base = ExtEnv("train", "placebo", "REP", seed=seed)

        class Ruin(gym.Wrapper):
            def step(self, action):
                o, r, d, t, i = self.env.step(action)
                return o, r - (strength if i["bust"] else 0.0), d, t, i

        return Ruin(base)
    if arm == "FINDER_DEF":
        return ExtEnv("train", "placebo", "WIDTH", seed=seed, positions=DEFAULT_POSITIONS)
    mode = {"FINDER": "WIDTH"}.get(arm, arm)
    return ExtEnv("train", "placebo", mode, seed=seed)


def ext_job(out_dir: str, arm: str, seed: int, steps: int, smoke: bool) -> str:
    out = job_path(Path(out_dir), arm, seed)
    if out.exists():
        return f"skip {out.name}"
    sys.path.insert(0, str(HERE))
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from envs_ext import POSITIONS, ExtEnv, split_moves
    t0 = time.time()
    np.random.seed(seed)
    fns = [(lambda s: (lambda: build_env(arm, s)))(seed * 100 + i) for i in range(N_ENVS)]
    venv = VecNormalize(DummyVecEnv(fns), norm_obs=False, norm_reward=True, gamma=GAMMA)
    model = PPO("MlpPolicy", venv, seed=seed, n_steps=1024, batch_size=256, gamma=GAMMA, policy_kwargs=dict(net_arch=[64, 64]), device="cpu", verbose=0)
    model.learn(total_timesteps=steps)

    positions = DEFAULT_POSITIONS if arm == "FINDER_DEF" else POSITIONS
    pos = np.array(positions, dtype=float)

    def greedy(o):
        return int(model.predict(np.asarray(o, dtype=np.float32), deterministic=True)[0])

    probes = {}
    for w in PROBE_W:
        for z in (-1.0, 1.0):
            for rem in PROBE_REM:
                for p in PROBE_POS:
                    probes[f"{w:g}|{z:+g}|{rem:g}|{p:+g}"] = float(pos[greedy([z, w, p, rem])])
    vals = np.abs(np.array(list(probes.values())))
    x10 = float((vals == 10).mean())           # the exploit share: probes at maximal leverage (0 by construction in the default configuration)

    def test(kind, s):
        env = ExtEnv("test", kind, "REP", seed=s, positions=positions)
        starts = np.random.default_rng(s).integers(0, 2**31 - 1, size=4 if smoke else N_TEST)
        rows = []
        inside = 0
        n_steps = 0
        for st in starts:
            o, _ = env.reset(options={"start": int(st)})
            R = B = LO = TIR = SPR = LINE = WIDTH = 0.0
            done = trunc = False
            lev = []
            ok = 1
            while not (done or trunc):
                a = greedy(o)
                lev.append(abs(pos[a]))
                r_twin = env.query(a, -1.0)
                c = max(-env.query(a, 0.0), 0.0)
                pp, p2, mm, m2 = (env.query(a, t) for t in (1e9, 2e9, -1e9, -2e9))
                ap, am = (p2 - pp) / 1e9, (mm - m2) / 1e9
                g = (ap + am) if ap * am < 0.0 else (ap if abs(ap) >= abs(am) else am)
                o, _, done, trunc, info = env.step(a)
                r, x = info["reported"], info["booked"]
                R += r
                B += x
                LO += -r_twin - 2.0 * c
                SPR += 0.5 * (r - r_twin) - c
                TIR += np.sign(r - r_twin) * (max(r, r_twin) + c) - c if abs(r - r_twin) > 1e-9 else -c
                LINE += g - c
                WIDTH += r + r_twin + 2.0 * c
                ok &= int(-r_twin - 2.0 * c <= x + 1e-6 and x <= r + 1e-6)
                n_steps += 1
            inside += ok
            rows.append((B, R, float(info["bust"]), float(np.mean(lev)), TIR, SPR, LINE, LO, WIDTH))
        r = np.array(rows)
        booked = r[:, 0]
        return dict(booked=float(booked.mean()), booked_se=float(booked.std(ddof=1) / np.sqrt(len(r))), reported=float(r[:, 1].mean()), bust=float(r[:, 2].mean()), mean_abs_position=float(r[:, 3].mean()),
                    episodes_steps_inside_bracket=int(inside), episodes=len(r), steps=n_steps, lower=float(r[:, 7].mean()), width=float(r[:, 8].mean()), max_width=float(r[:, 8].max()),
                    mae_tir=float(np.abs(r[:, 4] - booked).mean()), mae_spr=float(np.abs(r[:, 5] - booked).mean()), mae_line=float(np.abs(r[:, 6] - booked).mean()), mae_reported=float(np.abs(r[:, 1] - booked).mean()),
                    max_err_line=float(np.abs(r[:, 6] - booked).max()), booked_episodes=booked.tolist(), reported_episodes=r[:, 1].tolist())

    res = dict(arm=arm, seed=seed, steps=steps, x10=x10, mean_abs_probe=float(vals.mean()), probes=probes, test_placebo=test("placebo", 9500), seconds=None)
    res["seconds"] = round(time.time() - t0, 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    tp = res["test_placebo"]
    return f"done {out.name} ({res['seconds']}s): X10 {x10:.2f} |pos| {vals.mean():.1f}; placebo reported {tp['reported']:+.1f} booked {tp['booked']:+.1f} bust {tp['bust']:.2f} mean|pos| {tp['mean_abs_position']:.1f}; audit MAE line {tp['mae_line']:.3f} tir {tp['mae_tir']:.2f} spr {tp['mae_spr']:.2f} rep {tp['mae_reported']:.1f}"


def _init_worker():
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    a = ap.parse_args()
    steps = 20_000 if a.smoke else a.steps
    out = Path(a.out) if a.out else HERE / "results_ext"
    seeds = a.seeds or (SEEDS[:1] if a.smoke else SEEDS)
    jobs = [(arm, s) for s in seeds for arm in (a.arms or ARMS) if a.seeds or a.smoke or s in SEEDS_OF.get(arm, SEEDS)]
    todo = [j for j in jobs if not job_path(out, *j).exists()]
    print(time.strftime("%Y-%m-%d %H:%M:%S"), f"{len(jobs)} jobs, {len(jobs) - len(todo)} complete, running {len(todo)} with {a.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init_worker) as ex:
        futs = {ex.submit(ext_job, str(out), arm, s, steps if a.smoke else STEPS_OF.get(arm, steps), a.smoke): (arm, s) for arm, s in todo}
        for f in as_completed(futs):
            try:
                print(time.strftime("%H:%M:%S"), f.result(), flush=True)
            except Exception as e:  # noqa: BLE001
                print(time.strftime("%H:%M:%S"), "FAILED", futs[f], repr(e)[:400], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
