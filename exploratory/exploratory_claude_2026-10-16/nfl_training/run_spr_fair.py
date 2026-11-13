"""Stage 4 (EXPLORATORY, PLAN.md): sign-paired reward (SPR) in the paper's registered fair-market environment.

Study O/Q set-up exactly (study_o/envs_o.PartialEnv with alpha = 0, i.e. the zero-floor reward; placebo NE market; equity as the only
input; 8 envs; reward normalization; gamma 0.99; 1M steps; PPO as Study O, A2C-n64 and DQN-default as Study Q). The only change: the
training reward is SPR. Before each step the simulator is cloned, the clone takes the same action with the sign of this quarter's
spread reversed, and the reward is (r - r_twin) / 2 - own fee. The trainer uses the reward channel, a clone, a sign override of the
exogenous input and its own fee schedule; it never reads the ledger or the accounting rule.

    python run_spr_fair.py --workers 4
    python run_spr_fair.py --smoke
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LIAB = ROOT / "LIABILITY_STUDY_2026-09-29"
STUDY_O = ROOT / "ACCOUNTING_OPTIONS_2026-10-14" / "study_o"
OUT = HERE / "results_fair"
LEARNERS = {"PPO": ("PPO", dict(n_steps=1024, batch_size=256)),
            "A2C-n64": ("A2C", dict(n_steps=64, gae_lambda=0.95, normalize_advantage=True)),
            "DQN-default": ("DQN", {})}
SEEDS = [85, 86, 87, 88, 89]
REGIONS = ["DK1", "DK2"]
STEPS = 1_000_000
N_ENVS = 8
N_DRAWS = 20
GAMMA = 0.99
EGRID = (0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)


class _FlipOne:
    """Array view with element t sign-reversed (the one-step counterfactual exogenous input)."""

    def __init__(self, a, t):
        self.a, self.t = a, t

    def __getitem__(self, i):
        return -self.a[i] if i == self.t else self.a[i]


class _FlippedMarket:
    def __init__(self, m, t):
        self._m, self.spread = m, _FlipOne(m.spread, t)

    def __getattr__(self, name):
        return getattr(self._m, name)


def make_spr(env):
    import gymnasium as gym
    from envs_b import FEE, K0, LEV, Q1

    class SignPaired(gym.Wrapper):
        def step(self, action):
            base = self.env.unwrapped
            t = base.t
            q = 0.0 if base.closed else float(np.clip(float(LEV[int(action)]) * Q1, -base.m.cap_mwh[t], base.m.cap_mwh[t]))
            fee = FEE * abs(q) / K0 * 100.0                      # own fee schedule, percent of K0
            twin = copy.copy(base)
            twin.m = _FlippedMarket(base.m, t)
            _, r_twin, _, _, _ = twin.step(action)
            obs, r, done, trunc, info = self.env.step(action)
            return obs, 0.5 * (r - r_twin) - fee, done, trunc, info

    return SignPaired(env)


def job_path(learner, seed, region, smoke=False, method="SPR"):
    alpha = 1 if method == "LEDGER" else 0
    return OUT / ("smoke" if smoke else "jobs") / f"{method}__{learner}__a{alpha}__s{seed}__{region}.json"


def spr_job(learner, seed, region, steps, smoke, method="SPR"):
    out = job_path(learner, seed, region, smoke, method)
    if out.exists():
        return f"skip {out.name}"
    sys.path[:0] = [str(LIAB), str(STUDY_O), str(HERE)]
    import gymnasium as gym
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import A2C, DQN, PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from envs_b import LEV, load_market, outcome
    from envs_o import PartialEnv
    algo, kw = LEARNERS[learner]
    t0 = time.time()
    mtr = load_market(region, "train", "NE")
    split = "train" if smoke else "test"

    class EquityOnly(gym.ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            self.observation_space = gym.spaces.Box(0, 5, shape=(1,), dtype=np.float32)

        def observation(self, obs):
            return np.array([obs[6]], dtype=np.float32)

    if method == "LEDGER":      # post-hoc same-seed control: alpha = 1, every loss charged (the booked reward), no SPR
        venv = DummyVecEnv([(lambda i: (lambda: EquityOnly(PartialEnv(mtr, 1.0, train=True, seed=seed * 100 + i))))(i)
                            for i in range(N_ENVS)])
    else:
        venv = DummyVecEnv([(lambda i: (lambda: EquityOnly(make_spr(PartialEnv(mtr, 0.0, train=True, seed=seed * 100 + i)))))(i)
                            for i in range(N_ENVS)])
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=GAMMA)
    Algo = {"PPO": PPO, "A2C": A2C, "DQN": DQN}[algo]
    model = Algo("MlpPolicy", venv, seed=seed, gamma=GAMMA, policy_kwargs=dict(net_arch=[64, 64]), device="cpu", verbose=0, **kw)
    model.learn(total_timesteps=steps)
    pick = lambda e: int(model.predict(np.array([min(e, 5.0)], np.float32), deterministic=True)[0])  # noqa: E731
    lev = {f"{e:g}": float(abs(LEV[pick(e)])) for e in EGRID}
    draws = []
    for k in range(3 if smoke else N_DRAWS):
        env = PartialEnv(load_market(region, split, "NE", draw=k), 0.0, train=False)
        o, _ = env.reset(options={"e0": 1.0})
        done = False
        while not done:
            o, _, done, _, _ = env.step(int(model.predict(np.array([o[6]], np.float32), deterministic=True)[0]))
        draws.append(outcome(env))
    res = dict(method=method, learner=learner, algo=algo, kwargs=kw, alpha=1.0 if method == "LEDGER" else 0.0, seed=seed, region=region, steps=steps, eval_split=split,
               lev=lev, drawmean_reported_return_pct=float(np.mean([d["reported_return_pct"] for d in draws])),
               drawmean_ledger_return_pct=float(np.mean([d["ledger_return_pct"] for d in draws])),
               drawmean_mean_abs_lev_open=float(np.mean([d["mean_abs_lev_open"] for d in draws])),
               seconds=round(time.time() - t0, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    return (f"done {out.name} ({res['seconds']}s): L(0.02)={lev['0.02']:g} L(0.1)={lev['0.1']:g} L(2)={lev['2']:g} "
            f"reported {res['drawmean_reported_return_pct']:+.1f}% ledger {res['drawmean_ledger_return_pct']:+.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--learners", nargs="+", default=list(LEARNERS))
    ap.add_argument("--method", default="SPR", choices=["SPR", "LEDGER"])
    a = ap.parse_args()
    steps = 20_000 if a.smoke else STEPS
    jobs = [(ln, s, r) for s in SEEDS for ln in a.learners for r in REGIONS]
    if a.smoke:
        jobs = [(ln, SEEDS[0], "DK1") for ln in a.learners]
    print(f"{len(jobs)} jobs, {a.workers} workers, {steps} steps", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(spr_job, ln, s, r, steps, a.smoke, a.method): (ln, s, r) for ln, s, r in jobs}
        for f in as_completed(futs):
            try:
                print(time.strftime("%H:%M:%S"), f.result(), flush=True)
            except Exception as e:  # noqa: BLE001
                print(time.strftime("%H:%M:%S"), "FAILED", futs[f], repr(e)[:400], flush=True)


if __name__ == "__main__":
    main()
