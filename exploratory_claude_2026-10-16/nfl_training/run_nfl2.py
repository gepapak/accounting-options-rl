"""NFL stage-2 runner (EXPLORATORY): four accounting rules and a KL-to-reference baseline.

Arms
    PLAIN        reported reward, 8 real envs
    NFL          4 real envs (reported reward) + 4 placebo envs (reward -(reported + own cost))
    ORACLE       booked reward (needs the ledger), 8 real envs
    KL<beta>     reported reward minus beta * [log pi(a|s) - log pi_ref(a|s)] (per-step KL penalty toward a safe reference:
                 follow the signal at leverage 1 with probability 0.7, the other eight actions share 0.3), 8 real envs

    python run_nfl2.py --workers 4 --accounting ZF LL TERM LAG --arms PLAIN NFL ORACLE
    python run_nfl2.py --workers 4 --accounting ZF --arms KL0.003 KL0.03 KL0.3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results2"
N_ENVS = 8
GAMMA = 0.99
N_TEST = 40
PROBE_E = (-0.5, -0.1, 0.0, 0.02, 0.05, 0.1, 0.25, 1.0)
IDX_LONG1, IDX_SHORT1 = 6, 2          # LEV[6] = +1, LEV[2] = -1


def make_sign_paired(env):
    """SPR: clone the simulator, step the clone with the opposite sign of the same exogenous change, train on
    (r - r_twin) / 2 - own cost. Needs clone + sign override + own cost; not the ledger or the accounting rule."""
    import copy
    import gymnasium as gym
    from envs_nfl2 import K0

    class SignPaired(gym.Wrapper):
        def step(self, action):
            twin = copy.copy(self.env)
            twin.flip_next = True
            _, r_twin, _, _, _ = twin.step(action)
            obs, r, done, trunc, info = self.env.step(action)
            return obs, 0.5 * (r - r_twin) - info["cost"] / K0 * 100.0, done, trunc, info

    return SignPaired(env)


def ref_probs(obs, actions):
    """Reference policy: leverage 1 in the direction of the signal with probability 0.7, others 0.0375 each."""
    target = np.where(obs[:, 1] > 0, IDX_LONG1, IDX_SHORT1)
    return np.where(actions == target, 0.7, 0.3 / 8)


def make_kl_wrapper():
    import torch
    from stable_baselines3.common.vec_env import VecEnvWrapper

    class KLPenalty(VecEnvWrapper):
        def __init__(self, venv, beta):
            super().__init__(venv)
            self.beta, self.model, self.last_obs, self.actions = beta, None, None, None

        def reset(self):
            self.last_obs = self.venv.reset()
            return self.last_obs

        def step_async(self, actions):
            self.actions = np.asarray(actions)
            self.venv.step_async(actions)

        def step_wait(self):
            obs, rew, done, info = self.venv.step_wait()
            if self.model is not None:
                with torch.no_grad():
                    dist = self.model.policy.get_distribution(torch.as_tensor(self.last_obs, dtype=torch.float32))
                    logp = dist.log_prob(torch.as_tensor(self.actions)).numpy()
                rew = rew - self.beta * (logp - np.log(ref_probs(self.last_obs, self.actions)))
            self.last_obs = obs
            return obs, rew, done, info

    return KLPenalty


def job_path(arm, seed, region, acc, smoke, algo="PPO", p_ins=0.0):
    tag = ("" if algo == "PPO" else f"_{algo}") + ("" if p_ins == 0.0 else f"_ins{p_ins:g}")
    return OUT / ("smoke" if smoke else "jobs") / f"{arm}__{acc}{tag}__s{seed}__{region}.json"


def evaluate(model, region, acc, kind, n_ep, seed, p_ins=0.0):
    from envs_nfl2 import K0, NFLEnv2
    env = NFLEnv2(region, "test", kind, "reported", acc, seed=seed, p_insolvent=p_ins)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_ep):
        if p_ins > 0.0 and rng.random() < p_ins:          # same start distribution as training (stage 3)
            e0 = -float(rng.uniform(0.0, 0.5))
        else:
            e0 = float(np.exp(rng.uniform(np.log(0.05), np.log(2.0))))
        o, _ = env.reset(options={"e0": e0})
        done = False
        rep = 0.0
        while not done:
            o, r, done, _, info = env.step(int(model.predict(o, deterministic=True)[0]))
            rep += r
        rows.append(((env.ledger - env.e0) / K0 * 100.0, rep, env.forgiven / K0 * 100.0))
    r = np.array(rows)
    return {"booked_pct_K0": float(r[:, 0].mean()), "booked_se": float(r[:, 0].std(ddof=1) / np.sqrt(len(r))),
            "reported_pct_K0": float(r[:, 1].mean()), "reported_se": float(r[:, 1].std(ddof=1) / np.sqrt(len(r))),
            "forgiven_pct_K0": float(r[:, 2].mean())}


def nfl2_job(arm, seed, region, acc, steps, smoke, algo="PPO", p_ins=0.0):
    out = job_path(arm, seed, region, acc, smoke, algo, p_ins)
    if out.exists():
        return f"skip {out.name}"
    sys.path.insert(0, str(HERE))
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import A2C, PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from envs_nfl2 import LEV, NFLEnv2
    t0 = time.time()
    mk = lambda kind, mode, s: (lambda: NFLEnv2(region, "train", kind, mode, acc, seed=s, p_insolvent=p_ins))  # noqa: E731
    if arm == "NFL":
        fns = [mk("edge", "reported", seed * 100 + i) for i in range(4)] + [mk("placebo", "nfl_placebo", seed * 100 + 50 + i) for i in range(4)]
    elif arm == "SPR":
        fns = [(lambda s: (lambda: make_sign_paired(NFLEnv2(region, "train", "edge", "reported", acc, seed=s, p_insolvent=p_ins))))(seed * 100 + i)
               for i in range(N_ENVS)]
    else:
        mode = "booked" if arm == "ORACLE" else "reported"
        fns = [mk("edge", mode, seed * 100 + i) for i in range(N_ENVS)]
    venv = DummyVecEnv(fns)
    kl = None
    if arm.startswith("KL"):
        kl = make_kl_wrapper()(venv, float(arm[2:]))
        venv = kl
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=GAMMA)
    if algo == "A2C":     # Study Q's A2C-n64 settings
        model = A2C("MlpPolicy", venv, seed=seed, n_steps=64, gae_lambda=0.95, normalize_advantage=True, gamma=GAMMA,
                    policy_kwargs=dict(net_arch=[64, 64]), device="cpu", verbose=0)
    else:
        model = PPO("MlpPolicy", venv, seed=seed, n_steps=1024, batch_size=256, gamma=GAMMA, policy_kwargs=dict(net_arch=[64, 64]),
                    device="cpu", verbose=0)
    if kl is not None:
        kl.model = model
    model.learn(total_timesteps=steps)
    lev = {}
    for e in PROBE_E:
        for z in (-1, 1):
            for rem in (0.05, 0.5):
                for clock in (0.0, 0.5):
                    a = int(model.predict(np.array([e, z, rem, clock], np.float32), deterministic=True)[0])
                    lev[f"{e:g}|{z:+d}|{rem:g}|{clock:g}"] = float(LEV[a])
    n_ep = 4 if smoke else N_TEST
    res = dict(arm=arm, seed=seed, region=region, accounting=acc, algo=algo, p_insolvent=p_ins, steps=steps, lev=lev,
               test_edge=evaluate(model, region, acc, "edge", n_ep, 9000 + seed, p_ins),
               test_placebo=evaluate(model, region, acc, "placebo", n_ep, 9500 + seed, p_ins),
               seconds=round(time.time() - t0, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    return (f"done {out.name} ({res['seconds']}s): edge booked {res['test_edge']['booked_pct_K0']:+.1f}%  "
            f"placebo reported {res['test_placebo']['reported_pct_K0']:+.1f}% (forgiven {res['test_placebo']['forgiven_pct_K0']:.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--arms", nargs="+", default=["PLAIN", "NFL", "ORACLE"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[80, 81, 82])
    ap.add_argument("--regions", nargs="+", default=["DK1"])
    ap.add_argument("--accounting", nargs="+", default=["ZF", "LL", "TERM", "LAG"])
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--algo", default="PPO", choices=["PPO", "A2C"])
    ap.add_argument("--insolvent", type=float, default=0.0)
    a = ap.parse_args()
    steps = 20_000 if a.smoke else a.steps
    jobs = [(arm, s, r, acc) for acc in a.accounting for arm in a.arms for s in a.seeds for r in a.regions]
    print(f"{len(jobs)} jobs, {a.workers} workers, {steps} steps", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(nfl2_job, arm, s, r, acc, steps, a.smoke, a.algo, a.insolvent): (arm, s, r, acc) for arm, s, r, acc in jobs}
        for f in as_completed(futs):
            try:
                print(time.strftime("%H:%M:%S"), f.result(), flush=True)
            except Exception as e:  # noqa: BLE001
                print(time.strftime("%H:%M:%S"), "FAILED", futs[f], repr(e)[:400], flush=True)


if __name__ == "__main__":
    main()
