"""Registered Study R runner. Read PREREGISTRATION_R.md; no outcome-based stopping.

Resume skips only complete job files with matching settings and registration hash.
Interrupted jobs restart at the same seed. All model/checkpoint outputs stay here.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "LIABILITY_STUDY_2026-09-29")]
OUT = HERE / "results_r"
FEES, ALPHAS, SEEDS, ZONES = (.4, 1.5), (0.0, .8, 1.0), tuple(range(60, 65)), ("DK1", "DK2")
STEPS, N_ENVS, GAMMA = 1_000_000, 8, .99
EGRID = (0, .02, .05, .1, .25, .5, .75, 1, 1.25, 1.5, 2, 2.5, 3)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_registration():
    manifest = HERE / "PREREGISTRATION_HASHES_R.txt"
    if not manifest.exists():
        raise RuntimeError("Registration is not frozen; full training prohibited")
    lines = manifest.read_text(encoding="utf-8").splitlines()
    prefix_bytes = int(next(line.split()[-1] for line in lines if line.startswith("# RUN_LOG_PREFIX_BYTES ")))
    for line in lines:
        if not line or line.startswith("#"):
            continue
        digest, relative = line.split("  ", 1)
        target = ROOT / relative.lstrip("*")
        content = target.read_bytes()
        if target.name == "RUN_LOG_R.md":
            content = content[:prefix_bytes]
        if hashlib.sha256(content).hexdigest() != digest:
            raise RuntimeError(f"Registered input changed: {relative}")
    return sha(manifest)


def job_path(fee, alpha, seed, zone):
    return OUT / "jobs" / f"fee{fee:g}__a{alpha:g}__PPO__s{seed}__{zone}.json"


def validate_completed(value, fee, alpha, seed, zone, registration_sha):
    settings = dict(fee=fee, alpha=alpha, seed=seed, region=zone, steps=STEPS,
                    registration_sha256=registration_sha, algo="PPO", gamma=GAMMA, obsset="EQ")
    if any(value.get(k) != v for k, v in settings.items()):
        raise RuntimeError("Completed summary has incompatible settings")
    expected_steps = int(np.ceil(STEPS / (N_ENVS * 1024)) * (N_ENVS * 1024))
    if value.get("actual_timesteps") != expected_steps:
        raise RuntimeError("Incomplete training budget")
    if set(value.get("lev", {})) != {f"{e:g}" for e in EGRID}:
        raise RuntimeError("Missing policy probes")
    if any(v not in (0, .25, 1, 4, 16) for v in value["lev"].values()):
        raise RuntimeError("Invalid leverage probes")
    visits = np.asarray(value.get("visit_action_counts", []))
    if visits.shape != (8, 9) or visits.sum() != expected_steps or (visits < 0).any():
        raise RuntimeError("Incomplete visitation counters")
    artifacts = value.get("model_artifact_sha256", {})
    if len(artifacts) != 3:
        raise RuntimeError("Missing trained policy artifacts")
    for name, digest in artifacts.items():
        target = OUT / "models" / name
        if not target.is_file() or sha(target) != digest:
            raise RuntimeError(f"Missing/changed trained artifact: {name}")


def run_job(fee, alpha, seed, zone, registration_sha):
    path = job_path(fee, alpha, seed, zone)
    if path.exists():
        old = json.loads(path.read_text())
        validate_completed(old, fee, alpha, seed, zone, registration_sha)
        return f"skip {path.name}"
    import gymnasium as gym
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from envs_b import LEV, load_market
    from envs_r import CostPartialEnv, VISIT_EDGES
    from run_b import greedy, network_layers

    class EquityOnly(gym.ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            self.observation_space = gym.spaces.Box(0, 5, shape=(1,), dtype=np.float32)

        def observation(self, obs):
            return np.asarray([obs[6]], dtype=np.float32)

    started = time.perf_counter()
    market = load_market(zone, "train", "NE")
    raw = DummyVecEnv([(lambda i: lambda: EquityOnly(CostPartialEnv(
        market, alpha, fee, True, seed * 100 + i)))(i) for i in range(N_ENVS)])
    venv = VecNormalize(raw, norm_obs=False, norm_reward=True, gamma=GAMMA)
    model = PPO("MlpPolicy", venv, seed=seed, n_steps=1024, batch_size=256, gamma=GAMMA,
                policy_kwargs=dict(net_arch=[64, 64]), device="cpu", verbose=0)
    model.learn(total_timesteps=STEPS)
    layers = network_layers(model, "PPO")
    probes = np.asarray([[min(e, 5)] for e in EGRID], dtype=np.float32)
    actions = greedy(layers, probes)
    sb3, _ = model.predict(probes, deterministic=True)
    assert np.array_equal(actions, sb3), "saved policy forward-pass mismatch"
    lev = {f"{e:g}": float(abs(LEV[a])) for e, a in zip(EGRID, actions)}
    path.parent.mkdir(parents=True, exist_ok=True)
    model_dir = OUT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    stem = path.stem
    model.save(model_dir / f"{stem}.zip")
    venv.save(str(model_dir / f"{stem}__normalization.pkl"))
    # Save network weights for later critic inspection without new training.
    saved_layers = {}
    for i, (kind, weights, bias) in enumerate(layers):
        saved_layers[f"kind_{i}"] = np.asarray(kind)
        if weights is not None:
            saved_layers[f"weights_{i}"] = weights
            saved_layers[f"bias_{i}"] = bias
    np.savez_compressed(model_dir / f"{stem}__policy_layers.npz", **saved_layers)
    visits = np.sum([env.unwrapped.visit_counts for env in raw.envs], axis=0)
    sums = np.sum([env.unwrapped.training_sums for env in raw.envs], axis=0)
    value = dict(fee=fee, alpha=alpha, seed=seed, region=zone, algo="PPO", steps=STEPS,
                 actual_timesteps=int(model.num_timesteps), gamma=GAMMA, obsset="EQ", lev=lev,
                 seconds=time.perf_counter() - started, registration_sha256=registration_sha,
                 visit_bin_left_edges=VISIT_EDGES[:-1].tolist(), visit_action_counts=visits.tolist(),
                 training_sums_eur=dict(zip(("pnl", "cost", "injections", "reward"), sums.tolist())),
                 primary_endpoint="greedy absolute leverage at 2% equity; no TEST outcome used",
                 eval_split="fixed equity probes only; no stochastic TEST evaluation")
    value["model_artifact_sha256"] = {name: sha(model_dir / name) for name in
        (f"{stem}.zip", f"{stem}__normalization.pkl", f"{stem}__policy_layers.npz")}
    validate_completed(value, fee, alpha, seed, zone, registration_sha)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)
    venv.close()
    return f"done {path.name} ({value['seconds']:.1f}s) L(.02)={lev['0.02']:g}"


def main():
    global STEPS, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--smoke", action="store_true", help="Pre-freeze software check: one 8192-step booked control only")
    args = parser.parse_args()
    if not 1 <= args.workers <= 7:
        raise ValueError("workers must be 1..7")
    if args.smoke:
        if (HERE / "PREREGISTRATION_HASHES_R.txt").exists():
            raise RuntimeError("Development smoke is prohibited after registration freeze")
        STEPS, OUT = 8192, HERE / "development_smoke"
        print(run_job(.4, 1., 60, "DK1", "development-only-unregistered"), flush=True)
        return 0
    registration = verify_registration()
    jobs = [(f, a, s, z) for s in SEEDS for f in FEES for a in ALPHAS for z in ZONES]
    complete = sum(job_path(*j).exists() for j in jobs)
    print(f"Study R: {len(jobs)} jobs, {complete} complete; {args.workers} workers; registration {registration}", flush=True)
    failures = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_job, *j, registration): j for j in jobs}
        for future in as_completed(futures):
            try:
                print(time.strftime("%Y-%m-%d %H:%M:%S"), future.result(), flush=True)
            except Exception as error:
                failures += 1
                print(time.strftime("%Y-%m-%d %H:%M:%S"), "FAILED", futures[future], repr(error), flush=True)
    if failures:
        print(f"{failures} jobs failed; restart unchanged to retry same seeds", flush=True)
        return 1
    # The registered analysis refuses incomplete cohorts and never overwrites verdicts.
    from aggregate_r import main as aggregate
    return aggregate()


if __name__ == "__main__":
    raise SystemExit(main())
