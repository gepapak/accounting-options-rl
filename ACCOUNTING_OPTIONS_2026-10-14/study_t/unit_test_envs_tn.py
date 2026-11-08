"""Check that envs_tn.LagEnvT reproduces the registered Study N environment (study_n/envs_n.LagEnv) step for step.

    python unit_test_envs_tn.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LIAB = HERE.parents[1] / "LIABILITY_STUDY_2026-09-29"
sys.path[:0] = [str(LIAB), str(HERE)]
import envs_tn  # noqa: E402

spec = importlib.util.spec_from_file_location("envs_n_registered", HERE.parent / "study_n" / "envs_n.py")
envs_n = importlib.util.module_from_spec(spec)
spec.loader.exec_module(envs_n)                      # its hard-coded LIAB path is harmless: envs_b is already importable

mk = envs_tn.load_market("DK1", "train", "NE")
bad = 0
total = 0
for arm in envs_tn.ARMS:
    for seed in (1, 2, 3):
        a, b = envs_n.LagEnv(mk, arm, train=True, seed=seed), envs_tn.LagEnvT(mk, arm, train=True, seed=seed)
        oa, _ = a.reset()
        ob, _ = b.reset()
        assert np.allclose(oa, ob)
        rng = np.random.default_rng(seed)
        for _ in range(3000):
            act = int(rng.integers(9))
            ra = a.step(act)
            rb = b.step(act)
            total += 1
            same = np.allclose(ra[0], rb[0]) and abs(ra[1] - rb[1]) < 1e-12 and ra[2] == rb[2] and ra[3] == rb[3] and abs(ra[4]["pnl"] - rb[4]["pnl"]) < 1e-9
            bad += not same
            if ra[2] or ra[3]:
                oa, _ = a.reset()
                ob, _ = b.reset()
print(f"{total} steps compared across 4 arms and 3 seeds; mismatches: {bad}")
assert bad == 0
print("OK: LagEnvT reproduces the registered environment")
