"""Exploratory context for Study T (not registered): booked edge of the ORPO reference policy and of simple fixed policies on the same
100 test windows and start equities that every Study T run is evaluated on (edge market, test split, evaluation seed 9000).

    python reference_policy_edge.py
"""
import sys
from pathlib import Path

import numpy as np

STUDY_T = Path(__file__).resolve().parents[2] / "ACCOUNTING_OPTIONS_2026-10-14" / "study_t"
sys.path[:0] = [str(STUDY_T)]
from envs_t import K0, LEV, SETTINGS, EdgeEnvT  # noqa: E402
from orpo_t import ref_actions  # noqa: E402


def evaluate(setting, policy, s=9000, n_ep=100):
    env = EdgeEnvT(setting, "test", "edge", "reported", seed=s)
    rng = np.random.default_rng(s)                 # the test-window generator of run_t.py
    act_rng = np.random.default_rng(s + 1)
    out = []
    for _ in range(n_ep):
        o, _ = env.reset(options={"e0": float(np.exp(rng.uniform(np.log(0.05), np.log(2.0))))})
        done = False
        while not done:
            o, r, done, _, _ = env.step(int(policy(o, act_rng)))
        out.append((env.ledger - env.e0) / K0 * 100.0)
    return float(np.mean(out))


def follow(level_idx_pos, level_idx_neg):
    return lambda o, rng: level_idx_pos if o[1] > 0 else level_idx_neg


if __name__ == "__main__":
    print("booked edge, % of K0 per one-week episode (mean of 100 test episodes)")
    for st in SETTINGS:
        ref = evaluate(st, lambda o, rng: int(ref_actions(np.asarray(o)[None, :], rng)[0]))
        f1 = evaluate(st, follow(6, 2))
        f4 = evaluate(st, follow(7, 1))
        f16 = evaluate(st, follow(8, 0))
        print(f"{st:7s} reference policy {ref:+6.2f} | follow z at leverage 1: {f1:+6.2f}  4: {f4:+6.2f}  16: {f16:+6.2f}")
