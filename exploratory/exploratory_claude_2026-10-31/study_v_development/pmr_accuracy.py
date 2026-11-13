"""Exploratory: accuracy of the clone-free pooled-magnitude reward (PMR) against the booked P&L on random-action rollouts of the edge sleeve.

On every step the booked P&L x is compared with the reward each estimator assigns (all in percent of K0): the reported reward, the sign-paired
reward (SPR), the twin-identified reward (TIR) and PMR. Credited steps are those with a forgiven loss. Reported: mean absolute error on credited
and on uncredited steps, and the mean signed error (bias) over all steps, which is what a policy-gradient learner sees on average.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / "ACCOUNTING_OPTIONS_2026-10-14"
sys.path[:0] = [str(ROOT / "study_v"), str(ROOT / "study_t"), str(ROOT.parent / "LIABILITY_STUDY_2026-09-29")]
from envs_b import LEV  # noqa: E402
from envs_t import K0, EdgeEnvT  # noqa: E402
from rewards_v import MagnitudePool, make_pmr  # noqa: E402
from spr_t import make_spr, make_tir  # noqa: E402


def run(setting: str, n_ep: int, seed: int):
    # one pass per estimator, the same seeds, so every estimator sees the same episodes and actions
    out = {}
    for name in ("REP", "SPR", "TIR", "PMR"):
        rng = np.random.default_rng(seed)
        pool = MagnitudePool()
        base = EdgeEnvT(setting, "train", "edge", "reported", seed=seed)
        env = {"REP": base, "SPR": make_spr(base), "TIR": make_tir(base), "PMR": make_pmr(base, pool)}[name]
        steps = []
        for _ in range(n_ep):
            env.reset()
            done = False
            while not done:
                a = int(rng.integers(0, 9))
                eq = base.reported / K0
                _, r, done, _, info = env.step(a)
                steps.append((r, info["pnl"] / K0 * 100.0, info["forgiven"] / K0 * 100.0, eq, abs(LEV[a])))
        out[name] = np.array(steps)
    return out


def main():
    for setting in ("S_DK1", "T_DK2"):
        out = run(setting, 150, 77)
        x = out["REP"][:, 1]
        cred = out["REP"][:, 2] > 0
        print(f"\n{setting}: {len(x)} steps, {cred.mean() * 100:.1f}% credited")
        print(f"{'':6s} {'MAE credited':>13s} {'MAE uncredited':>15s} {'mean err (all)':>15s} {'mean err (credited)':>20s}")
        for name in ("REP", "SPR", "TIR", "PMR"):
            r = out[name][:, 0]
            e = r - x
            print(f"{name:6s} {np.abs(e[cred]).mean():13.4f} {np.abs(e[~cred]).mean():15.4f} {e.mean():+15.5f} {e[cred].mean():+20.4f}")
        print(f"mean booked x: {x.mean():+.5f}; mean PMR reward {out['PMR'][:, 0].mean():+.5f}")
        eq, lev = out["REP"][:, 3], out["REP"][:, 4]
        print("conditional on the state and action (what a learner sees): mean reward, percent of K0 per quarter-hour")
        print(f"{'bucket':32s} {'n':>6s} {'booked':>9s} {'REP':>9s} {'SPR':>9s} {'TIR':>9s} {'PMR':>9s}")
        for label, m in (("equity<=0.1, |lev|=16", (eq <= 0.1) & (lev == 16)), ("equity<=0.1, |lev|=4", (eq <= 0.1) & (lev == 4)),
                         ("equity<=0.1, |lev|=1", (eq <= 0.1) & (lev == 1)), ("equity>1, |lev|=16", (eq > 1) & (lev == 16))):
            print(f"{label:32s} {int(m.sum()):6d} {x[m].mean():+9.4f} " + " ".join(f"{out[n][m, 0].mean():+9.4f}" for n in ("REP", "SPR", "TIR", "PMR")))


if __name__ == "__main__":
    main()
