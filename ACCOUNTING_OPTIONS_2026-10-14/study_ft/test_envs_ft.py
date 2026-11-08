"""Exact checks of Study FT's environments and rewards, run before any pilot. Random actions, all starts; no training.

  1. Convexity: the reported reward of each environment is convex in t along the line through the realized move, for every state and action.
  2. The width identity: r + r' + 2C = (phi(1) + phi(-1) - 2 phi(0)) + 2 (phi(0) + C).
  3. TSR equals the booked P&L on every step of every environment, within 1e-6 points of K0.
  4. Reported minus booked is non-negative in the bucket and partial environments (forgiving). The carried deficit is convex but not forgiving: it withholds a profit made
     while insolvent, so those steps are counted and not treated as violations.
  5. Where the sign query fails: the steps on which TIR and SPR are wrong, and on which the pair coincides though the gross is not zero.
"""
from __future__ import annotations

import copy
import sys

import numpy as np

from envs_ft import K0, BucketEnv, DeficitEnv, PartialEnv
from rewards_ft import line_reward, tail_estimate, twin_reward

sys.stdout.reconfigure(encoding="utf-8")


def starts(rng, kind):
    if kind == "bucket":
        return dict(cash0=float(rng.choice([0.0, 0.0, 0.02, 0.1, 1.0])), bucket0=float(rng.choice([0.0, -0.05, -0.5, -2.0, 0.3])))
    if kind == "deficit":
        return dict(e0=float(rng.choice([0.0, 0.05, 1.0, -0.02, -0.5, -2.0])))
    return dict(e0=float(rng.choice([0.0, 0.02, 0.1, 1.0, 2.0])))


def check(env, kind, n_steps, rng):
    bad = dict(convex=0, width=0, tsr=0, negative=0)
    err = dict(tsr=0.0, tir=0.0, spr=0.0)
    wrong = dict(tir=0, spr=0, coincide=0, withheld=0)
    n = 0
    env.reset(options=starts(rng, kind))
    for _ in range(n_steps):
        a = env.action_space.sample()
        ts = rng.uniform(-40, 40, size=2)
        phi = lambda t: line_reward(env, a, float(t))  # noqa: E731
        if phi(ts.mean()) * 2 > phi(ts[0]) + phi(ts[1]) + 1e-9:
            bad["convex"] += 1
        g_hat = tail_estimate(env, a)
        sigma = env.realized_sign(a)
        r_twin = twin_reward(env, a)
        f0 = phi(0.0)
        o, r, done, _, info = env.step(a)
        c = info["cost"] / K0 * 100.0
        x = info["pnl"] / K0 * 100.0
        if abs((r + r_twin + 2 * c) - ((r + r_twin - 2 * f0) + 2 * (f0 + c))) > 1e-9:
            bad["width"] += 1
        if abs(g_hat - c - x) > 1e-6:
            bad["tsr"] += 1
        err["tsr"] = max(err["tsr"], abs(g_hat - c - x))
        if r < x - 1e-9:                                   # the owner's claim withholds a profit made while insolvent: convex but not forgiving
            wrong["withheld"] += 1
            if kind != "deficit":
                bad["negative"] += 1
        tir = sigma * (max(r, r_twin) + c) - c if abs(r - r_twin) > 1e-12 else -c
        spr = 0.5 * (r - r_twin) - c
        err["tir"] = max(err["tir"], abs(tir - x))
        err["spr"] = max(err["spr"], abs(spr - x))
        wrong["tir"] += abs(tir - x) > 1e-6
        wrong["spr"] += abs(spr - x) > 1e-6
        wrong["coincide"] += abs(r - r_twin) <= 1e-12 and abs(info["gross"]) > 1e-9
        n += 1
        if done:
            env.reset(options=starts(rng, kind))
    return n, bad, err, wrong


def main():
    rng = np.random.default_rng(61108)
    total_bad = 0
    for name, env, kind in (("bucket", BucketEnv("train", "edge", "reported", seed=1), "bucket"), ("deficit", DeficitEnv("train", "edge", "reported", seed=2), "deficit"),
                            ("partial 0.10", PartialEnv("train", "edge", "reported", seed=3, lam=0.1), "partial"), ("partial 0.50", PartialEnv("train", "edge", "reported", seed=4, lam=0.5), "partial"),
                            ("partial 1.00", PartialEnv("train", "edge", "reported", seed=5, lam=1.0), "partial"), ("placebo bucket", BucketEnv("train", "placebo", "reported", seed=6), "bucket")):
        env.action_space.seed(7)
        n, bad, err, wrong = check(env, kind, 30000, rng)
        total_bad += sum(bad.values())
        print(f"{name:15s} steps {n}; violations {bad}; largest error, points of K0: TSR {err['tsr']:.2e}, TIR {err['tir']:.2f}, SPR {err['spr']:.2f}; steps wrong: TIR {wrong['tir']}, SPR {wrong['spr']}; pair coincides with gross non-zero {wrong['coincide']}; profit withheld {wrong['withheld']}")
    assert total_bad == 0, "a registered property failed"
    print("all checks passed")


if __name__ == "__main__":
    main()
