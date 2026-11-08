"""Exact checks of Study EXT's harness, run before any registered training. No learning. The wealth is used only to score.

 1. The vendored source equals the pinned commit, byte for byte (SOURCE_MANIFEST.json); the example data equal the pinned bytes.
 2. clone_core gives the same reward and valuation as copy.deepcopy(core) at t = 1, -1, 0 and far along the line, and queries leave the running engine unchanged.
 3. The accounting: the reward equals the change of wealth on every step that does not take the wealth to zero or below, and is 0 on a step that does.
 4. The bracket -r' - 2C <= x <= r holds on every step, with C read from phi(0).
 5. TSR equals the booked change on every step within 1e-6 points; TIR wherever the favourable branch survives.
"""
from __future__ import annotations

import copy

import numpy as np

from envs_ext import POSITIONS, T_FAR, ExtEnv, daily_returns, verify_vendor


def main(n_steps=60000):
    print("vendored files verified:", verify_vendor())
    daily_returns()                                                       # asserts the data's hash
    rng = np.random.default_rng(20261107)
    n = dict(steps=0, busts=0, deep=0, bracket=0, tsr=0, tir=0, tir_tried=0, reward_ok=0)
    for kind, positions, seed in (("placebo", POSITIONS, 1), ("edge", POSITIONS, 2), ("placebo", (0, 1), 3)):
        env = ExtEnv("train", kind, "REP", seed=seed, positions=positions)
        env.reset()
        for k in range(n_steps // 3):
            a = int(rng.integers(len(positions)))
            if k % 50 == 0:                                                 # clones against a true deep copy (slow, so on a sample)
                for t in (1.0, -1.0, 0.0, 37.0, -T_FAR):
                    d = copy.deepcopy(env.core)
                    p = d._price_array.copy()
                    p[d._idx + 1] = p[d._idx] + t * (p[d._idx + 1] - p[d._idx])
                    d._price_array = p
                    r_deep = float(d.step(a)[1])
                    assert abs(env.query(a, t) - r_deep) <= 1e-9 * (1.0 + abs(r_deep)), "clone differs from a deep copy"
                    n["deep"] += 1
            before = (env.core._idx, env.wealth, float(env.core._portfolio.asset), float(env.core._portfolio.fiat), env.core._position)
            r_twin = env.query(a, -1.0)
            c = max(-env.query(a, 0.0), 0.0)
            pp, p2, mm, m2 = (env.query(a, t) for t in (T_FAR, 2.0 * T_FAR, -T_FAR, -2.0 * T_FAR))
            assert before == (env.core._idx, env.wealth, float(env.core._portfolio.asset), float(env.core._portfolio.fiat), env.core._position), "a query changed the engine"
            ap, am = (p2 - pp) / T_FAR, (mm - m2) / T_FAR
            g = (ap + am) if ap * am < 0.0 else (ap if abs(ap) >= abs(am) else am)
            _, _, d, tr, info = env.step(a)
            r, x = info["reported"], info["booked"]
            n["steps"] += 1
            if d:
                n["busts"] += 1
                assert r == 0.0, "the engine rewarded a bust step"
                assert env.wealth <= 0.0
            else:
                assert abs(r - x) <= 1e-9 * (1.0 + abs(x)), "reward and wealth change differ on a surviving step"
            n["reward_ok"] += 1
            assert -r_twin - 2.0 * c <= x + 1e-6 and x <= r + 1e-6, "bracket"
            n["bracket"] += 1
            assert abs(g - c - x) <= 1e-6, f"TSR {g - c} booked {x}"
            n["tsr"] += 1
            if abs(r - r_twin) > 1e-9:
                n["tir_tried"] += 1
                n["tir"] += abs(np.sign(r - r_twin) * (max(r, r_twin) + c) - c - x) <= 1e-6
            if d or tr:
                env.reset()
    print(n)
    print("all checks passed")


if __name__ == "__main__":
    main()
