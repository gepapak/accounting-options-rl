"""Exact checks of the Study Z environments and of the split-account proposition (run before any training; `python test_envs_z.py`).

1  every step: reported change = booked P&L + forgiven, forgiven >= 0 (both structures are forgiving accountings)
2  for the same state, action and sign, the split structure forgives at least as much as the pooled one (sub-additivity of the floor)
3  a perfect hedge (+16 in one account, -16 in the other) from two empty accounts: the portfolio reward is the same for both signs, equal to
   (q|s| - c)^+ - ... the winning leg's net gain, while the booked P&L is minus both costs; its sign twin returns the same reward, so SPR and TIR
   equal the booked P&L
4  park-and-gamble: with one account at positive equity and the other empty, no step that trades only from the empty account changes the
   portfolio from positive to zero or lowers it, so a portfolio-level ruin (or loss) penalty never fires; the account-level count does
5  twin bracket: -r_twin - 2C <= booked <= r on every step (Proposition 10), for random states and actions of both structures
6  SPR is fair-sign unbiased for any state and action: the mean of SPR over the two signs equals minus the own cost
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from envs_z import K0, LEV, NLEV, PooledEnv, SplitEnv  # noqa: E402

TOL = 1e-9
I16P, I16N, I0 = int(np.where(LEV == 16.0)[0][0]), int(np.where(LEV == -16.0)[0][0]), int(np.where(LEV == 0.0)[0][0])


def one_step(env, action, flip=False):
    e = copy.copy(env)
    e.flip_next = flip
    _, r, _, _, info = e.step(action)
    return r, info, e


def random_state(env, rng):
    env.reset(options={"ea": float(rng.choice([0.0, rng.uniform(0, 0.02), rng.uniform(0, 0.5), rng.uniform(0, 2)])),
                       "eb": float(rng.choice([0.0, rng.uniform(0, 0.02), rng.uniform(0, 0.5), rng.uniform(0, 2)]))})
    env.t = int(rng.integers(env.t0, env.t_end - 1))
    return env


def main() -> int:
    rng = np.random.default_rng(20261103)
    n_fail = 0
    counts = dict(identity=0, subadd=0, hedge=0, park=0, bracket=0, spr=0)
    split = SplitEnv("train", "edge", "reported", seed=1)
    pooled = PooledEnv("train", "edge", "reported", seed=1)
    split.reset()
    pooled.reset()
    for _ in range(20_000):
        random_state(split, rng)
        a = (int(rng.integers(NLEV)), int(rng.integers(NLEV)))
        pooled.reset(options={"ea": split.ea / K0, "eb": split.eb / K0, "t0": split.t0})
        pooled.t = split.t
        pooled.eps_ep = split.eps_ep
        for flip in (False, True):
            r_s, i_s, _ = one_step(split, a, flip)
            r_p, i_p, _ = one_step(pooled, a, flip)
            # 1 identity and forgiving
            for r, i in ((r_s, i_s), (r_p, i_p)):
                ok = abs(r * K0 / 100.0 - (i["pnl"] + i["forgiven"])) < 1e-6 and i["forgiven"] >= -TOL
                n_fail += not ok
                counts["identity"] += 1
            # 2 sub-additivity (same booked P&L, split forgives at least as much)
            ok = abs(i_s["pnl"] - i_p["pnl"]) < 1e-9 and i_s["forgiven"] >= i_p["forgiven"] - 1e-9
            n_fail += not ok
            counts["subadd"] += 1
        # 5 bracket and 6 SPR unbiasedness on both structures
        for env in (split, pooled):
            r, info, _ = one_step(env, a, False)
            r_t, info_t, _ = one_step(env, a, True)
            c = info["cost"] / K0 * 100.0
            x = info["pnl"] / K0 * 100.0
            x_t = info_t["pnl"] / K0 * 100.0
            ok = (-r_t - 2 * c <= x + 1e-9) and (x <= r + 1e-9)
            n_fail += not ok
            counts["bracket"] += 1
            spr = 0.5 * (r - r_t) - c
            spr_t = 0.5 * (r_t - r) - c
            ok = abs(0.5 * (spr + spr_t) + c) < 1e-9 and abs(0.5 * (x + x_t) + c) < 1e-9
            n_fail += not ok
            counts["spr"] += 1
    # 3 the perfect hedge from two empty accounts
    for _ in range(2_000):
        split.reset(options={"ea": 0.0, "eb": 0.0})
        split.t = int(rng.integers(split.t0, split.t_end - 1))
        for a in ((I16P, I16N), (I16N, I16P)):
            r, info, _ = one_step(split, a, False)
            r_t, info_t, _ = one_step(split, a, True)
            qa, qb, ca, cb = split.legs(a)
            g = abs(qa) * split.abs_s[split.t]
            expect = max(g - ca, 0.0) / K0 * 100.0                # the winning leg's net gain; the losing leg is floored
            c = info["cost"] / K0 * 100.0
            x = info["pnl"] / K0 * 100.0
            tir = 0.0 * 0.0 - c                                   # sigma = sign(qa + qb) * eps = 0 for a perfect hedge
            ok = (abs(r - r_t) < 1e-9 and abs(r - expect) < 1e-9 and abs(x + c) < 1e-9 and abs(0.5 * (r - r_t) - c - x) < 1e-9
                  and abs(tir - x) < 1e-9)
            n_fail += not ok
            counts["hedge"] += 1
    # 4 park-and-gamble: account B parked at positive equity, gamble only from the empty account A
    for _ in range(2_000):
        split.reset(options={"ea": 0.0, "eb": float(rng.uniform(0.01, 2.0))})
        split.t = int(rng.integers(split.t0, split.t_end - 1))
        for lev_a in range(NLEV):
            for flip in (False, True):
                r, info, _ = one_step(split, (lev_a, I0), flip)
                ok = info["ruin_portfolio"] == 0 and r >= -1e-12          # never a portfolio ruin, never a portfolio loss
                n_fail += not ok
                counts["park"] += 1
        # from A > 0 (after a win) and B > 0, a losing gamble in A ruins A (account count) but not the portfolio
        split.reset(options={"ea": float(rng.uniform(0.001, 0.01)), "eb": float(rng.uniform(0.01, 2.0))})
        split.t = int(rng.integers(split.t0, split.t_end - 1))
        for flip in (False, True):
            _, info, e2 = one_step(split, (I16P, I0), flip)
            ok = info["ruin_portfolio"] == 0 and info["ruin_accounts"] == int(e2.ea <= 0.0)
            n_fail += not ok
            counts["park"] += 1
    print("checks:", counts, "failures:", n_fail)
    print("ALL PASS" if n_fail == 0 else "FAILURES")
    return int(n_fail > 0)


if __name__ == "__main__":
    sys.exit(main())
