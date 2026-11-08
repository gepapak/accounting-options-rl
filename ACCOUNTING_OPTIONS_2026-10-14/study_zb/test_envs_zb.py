"""Exact checks of the Study Z, Part B environment (run before any training; `python test_envs_zb.py`).

1  every step, both modes and signs: reported change = booked P&L + forgiven, forgiven >= 0 (a forgiving accounting)
2  single censor: the reward is max(x, tau) with tau = -min(m, P) in isolated mode and -P in cross mode, so TIR equals the booked P&L on every
   step whose pair differs, and the pair differs whenever the move is not tiny relative to the cost at an account below its cost
3  an isolated position never takes an account with P > m to zero (a penalty on the account's ruin never fires), while it is liquidated
   exactly when its loss exceeds m
4  twin bracket: -r_twin - 2C <= booked <= r on every step (Proposition 10)
5  SPR is fair-sign unbiased: its mean over the two signs equals minus the own cost
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from envs_zb import K0, LEV, M_ISO, NLEV, IsolatedEnv  # noqa: E402


def one(env, a, flip):
    e = copy.copy(env)
    e.flip_next = flip
    _, r, _, _, info = e.step(a)
    return r, info


def main() -> int:
    rng = np.random.default_rng(20261103)
    env = IsolatedEnv("train", "edge", "reported", seed=3)
    env.reset()
    counts = dict(identity=0, single_floor=0, tir_exact=0, isolated_no_ruin=0, liquidation=0, bracket=0, spr=0)
    fails = 0
    for _ in range(40_000):
        e0 = float(rng.choice([0.0, rng.uniform(0, 0.01), rng.uniform(0, 0.5), rng.uniform(0, 2.0)]))
        env.reset(options={"e0": e0})
        env.t = int(rng.integers(env.t0, env.t_end - 1))
        a = (int(rng.integers(NLEV)), int(rng.integers(2)))
        q, cost, iso = env.position(a)
        P = env.reported
        res = {f: one(env, a, f) for f in (False, True)}
        for flip, (r, info) in res.items():
            x = info["pnl"] / K0 * 100.0
            ok = abs(r * K0 / 100.0 - (info["pnl"] + info["forgiven"])) < 1e-6 and info["forgiven"] >= -1e-9
            fails += not ok
            counts["identity"] += 1
            tau = -(min(cost + M_ISO * abs(q), P) if (iso and q != 0.0) else P) / K0 * 100.0
            ok = abs(r - max(x, tau)) < 1e-9
            fails += not ok
            counts["single_floor"] += 1
            if iso and q != 0.0:
                m = cost + M_ISO * abs(q)
                ok = info["liquidation"] == int(info["pnl"] < -min(m, P))
                fails += not ok
                counts["liquidation"] += 1
                if P > m:
                    ok = info["ruin_portfolio"] == 0
                    fails += not ok
                    counts["isolated_no_ruin"] += 1
        (r, info), (r_t, info_t) = res[False], res[True]
        c = info["cost"] / K0 * 100.0
        x = info["pnl"] / K0 * 100.0
        x_t = info_t["pnl"] / K0 * 100.0
        ok = (-r_t - 2 * c <= x + 1e-9) and (x <= r + 1e-9)
        fails += not ok
        counts["bracket"] += 1
        spr, spr_t = 0.5 * (r - r_t) - c, 0.5 * (r_t - r) - c
        ok = abs(0.5 * (spr + spr_t) + c) < 1e-9 and abs(0.5 * (x + x_t) + c) < 1e-9
        fails += not ok
        counts["spr"] += 1
        if abs(r - r_t) > 1e-12:                                       # TIR identifies the step: it must equal the booked P&L
            sigma = float(np.sign(q)) * float(env.eps_ep[env.t - env.t0])
            tir = sigma * (max(r, r_t) + c) - c
            ok = abs(tir - x) < 1e-9
            fails += not ok
            counts["tir_exact"] += 1
    print("checks:", counts, "failures:", fails)
    print("ALL PASS" if fails == 0 else "FAILURES")
    return int(fails > 0)


if __name__ == "__main__":
    sys.exit(main())
