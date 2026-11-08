"""The reward diagram of the official Gym-Trading-Env, read from its reward alone (Study EXT, Part S). No learning. The wealth is used only to score the reading.

States come from random-policy episodes of the engine in the placebo market, in two configurations: the documented default position set [0, 1] and the leverage set of the engine's tutorial
discussion [-10, ..., 10]. At every sampled (state, action) the reward is read along the line through the next price on clones:

    - tail reading: gross from the slopes at +-T, +-2T; cost from phi(0); booked estimate = gross - cost;
    - jump search: where the slopes stop being convex the interval is bisected; a difference that does not shrink with the interval is a jump (the reward replaced on a branch);
    - the position and the balance read from the same queries: slope = position * move, jump where the wealth reaches zero.

Output: results_ext/scan_ext.json with the counts registered in PREREGISTRATION_EXT.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from envs_ext import POSITIONS, T_FAR, W0, ExtEnv, clone_core  # noqa: E402

GRID = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0)
N_STATES = 4000


def leave_line(q, r0, g, a, b, tol, iters=80):
    """phi lies on the line r0 + g * t at a and off it at b: bisect for the point where it leaves. Returns (location, phi just outside minus phi just inside): a change that does not
    shrink with the interval is a jump; a convex kink gives a change near zero."""
    on = lambda t: abs(q(t) - (r0 + g * t)) <= tol  # noqa: E731
    assert on(a) and not on(b)
    for _ in range(iters):
        m = 0.5 * (a + b)
        if on(m):
            a = m
        else:
            b = m
        if abs(b - a) < 1e-10 * max(1.0, abs(a)):
            break
    return 0.5 * (a + b), q(b) - q(a)


def read_state(env, a):
    q = lambda t: env.query(a, t)  # noqa: E731
    r1, rm, r0 = q(1.0), q(-1.0), q(0.0)
    c = max(-r0, 0.0)
    pp, p2, mm, m2 = q(T_FAR), q(2.0 * T_FAR), q(-T_FAR), q(-2.0 * T_FAR)
    ap, am = (p2 - pp) / T_FAR, (mm - m2) / T_FAR
    g = (ap + am) if ap * am < 0.0 else (ap if abs(ap) >= abs(am) else am)
    ts = sorted({0.0} | {t for t in GRID} | {-t for t in GRID})
    phi = np.array([q(t) for t in ts])
    s = np.diff(phi) / np.diff(ts)
    # a convexity violation on the grid means a jump somewhere: floors alone cannot produce it
    tol = 1e-7 * (1.0 + np.abs(phi).max())
    viol = np.flatnonzero(np.diff(s) * np.minimum(np.diff(ts)[:-1], np.diff(ts)[1:]) < -tol)
    jump = None
    if abs(g) > 1e-9:
        # walk outward from t = 0 on each side; where phi first leaves its affine piece, bisect and measure the change there
        line_tol = 1e-7 * (1.0 + abs(g)) * 1024.0 * 1e-3
        best = None
        for side in (1.0, -1.0):
            prev = 0.0
            for t in GRID:
                tt = side * t
                if abs(q(tt) - (r0 + g * tt)) > line_tol:
                    loc, size = leave_line(q, r0, g, prev, tt, line_tol)
                    if best is None or abs(loc) < abs(best[0]):
                        best = (loc, size)
                    break
                prev = tt
        jump = best
    width = r1 + rm + 2.0 * c
    return dict(r1=r1, rm=rm, c=c, g=g, x_hat=g - c, width=width, jump=jump, nonconvex=bool(len(viol)), ap=ap, am=am)


def run_config(name, positions, seed, n_states):
    env = ExtEnv("test", "placebo", "REP", seed=seed, positions=positions)
    rng = np.random.default_rng(seed)
    rows = []
    o, _ = env.reset()
    while len(rows) < n_states:
        a = int(rng.integers(len(positions)))
        w_old = env.wealth
        rec = read_state(env, a)
        # ground truth, from the engine's public wealth on two clones
        c1 = clone_core(env.core)
        c1.step(a)
        x_true = (float(c1.historical_info["portfolio_valuation", -1]) - w_old) / W0 * 100.0
        c0 = clone_core(env.core, price_line=0.0)
        c0.step(a)
        c_true = -(float(c0.historical_info["portfolio_valuation", -1]) - w_old) / W0 * 100.0
        move = float(env.core._price_array[env.core._idx + 1] / env.core._price_array[env.core._idx] - 1.0)
        # the wealth after the trade and the holdings at the next price: the true jump location
        asset = float(c0._portfolio.asset)
        fiat = float(c0._portfolio.fiat)
        p, dp = float(env.core._price_array[env.core._idx]), float(env.core._price_array[env.core._idx + 1] - env.core._price_array[env.core._idx])
        t_star = -(asset * p + fiat) / (asset * dp) if abs(asset * dp) > 1e-12 else np.nan
        rows.append((abs(positions[a]), rec["x_hat"], x_true, rec["c"], c_true, rec["r1"], rec["rm"], rec["width"], rec["nonconvex"], np.nan if rec["jump"] is None else rec["jump"][0], np.nan if rec["jump"] is None else rec["jump"][1], t_star, w_old, move, rec["g"]))
        _, _, d, tr, _ = env.step(a)
        if d or tr:
            o, _ = env.reset()
    R = np.array(rows, dtype=float)
    lev, xh, xt, ch, ct, r1, rm, width, nonconv, jt, js, tstar, wold, mv, g = R.T
    moved = lev > 0
    err = np.abs(xh - xt)
    r_err = np.abs(r1 - xt)
    tir = np.where(np.abs(r1 - rm) > 1e-9, np.sign(r1 - rm) * (np.maximum(r1, rm) + ch) - ch, -ch)
    spr = 0.5 * (r1 - rm) - ch
    near = np.isfinite(jt) & np.isfinite(tstar) & (np.abs(jt - tstar) <= 1e-4 * np.maximum(1.0, np.abs(tstar)))
    big_jump = np.isfinite(js) & (np.abs(js) > 1e-3)
    bust_step = xt <= -(wold / W0 * 100.0) + 1e-9        # the real step takes the wealth to zero or below
    within16 = np.isfinite(tstar) & (np.abs(tstar) <= 16.0)
    out = dict(config=name, positions=list(positions), states=len(R), moved=int(moved.sum()), exact_within_1e_6=int((err < 1e-6).sum()), max_err=float(err.max()),
               reported_exact=int((r_err < 1e-6).sum()), tir_exact=int((np.abs(tir - xt) < 1e-6).sum()), spr_exact=int((np.abs(spr - xt) < 1e-6).sum()), cost_exact=int((np.abs(ch - ct) < 1e-9).sum()),
               nonconvex=int(nonconv.sum()), left_the_line=int(np.isfinite(jt).sum()), jumps=int(big_jump.sum()), jumps_at_true_location=int((near & big_jump).sum()), median_jump=float(np.median(np.abs(js[big_jump]))) if big_jump.any() else None,
               jump_equals_wealth=int((np.abs(np.abs(js[big_jump]) - (wold[big_jump] / W0 * 100.0)) <= 0.05 * (wold[big_jump] / W0 * 100.0) + 1e-3).sum()) if big_jump.any() else 0,
               median_abs_t_star_by_position={str(int(L)): float(np.nanmedian(np.abs(tstar[lev == L]))) for L in sorted(set(lev)) if L > 0},
               share_kink_within_16_by_position={str(int(L)): float(np.mean(within16[lev == L])) for L in sorted(set(lev)) if L > 0},
               width_nonzero=int((np.abs(width) > 1e-7).sum()), bust_steps=int(bust_step.sum()), width_nonzero_on_bust=int((np.abs(width[bust_step]) > 1e-7).sum()), width_nonzero_off_bust=int((np.abs(width[~bust_step]) > 1e-7).sum()))
    return out


def main():
    res = {}
    for name, positions, seed in (("default_0_1", (0, 1), 20261107), ("leverage_10", POSITIONS, 20261108)):
        res[name] = run_config(name, positions, seed, 40 if "--smoke" in sys.argv else N_STATES)
        print(json.dumps(res[name], indent=1))
    outdir = HERE / ("smoke_results" if "--smoke" in sys.argv else "results_ext")
    outdir.mkdir(exist_ok=True)
    (outdir / "scan_ext.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
