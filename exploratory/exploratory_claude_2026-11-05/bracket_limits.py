"""Exploratory: where the twin bracket and TIR stop, and how the bracket's width compares with the forgiveness a policy actually consumed.

    python bracket_limits.py            # writes bracket_limits.json
    python bracket_limits.py --print    # prints the results, writes nothing

None of this contradicts the propositions, which state their premises; it measures what happens outside them, on the repository's own simulators:
  1. a carried deficit (Study N's settlement-lag environment) reports less than the booked profit while insolvent, so its credit is negative there;
  2. split accounts with legs of opposite sign: the favourable portfolio branch can itself contain a credit, so an unequal pair does not identify x;
  3. a cost charged on one branch only (a liquidation fee): the bracket's lower end can exceed the booked P&L while the width stays positive;
  4. a payoff convex in the sign: positive width with nothing forgiven;
  5. the width against the realized forgiveness, from the stored results of Studies Z and RS.
Trains nothing; random actions use fixed seeds.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ACC = REPO / "ACCOUNTING_OPTIONS_2026-10-14"
sys.path[:0] = [str(ACC / "study_t"), str(ACC / "study_z"), str(REPO / "LIABILITY_STUDY_2026-09-29"), str(REPO / "GENERALIZATION_STUDY_2026-09-29")]
from envs_tn import K0, LEV, LagEnvT, load_market  # noqa: E402
from envs_z import LEV as LEV_Z  # noqa: E402
from envs_z import K0 as K0_Z  # noqa: E402
from envs_z import SplitEnv  # noqa: E402

TOL = 1e-9


def carried_deficit() -> dict:
    """Random actions in Study N's placebo market, half of the episodes starting insolvent (its training starts). ZFG: settlement lag, TFG: terminal."""
    m = load_market("DK2", "test", "NE", draw=0)
    out = {}
    for arm in ("ZFG", "TFG"):
        env = LagEnvT(m, arm, train=True, seed=2)
        g = np.random.default_rng(3)
        steps = upper = lower = eps = eps_upper = eps_lower = 0
        for _ in range(300):
            env.reset()
            sx = sr = slo = 0.0
            done = False
            while not done:
                a = int(g.integers(0, len(LEV)))
                twin = copy.copy(env)
                twin.flip_next = True
                rp = twin.step(a)[1]
                _, r, term, trunc, info = env.step(a)
                x, c = info["pnl"] / K0 * 100, info["cost"] / K0 * 100
                lo = -rp - 2 * c
                steps += 1
                upper += x > r + TOL
                lower += lo > x + TOL
                sx, sr, slo = sx + x, sr + r, slo + lo
                done = term or trunc
            eps += 1
            eps_upper += sx > sr + 1e-6
            eps_lower += slo > sx + 1e-6
        out[arm] = dict(steps=steps, step_upper_fails=int(upper), step_lower_fails=int(lower), episodes=eps, episode_upper_fails=int(eps_upper), episode_lower_fails=int(eps_lower))
    env = LagEnvT(m, "ZFG", train=True, seed=1)
    env.reset()
    env.L, env.k, env.t = -0.5 * K0, 50, 1000
    while abs(m.spread[env.t]) < 20:
        env.t += 1
    a = int(np.argmax(LEV)) if m.spread[env.t] > 0 else int(np.argmin(LEV))
    twin = copy.copy(env)
    twin.flip_next = True
    rp = twin.step(a)[1]
    _, r, _, _, info = env.step(a)
    out["one_zombie_step"] = dict(booked=info["pnl"] / K0 * 100, reward=r, twin_reward=rp, cost=info["cost"] / K0 * 100)
    return out


def mixed_legs() -> dict:
    env = SplitEnv("test", "edge", "reported", seed=1)
    env.reset(options=dict(t0=500, ea=1.0, eb=0.0))
    while env.abs_s[env.t] < 30:
        env.t += 1
    long_first = env.eps_ep[env.t - env.t0] > 0
    act = (int(np.where(LEV_Z == (16 if long_first else -16))[0][0]), int(np.where(LEV_Z == (-4 if long_first else 4))[0][0]))
    qa, qb, _, _ = env.legs(act)
    sign = float(env.eps_ep[env.t - env.t0])
    twin = copy.copy(env)
    twin.flip_next = True
    rp = twin.step(act)[1]
    _, r, _, _, info = env.step(act)
    x, c = info["pnl"] / K0_Z * 100, info["cost"] / K0_Z * 100
    sigma = float(np.sign(qa + qb)) * sign
    example = dict(q_funded=qa, q_empty=qb, booked=x, reward=r, twin_reward=rp, tir=sigma * (max(r, rp) + c) - c, lower=-rp - 2 * c)
    g = np.random.default_rng(9)
    env = SplitEnv("test", "edge", "reported", seed=2)
    steps = unequal = wrong = bracket_fails = 0
    errs = []
    for _ in range(50):
        env.reset(options=dict(ea=float(g.choice([0, .02, .1, 1])), eb=float(g.choice([0, .02, .1, 1]))))
        done = False
        while not done:
            act = (int(g.integers(0, 9)), int(g.integers(0, 9)))
            qa, qb, _, _ = env.legs(act)
            sign = float(env.eps_ep[env.t - env.t0])
            twin = copy.copy(env)
            twin.flip_next = True
            rp = twin.step(act)[1]
            _, r, done, _, info = env.step(act)
            x, c = info["pnl"] / K0_Z * 100, info["cost"] / K0_Z * 100
            steps += 1
            bracket_fails += (-rp - 2 * c > x + TOL) or (x > r + TOL)
            if abs(r - rp) > 1e-12:
                unequal += 1
                t = float(np.sign(qa + qb)) * sign * (max(r, rp) + c) - c
                if abs(t - x) > TOL:
                    wrong += 1
                    errs.append(abs(t - x))
    z = pd.read_csv(ACC / "study_z" / "runs_z.csv")
    return dict(example=example, random_actions=dict(steps=steps, unequal_pairs=unequal, tir_wrong=wrong, largest_error=float(max(errs)), median_error=float(np.median(errs)), bracket_fails=int(bracket_fails)),
                trained_agents_largest_tir_error_placebo=float(z.err_tir_plac.max()))


def liquidation_fee() -> dict:
    """A long of 4 MWh entered at 100 EUR/MWh with a margin of 100 EUR/MWh and a prepaid cost of 0.10|q| + 4q^2 = 64.40 EUR, whose liquidation carries a
    fee of 1 EUR/MWh on the liquidated branch only. Whole-trade pair (path 74, -10 against its reflection 126, 210), and one tick from a pending liquidation."""
    q, margin, cost, fee = 4.0, 400.0, 64.40, 4.0
    gross = q * (-10.0 - 100.0)
    residual = margin + gross - fee
    credit = max(-residual, 0.0)
    booked = gross - cost - fee
    r = booked + credit
    r_twin = q * (210.0 - 100.0) - cost
    whole = dict(booked=booked, reward=r, twin_reward=r_twin, declared_cost=cost, lower=-r_twin - 2 * cost, upper=r, width=r + r_twin + 2 * cost, true_credit=credit)
    # one tick from the pending state: marked at 74, executed at -10; the twin moves by the opposite amount, to 158
    x = q * (-10.0 - 74.0) - fee
    equity_before = margin + q * (74.0 - 100.0)
    r1 = -equity_before
    r1_twin = q * (158.0 - 74.0) - fee
    tick = dict(booked=x, reward=r1, twin_reward=r1_twin, lower_fee_declared=-r1_twin - 2 * fee, lower_fee_undeclared=-r1_twin, width_fee_undeclared=r1 + r1_twin)
    return dict(whole_trade=whole, one_tick=tick)


def convex_payoff() -> dict:
    move, premium = 10.0, 3.0
    r, r_twin = max(move, 0.0) - premium, max(-move, 0.0) - premium
    return dict(reward=r, twin_reward=r_twin, width=r + r_twin + 2 * premium)


def width_vs_realized() -> dict:
    out = {}
    b = pd.read_csv(ACC / "study_zb" / "runs_zb.csv").groupby("arm")[["width_edge", "edge_forgiven"]].mean()
    for arm in ("FINDER", "LEDGER", "TIR", "REP"):
        out[f"ZB {arm}, edge"] = dict(width=float(b.width_edge[arm]), realized=float(b.edge_forgiven[arm]))
    z = pd.read_csv(ACC / "study_z" / "runs_z.csv").groupby("arm")[["width_plac", "placebo_forgiven"]].mean()
    for arm in ("LEDGER", "REP"):
        out[f"Z {arm}, placebo"] = dict(width=float(z.width_plac[arm]), realized=float(z.placebo_forgiven[arm]))
    for arm in ("ORACLE", "TIR", "PLAIN"):
        rows = []
        for s in range(300, 316):
            d = json.loads((ACC / "study_rs" / "results_rs" / f"RS_DK2__{arm}__s{s}.json").read_text())
            rows.append((d["audit_edge"]["reported"] - d["audit_edge"]["lower"], d["test_edge"]["forgiven_pct_K0"]))
        out[f"RS {arm}, edge"] = dict(width=float(np.mean([r[0] for r in rows])), realized=float(np.mean([r[1] for r in rows])))
    for v in out.values():
        v["width_over_realized"] = v["width"] / v["realized"]
    return out


def analysis() -> dict:
    return dict(carried_deficit=carried_deficit(), mixed_legs=mixed_legs(), liquidation_fee=liquidation_fee(), convex_payoff=convex_payoff(), width_vs_realized=width_vs_realized())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", dest="print_only", action="store_true")
    a = ap.parse_args()
    res = analysis()
    print(json.dumps(res, indent=1))
    if not a.print_only:
        (HERE / "bracket_limits.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
        print("wrote bracket_limits.json")


if __name__ == "__main__":
    main()
