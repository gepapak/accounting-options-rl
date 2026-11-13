"""Brute-force, data-free check of the corrected identification statements (exploratory; imports no project code).

1. Identified set (the paper's twin-identification proposition): for each observed pair of twin rewards, enumerate every magnitude g', threshold tau'
   (including -infinity) and sign that reproduces the unordered pair with the same known cost, and compare with the claimed set
   {max(r, r') + C} (unequal pair) or [0, r + C] (equal pair).
2. SPR is not the sharp lower endpoint: g = 10, C = 1, tau = 0 gives the pair (9, 0), SPR magnitude 4.5, identified g = 10.
3. TIR's error never exceeds SPR's (grid of rational cases).
4. Minimax regret of the identification-failure proposition: brute-force search over deterministic policies q(z) against the closed form
   Q*A*B/(A+B).
5. Finite-pool counterexamples of the pooled-magnitude proposition in exact fractions.

    python check_identification_statements.py        # exit code 0 = every check passes
"""
from fractions import Fraction as F
from itertools import product
import sys

import numpy as np

FAILS = []


def pair(g, c, tau, sigma):
    """Twin rewards (r, r'); tau None = -infinity (no credit)."""
    a, b = sigma * g - c, -sigma * g - c
    if tau is None:
        return a, b
    return max(a, tau), max(b, tau)


def check_identified_sets():
    cands = [F(i, 4) for i in range(0, 41)]                       # g' in 0..10
    taus = [F(i, 4) for i in range(-60, 41)] + [None]
    n = bad = 0
    for gi, ci, ti, sigma in product(range(0, 21, 3), (0, 1, 2), range(-8, 12, 2), (-1, 1)):
        g, c, tau = F(gi, 2), F(ci), F(ti)
        r, rt = pair(g, c, tau, sigma)
        obs = sorted((r, rt))
        consistent = {g2 for g2 in cands for t2 in taus for s2 in (-1, 1) if sorted(pair(g2, c, t2, s2)) == obs}
        claimed = {max(r, rt) + c} if r != rt else {x for x in cands if 0 <= x <= r + c}
        n += 1
        if consistent != claimed:
            bad += 1
    print(f"identified sets: {n} observed pairs, {bad} mismatches")
    if bad:
        FAILS.append("identified sets")


def check_spr_example():
    r, rt = pair(F(10), F(1), F(0), 1)
    ok = (r, rt) == (9, 0) and abs(r - rt) / 2 == F(9, 2) and max(r, rt) + 1 == 10
    print(f"SPR example: pair {(r, rt)}, SPR magnitude {abs(r - rt) / 2}, identified g {max(r, rt) + 1}: {'ok' if ok else 'FAIL'}")
    if not ok:
        FAILS.append("SPR example")


def check_dominance():
    n = bad = 0
    for gi, ci, ti, sigma in product(range(0, 25), range(0, 4), range(-12, 14), (-1, 1)):
        g, c, tau = F(gi, 2), F(ci, 2), F(ti, 2)
        r, rt = pair(g, c, tau, sigma)
        x = sigma * g - c
        spr = (r - rt) / 2 - c
        tir = sigma * (max(r, rt) + c) - c if r != rt else -c
        n += 1
        bad += abs(tir - x) > abs(spr - x)
    print(f"TIR error dominance: {n} grid points, {bad} violations")
    if bad:
        FAILS.append("dominance")


def minimax_brute(Q, c, d, a, b, grid=201):
    qs = np.linspace(-Q, Q, grid)
    best = None
    for qp in qs:
        for qm in qs:
            regs = []
            for m in (a, b):
                value = 0.5 * (qp * d * m - c * abs(qp)) + 0.5 * (qm * (-d) * m - c * abs(qm))
                regs.append(Q * max(d * m - c, 0.0) - value)
            reg = max(regs)
            best = reg if best is None else min(best, reg)
    return best


def check_minimax():
    for Q, c, d, a, b in [(1, 1, 0.2, 1, 10), (2, 0.5, 0.3, 1, 4), (1, 2, 0.5, 2, 9), (3, 1, 0.1, 5, 40)]:
        A, B = c - d * a, d * b - c
        closed = Q * A * B / (A + B)
        brute = minimax_brute(Q, c, d, a, b)
        ok = abs(brute - closed) < 0.01 * max(closed, 1.0)
        print(f"minimax regret Q={Q} c={c} d={d} a={a} b={b}: brute force {brute:.4f}, closed form {closed:.4f}: {'ok' if ok else 'FAIL'}")
        if not ok:
            FAILS.append(f"minimax {Q, c, d, a, b}")
    assert F(4, 5) * F(1) / (F(4, 5) + 1) == F(4, 9)


def check_pools():
    stopped = F(1, 2) * 0 + F(1, 2) * (F(1) + F(2)) / 2                # stop after a first draw of 0; otherwise two draws {0,2}
    ok = stopped == F(3, 4) < 1
    print(f"stopped pool mean {stopped} (population mean 1): {'ok' if ok else 'FAIL'}")
    # m in {0, 4}, C = 0, q = 1, tau = 1, realized m = 0 -> U = max(0, tau) = 1 < mu = 2
    u = max(F(0), F(1))
    ok2 = u == 1 < 2
    print(f"censored single draw U = {u} below mean 2: {'ok' if ok2 else 'FAIL'}")
    # trade only if the one-draw pool equals 2; current favourable probability 3/4 -> E[sigma] = 1/2; PMR = 1 * 1 = 1, booked = 1/2 * 1
    pmr, booked = F(1, 2) * 0 + 1 * F(2) / 2 * 1, F(1, 2) * 1
    ok3 = pmr == 1 and booked == F(1, 2)
    print(f"selection by a one-draw pool: PMR given a trade {pmr}, booked {booked}: {'ok' if ok3 else 'FAIL'}")
    for flag, name in ((ok, "stopped pool"), (ok2, "censored pool"), (ok3, "selection")):
        if not flag:
            FAILS.append(name)


if __name__ == "__main__":
    check_identified_sets()
    check_spr_example()
    check_dominance()
    check_minimax()
    check_pools()
    print("all checks pass" if not FAILS else f"FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)
