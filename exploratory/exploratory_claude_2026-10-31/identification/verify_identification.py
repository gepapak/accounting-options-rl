"""EXPLORATORY (31 Oct 2026): exact checks of the censored-reward identification theory for sign twins.

Model. One step: gross magnitude g >= 0 (hidden), known own cost c >= 0, realized gross sign sigma in {+1, -1} (known), booked P&L
b = sigma*g - c. Every accounting rule of the ladder reports  r = max(b, tau)  with a hidden threshold tau (tau = -equity under the
zero floor and limited liability; tau = the carried deficit at a terminal or detection step; tau = -infinity at steps without credit).
The credit is  max(tau - b, 0) >= 0.  Twin query: the same step with the gross sign reversed, reward r' = max(-sigma*g - c, tau).

Checks (all in exact rational arithmetic):
  1. favourable branch uncensored (tau <= g - c): g_U = max(r, r') + c equals g exactly, so b_hat = sigma*g_U - c equals b exactly;
  2. both branches censored: r = r' = tau and g lies in [0, tau + c]; g_U is the sharp upper end, SPR's g_L = |r - r'|/2 = 0 the lower end;
  3. g_L = |r - r'|/2 <= g always, with equality iff the unfavourable branch is uncensored (tau <= -g - c);
  4. the identified set of g given (r, r') is a point iff r != r' (or r = r' = -c, g = 0);
  5. the worst-case regret lower bound 4/9 of a decision rule that sees only (r, r') when a deficit of 10 is carried (LP check).

    python verify_identification.py
"""
import random
from fractions import Fraction as F

import numpy as np
from scipy.optimize import linprog


def rewards(g, c, sigma, tau):
    b = sigma * g - c
    b_twin = -sigma * g - c
    r = max(b, tau)
    r_twin = max(b_twin, tau)
    return b, r, r_twin


def g_upper(r, r_twin, c):
    return max(r, r_twin) + c


def g_lower(r, r_twin):
    return abs(r - r_twin) / 2


def rnd(rng, lo, hi, den=8):
    return F(rng.randint(lo * den, hi * den), den)


def run_random_checks(n=200_000, seed=0):
    rng = random.Random(seed)
    stats = dict(uncensored=0, both=0, lose_only=0, exact_upper=0, lower_tight=0, bad=0, ident_point=0)
    for _ in range(n):
        g = rnd(rng, 0, 12)
        c = rnd(rng, 0, 4)
        sigma = rng.choice((1, -1))
        tau = F(-10 ** 9) if rng.random() < 0.15 else rnd(rng, -6, 14)     # 15%: no credit at all
        b, r, rt = rewards(g, c, sigma, tau)
        gu, gl = g_upper(r, rt, c), g_lower(r, rt)
        fav_unc = tau <= g - c                                              # the favourable branch is uncensored
        unfav_unc = tau <= -g - c
        assert gl <= g, ("lower bound violated", g, c, tau)
        assert gu >= g, ("upper bound violated", g, c, tau)
        if fav_unc:
            stats["uncensored"] += 1
            assert gu == g and sigma * gu - c == b, ("TMR not exact", g, c, sigma, tau)
            stats["exact_upper"] += 1
        else:
            stats["both"] += 1
            assert r == rt == tau and 0 <= g <= tau + c and gu == tau + c, ("both-censored structure", g, c, tau)
        assert (gl == g) == unfav_unc or g == 0, ("SPR tightness iff unfavourable branch uncensored", g, c, tau)
        if unfav_unc:
            stats["lower_tight"] += 1
        elif fav_unc:
            stats["lose_only"] += 1
        # identified set is a point iff r != r' (or r = r' = -c)
        point = (r != rt) or (r == -c)
        if point:
            stats["ident_point"] += 1
            assert gu == g or (r == rt == -c and g == 0) or r != rt and gu == g, ("point identification", g, c, tau)
    return stats


def regret_lower_bound():
    """Carried deficit D = 10 (tau = 10): every query returns 10 in both worlds g in {1, 10}. Signal z = +-1 with E[eps|z] = 0.2 z,
    own cost |q|, q in [-1, 1]. A policy is a pair of distributions over q given z; by symmetry optimize x = E[q z], y = E|q| >= |x|.
    Regret(world g) = best_g - (0.2*g*x - y); best_1 = 0 (flat), best_10 = 1 (q = z). Minimize the maximum regret (LP in x, y, t)."""
    # variables [x, y, t]; minimize t  s.t.  regret_1 = y - 0.2x <= t ; regret_10 = 1 - 2x + y <= t ; y >= x ; y >= -x ; |x| <= 1, 0 <= y <= 1
    res = linprog(c=[0, 0, 1],
                  A_ub=[[-0.2, 1, -1], [-2, 1, -1], [1, -1, 0], [-1, -1, 0]],
                  b_ub=[0, -1, 0, 0],
                  bounds=[(-1, 1), (0, 1), (None, None)])
    x, y, t = res.x
    # a flat-query check that the deficit is observable but g is not: every (action, sign) reward equals max(b, 10) = 10
    seen = {max(sg * g * abs(q) - abs(q), 10) for g in (1, 10) for q in np.linspace(-1, 1, 21) for sg in (1, -1)}
    return t, (x, y), seen


if __name__ == "__main__":
    s = run_random_checks()
    print("random exact checks passed (200,000 draws):", s)
    t, (x, y), seen = regret_lower_bound()
    print(f"worst-case regret LP: {t:.6f} (4/9 = {4 / 9:.6f}) at x = E[qz] = {x:.4f}, y = E|q| = {y:.4f}")
    print("distinct rewards over all actions, signs and both worlds with a carried deficit of 10:", seen)
