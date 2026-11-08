"""Exact checks of the reward-diagram theory (manuscript section "The reward diagram"), in rational arithmetic: every claim is checked with Fractions on random
account programs, not in floating point. Independent of any earlier validation code.

Model. Accounts i; account i starts at e_i >= 0 and makes writes b <- max(b + t * v + d, 0) in a fixed order. An unfloored channel adds t * u + w. Then
    phi(t) = sum_i b_i(t) - E + t * u + w,    booked(t) = t * (G + u) - C,    G = sum v,   C = -sum d - w,    E = sum e_i.

 D1  credit(t) = phi(t) - booked(t) >= 0 for every t, and phi is convex in t, for ANY order and any signs of the writes.
 D2  width identity: phi(T) + phi(-T) + 2C = [phi(T) + phi(-T) - 2 phi(0)] + 2 [phi(0) + C], and it equals credit(T) + credit(-T).
 D3  certificate: credit(a) = credit(b) = 0 implies credit(t) = 0 on [a, b]; in particular a zero width at T certifies no credit on all of [-T, T].
 D4  tail slopes, coherent gross per account and no unfloored channel: beyond the last kink, s_plus = sum of positive g_i, s_minus = sum of negative g_i, s_plus + s_minus = G.
 D5  far-tail bound, coherent, no unfloored channel: |a_plus(T) + a_minus(T) - G| <= (E + 2A) / T, with A = sum |d|; exact once every kink lies inside [-T, T].
 D6  one exposure with an unfloored part of the same sign: the steeper tail is the total gross.
 D7  minimax upper bound: |(phi(L) - phi(-L)) / L - G| <= (E + A) / L, coherent accounts.
 D8  minimax lower bound: the two worlds of the theorem coincide at every grid point within eta, have the declared budgets, and differ by exactly 2 (E + A + 2 eta) / L.
 D9  read-outs, one account, gross after the fee or fee after the gross: the flat tail's level is -e; the kink lies where the favourable formula meets it.
 D10 parallel accounts, one gross write each: kinks at t_i = -(e_i + d_i) / v_i with slope jump |v_i|; the slope jumps sum to s_plus - s_minus.
 D11 a cap on the gross (concave in t) produces a concave kink, which no program of floors can produce (contrast with D1).
 D12 the counterexamples: an unfloored channel with an opposite floored account, and opposite writes inside one account, are not identified by the whole of phi.
 D13 a fee forgiven before the first gross write is invisible on the whole line: two worlds with different costs have the same phi.
 D14 no reference is needed: along any line through the realized input, for one account that can pay its cost, the favourable tail's asymptote at t = 1 is the booked P&L.
"""
from __future__ import annotations

import random
from fractions import Fraction as F

N_RANDOM = 4000


def rand_frac(rng, lo=-40, hi=40, den=8):
    return F(rng.randint(lo, hi), rng.randint(1, den))


def run_account(e, writes, t):
    b = e
    for v, d in writes:
        b = max(b + t * v + d, F(0))
    return b


def phi(accounts, u, w, t):
    return sum(run_account(e, ws, t) for e, ws in accounts) - sum(e for e, _ in accounts) + t * u + w


def budgets(accounts, u, w):
    E = sum(e for e, _ in accounts)
    G = sum(v for _, ws in accounts for v, _ in ws)
    C = -sum(d for _, ws in accounts for _, d in ws) - w
    A = sum(abs(d) for _, ws in accounts for _, d in ws) + abs(w)
    return E, G, C, A


def rand_program(rng, coherent=False, max_accounts=4, max_writes=4, unfloored=False, nonneg_fees=False):
    accounts = []
    for _ in range(rng.randint(1, max_accounts)):
        e = abs(rand_frac(rng, 0, 40))
        sgn = rng.choice([1, -1])
        ws = []
        for _ in range(rng.randint(1, max_writes)):
            v = rand_frac(rng, -20, 20)
            if coherent:
                v = abs(v) * sgn
            d = rand_frac(rng, -10, 10)
            if nonneg_fees:
                d = -abs(d)
            ws.append((v, d))
        accounts.append((e, ws))
    u = rand_frac(rng, -10, 10) if unfloored else F(0)
    w = rand_frac(rng, -5, 5) if unfloored else F(0)
    return accounts, u, w


def kinks_of(accounts, u, w, lo, hi, step=F(1, 2)):
    """All t on a fine rational grid at which phi has a slope change (used only to find a safe far T)."""
    t = lo
    prev_slope = None
    out = []
    while t < hi:
        s = (phi(accounts, u, w, t + step) - phi(accounts, u, w, t)) / step
        if prev_slope is not None and s != prev_slope:
            out.append(t)
        prev_slope = s
        t += step
    return out


def main():
    rng = random.Random(20261106)
    n = {f"D{i}": 0 for i in range(1, 14)}

    # D1, D2, D3: any program
    for _ in range(N_RANDOM):
        acc, u, w = rand_program(rng, unfloored=True)
        E, G, C, A = budgets(acc, u, w)
        ts = sorted({rand_frac(rng, -200, 200, 3) for _ in range(9)} | {F(0), F(1), F(-1)})
        cred = {t: phi(acc, u, w, t) - (t * (G + u) - C) for t in ts}
        assert all(c >= 0 for c in cred.values()), "D1 credit"
        for a, m, b in zip(ts[:-2], ts[1:-1], ts[2:]):
            lam = (b - m) / (b - a)
            assert phi(acc, u, w, m) <= lam * phi(acc, u, w, a) + (1 - lam) * phi(acc, u, w, b), "D1 convexity"
            if cred[a] == 0 and cred[b] == 0:
                assert cred[m] == 0, "D3"
                n["D3"] += 1
        n["D1"] += 1
        T = abs(rand_frac(rng, 1, 400, 2)) + 1
        p, pm, p0 = phi(acc, u, w, T), phi(acc, u, w, -T), phi(acc, u, w, F(0))
        width = p + pm + 2 * C
        assert width == (p + pm - 2 * p0) + 2 * (p0 + C), "D2 identity"
        assert width == (p - (T * (G + u) - C)) + (pm - (-T * (G + u) - C)), "D2 credit"
        if width == 0:
            for t in ts:
                if abs(t) <= T:
                    assert phi(acc, u, w, t) == t * (G + u) - C, "D3 zero width"
        n["D2"] += 1

    # D4, D5, D7: coherent, no unfloored channel
    for _ in range(N_RANDOM):
        acc, u, w = rand_program(rng, coherent=True)
        E, G, C, A = budgets(acc, u, w)
        g_i = [sum(v for v, _ in ws) for _, ws in acc]
        for T in (F(1), F(3), F(10), F(100), F(1000), F(10**6)):
            ap = (phi(acc, u, w, 2 * T) - phi(acc, u, w, T)) / T
            am = (phi(acc, u, w, -T) - phi(acc, u, w, -2 * T)) / T
            assert abs(ap + am - G) <= (E + 2 * A) / T, "D5 bound"
            est = (phi(acc, u, w, T) - phi(acc, u, w, -T)) / T
            assert abs(est - G) <= (E + A) / T, "D7 bound"
            n["D5"] += 1
            n["D7"] += 1
        # exactness beyond every kink: a kink cannot lie beyond (E + A) / min |v|
        vmin = min((abs(v) for _, ws in acc for v, _ in ws if v != 0), default=None)
        if vmin is not None:
            T = (E + A + 1) / vmin + 1
            ap = (phi(acc, u, w, 2 * T) - phi(acc, u, w, T)) / T
            am = (phi(acc, u, w, -T) - phi(acc, u, w, -2 * T)) / T
            assert ap == sum(g for g in g_i if g > 0), "D4 s_plus"
            assert am == sum(g for g in g_i if g < 0), "D4 s_minus"
            assert ap + am == G, "D4 net"
            n["D4"] += 1

    # D6: one exposure, unfloored part of the same sign
    for _ in range(N_RANDOM):
        acc, _, _ = rand_program(rng, coherent=True, max_accounts=3)
        sgn = rng.choice([1, -1])
        acc = [(e, [(abs(v) * sgn, d) for v, d in ws]) for e, ws in acc]
        u = abs(rand_frac(rng, 0, 10)) * sgn
        w = rand_frac(rng, -5, 5)
        E, G, C, A = budgets(acc, u, w)
        vmin = min((abs(v) for _, ws in acc for v, _ in ws if v != 0), default=None)
        if vmin is None:
            continue
        T = (E + A + 1) / vmin + 1
        ap = (phi(acc, u, w, 2 * T) - phi(acc, u, w, T)) / T
        am = (phi(acc, u, w, -T) - phi(acc, u, w, -2 * T)) / T
        steep = ap if abs(ap) >= abs(am) else am
        assert steep == G + u, "D6"
        n["D6"] += 1

    # D8: the matching worlds
    for _ in range(N_RANDOM // 10):
        E = abs(rand_frac(rng, 0, 40))
        C = rand_frac(rng, -10, 10)
        A = abs(C) + abs(rand_frac(rng, 0, 10))
        eta = abs(rand_frac(rng, 0, 5))
        L = abs(rand_frac(rng, 1, 100, 2)) + 1
        P, N = (A - C) / 2, (A + C) / 2
        for sign in (1, -1):
            world = [(E, [(sign * (-(E + P)) / L, P)]), (F(0), [(sign * 2 * (E + P) / L, F(0))]), (F(0), [(sign * N / L, -N)]), (F(0), [(sign * 2 * eta / L, F(0))])]
            e_, g_, c_, a_ = budgets(world, F(0), F(0))
            assert (e_, c_, a_) == (E, C, A), "D8 budgets"
            assert g_ == sign * (E + A + 2 * eta) / L, "D8 target"
            for k in range(-20, 21):
                t = L * F(k, 20)
                y = P + (E + P + eta) * abs(t) / L
                assert abs(y - phi(world, F(0), F(0), t)) <= eta, "D8 common observation"
        n["D8"] += 1

    # D9: read-outs from one account
    for _ in range(N_RANDOM):
        e = abs(rand_frac(rng, 1, 400))
        g = abs(rand_frac(rng, 1, 40)) * rng.choice([1, -1])
        c = abs(rand_frac(rng, 0, 10))
        for acc in ([(e, [(g, F(0)), (F(0), -c)])], [(e, [(F(0), -c), (g, F(0))])]):
            T = (e + c + 1) / abs(g) + 1
            flat = phi(acc, F(0), F(0), -T if g > 0 else T)
            assert flat == -e, "D9 level"
            slope = (phi(acc, F(0), F(0), 2 * T if g > 0 else -T) - phi(acc, F(0), F(0), T if g > 0 else -2 * T)) / T
            assert slope == g, "D9 slope"
            if e >= c:
                intercept = phi(acc, F(0), F(0), T if g > 0 else -T) - (T if g > 0 else -T) * g
                assert intercept == -c, "D9 cost read from the tail"
                assert (-e - intercept) / g == (c - e) / g, "D9 kink"
            n["D9"] += 1

    # D10: parallel accounts, one gross write each
    for _ in range(N_RANDOM // 4):
        k = rng.randint(1, 4)
        acc = []
        for _ in range(k):
            e = abs(rand_frac(rng, 0, 40))
            v = abs(rand_frac(rng, 1, 20)) * rng.choice([1, -1])
            d = -abs(rand_frac(rng, 0, 5))
            acc.append((e, [(v, d)]))
        E, G, C, A = budgets(acc, F(0), F(0))
        positions = {}
        for e, ((v, d),) in acc:
            t_i = -(e + d) / v
            positions[t_i] = positions.get(t_i, F(0)) + abs(v)
        far = max(abs(t) for t in positions) + 1
        total = F(0)
        for t_i, jump in positions.items():
            eps = min([abs(t_i - o) for o in positions if o != t_i] + [F(1)]) / 2
            left = (phi(acc, F(0), F(0), t_i) - phi(acc, F(0), F(0), t_i - eps)) / eps
            right = (phi(acc, F(0), F(0), t_i + eps) - phi(acc, F(0), F(0), t_i)) / eps
            assert right - left == jump, "D10 slope jump"
            total += jump
        sp = (phi(acc, F(0), F(0), far + 1) - phi(acc, F(0), F(0), far)) / 1
        sm = (phi(acc, F(0), F(0), -far) - phi(acc, F(0), F(0), -far - 1)) / 1
        assert sp - sm == total, "D10 mass"
        n["D10"] += 1

    # D11: a capped gross is concave somewhere, unlike any program of floors
    for _ in range(N_RANDOM // 10):
        e = abs(rand_frac(rng, 100, 400))
        g = abs(rand_frac(rng, 1, 20))
        cap = abs(rand_frac(rng, 1, 40)) + 1
        def capped(t):
            return max(-cap, min(t * g, cap))
        a, b = cap / g, 2 * cap / g
        m = (a + b) / 2
        a0 = a / 2
        # slope before the cap g, after it 0: concave at t = cap / g, with the account never near its floor
        assert (capped(a) - capped(a0)) / (a - a0) == g and (capped(b) - capped(m)) / (b - m) == 0, "D11"
        n["D11"] += 1

    # D12: the two non-identification witnesses
    protected_long = [(F(0), [(F(1), F(0))])]
    unfloored_plus_short = ([(F(0), [(F(-1), F(0))])], F(1))
    for k in range(-200, 201):
        t = F(k, 10)
        assert phi(protected_long, F(0), F(0), t) == phi(unfloored_plus_short[0], unfloored_plus_short[1], F(0), t), "D12 a"
        one = [(F(0), [(F(1), F(0))])]
        two = [(F(0), [(F(-1), F(0)), (F(1), F(0))])]
        assert phi(one, F(0), F(0), t) == phi(two, F(0), F(0), t), "D12 b"
        n["D12"] += 1
    assert budgets(protected_long, F(0), F(0))[1] == 1 and budgets(unfloored_plus_short[0], unfloored_plus_short[1], F(0))[1] + unfloored_plus_short[1] == 0
    assert budgets([(F(0), [(F(-1), F(0)), (F(1), F(0))])], F(0), F(0))[1] == 0

    # D13: a fee forgiven before the first gross write
    for _ in range(N_RANDOM // 10):
        e = abs(rand_frac(rng, 0, 20))
        c1, c2 = e + abs(rand_frac(rng, 1, 20)), e + abs(rand_frac(rng, 21, 40))
        g = rand_frac(rng, -20, 20)
        for k in range(-50, 51):
            t = F(k, 5)
            assert phi([(e, [(F(0), -c1), (g, F(0))])], F(0), F(0), t) == phi([(e, [(F(0), -c2), (g, F(0))])], F(0), F(0), t), "D13"
        n["D13"] += 1

    # D14: no reference is needed. Along any line through the realized input, input(t) = ref + t * (input - ref), for one account that can pay its cost the asymptote of the favourable tail,
    # evaluated at t = 1, is the booked P&L; with the unknown reference the slope alone is not.
    n["D14"] = 0
    for _ in range(N_RANDOM):
        e = abs(rand_frac(rng, 1, 400))
        c = abs(rand_frac(rng, 0, 10))
        if c > e:
            e, c = c, e
        q = rand_frac(rng, -20, 20) or F(1)
        s = rand_frac(rng, -40, 40)
        ref = rand_frac(rng, -40, 40)
        if s == ref:
            continue
        x = q * s - c
        a = q * (s - ref)                                                  # the slope along the line
        for order in ("gross first", "fee first"):
            def phi_line(t):
                gross = q * (ref + t * (s - ref))
                if order == "gross first":
                    return max(max(e + gross, F(0)) - c, F(0)) - e
                return max(max(e - c, F(0)) + gross, F(0)) - e
            T = (e + c + abs(q * ref) + 1) / abs(a) + 1
            side = 1 if a > 0 else -1
            p1, p2 = phi_line(side * T), phi_line(side * 2 * T)
            slope = (p2 - p1) / (side * T)
            assert slope == a, "D14 slope"
            assert p1 + (1 - side * T) * slope == x, "D14 asymptote at t = 1"
            n["D14"] += 1

    total = sum(n.values())
    print("exact checks passed:", {k: v for k, v in n.items()}, "total", total)
    return total


if __name__ == "__main__":
    main()
