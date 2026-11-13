"""Exact check of the range-by-path statement (PROTOCOL_HORIZON.md), in rational arithmetic.

Floored program: accounts with balances e_i >= 0, each making a fixed sequence of writes b <- max(b + t v_ij + d_ij, 0), with the
gross writes of each account of one sign (coherent). Reward phi(t) = sum of terminal balances - E (no unfloored channel here).
Path: the same program applied k times with t = 1 each, the terminal balances of one application being the starting balances of
the next; the path reward is the sum of the k step rewards, which telescopes to terminal balances - E.

Claims checked:
  (1) with no fixed writes (d = 0): path reward over k unit steps == phi(k), for every k;
  (2) with fixed writes: |path reward - phi(k)| <= (k - 1) * A, A = sum |d_ij|;
  (3) same-sign steps of unequal size t_1..t_k with no fixed writes: path reward == phi(sum t_j).
"""
from __future__ import annotations

import random
from fractions import Fraction as F

random.seed(20261107)


def program():
    accounts = []
    for _ in range(random.randint(1, 3)):
        e = F(random.randint(0, 20))
        sign = random.choice([-1, 1])
        writes = [(sign * F(random.randint(1, 9)), F(random.randint(-4, 4))) for _ in range(random.randint(1, 3))]
        accounts.append((e, writes))
    return accounts


def run(accounts, t, use_fixed=True):
    """One application at scale t; returns terminal balances."""
    out = []
    for e, writes in accounts:
        b = e
        for v, d in writes:
            b = max(b + t * v + (d if use_fixed else 0), F(0))
        out.append(b)
    return out


def phi(accounts, t, use_fixed=True):
    return sum(run(accounts, t, use_fixed)) - sum(e for e, _ in accounts)


def path(accounts, ts, use_fixed=True):
    bal = [e for e, _ in accounts]
    for t in ts:
        bal = run([(b, w) for b, (_, w) in zip(bal, accounts)], t, use_fixed)
    return sum(bal) - sum(e for e, _ in accounts)


def main() -> None:
    n1 = n2 = n3 = 0
    for _ in range(20000):
        acc = program()
        k = random.randint(1, 8)
        # (1) unit steps, no fixed writes
        assert path(acc, [F(1)] * k, False) == phi(acc, F(k), False)
        assert path(acc, [F(-1)] * k, False) == phi(acc, F(-k), False)
        n1 += 2
        # (2) unit steps with fixed writes: bounded difference
        A = sum(abs(d) for _, w in acc for _, d in w)
        for sgn in (1, -1):
            diff = abs(path(acc, [F(sgn)] * k, True) - phi(acc, F(sgn * k), True))
            assert diff <= (k - 1) * A, (acc, k, diff, A)
            n2 += 1
        # (3) unequal same-sign steps, no fixed writes
        ts = [F(random.randint(1, 5), random.randint(1, 3)) for _ in range(k)]
        sgn = random.choice([1, -1])
        assert path(acc, [sgn * t for t in ts], False) == phi(acc, sgn * sum(ts), False)
        n3 += 1
    print(f"checks passed: (1) {n1}, (2) {n2}, (3) {n3}")


if __name__ == "__main__":
    main()
