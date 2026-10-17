"""Exact, post-hoc verification of the 10 October 2026 accounting revision.

No training, network access or mutation of registered outputs. Enumerates finite
fair-sign markets using rational arithmetic, including adaptive positions.
"""
from fractions import Fraction as Q
from itertools import product
from math import sqrt, exp, log


def rollout(signs, policy, initial, fee, gamma):
    equity = ledger = Q(initial)
    regulator = costs = variation = Q(0)
    reported_discounted = booked_discounted = subsidy_discounted = Q(0)
    weighted_cost = Q(0)
    for t, sign in enumerate(signs):
        position = Q(policy(t, equity, signs[:t]))
        magnitude = Q(t % 3 + 1)
        cost = Q(fee) * abs(position)
        pnl = position * sign * magnitude - cost
        injection = max(-equity - pnl, Q(0))
        reported_reward = pnl + injection
        equity += reported_reward
        ledger += pnl
        regulator += injection
        costs += cost
        variation += position ** 2 * magnitude ** 2
        weight = Q(gamma) ** t
        reported_discounted += weight * reported_reward
        booked_discounted += weight * pnl
        subsidy_discounted += weight * injection
        weighted_cost += weight * cost
        assert equity == ledger + regulator
    assert reported_discounted - booked_discounted == subsidy_discounted
    return dict(equity=equity, ledger=ledger, regulator=regulator,
                costs=costs, variation=variation, booked=booked_discounted,
                weighted_cost=weighted_cost)


def mean(rows, key):
    return sum((row[key] for row in rows), Q(0)) / len(rows)


def main():
    policies = {
        'flat': lambda t, e, past: 0,
        'fixed': lambda t, e, past: 1,
        'low_equity_barrier': lambda t, e, past: 2 if e < 2 else 0,
        'adaptive_direction': lambda t, e, past: (past[-1] if past else 1),
        'adaptive_size': lambda t, e, past: 1 if e > 1 else 3,
    }
    cases = 0
    for initial, fee, gamma in product((Q(0), Q(1), Q(3)),
                                      (Q(0), Q(1, 5), Q(2)),
                                      (Q(1), Q(99, 100))):
        for name, policy in policies.items():
            rows = [rollout(signs, policy, initial, fee, gamma)
                    for signs in product((-1, 1), repeat=6)]
            assert mean(rows, 'booked') == -mean(rows, 'weighted_cost'), name
            cap = sqrt(float(mean(rows, 'variation')))
            assert float(mean(rows, 'equity') - initial) <= cap + 1e-12
            assert float(mean(rows, 'regulator')) <= float(mean(rows, 'costs')) + cap + 1e-12
            cases += 1
    print(f'PASS: pathwise reflection and discounted reward identities, and exact no-edge '
          f'expectations, in {cases} adaptive-policy configurations (64 paths each).')

    for alpha in (Q(0), Q(1, 2), Q(9, 10), Q(99, 100), Q(999, 1000)):
        a = Q(1)
        fee = (1-alpha) / (2*(1+alpha))
        rewards = []
        for sign in (-1, 1):
            pnl = sign*a-fee
            injection = max(-pnl, Q(0))
            rewards.append(pnl+(1-alpha)*injection)
        expected = sum(rewards)/2
        formula = ((1-alpha)*a-(1+alpha)*fee)/2
        assert expected == formula and expected > 0
        assert -fee < 0
    print('PASS: every tested partial charge admits positive proxy reward and negative booked profit.')

    # Fees need to be included in the inflation bound. Four periods, x = +/-1 - 2.
    gaps = []
    for signs in product((-1, 1), repeat=4):
        e = l = Q(0)
        for sign in signs:
            x = Q(sign-2)
            l += x
            e = max(e+x, Q(0))
        gaps.append(e-l)
        assert e == 0
    assert sum(gaps)/len(gaps) == 8 > sqrt(4)
    print('PASS: cost counterexample: expected reported profit 0, inflation 8, gross-variance cap 2.')

    alpha = Q(1)
    fee = Q(1, 5)
    booked = -fee
    subsidy = (1+fee)/2
    reported = booked+subsidy
    assert booked < 0 < reported
    assert booked + (1-alpha)*subsidy == booked
    print('PASS: a one-period risky position loses in the ledger but outranks flat under the floor; '
          'full charging restores its booked score.')
    # Check the concentration argument against the entire distribution of adaptive paths.
    b_squared = sum((3 * (t % 3 + 1)) ** 2 for t in range(6))
    for policy in policies.values():
        rows = [rollout(signs, policy, Q(1), Q(1, 5), Q(1))
                for signs in product((-1, 1), repeat=6)]
        gross = [float(r['ledger'] - 1 + r['costs']) for r in rows]
        assert all(abs(float(r['equity'] - 1 - r['regulator'] + r['costs']) - g) < 1e-12
                   for r, g in zip(rows, gross))
        for theta in (-0.1, -0.01, 0.01, 0.1):
            mgf = sum(exp(theta * g) for g in gross) / len(gross)
            assert log(mgf) <= theta ** 2 * b_squared / 2 + 1e-12
        for delta in (0.5, 0.1, 0.05, 0.01):
            cutoff = sqrt(2 * b_squared * log(1 / delta))
            assert sum(g > cutoff for g in gross) / len(gross) <= delta
    print('PASS: adaptive-path exponential-moment and reconciled placebo tail bounds.')
    print('ALL ACCOUNTING REVISION CHECKS PASSED (analytical verification).')


if __name__ == '__main__':
    main()
