"""Exact toy checks of a cash-inflow lower endpoint; no native imports/data.

All arithmetic is Fraction. These are finite consistency checks, not native
evidence, new confidence limits, or a replacement for the accompanying proof.
Output is exclusive in this new unfrozen validation directory.
"""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import hashlib
import json


def positive(x):
    return max(x, F(0))


def weighted(values, weights):
    return sum((x * w for x, w in zip(values, weights)), F(0))


def parts(cash, weights):
    lhs = weighted([cash[t + 1] - cash[t] for t in range(len(weights))], weights)
    rhs = weights[-1] * cash[-1] - weights[0] * cash[0]
    rhs += sum(((weights[t - 1] - weights[t]) * cash[t]
                for t in range(1, len(weights))), F(0))
    assert lhs == rhs
    return lhs


def rational(x):
    return str(x.numerator) + "/" + str(x.denominator)


def run():
    counts = dict(nonnegative_step_paths=0, capped_step_checks=0,
                  event_credit_paths=0, event_paths_with_negative_actual_cash=0,
                  ordered_fair_floor_laws=0, partial_charge_counterexamples=0)
    for horizon in (1, 2, 3):
        for gamma in (F(1), F(1, 2)):
            weights = [gamma ** t for t in range(horizon)]
            for cash in product((F(0), F(1, 2), F(2)), repeat=horizon + 1):
                assert parts(cash, weights) >= -weights[0] * cash[0]
                for excluded in product((F(-1), F(0), F(1)), repeat=horizon):
                    financial = [cash[t + 1] - cash[t] - excluded[t] for t in range(horizon)]
                    y = weighted(financial, weights)
                    k = weights[0] * cash[0] + weighted([positive(d) for d in excluded], weights)
                    assert y >= -k
                    for cap in (F(0), F(1)):
                        capped = min(y, cap)
                        assert -k <= capped <= cap and capped <= y
                        counts["capped_step_checks"] += 1
                    counts["nonnegative_step_paths"] += 1

    # Eventwise positive-credit version: actual cash may dip below zero at
    # nonfinancial events, but every financial POST-event cash is nonnegative.
    for horizon in (1, 2, 3):
        for gamma in (F(1), F(1, 2)):
            weights = [gamma ** t for t in range(horizon)]
            for initial in (F(0), F(1)):
                for kinds in product(("financial", "nonfinancial"), repeat=horizon):
                    for increments in product((F(-1), F(0), F(1)), repeat=horizon):
                        actual, virtual = [initial], [initial]
                        admissible, credit, y = True, F(0), F(0)
                        for kind, change, weight in zip(kinds, increments, weights):
                            actual.append(actual[-1] + change)
                            if kind == "financial":
                                if actual[-1] < 0:
                                    admissible = False
                                    break
                                virtual.append(virtual[-1] + change)
                                y += weight * change
                            else:
                                virtual.append(virtual[-1] + positive(change))
                                credit += weight * positive(change)
                            assert virtual[-1] >= 0
                            assert virtual[-1] >= actual[-1]
                        if not admissible:
                            continue
                        assert parts(virtual, weights) == y + credit
                        assert y >= -initial - credit
                        counts["event_credit_paths"] += 1
                        counts["event_paths_with_negative_actual_cash"] += any(e < 0 for e in actual)

    # Native ordering may change forgiveness, but alpha0 actual cash increments
    # telescope in either fee/maturity order. Booked fair gross has mean zero.
    for initial, magnitude, cost, order in product(
            (F(0), F(1, 2), F(2)), (F(1, 2), F(2)),
            (F(0), F(1, 4), F(7)), ("fee_first", "maturity_first")):
        ys, books = [], []
        for sign in (F(-1), F(1)):
            gross = sign * magnitude
            events = [-cost, gross] if order == "fee_first" else [gross, -cost]
            cash, total, booked = initial, F(0), gross - cost
            for change in events:
                injection = positive(-cash - change)
                increment = change + injection
                cash += increment
                total += increment
                assert cash >= 0
            assert total == cash - initial and total >= -initial
            assert -initial <= min(total, F(1)) <= F(1)
            ys.append(total)
            books.append(booked)
        assert sum(books, F(0)) / 2 == -cost
        assert sum((min(y, F(1)) for y in ys), F(0)) / 2 <= sum(ys, F(0)) / 2
        counts["ordered_fair_floor_laws"] += 1

    counterexamples = []
    # Same-step net excluded flow is zero, financial pre/post cash are 1/0,
    # but the final excluded debit leaves negative boundary cash.
    actual = (F(0), F(1), F(0), F(-1))
    financial, net_excluded = F(-1), F(0)
    assert actual[1] >= 0 and actual[2] >= 0 and actual[-1] < 0
    assert financial < -actual[0] - positive(net_excluded)
    assert financial >= -actual[0] - F(1)  # all-positive-event credit budget works
    counterexamples.append(dict(name="financial_floor_only_does_not_justify_net_step_budget",
        actual_cash=list(map(rational, actual)), financial=rational(financial),
        net_step_excluded=rational(net_excluded), positive_event_credit=rational(F(1))))

    assert parts((F(0), F(1), F(0)), (F(1), F(2))) == -1
    counterexamples.append(dict(name="increasing_weights", weighted_financial="-1/1", proposed_lower="0/1"))
    assert parts((F(0), F(-1), F(0)), (F(1), F(1, 2))) == F(-1, 2)
    counterexamples.append(dict(name="terminal_nonnegativity_only", weighted_financial="-1/2", proposed_lower="0/1"))
    assert F(-2) < -F(1)  # nonnegative financial PRE cash alone, no post floor
    counterexamples.append(dict(name="financial_pre_only_without_post_floor", pre_cash="1/1", post_cash="-1/1", financial="-2/1"))
    assert F(-2) < 0 and F(-2) >= -F(2)
    counterexamples.append(dict(name="omitted_positive_excluded_transfer", true_credit_budget="2/1", financial="-2/1", invalid_lower="0/1"))

    for alpha, cost in product((F(1, 4), F(1, 2), F(1)), (F(1), F(2), F(100))):
        # G=0, e0=0, cost C and native injection I=C leave actual cash at0.
        raw = -cost + cost
        charged = -cost + (1 - alpha) * cost
        assert raw == 0 and charged == -alpha * cost and charged < 0
        counts["partial_charge_counterexamples"] += 1
    counterexamples.append(dict(name="partial_charge_same_dynamics", gross="0/1", initial_cash="0/1",
        actual_cash_increment="0/1", excluded_credit="0/1", general_charged_return="-alpha*C; unbounded below for alpha>0"))
    assert counts["event_paths_with_negative_actual_cash"] > 0
    return {"status": "PASS EXACT TOY CHECKS", "arithmetic": "fractions.Fraction only",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "counts": counts, "counterexamples": counterexamples,
            "native_queries": 0, "endpoint_reads": 0, "new_confidence_limits": 0,
            "interpretation": "Finite consistency checks of the accompanying cash-account proof, not native experiment evidence or generic mathematical novelty"}


if __name__ == "__main__":
    result = run()
    output = Path(__file__).with_suffix(".json")
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "counts": result["counts"], "output": str(output)}))
