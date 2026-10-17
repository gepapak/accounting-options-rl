"""Independent exact checks of cash controls; no native/model/endpoint access."""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import hashlib
import json


def check():
    laws = paths = identities = caps_checked = 0
    for e0, g0, g1, c0, c1, credit, old, beta, budget in product(
            (0, 1, 2), (1, 3), (2, 4), (0, 1), (0, 1), (0, 2),
            (False, True), (F(0), F(1, 2), F(1), F(2)), (0, 1, 3, 9)):
        if old and c0:
            continue  # The native inherited-maturity event has zero direct cost.
        K = F(e0 + credit + (g0 if old else 0))
        total_m = total_raw = F(0)
        capped = {cap: F(0) for cap in (0, 1, 4, 16)}
        for signs in product((-1, 1), repeat=2):
            cash, remaining = F(e0), F(budget)
            raw = paired = control = local_full = F(0)
            for k, (g, cost, sign, weight) in enumerate(zip(
                    (g0, g1), (c0, c1), signs, (F(1), F(1, 2)))):
                plus = max(-cash-g+cost, F(0))
                minus = max(-cash+g+cost, F(0))
                injection = plus if sign == 1 else minus
                excluded = old and k == 0
                actual_raw = -F(cost) if excluded else sign*g-cost+injection
                pair = -F(cost) if excluded else (plus+minus)/2-cost
                local = F(0) if excluded else weight*(g+(plus-minus)/2)
                # Deliberately inaccurate finite common-branch continuation.
                after_plus = max(cash+g-cost, F(0))
                after_minus = max(cash-g-cost, F(0))
                qplus = 3*after_plus**2+2-cash if k == 0 else F(0)
                qminus = 3*after_minus**2-2-cash if k == 0 else F(0)
                delta = local+(qplus-qminus)/2
                proposed = beta*delta
                amplitude = min(abs(proposed), remaining)
                coefficient = amplitude if proposed >= 0 else -amplitude
                remaining -= amplitude  # Before using this current sign.
                control += sign*coefficient
                local_full += sign*local
                raw += weight*actual_raw
                paired += weight*pair
                cash = max(cash+sign*g-cost, F(0))
                if k == 0:
                    cash += credit
            assert cash >= 0 and raw >= -K
            assert abs(control) <= budget and remaining >= 0
            assert raw-local_full == paired
            identities += 1
            assert raw-control >= -K-budget
            for cap in capped:
                statistic = min(raw-control, F(cap))
                assert -K-budget <= statistic <= cap
                capped[cap] += statistic/4
                caps_checked += 1
            total_m += control/4
            total_raw += raw/4
            paths += 1
        assert total_m == 0
        assert all(value <= total_raw for value in capped.values())
        laws += 1
    return dict(status="PASS EXACT CASH-CONTROL CHECKS", arithmetic="fractions.Fraction",
                fair_two_event_laws=laws, complete_paths=paths,
                raw_minus_full_local_equals_paired=identities, capped_support_checks=caps_checked,
                native_queries=0, actor_calls=0, model_fits=0, statistical_endpoint_reads=0,
                interpretation="Finite mathematical consistency checks, not empirical power or novelty evidence",
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__ == "__main__":
    result = check()
    output = Path(__file__).with_suffix(".json")
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result))
