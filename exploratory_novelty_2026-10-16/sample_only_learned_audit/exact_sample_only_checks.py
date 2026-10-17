"""Exact rational checks for sample-only financial-event certificates.

No Monte Carlo, trained actor/critic, empirical panel, or older result is used.
The contextual laws are known ONLY to this enumerated mathematical verifier;
the proposed trajectory statistic itself queries no conditional moment.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
H, GAMMA, INITIAL = 3, F(9, 10), F(3, 4)
WEIGHTS = tuple(GAMMA**t for t in range(H))
FLOOR_CMAX = F(7, 20) * sum(WEIGHTS)
POLICIES = ("flat", "constant1", "adaptive", "context_gate")
ORDERS = ("fee_first", "maturity_first")
ALPHAS = (F(0), F(4, 5), F(1))
BUDGETS = (F(0), F(1, 2), F(2))
CAPS = (F(0), F(1, 2), F(2))


def pos(x):
    return max(F(0), x)


def debt(x):
    return pos(-x)


def rational(x):
    x = F(x)
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def mean(paths, values):
    return sum((p["probability"] * value for p, value in zip(paths, values)), F(0))


def variance(paths, values):
    m = mean(paths, values)
    return mean(paths, [(v-m)**2 for v in values])


def floor_event(cash, amount):
    injection = debt(cash + amount)
    after = cash + amount + injection
    assert after >= 0
    assert after-cash == amount+injection
    return after, injection


def action(policy, t, cash, context):
    if policy == "flat":
        return F(0)
    if policy == "constant1":
        return F(1)
    if policy == "adaptive":
        return F(2) if cash <= 1 and context == 1 else F(1) if cash <= 1 else F(0)
    if policy == "context_gate":
        return F(1) if t == 0 else F(2) if context == 0 and cash < F(3, 2) else F(0)
    raise ValueError(policy)


def context_rows(context):
    # Neither the statistic nor the branch predictor uses these probabilities.
    return ((F(1, 2), F(3, 4)), (F(3, 2), F(1, 4))) if context == 0 else (
        (F(1, 4), F(1, 3)), (F(2), F(2, 3)))


def nonfinancial_flow(t, context):
    if t == 0:
        return F(0)
    if t == 1:
        return F(3, 10) if context == 0 else -F(1, 2)
    return F(1, 10) if context == 0 else -F(1, 4)


def enumerate_floor(policy, order):
    leaves = []

    def walk(t, cash, context, probability, events):
        if t == H:
            leaves.append(dict(probability=probability, events=tuple(events)))
            return
        flow = nonfinancial_flow(t, context)
        after_flow, nonfinancial_injection = floor_event(cash, flow)
        a = action(policy, t, after_flow, context)
        assert 0 <= a <= 2
        for row_id, (magnitude, row_probability) in enumerate(context_rows(context)):
            z = a*magnitude
            fee = a*(F(1, 10)+F(context, 20)+F(row_id, 40))
            assert 0 <= fee <= F(7, 20)
            if order == "fee_first":
                pre_coin, fee_injection_before = floor_event(after_flow, -fee)
            else:
                pre_coin, fee_injection_before = after_flow, F(0)
            plus_cash, plus_injection = floor_event(pre_coin, z)
            minus_cash, minus_injection = floor_event(pre_coin, -z)
            paired_coin_injection = (plus_injection+minus_injection)/2
            assert paired_coin_injection >= 0
            common = dict(t=t, cash=pre_coin, context=context, action=a, z=z,
                          fee=fee, fee_first=order == "fee_first")
            for epsilon in (1, -1):
                gross = epsilon*z
                after_coin = plus_cash if epsilon == 1 else minus_cash
                coin_injection = plus_injection if epsilon == 1 else minus_injection
                if order == "maturity_first":
                    final_cash, fee_injection_after = floor_event(after_coin, -fee)
                else:
                    final_cash, fee_injection_after = after_coin, F(0)
                fee_injection = fee_injection_before+fee_injection_after
                financial_injection = coin_injection+fee_injection
                paired_financial_injection = paired_coin_injection+fee_injection
                # Fee events stay at their ACTUAL pre-event cash. They are not
                # recomputed at an averaged/counterfactual physical state.
                assert paired_financial_injection >= 0
                cash_change = final_cash-cash
                assert cash_change == flow+nonfinancial_injection+gross-fee+financial_injection
                # On a native floor the event-level Tanaka correction is the
                # ordinary injection, not a distinct estimator.
                assert debt(pre_coin+gross)-debt(pre_coin)+(gross if pre_coin < 0 else 0) == coin_injection
                event = dict(common=common, epsilon=epsilon, gross=gross, fee=fee,
                    actual_financial_injection=financial_injection,
                    paired_financial_injection=paired_financial_injection,
                    nonfinancial_flow=flow, nonfinancial_injection=nonfinancial_injection,
                    cash_change=cash_change)
                walk(t+1, final_cash, 0 if epsilon == 1 else 1,
                     probability*row_probability/2, events+(event,))

    for initial_context in (0, 1):
        walk(0, INITIAL, initial_context, F(1, 2), ())
    assert sum(p["probability"] for p in leaves) == 1
    return leaves


def q_zero(common, branch):
    return F(0)


def q_linear(common, branch):
    return (2*common["cash"]-F(common["context"], 3)+F(common["t"], 7)
            +branch*(common["z"]-common["fee"]+common["cash"]/2))


def q_tiny_relu(common, branch):
    # Fixed rational two-layer neural predictor: no training or accuracy claim.
    h0 = pos(common["cash"]/2+branch*common["z"]-common["fee"])
    h1 = pos(1-common["cash"]+F(branch, 3)+F(common["context"], 2))
    h2 = pos(common["action"]/2-F(common["context"], 4)-F(branch, 2))
    return F(3, 2)*h0-2*h1+F(5, 4)*h2+F(common["t"], 5)


def q_wrong_model(common, branch):
    # Deliberately reverses the gross direction and treats fees as a subsidy.
    fake_cash = pos(common["cash"]-3*branch*common["z"]+common["fee"]+2)
    return 1000*fake_cash+400*branch-50*common["context"]


PREDICTORS = dict(zero=q_zero, linear=q_linear, tiny_relu=q_tiny_relu,
                  deliberately_wrong_model=q_wrong_model)


def control_on_path(path, predictor, budget):
    remaining, total, used = budget, F(0), F(0)
    amplitudes = []
    for event in path["events"]:
        common = event["common"]
        # Both predictions use exactly the same PRE-SIGN information.
        qplus, qminus = predictor(common, 1), predictor(common, -1)
        assert isinstance(qplus, F) and isinstance(qminus, F)
        delta = (qplus-qminus)/2
        amplitude = min(abs(delta), remaining)
        amplitude = amplitude if delta >= 0 else -amplitude
        # Debit ABSOLUTE amplitude BEFORE reading the current coin.
        remaining -= abs(amplitude)
        used += abs(amplitude)
        assert 0 <= remaining <= budget and used <= budget
        total += event["epsilon"]*amplitude
        amplitudes.append(amplitude)
    assert abs(total) <= used <= budget
    return total, used, tuple(amplitudes)


def floor_checks():
    records, family_records = [], []
    for order in ORDERS:
        for policy in POLICIES:
            paths = enumerate_floor(policy, order)
            costs = [sum(WEIGHTS[e["common"]["t"]]*e["fee"] for e in p["events"]) for p in paths]
            grosses = [sum(WEIGHTS[e["common"]["t"]]*e["gross"] for e in p["events"]) for p in paths]
            actual_a = [sum(WEIGHTS[e["common"]["t"]]*e["actual_financial_injection"] for e in p["events"]) for p in paths]
            paired_a = [sum(WEIGHTS[e["common"]["t"]]*e["paired_financial_injection"] for e in p["events"]) for p in paths]
            flows = [sum(WEIGHTS[e["common"]["t"]]*e["nonfinancial_flow"] for e in p["events"]) for p in paths]
            nonfin_inj = [sum(WEIGHTS[e["common"]["t"]]*e["nonfinancial_injection"] for e in p["events"]) for p in paths]
            assert all(0 <= c <= FLOOR_CMAX for c in costs)
            assert mean(paths, grosses) == 0
            assert mean(paths, paired_a) == mean(paths, actual_a)
            family_records.append(dict(order=order, policy=policy, enumerated_paths=len(paths),
                probability_sum="1", expected_financial_cost=rational(mean(paths, costs)),
                expected_booked_financial_return=rational(-mean(paths, costs)),
                expected_nonfinancial_flow=rational(mean(paths, flows)),
                expected_excluded_nonfinancial_injection=rational(mean(paths, nonfin_inj)),
                exact_pairing_identity=True, native_floor_tanaka_is_existing_injection=True))
            for alpha in ALPHAS:
                raw = [g-c+(1-alpha)*i for g, c, i in zip(grosses, costs, actual_a)]
                paired = [(1-alpha)*a-c for a, c in zip(paired_a, costs)]
                target = mean(paths, raw)
                assert mean(paths, paired) == target
                for predictor_name, predictor in PREDICTORS.items():
                    for budget in BUDGETS:
                        evaluations = [control_on_path(p, predictor, budget) for p in paths]
                        controls = [value[0] for value in evaluations]
                        used = [value[1] for value in evaluations]
                        assert mean(paths, controls) == 0
                        corrected = [s-m for s, m in zip(paired, controls)]
                        assert mean(paths, corrected) == target
                        lower = -FLOOR_CMAX-budget
                        assert min(corrected) >= lower
                        for cap in CAPS:
                            capped = [min(value, cap) for value in corrected]
                            capped_mean = mean(paths, capped)
                            assert lower <= min(capped) <= max(capped) <= cap
                            assert capped_mean <= target
                            records.append(dict(order=order, policy=policy, alpha=rational(alpha),
                                predictor=predictor_name, control_budget=rational(budget), upper_cap=rational(cap),
                                charged_financial_mean=rational(target), uncapped_paired_mean=rational(mean(paths, paired)),
                                control_mean="0", uncapped_corrected_mean=rational(mean(paths, corrected)),
                                capped_corrected_mean=rational(capped_mean), cap_lower_bias=rational(target-capped_mean),
                                expected_booked_financial_return=rational(-mean(paths, costs)),
                                certified_support_lower=rational(lower), certified_support_upper=rational(cap),
                                observed_capped_min=rational(min(capped)), observed_capped_max=rational(max(capped)),
                                maximum_spent_absolute_amplitude=rational(max(used)),
                                paired_variance=rational(variance(paths, paired)),
                                corrected_uncapped_variance=rational(variance(paths, corrected)),
                                range_and_mean_checks_pass=True))
    return family_records, records


def tanaka_increment(n, gross, cost):
    x = gross-cost
    value = debt(n+x)-debt(n)+(gross if n < 0 else 0)
    stable = debt(n+x) if n >= 0 else cost+pos(n+x)
    assert value == stable and value >= 0
    return value


def terminal_checks():
    records = []
    for initial in (F(-1), F(0), F(1, 2)):
        for policy in ("flat", "constant1", "adaptive"):
            leaves = []

            def walk(t, ledger, probability, events):
                if t == H:
                    leaves.append(dict(probability=probability, events=events, final=ledger))
                    return
                a = F(0) if policy == "flat" else F(2) if policy == "adaptive" and ledger < 0 else F(1)
                cost = a/5
                # Mean-zero gross but NOT a fair-sign/symmetric distribution.
                for base_gross, chance in ((F(2), F(1, 3)), (F(-1), F(2, 3))):
                    gross = a*base_gross
                    y = tanaka_increment(ledger, gross, cost)
                    walk(t+1, ledger+gross-cost, probability*chance,
                         events+((gross, cost, y),))

            walk(0, initial, F(1), ())
            assert sum(p["probability"] for p in leaves) == 1
            costs = [sum(WEIGHTS[t]*event[1] for t, event in enumerate(p["events"])) for p in leaves]
            grosses = [sum(WEIGHTS[t]*event[0] for t, event in enumerate(p["events"])) for p in leaves]
            sample_a = [WEIGHTS[-1]*sum(event[2] for event in p["events"]) for p in leaves]
            debt_change = [WEIGHTS[-1]*(debt(p["final"])-debt(initial)) for p in leaves]
            assert mean(leaves, sample_a) == mean(leaves, debt_change)
            assert mean(leaves, grosses) == 0
            cost_max = F(2, 5)*sum(WEIGHTS)
            for alpha in ALPHAS:
                raw = [g-c+(1-alpha)*d for g, c, d in zip(grosses, costs, debt_change)]
                sample = [(1-alpha)*a-c for a, c in zip(sample_a, costs)]
                target = mean(leaves, raw)
                assert mean(leaves, sample) == target
                assert min(sample) >= -cost_max
                for cap in CAPS:
                    capped = [min(value, cap) for value in sample]
                    assert -cost_max <= min(capped) <= max(capped) <= cap
                    assert mean(leaves, capped) <= target
                    records.append(dict(initial_ledger=rational(initial), policy=policy, alpha=rational(alpha),
                        cap=rational(cap), paths=len(leaves), gross_law="+2 with1/3; -1 with2/3, scaled by action",
                        expected_charged_relative_flat=rational(target), sample_tanaka_mean=rational(mean(leaves, sample)),
                        capped_mean=rational(mean(leaves, capped)), booked_financial_mean=rational(-mean(leaves, costs)),
                        lower_support=rational(-cost_max), upper_support=rational(cap), all_exact_checks_pass=True))
    return records


def counterexamples():
    # Wrong counterfactual model: a true actual next-state value is centered
    # against two model predictions, rather than one common branch predictor.
    true_next = (F(2), F(0))
    modeled_center = F(2)
    mixed_control = [v-modeled_center for v in true_next]
    mixed_mean = sum(mixed_control)/2
    false_positive_mean = sum(min(-v, F(2)) for v in mixed_control)/2
    assert mixed_mean == -1 and false_positive_mean == 1
    # Any finite COMMON model is safe when only its branch DIFFERENCE is used.
    correctly_centered_common_model_mean = F(0)

    # Native event order cannot be replaced by a net clamp.
    cash, fee, magnitude = F(1, 5), F(1, 2), F(1)
    after_fee, fee_injection = floor_event(cash, -fee)
    ordered_injections = [fee_injection+floor_event(after_fee, eps*magnitude)[1] for eps in (1, -1)]
    aggregate_injections = [debt(cash+eps*magnitude-fee) for eps in (1, -1)]
    ordered_mean, aggregate_mean = sum(ordered_injections)/2, sum(aggregate_injections)/2
    assert ordered_mean == F(4, 5) and aggregate_mean == F(13, 20)

    # A flat financial sleeve has zero financial return despite an unrelated
    # physical floor injection. Counting that injection produces false value.
    _, unrelated_injection = floor_event(F(1, 5), -F(1, 2))
    assert unrelated_injection == F(3, 10)
    # EG=0 does not imply a fair sign, so an epsilon control need not center.
    asymmetric_gross_mean = F(1, 3)*2+F(2, 3)*(-1)
    asymmetric_sign_mean = F(1, 3)-F(2, 3)
    assert asymmetric_gross_mean == 0 and asymmetric_sign_mean == -F(1, 3)
    return dict(actual_model_mixed=dict(true_next_cash=["2", "0"], common_model_branch_predictions=["2", "2"],
                    invalid_control_mean=rational(mixed_mean), true_charged_financial_mean="0",
                    invalid_corrected_capped_mean=rational(false_positive_mean),
                    correctly_centered_model_control_mean=rational(correctly_centered_common_model_mean),
                    independent_model_fitting_does_not_repair_wrong_centering=True),
                aggregate_net_clamp=dict(ordered_financial_injection_mean=rational(ordered_mean),
                    incorrectly_aggregated_mean=rational(aggregate_mean), not_equivalent=True),
                unrelated_floor_injection=dict(flat_financial_target="0",
                    false_target_if_unrelated_injection_included=rational(unrelated_injection)),
                fair_mean_without_fair_sign=dict(expected_gross="0", naive_sign_control_mean=rational(asymmetric_sign_mean)))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE/"exact_sample_only_checks.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve existing output; choose a new --output path to reproduce")
    started = time.perf_counter()
    families, floor = floor_checks()
    terminal = terminal_checks()
    examples = counterexamples()
    report = dict(status="EXACT_CHECKS_PASS", method="Fraction enumeration; no simulation or fitted model",
        scope="Toy mathematical checks only; no learned-policy, market, power or novelty result",
        horizon=H, gamma=rational(GAMMA), floor_initial_cash=rational(INITIAL),
        floor_discounted_financial_cost_upper=rational(FLOOR_CMAX),
        floor_path_families=len(families), floor_certificate_cases=len(floor),
        terminal_certificate_cases=len(terminal), distinct_predictors=list(PREDICTORS),
        floor_families=families, floor_cases=floor, terminal_cases=terminal,
        invalid_constructions=examples, every_exact_assertion_passed=True,
        script_sha256=sha(__file__), theory_sha256=sha(HERE/"SAMPLE_ONLY_THEORY.md"),
        elapsed_seconds=time.perf_counter()-started,
        old_artifacts_read=False, old_artifacts_changed=False, monte_carlo_draws=0, learners_trained=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "floor_path_families", "floor_certificate_cases",
        "terminal_certificate_cases", "distinct_predictors", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    main()
