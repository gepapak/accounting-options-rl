"""Exploratory exact predictable accounting checks; no Monte Carlo, RL or DP.

Only finite-law fixed-policy trajectory enumeration is used. This script reads
no earlier studies or artifacts and writes its JSON beside this new source.
"""
from __future__ import annotations

from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ZERO, ONE, GAMMA = F(0), F(1), F(99, 100)
LAWS = {
    "flat": ((ZERO, ZERO, ONE),),
    "trade": ((F(-1), F(2, 5), F(1, 2)), (ONE, F(2, 5), F(1, 2))),
    "small": ((F(-1, 2), F(1, 10), F(1, 2)), (F(1, 2), F(1, 10), F(1, 2))),
}


def num(x):
    return {"exact": str(x), "decimal": float(x)}


def d(n):
    return max(-n, ZERO)


def cost(law):
    return sum(c * p for g, c, p in law)


def h(law, n):
    return sum(d(n + g - c) * p for g, c, p in law)


def delta(law, n):
    return h(law, n) - d(n)


def choose(policy, t, n, initial):
    if policy == "flat":
        return "flat"
    if policy == "constant_trade":
        return "trade"
    if policy == "prefix_gate":
        return "trade" if t == 0 or n <= F(1, 2) else "flat"
    if policy == "barrier":
        return "trade" if n <= initial else "flat"
    if policy == "random_fee_feedback":
        return "random_fee" if t == 0 else ("fixed_fee" if n == initial else "flat")
    raise ValueError(policy)


def enumerate_policy(laws, policy, initial, horizon, alpha, accounting, gamma=GAMMA):
    """Enumerate trajectories of ONE fixed controller, not a Bellman search."""
    paths = []
    terminal_weight = gamma ** (horizon - 1)

    def visit(t, n, probability, gross_sum, cost_sum, forgiveness_sum,
              predictable_forgiveness, predictable_cost, trace):
        if t == horizon:
            actual_forgiveness = (terminal_weight * (d(n) - d(initial))
                                  if accounting == "terminal" else forgiveness_sum)
            actual_residual = (1 - alpha) * actual_forgiveness - cost_sum
            charged = gross_sum + actual_residual
            predictable = (1 - alpha) * predictable_forgiveness - predictable_cost
            paths.append(dict(probability=probability, charged=charged,
                              residual=actual_residual, compensator=predictable,
                              gross=gross_sum, booked=gross_sum-cost_sum,
                              actual_forgiveness=actual_forgiveness,
                              predictable_forgiveness=predictable_forgiveness,
                              actual_cost=cost_sum, predictable_cost=predictable_cost,
                              trace=trace))
            return
        action = choose(policy, t, n, initial)
        law, weight = laws[action], gamma ** t
        curve = delta(law, n) if accounting == "terminal" else h(law, n)
        a_weight = terminal_weight if accounting == "terminal" else weight
        for gross, fee, outcome_probability in law:
            injection = d(n + gross-fee) if accounting == "floor" else ZERO
            nxt = n + gross-fee + injection
            visit(t+1, nxt, probability*outcome_probability,
                  gross_sum+weight*gross, cost_sum+weight*fee,
                  forgiveness_sum+weight*injection,
                  predictable_forgiveness+a_weight*curve,
                  predictable_cost+weight*cost(law),
                  trace+[(action, n, gross, fee)])

    visit(0, initial, ONE, ZERO, ZERO, ZERO, ZERO, ZERO, [])
    assert sum(p["probability"] for p in paths) == 1
    fields = ("charged", "residual", "compensator", "gross", "booked",
              "actual_forgiveness", "predictable_forgiveness", "actual_cost", "predictable_cost")
    means = {key: sum(p["probability"]*p[key] for p in paths) for key in fields}
    assert means["gross"] == 0
    assert means["charged"] == means["residual"] == means["compensator"]
    assert means["actual_forgiveness"] == means["predictable_forgiveness"]
    assert means["actual_cost"] == means["predictable_cost"]
    assert means["booked"] == -means["predictable_cost"]
    variances = {key: sum(p["probability"]*(p[key]-means[key])**2 for p in paths)
                 for key in ("charged", "residual", "compensator")}
    cmax = max(cost(law) for law in laws.values())
    low = -sum(gamma**t*cmax for t in range(horizon))
    high = sum(max((1-alpha)*(terminal_weight if accounting == "terminal" else gamma**t)*h(law, ZERO)
                   - gamma**t*cost(law) for law in laws.values()) for t in range(horizon))
    for p in paths:
        assert low <= p["compensator"] <= high
    return {
        "accounting": accounting, "policy": policy, "initial_ledger": num(initial),
        "horizon": horizon, "alpha": num(alpha), "gamma": num(gamma),
        "enumerated_paths": len(paths), "expected": {key: num(v) for key, v in means.items()},
        "variance": {key: num(v) for key, v in variances.items()},
        "safe_additive_compensator_range": [num(low), num(high)],
        "actual_compensator_range": [num(min(p["compensator"] for p in paths)),
                                     num(max(p["compensator"] for p in paths))],
        "range_caveat": "Sharp per-state/action bounds need not be jointly attainable by a full trajectory.",
        "same_account_flat_terminal_claim_subtracted": accounting == "terminal",
    }, paths


def drift_checks():
    states = tuple(map(F, (-4, -2, -1, 0, 1, 2, 4))) + (F(-1, 10), F(1, 10), F(7, 10))
    records = []
    for name, law in LAWS.items():
        c, h0 = cost(law), h(law, ZERO)
        assert sum(g*p for g, fee, p in law) == 0
        assert sum(p for g, fee, p in law) == 1
        assert h0 >= c
        positive_net_mean = sum(max(g-fee, ZERO)*p for g, fee, p in law)
        assert h0 == c + positive_net_mean
        values = []
        for n in sorted(states):
            drift = delta(law, n)
            assert ZERO <= drift <= h0
            if n < 0:
                assert drift == c + sum(max(n+g-fee, ZERO)*p for g, fee, p in law)
                assert drift >= c
            else:
                assert drift == h(law, n)
            values.append(dict(ledger=num(n), delta=num(drift), h=num(h(law, n))))
        assert delta(law, ZERO) == h0
        assert h(law, F(4)) == 0
        assert delta(law, F(-4)) == c
        records.append(dict(action=name, expected_cost=num(c), h_at_zero=num(h0),
                            positive_net_mean=num(positive_net_mean), state_checks=values))
    return records


def repair_checks():
    results = []
    for names in (("flat", "trade"), ("flat", "trade", "small")):
        laws = {name: LAWS[name] for name in names}
        nonflat = [name for name in names if cost(laws[name]) > 0]
        rho = max(h(laws[a], ZERO)/cost(laws[a]) for a in nonflat)
        alpha_star = 1-1/rho
        best = max(nonflat, key=lambda a: h(laws[a], ZERO)/cost(laws[a]))
        checked = 0
        for alpha in (alpha_star, F(4, 5), ONE):
            for horizon in (1, 2, 3):
                for t in range(horizon):
                    for name, law in laws.items():
                        for n in (F(-2), F(-1, 2), ZERO, F(1, 2), F(2)):
                            terminal_drift = ((1-alpha)*GAMMA**(horizon-1)*delta(law,n)
                                              - GAMMA**t*cost(law))
                            assert terminal_drift <= 0
                            if n >= 0:
                                assert (1-alpha)*h(law,n)-cost(law) <= 0
                            checked += 1
        below = alpha_star-F(1, 100)
        witness = (1-below)*h(laws[best], ZERO)-cost(laws[best])
        assert witness > 0
        at_boundary = (1-alpha_star)*h(laws[best], ZERO)-cost(laws[best])
        assert at_boundary == 0
        results.append(dict(actions=list(names), rho=num(rho), alpha_star=num(alpha_star),
                            pointwise_checks=checked, sharp_action=best,
                            alpha_below_boundary=num(below),
                            one_action_at_zero_reward_below_boundary=num(witness),
                            one_action_at_zero_reward_at_boundary=num(at_boundary),
                            terminal_deadline_H3_reward_below_boundary=num(GAMMA**2*witness),
                            scope="Uniform all-state/all-policy sufficient and sharp criterion for this action kernel; not minimal for one fixed positive start."))
    free_law = ((F(-1), ZERO, F(1, 2)), (ONE, ZERO, F(1, 2)))
    assert cost(free_law) == 0 and h(free_law, ZERO) == F(1, 2)
    return results, dict(expected_cost=num(ZERO), h0=num(h(free_law, ZERO)),
                         implication="rho infinite; every alpha<1 leaves a positive zero-equity one-action incentive, with booked mean zero rather than strict harm")


def counterchecks():
    fee_laws = {
        "flat": LAWS["flat"],
        "random_fee": ((ZERO, ZERO, F(1, 2)), (ZERO, ONE, F(1, 2))),
        "fixed_fee": ((ZERO, ONE, ONE),),
    }
    variance, paths = enumerate_policy(fee_laws, "random_fee_feedback", F(2), 2, ZERO, "floor", gamma=ONE)
    assert variance["variance"]["residual"]["exact"] == "0"
    assert variance["variance"]["compensator"]["exact"] == "1/4"
    variance["paths"] = [dict(probability=num(p["probability"]),
                              residual=num(p["residual"]), compensator=num(p["compensator"]),
                              realized_cost=num(p["actual_cost"]), predictable_cost=num(p["predictable_cost"]))
                         for p in paths]
    at_zero = ((F(-1), F(1, 10), F(1, 2)), (ONE, F(1, 10), F(1, 2)))
    at_one = ((F(-4), F(1, 10), F(1, 2)), (F(4), F(1, 10), F(1, 2)))
    assert h(at_zero, ZERO) == F(11, 20)
    assert delta(at_one, ONE) == F(31, 20) > h(at_zero, ZERO)
    assert h(at_one, ZERO) == F(41, 20)
    edge_law = ((ONE, F(1, 10), ONE),)
    assert delta(edge_law, F(-1)) == F(-9, 10)
    return {
        "no_universal_variance_reduction": variance,
        "cash_dependent_kernel": {
            "zero_state_law": "fair gross +/-1; cost .1", "one_state_law": "fair gross +/-4; cost .1",
            "actual_zero_state_h0": num(h(at_zero, ZERO)),
            "delta_at_ledger_one": num(delta(at_one, ONE)),
            "counterfactual_zero_under_current_one_state_law": num(h(at_one, ZERO)),
            "actual_zero_state_bound_fails": True,
        },
        "conditional_gross_edge": {
            "ledger": num(F(-1)), "deterministic_gross": num(ONE), "cost": num(F(1, 10)),
            "delta": num(delta(edge_law, F(-1))), "nonnegative_drift_bound_fails": True,
        },
    }


def main():
    cases = (
        ("floor", "constant_trade", F(7, 10), 1),
        ("floor", "prefix_gate", F(7, 10), 2),
        ("floor", "prefix_gate", F(7, 10), 3),
        ("floor", "barrier", F(3, 2), 3),
        ("terminal", "barrier", F(7, 10), 3),
        ("terminal", "prefix_gate", F(3, 2), 2),
        ("terminal", "constant_trade", F(-1, 2), 2),
        ("terminal", "flat", F(-1, 2), 2),
    )
    scenarios, families = [], []
    for accounting, policy, initial, horizon in cases:
        records = [enumerate_policy(LAWS, policy, initial, horizon, alpha, accounting)[0]
                   for alpha in (ZERO, F(1, 5), F(4, 5), ONE)]
        scenarios.extend(records)
        a = F(records[0]["expected"]["predictable_forgiveness"]["exact"])
        b = F(records[0]["expected"]["predictable_cost"]["exact"])
        for r in records:
            alpha = F(r["alpha"]["exact"])
            assert F(r["expected"]["compensator"]["exact"]) == (1-alpha)*a-b
        families.append(dict(accounting=accounting, policy=policy, initial_ledger=num(initial),
                             horizon=horizon, expected_predictable_forgiveness=num(a),
                             expected_predictable_cost=num(b),
                             strict_harmful_alpha_upper=num(1-b/a) if a > b > 0 else None,
                             fixed_policy_affine_identity_checked=True))
    single = scenarios[0]
    assert single["variance"]["compensator"]["exact"] == "0"
    assert single["expected"]["compensator"]["exact"] == "-1/20"
    floor_two = scenarios[4]
    assert floor_two["expected"]["compensator"]["exact"] == "197/2000"
    repair, free = repair_checks()
    result = {
        "status": "exploratory exact finite-law predictable-compensator checks",
        "no_monte_carlo": True, "learners_trained": 0, "dp_solved": False,
        "inputs_read": [], "drift_bound_checks": drift_checks(),
        "fixed_policy_scenarios": scenarios, "fixed_policy_alpha_families": families,
        "uniform_partial_charge_checks": repair, "zero_cost_exception": free,
        "counterexamples": counterchecks(),
        "scope": [
            "Exact conditional moments and known uniform first-moment bounds are required; unknown black-box curves do not inherit this guarantee.",
            "Per-state/action ranges are sharp; summed whole-path ranges can be conservative due to reachability.",
            "Uniform partial-charge result is kernel/action-set specific; it is not a universal constant across arbitrary markets.",
            "Predictable compensation is established martingale-control theory; no general AI-method priority is claimed.",
        ],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (HERE / "exact_predictable_checks.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"fixed_policy_scenarios": len(scenarios),
                      "alpha_families": len(families),
                      "uniform_thresholds": [{"actions": r["actions"], "alpha_star": r["alpha_star"]} for r in repair],
                      "single_action_compensator_variance": single["variance"]["compensator"],
                      "variance_increase_counterexample": result["counterexamples"]["no_universal_variance_reduction"]["variance"],
                      "all_assertions_passed": True}, indent=2))


if __name__ == "__main__":
    main()
