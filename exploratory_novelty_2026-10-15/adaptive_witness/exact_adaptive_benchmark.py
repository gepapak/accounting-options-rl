"""EXPLORATORY exact adaptive accounting witnesses; no learning or Bellman DP.

Enumerates the paths of a fixed, small policy library using rational arithmetic.
It also verifies a closed-form unbounded-support clipping example. This file
does not read market data, registered artifacts or Study R.
"""
from __future__ import annotations

from fractions import Fraction as F
import hashlib
from itertools import product
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
GAMMA = F(99, 100)
ZERO = F(0)


def number(x):
    return {"exact": str(x), "decimal": float(x)}


def myopic_value(equity, cost, alpha=ZERO):
    """One additional trade versus flat; gross is an independent fair +/-1."""
    injection = sum(max(-equity - (sign - cost), ZERO) for sign in (-1, 1)) / 2
    return -cost + (1 - alpha) * injection


def action(name, t, state, initial, cost):
    if name == "flat":
        return False
    if name.startswith("one_action_at_"):
        return t == int(name.rsplit("_", 1)[1])
    if name == "constant_trade":
        return True
    if name == "myopic_positive_gate":
        return myopic_value(state, cost) > 0
    if name == "forced_prefix_then_floor_gate":
        return t == 0 or state <= 0
    if name == "forced_prefix_then_positive_gate":
        # Gate is computed at design alpha=0 and then frozen for alpha contrasts.
        return t == 0 or myopic_value(state, cost) > 0
    raise ValueError(name)


def path(policy, signs, initial, cost, alpha, accounting):
    horizon = len(signs)
    state = initial
    gross_discounted = ZERO
    cost_discounted = ZERO
    floor_discounted = ZERO
    charged_discounted = ZERO
    trace = []
    for t, sign in enumerate(signs):
        trade = action(policy, t, state, initial, cost)
        gross, fee = (F(sign), cost) if trade else (ZERO, ZERO)
        before = state
        injection = max(-(before + gross - fee), ZERO) if accounting == "step_floor" else ZERO
        state = before + gross - fee + injection
        weight = GAMMA ** t
        gross_discounted += weight * gross
        cost_discounted += weight * fee
        floor_discounted += weight * injection
        charged_discounted += weight * (gross - fee + (1 - alpha) * injection)
        trace.append({"t": t, "sign": sign, "trade": trade, "state_before": number(before),
                      "state_after": number(state), "injection": number(injection)})
    if accounting == "terminal":
        assert initial >= 0
        floor_discounted = GAMMA ** (horizon - 1) * max(-state, ZERO)
        charged_discounted += (1 - alpha) * floor_discounted
    booked = gross_discounted - cost_discounted
    residual = (1 - alpha) * floor_discounted - cost_discounted
    assert charged_discounted == booked + (1 - alpha) * floor_discounted
    assert charged_discounted - gross_discounted == residual
    return {
        "charged": charged_discounted, "booked": booked, "gross": gross_discounted,
        "cost": cost_discounted, "floor": floor_discounted, "residual": residual,
        "trace": trace,
    }


def evaluate(policy, initial, cost, horizon, alpha, accounting):
    probability = F(1, 2 ** horizon)
    paths = [path(policy, signs, initial, cost, alpha, accounting)
             for signs in product((-1, 1), repeat=horizon)]
    means = {key: sum(p[key] for p in paths) * probability
             for key in ("charged", "booked", "gross", "cost", "floor", "residual")}
    assert means["gross"] == 0
    assert means["booked"] == -means["cost"]
    assert means["charged"] == means["residual"]
    variances = {key: sum((p[key] - means[key]) ** 2 for p in paths) * probability
                 for key in ("charged", "residual")}
    return {
        "policy": policy,
        "expected": {key: number(value) for key, value in means.items()},
        "variance": {key: number(value) for key, value in variances.items()},
        "positive_charged_negative_booked": means["charged"] > 0 and means["booked"] < 0,
        "path_count": len(paths),
        "paths": [{"probability": number(probability),
                   **{key: number(p[key]) for key in means}, "trace": p["trace"]} for p in paths],
    }


def scenario(name, initial, cost, horizon, alpha=ZERO, accounting="step_floor"):
    policies = ["flat", *[f"one_action_at_{t}" for t in range(horizon)],
                "myopic_positive_gate", "constant_trade", "forced_prefix_then_floor_gate",
                "forced_prefix_then_positive_gate"]
    results = [evaluate(p, initial, cost, horizon, alpha, accounting) for p in policies]
    best = max(results, key=lambda r: F(r["expected"]["charged"]["exact"]))
    chosen = next(r for r in results if r["policy"] == "forced_prefix_then_positive_gate")
    floor, charge = (F(chosen["expected"][key]["exact"]) for key in ("floor", "cost"))
    critical = (1 - charge / floor) if floor > 0 else None
    return {
        "name": name, "accounting": accounting, "initial_equity": number(initial),
        "cost_per_trade": number(cost), "horizon": horizon, "alpha": number(alpha),
        "policies": results, "best_in_finite_library": best["policy"],
        "library_decision": "WITNESS" if best["positive_charged_negative_booked"] else "ABSTAIN",
        "positive_gate_policy_fixed_alpha_critical": number(critical) if critical is not None else None,
        "critical_scope": "Strict harmful interval for this fixed policy only; not a global repair boundary.",
        "no_possible_floor_for_any_binary_policy": initial >= horizon * (1 + cost),
    }


def unbounded_terminal_example():
    """A has P(A=k)=2^-k, k>=1, and independent fair sign; E[A]=2."""
    initial, cost, cap = F(1), F(1, 10), F(1)
    # Trade at t=0 then flat, H=2: F_gamma=gamma*(A-.9) on negative signs.
    expected_floor = GAMMA * F(1, 2) * (2 + cost - initial)
    expected_charged = expected_floor - cost
    # Residual negative-sign values at k=1,2; k>=3 exceed cap.
    z1 = GAMMA * (1 + cost - initial) - cost
    z2 = GAMMA * (2 + cost - initial) - cost
    assert z1 < cap and z2 < cap
    assert GAMMA * (3 + cost - initial) - cost > cap
    clipped_mean = F(1, 2) * (-cost) + F(1, 2) * (F(1, 2) * z1 + F(1, 4) * z2 + F(1, 4) * cap)
    clipped_floor_mean = F(1, 2) * (F(1, 2) * GAMMA * (1 + cost - initial) + F(1, 2) * cap)
    assert expected_charged == F(889, 2000)
    assert clipped_mean == F(1587, 8000) > 0
    assert clipped_floor_mean == F(1099, 4000)
    # If iid confirmation is later performed, this n suffices for Hoeffding's
    # lower bound to be positive with probability >=1-delta, not a sampled result.
    delta = .05
    sufficient_n = math.floor(2 * float((cap + cost) ** 2) * math.log(1 / delta) / float(clipped_mean ** 2)) + 1
    return {
        "gross_distribution": "G=sign*A; P(A=k)=2^-k for integers k>=1; independent fair sign",
        "gross_mean": number(ZERO), "gross_absolute_mean": number(F(2)),
        "payoff_support_bounded": False, "cost_per_trade": number(cost),
        "initial_equity": number(initial), "horizon": 2,
        "policy": "trade at first time then flat; terminal liability",
        "expected_booked": number(-cost), "expected_discounted_floor": number(expected_floor),
        "expected_charged_alpha0": number(expected_charged),
        "residual_lower_bound": number(-cost), "residual_upper_clip": number(cap),
        "exact_expected_clipped_residual": number(clipped_mean),
        "raw_charged_on_negative_sign": "-(1-gamma)*k-cost-gamma*(initial-cost), unbounded below",
        "raw_charged_on_positive_sign": "k-cost, unbounded above",
        "clipped_floor_expectation_at_cap1": number(clipped_floor_mean),
        "population_truncation_harmful_alpha_interval_upper": number(1 - cost / clipped_floor_mean),
        "true_fixed_policy_harmful_alpha_interval_upper": number(1 - cost / expected_floor),
        "illustrative_hoeffding_delta": delta,
        "sufficient_iid_confirmation_n_for_positive_lcb_with_probability_1_minus_delta": sufficient_n,
        "confirmation_samples_actually_drawn": 0,
        "scope": "Closed-form exact capability example. The sample-size bound is theoretical, not an empirical certificate or registered experiment.",
    }


def clock_compiler_examples():
    """Reuse one local path set: no repeated transition simulation by placement."""
    initial, cost, local_horizon = F(3, 2), F(1, 10), 2
    policy = "forced_prefix_then_positive_gate"
    local = [path(policy, signs, initial, cost, ZERO, "terminal")
             for signs in product((-1, 1), repeat=local_horizon)]
    probability, records = F(1, len(local)), []
    for outer_horizon in (4, 16, 48):
        delay = outer_horizon - local_horizon
        scale = GAMMA ** delay
        early, late = [], []
        for p in local:
            terminal_floor = scale * p["floor"]
            early_raw = p["booked"] + terminal_floor
            late_raw = scale * p["booked"] + terminal_floor
            early_residual = terminal_floor - p["cost"]
            late_residual = terminal_floor - scale * p["cost"]
            assert late_raw == scale * p["charged"]
            assert late_residual == scale * p["residual"]
            assert late_residual >= early_residual
            early.append((early_raw, early_residual, p["booked"]))
            late.append((late_raw, late_residual, scale * p["booked"]))
        early_means = [sum(p[k] for p in early) * probability for k in range(3)]
        late_means = [sum(p[k] for p in late) * probability for k in range(3)]
        assert late_means[0] >= early_means[0]
        assert late_means[0] > 0 and late_means[2] < 0
        records.append({
            "outer_horizon": outer_horizon, "active_episode_length": local_horizon,
            "latest_start_delay": delay, "scaling_factor": number(scale),
            "early_expected_charged": number(early_means[0]),
            "late_expected_charged": number(late_means[0]),
            "late_expected_booked": number(late_means[2]),
            "residual_late_dominates_early_on_every_coupled_path": True,
            "latest_raw_and_booked_equal_scaled_local_pathwise": True,
            "new_active_transition_evaluations_for_placement_compilation": 0,
        })
    assert F(records[-1]["early_expected_charged"]["exact"]) < 0
    return {
        "policy": policy, "local_initial_equity": number(initial), "cost": number(cost),
        "local_episode_length": local_horizon, "local_expected_charged": number(F(19, 800)),
        "local_fixed_policy_alpha_strict_upper": number(F(95, 693)),
        "outer_horizon_results": records,
        "scope": "Stationary IID active-step kernel, no wait information/idle costs/early termination, policy uses active local history and clock only. Does not optimize episode length or dominate all global-clock policies.",
        "no_edge_assumption_counterexample": {
            "gross": number(F(1)), "cost": number(F(1, 10)),
            "initial_equity": number(F(3, 2)), "outer_horizon": 2,
            "early_charged": number(F(9, 10)), "late_charged": number(GAMMA * F(9, 10)),
            "explanation": "Positive predictable gross drift; no default payout. Early placement strictly beats late, violating no-edge dominance.",
        },
    }


def get(scenarios, name, policy):
    return next(p for p in next(s for s in scenarios if s["name"] == name)["policies"] if p["policy"] == policy)


def main():
    scenarios = []
    for horizon in (2, 3):
        scenarios.append(scenario(f"floor_prefix_H{horizon}", F(7, 10), F(2, 5), horizon))
        scenarios.append(scenario(f"solvent_prefix_H{horizon}", F(3, 2), F(1, 10), horizon))
    scenarios += [
        scenario("terminal_solvent_prefix_H2", F(3, 2), F(1, 10), 2, accounting="terminal"),
        scenario("floor_prefix_H2_alpha_one_fifth", F(7, 10), F(2, 5), 2, alpha=F(1, 5)),
        scenario("floor_prefix_H3_alpha_one_fifth", F(7, 10), F(2, 5), 3, alpha=F(1, 5)),
        scenario("full_booking_H3", F(7, 10), F(2, 5), 3, alpha=F(1)),
        scenario("high_capital_H3", F(5), F(2, 5), 3),
    ]
    a = get(scenarios, "floor_prefix_H2", "forced_prefix_then_positive_gate")
    assert a["expected"]["charged"]["exact"] == "197/2000"
    assert a["expected"]["booked"]["exact"] == "-299/500"
    assert a["expected"]["floor"]["exact"] == "1393/2000"
    assert a["variance"]["charged"]["exact"] == "1182427/4000000"
    assert a["variance"]["residual"]["exact"] == "1954507/4000000"
    assert get(scenarios, "floor_prefix_H2", "constant_trade")["expected"]["charged"]["exact"] == "-299/4000"
    b = get(scenarios, "solvent_prefix_H2", "forced_prefix_then_positive_gate")
    assert b["expected"]["charged"]["exact"] == "19/800"
    assert get(scenarios, "solvent_prefix_H2", "forced_prefix_then_floor_gate")["expected"]["charged"]["exact"] == "-1/10"
    assert get(scenarios, "terminal_solvent_prefix_H2", "forced_prefix_then_positive_gate")["expected"] == b["expected"]
    for s in scenarios:
        if s["name"] in ("floor_prefix_H2", "floor_prefix_H3", "solvent_prefix_H2", "solvent_prefix_H3", "terminal_solvent_prefix_H2"):
            assert s["library_decision"] == "WITNESS"
            for p in s["policies"]:
                if p["policy"].startswith("one_action_at_"):
                    assert F(p["expected"]["charged"]["exact"]) < 0
        if s["name"] in ("floor_prefix_H2_alpha_one_fifth", "floor_prefix_H3_alpha_one_fifth", "full_booking_H3", "high_capital_H3"):
            assert s["library_decision"] == "ABSTAIN"
    result = {
        "status": "retrospective exploratory exact synthetic capability benchmark",
        "gamma": number(GAMMA), "gross_per_trade": "independent fair +1/-1",
        "action_space": "flat or a unit trade; no observation of current sign before action",
        "policy_gates_design_alpha": 0, "policy_gates_frozen_across_alpha_contrasts": True,
        "scenarios": scenarios, "unbounded_support_certificate_example": unbounded_terminal_example(),
        "terminal_clock_compiler": clock_compiler_examples(),
        "scope": [
            "Finite library evaluation by trajectory enumeration, not Bellman dynamic programming or an optimal policy claim.",
            "The examples were constructed after seeing the manuscript and are not new empirical confirmation.",
            "ABSTAIN means no positive harmful witness in this library, except that full booking and the stated high-capital bound have analytic class-wide no-edge guarantees.",
            "Martingale subtraction can increase variance; the first benchmark explicitly exhibits this failure.",
            "No market data, learner training, protected artifact or Study R outcome was read.",
        ],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (HERE / "exact_adaptive_benchmark.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"scenarios": [{"name": s["name"], "decision": s["library_decision"],
                                    "best": s["best_in_finite_library"],
                                    "adaptive_charged": get(scenarios, s["name"], "forced_prefix_then_positive_gate")["expected"]["charged"]}
                                   for s in scenarios],
                      "unbounded_clipped_mean": result["unbounded_support_certificate_example"]["exact_expected_clipped_residual"]}, indent=2))


if __name__ == "__main__":
    main()
