"""EXPLORATORY: exact bounded matched-moment accounting counterexample.

No learner, market data, Monte Carlo, DP or acquisition threshold is used.
Run this file to reproduce matched_moments.json in the same directory.
"""
from __future__ import annotations

from fractions import Fraction as F
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
EQUITY, COST, ALPHA = F(1), F(1, 1000), F(4, 5)
AMPLITUDES = {
    "A": [(F(1, 4), F(1, 2)), (F(3, 4), F(1, 2))],
    "B": [(F(1, 4), F(1, 5)), (F(1, 2), F(3, 4)), (F(3, 2), F(1, 20))],
}


def number(x: F) -> dict[str, str | float]:
    return {"exact": str(x), "decimal": float(x)}


def evaluate(amplitudes):
    assert sum(p for _, p in amplitudes) == 1
    # Draw an independent fair sign for every magnitude.
    pmf = {}
    for amplitude, probability in amplitudes:
        for sign in (-1, 1):
            gross = sign * amplitude
            pmf[gross] = pmf.get(gross, F(0)) + probability / 2
    assert sum(pmf.values()) == 1
    mean = sum(g * p for g, p in pmf.items())
    second = sum(g * g * p for g, p in pmf.items())
    abs_mean = sum(abs(g) * p for g, p in pmf.items())
    booked = sum((g - COST) * p for g, p in pmf.items())
    net_abs_mean = sum(abs(g - COST) * p for g, p in pmf.items())
    injection = sum(max(-EQUITY - (g - COST), F(0)) * p for g, p in pmf.items())
    charged = sum(((g - COST) + (1 - ALPHA) * max(-EQUITY - (g - COST), F(0))) * p
                  for g, p in pmf.items())
    account = sum((max(EQUITY + g - COST, F(0)) - EQUITY) * p for g, p in pmf.items())
    assert mean == 0 and booked == -COST
    assert charged == -COST + (1 - ALPHA) * injection
    assert account == booked + injection
    variance = second - mean * mean
    net_variance = sum(((g - COST) - booked) ** 2 * p for g, p in pmf.items())
    assert net_variance == variance
    return {
        "amplitude_distribution": [dict(amplitude=number(a), probability=number(p)) for a, p in amplitudes],
        "gross_pmf": [dict(gross_payoff=number(g), probability=number(p)) for g, p in sorted(pmf.items())],
        "gross_mean": number(mean), "gross_variance": number(variance),
        "gross_absolute_mean": number(abs_mean), "expected_executed_cost": number(COST),
        "net_mean": number(booked), "net_variance": number(net_variance),
        "net_absolute_mean": number(net_abs_mean),
        "expected_injection": number(injection), "charged_one_step_reward": number(charged),
        "unadjusted_account_reward": number(account), "positive_charged_witness": charged > 0,
        "fully_booked_action_reward": number(booked), "fully_booked_gated_witness_value": number(F(0)),
    }


def main() -> int:
    records = {name: evaluate(pmf) for name, pmf in AMPLITUDES.items()}
    matched = ("gross_mean", "gross_variance", "gross_absolute_mean", "expected_executed_cost", "net_mean", "net_variance", "net_absolute_mean")
    for field in matched:
        assert records["A"][field] == records["B"][field], field
    assert records["A"]["charged_one_step_reward"]["exact"] == "-1/1000"
    assert records["B"]["charged_one_step_reward"]["exact"] == "301/200000"
    assert records["A"]["expected_injection"]["exact"] == "0"
    assert records["B"]["expected_injection"]["exact"] == "501/40000"
    assert not records["A"]["positive_charged_witness"] and records["B"]["positive_charged_witness"]
    result = {
        "status": "exploratory exact mathematical counterexample",
        "initial_equity": number(EQUITY), "alpha": number(ALPHA),
        "executed_position": number(F(1)), "nominal_fee_per_unit": number(COST),
        "common_gross_absolute_bound": number(F(3, 2)),
        "matched_fields": list(matched), "markets": records,
        "no_learning_prediction": True, "new_agents_trained": 0, "dp_solved": False,
        "scope": [
            "Same dose, cost, equity, gross mean, variance and gross/net absolute means do not determine the one-step accounting incentive.",
            "A supplies no positive witness for this specified one-action policy at this equity; it is not certified safe against multi-step policies.",
            "No claim is made about which learner acquires either incentive, or about empirical predictive superiority.",
            "This is an application of the classical stop-loss transform, not a new probability theorem.",
        ],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (HERE / "matched_moments.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"matched": {f: records["A"][f] for f in matched},
                      "charged_reward_A": records["A"]["charged_one_step_reward"],
                      "charged_reward_B": records["B"]["charged_one_step_reward"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
