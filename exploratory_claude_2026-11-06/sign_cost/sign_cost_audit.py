"""Exploratory sign-cost audit of the existing engine traces; no training.

The auditor uses reported reward, a lower bound on net own cost, and a
certificate that gross payoff is nonpositive. It never uses a twin reward.
The evaluation adapter derives direction from logged positions, prices and the
known payoff mode/maturity schedule, not recorded gross magnitudes. Costs still
come from instrumented traces; this is not a deployed black-box audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def upper_bound(
    reported: NDArray[np.float64],
    cost_lower: NDArray[np.float64],
    nonpositive_gross: NDArray[np.bool_],
) -> NDArray[np.float64]:
    return np.where(nonpositive_gross, np.minimum(reported, -cost_lower), reported)


def nonpositive_from_observables(
    columns: dict[str, NDArray[np.float64]], mode: str,
) -> NDArray[np.bool_]:
    spot = columns["spot"]
    if mode == "percent":
        previous = np.r_[spot[0], spot[:-1]]
        direction = np.sign(columns["position_pre"]) * np.sign(spot - previous)
    elif mode == "horizon":
        direction = np.zeros(len(spot))
        decision = np.arange(6, len(spot) - 6, 6)
        direction[decision + 6] = (
            np.sign(columns["position_post"][decision])
            * np.sign(columns["settlement"][decision] - columns["entry"][decision])
        )
    else:
        raise ValueError(f"Unknown payoff mode: {mode}")
    return direction <= 0


def self_test() -> dict[str, int]:
    rng = np.random.default_rng(20261106)
    n = 200_000
    gross = rng.normal(0, 100, n)
    cost = rng.uniform(-20, 100, n)
    lower_cost = cost - rng.uniform(0, 20, n)
    credit = rng.uniform(0, 200, n)
    booked = gross - cost
    reported = booked + credit
    known_nonpositive = (gross <= 0) & (rng.random(n) > 0.3)
    bound = upper_bound(reported, lower_cost, known_nonpositive)
    assert np.all(booked <= bound + 1e-10)
    assert np.all(reported - bound >= -1e-10)
    assert np.all(reported - bound <= credit + 1e-10)

    witness_gross = bound + lower_cost
    witness_credit = reported - bound
    assert np.all(witness_gross[known_nonpositive] <= 1e-10)
    assert np.all(witness_credit >= -1e-10)
    np.testing.assert_allclose(
        witness_gross - lower_cost + witness_credit, reported, atol=1e-10
    )

    exact_upper = upper_bound(reported, cost, gross <= 0)
    twin_credit = rng.uniform(0, 200, n)
    twin = -gross - cost + twin_credit
    twin_lower = np.where(gross >= 0, np.maximum(-twin - 2 * cost, -cost),
                          -twin - 2 * cost)
    assert np.all(twin_lower <= booked + 1e-10)
    assert np.all(twin_lower <= exact_upper + 1e-10)
    witness_at_upper = exact_upper + cost
    witness_twin_credit = twin + witness_at_upper + cost
    assert np.all(witness_twin_credit >= -1e-10)

    grid_cases = 0
    for lower in range(-2, 4):
        for known in (False, True):
            for reward in range(-5, 6):
                feasible = [
                    g - c for c in range(lower, lower + 5) for g in range(-20, 21)
                    if (not known or g <= 0) and g - c <= reward
                ]
                u = upper_bound(np.array([float(reward)]), np.array([float(lower)]),
                                np.array([known]))[0]
                assert max(feasible) == u
                grid_cases += 1
    twin_grid_cases = 0
    for c in range(4):
        for sigma in (-1, 1):
            for reward in range(-5, 11):
                for reflected in range(-5, 11):
                    feasible = [
                        sigma * g - c for g in range(21)
                        if reward >= sigma * g - c and reflected >= -sigma * g - c
                    ]
                    if not feasible:
                        continue
                    u = min(reward, -c) if sigma < 0 else reward
                    l = max(-reflected - 2 * c, -c) if sigma > 0 else -reflected - 2 * c
                    assert min(feasible) == l and max(feasible) == u
                    twin_grid_cases += 1

    # These are assumption failures, not cases the auditor can safely repair.
    wrong_sign_bound = upper_bound(np.array([9.]), np.array([1.]), np.array([True]))
    assert 9 > wrong_sign_bound[0]
    overstated_cost = upper_bound(np.array([-2.]), np.array([5.]), np.array([True]))
    assert -2 > overstated_cost[0]
    withheld_profit = upper_bound(np.array([0.]), np.array([1.]), np.array([False]))
    assert 9 > withheld_profit[0]

    no_cost_reported = np.array([10., 0.])
    no_cost_upper = upper_bound(no_cost_reported, np.zeros(2), np.array([False, True]))
    np.testing.assert_array_equal(no_cost_upper, no_cost_reported)
    assert no_cost_reported[1] - (-10.) == 10
    corrected_upper = upper_bound(np.array([9., 0.]), np.ones(2), np.array([False, True]))
    assert corrected_upper.mean() == 4 and np.mean([9., -11.]) == -1
    return {"random_worlds": n, "assumption_counterexamples": 3,
            "detection_and_training_counterexamples": 2,
            "enumerated_sign_cost_cases": grid_cases,
            "enumerated_twin_cases": twin_grid_cases}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def floor_bound_names(manifest: Path) -> set[str]:
    names: set[str] = set()
    bound_section = False
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            bound_section = "floor bound" in line
        elif line.strip() and bound_section:
            names.add(line.strip())
    return names


def analyze(campaign: Path, output: Path) -> dict[str, object]:
    tests = self_test()
    source = campaign / "results" / "engine_twin_runs.csv"
    original = pd.read_csv(source)
    assert len(original) == 115 and original.name.is_unique
    bound_names = floor_bound_names(campaign / "JOBS.txt")
    assert len(bound_names) == 54
    rows = []
    trace_hashes = {}
    source_direction_failures = 0
    for index, row in enumerate(original.itertuples(), 1):
        path = campaign / "traces" / f"{row.name}.npz"
        trace_hashes[row.name] = digest(path)
        with np.load(path, allow_pickle=False) as z:
            data = z["rows"]
            c = {str(name): data[:, i] for i, name in enumerate(z["columns"])}
            scale = 100.0 / float(z["e0"])
            cost = c["access"] + c["roll"] + c["transaction"] + c["funding"] - c["battery"]
            gross, reported = c["gross"], c["r"]
            booked = gross - cost
            nonpositive = nonpositive_from_observables(c, row.mode)
            invalid_direction = nonpositive & (gross > 1e-3)
            source_direction_failures += int(invalid_direction.sum())
            assert not invalid_direction.any(), row.name
            upper = upper_bound(reported, cost, nonpositive)
            oracle_upper = upper_bound(reported, cost, gross <= 0)
            lower = -c["r_twin"] - 2 * cost
            sign_lower = np.where(gross >= 0, np.maximum(lower, -cost), lower)
            assert np.all(booked <= upper + 1e-3), row.name
            assert np.all(booked >= sign_lower - 1e-3), row.name
            np.testing.assert_allclose(
                [reported.sum() * scale, booked.sum() * scale],
                [row.reported_pct, row.booked_pct], atol=1e-7, rtol=0
            )
            minimum_credit = (reported - upper).sum() * scale
            rows.append({
                "name": row.name, "variant": row.variant, "agent": row.agent,
                "seed": row.seed, "region": row.region, "steps": len(reported),
                "bound": row.name in bound_names,
                "cap_on": row.variant.endswith("_capon"),
                "reported_pct": row.reported_pct, "booked_pct": row.booked_pct,
                "gross_pct": gross.sum() * scale, "net_cost_pct": cost.sum() * scale,
                "no_twin_upper_pct": upper.sum() * scale,
                "minimum_real_credit_pp": minimum_credit,
                "oracle_sign_minimum_credit_pp": (reported - oracle_upper).sum() * scale,
                "twin_lower_pct": lower.sum() * scale,
                "sign_twin_lower_pct": sign_lower.sum() * scale,
                "flag_sign_cost": minimum_credit > 5,
                "flag_tir_gap": row.flag_tir_gap,
                "flag_ledger": row.flag_reconcile,
            })
        if index % 25 == 0:
            print(f"Analyzed {index}/115 traces", flush=True)

    d = pd.DataFrame(rows)
    a = d[~d.cap_on]
    b, n = a[a.bound], a[~a.bound]
    false_profit = a[(a.reported_pct > 0) & (a.booked_pct < 0)]
    flag_names = set(a[a.flag_sign_cost].name)
    l3 = a[(a.variant == "L3_price_taker_no_solvency") & (a.agent == "marl")]
    six = a[(a.variant == "L7_minus_percent_mtm") & (a.agent == "marl")]
    summary: dict[str, object] = {
        "status": "exploratory post-hoc; no training; logged price/position and instrumented cost inputs",
        "auditor_inputs": ["reported reward", "cost lower bound", "known nonpositive gross"],
        "direction_adapter": "price/position signs; known payoff mode and six-step horizon schedule",
        "direction_assumption_failures_over_0_001_dkk": source_direction_failures,
        "validation_inputs": "recorded gross is validation only; costs remain instrumented",
        "self_tests": tests,
        "source_table_sha256": digest(source),
        "script_sha256": digest(Path(__file__)),
        "runs": len(a), "steps": int(a.steps.sum()),
        "floor_bound": len(b), "comparison": len(n), "cap_on": int(d.cap_on.sum()),
        "sign_cost_flags_bound": int(b.flag_sign_cost.sum()),
        "sign_cost_flags_comparison": int(n.flag_sign_cost.sum()),
        "same_flags_as_tir": flag_names == set(a[a.flag_tir_gap].name),
        "same_flags_as_ledger": flag_names == set(a[a.flag_ledger].name),
        "positive_reported_negative_booked_runs": len(false_profit),
        "false_profits_certified_negative_without_twin":
            int((false_profit.no_twin_upper_pct < 0).sum()),
        "positive_booked_runs": int((a.booked_pct > 0).sum()),
        "positive_twin_lower_runs": int((a.twin_lower_pct > 0).sum()),
        "positive_sign_twin_lower_runs": int((a.sign_twin_lower_pct > 0).sum()),
        "l3_runs": len(l3), "l3_negative_upper": int((l3.no_twin_upper_pct < 0).sum()),
        "six_run_headline_runs": len(six),
        "six_run_headline_negative_upper": int((six.no_twin_upper_pct < 0).sum()),
        "cap_on_flags": int(d[d.cap_on].flag_sign_cost.sum()),
        "minimum_credit_pp_floor_bound_median": float(b.minimum_real_credit_pp.median()),
        "flagged_names": sorted(flag_names),
    }
    output.mkdir(parents=True, exist_ok=True)
    d.to_csv(output / "sign_cost_runs.csv", index=False, lineterminator="\n")
    (output / "sign_cost_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "input_trace_sha256.json").write_text(
        json.dumps(trace_hashes, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign", type=Path,
                    default=Path(__file__).resolve().parent.parent / "engine_twin_2026-11-05")
    ap.add_argument("--output", type=Path,
                    default=Path(__file__).resolve().parent / "results")
    ap.add_argument("--self-test-only", action="store_true")
    args = ap.parse_args()
    result = self_test() if args.self_test_only else analyze(args.campaign, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "flagged_names"},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
