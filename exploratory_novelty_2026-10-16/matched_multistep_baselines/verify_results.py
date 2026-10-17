"""Read-only independent arithmetic/integrity checks; no simulation or training."""
import hashlib
import json
import math
import os
from pathlib import Path
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    r = json.loads((HERE / "results.json").read_text())
    assert r["status"] == "EXPLORATORY_COMPLETE"
    assert r["all_fixed_confirmation_budgets_complete"]
    assert r["allocated_one_sided_lower_bounds"] == r["unique_policies"] == 9
    assert r["delta_per_bound"] == .01 / 9
    hashes = dict(r["input_sha256"])
    hashes[str((HERE / "matched_baselines.py").relative_to(ROOT))] = r["script_sha256"]
    hashes[str((HERE / "EXPLORATORY_PLAN.md").relative_to(ROOT))] = r["plan_sha256"]
    for path, expected in hashes.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
    panel = pd.read_pickle(ROOT / "GENERALIZATION_STUDY_2026-09-29/data_panel/panel_DK1.pkl")
    panel = panel.loc[panel.split == "train"].iloc[200:]
    spread = np.abs(panel.imb.to_numpy(float) - panel.spot.to_numpy(float))
    caps = .25 * .25 * panel.bal_mw.to_numpy(float)
    g, h, fee, cash = r["gamma"], r["horizon"], r["fee"], r["initial_ledger_eur"]
    scores = pd.read_csv(HERE / "cap_search_scores.csv")
    maximum_bound_error = maximum_exact_error = 0.
    positive = []
    for item in r["policy_results"]:
        p, b, summary = item["policy"], item["certificate"], item["summary"]
        assert b["n"] == summary["completed_confirmation_paths"] == 524288
        assert b["delta"] == .01 / 9
        cost_max = p["lev"] * .25 * fee * sum(g ** t for t in range(h - p["duration"], h))
        assert abs(cost_max - summary["cost_max"]) < 1e-10
        assert b["range_lower"] == -summary["cost_max"]
        assert b["range_upper"] == summary["cap"]
        log = math.log(2 / b["delta"])
        radius = math.sqrt(2 * b["variance"] * log / b["n"]) + 7 * (b["range_upper"] - b["range_lower"]) * log / (3 * (b["n"] - 1))
        maximum_bound_error = max(maximum_bound_error, abs(radius - b["radius"]), abs(b["mean"] - radius - b["lower"]))
        assert maximum_bound_error < 1e-11
        subset = scores.loc[scores.policy == p["name"]]
        assert len(subset) == 4
        winner = subset.loc[subset.projected_confirmation_lower.idxmax()]
        assert winner.cap == summary["cap"]
        q = np.minimum(.25 * p["lev"], caps)
        expected_cost = float((fee * q).mean())
        expected_booked = -expected_cost * (1 if p["kind"] == "adaptive" else sum(g ** t for t in range(h - p["duration"], h)))
        assert abs(expected_booked - summary["expected_booked_return_upper_or_exact"]) < 1e-10
        assert summary["positive_witness_confirmed"] == (b["lower"] > 0)
        if b["lower"] > 0:
            positive.append(p["name"])
        for field in ("raw", "residual", "booked", "gross", "cost", "forgiveness"):
            assert math.isfinite(summary["mean_" + field])
        assert abs(summary["mean_raw"] - summary["mean_residual"] - summary["mean_gross"]) < 1e-10
    direct = {}
    for lev in (.25, 1, 4, 16):
        q = np.minimum(.25 * lev, caps)
        c, a = fee * q, q * spread
        value = g ** (h - 1) * float((.5 * (np.maximum(c - a - cash, 0) + np.maximum(c + a - cash, 0)) - c).mean())
        direct[str(lev)] = value
        maximum_exact_error = max(maximum_exact_error, abs(value - r["exact_isolated_action_values"][str(lev)]))
    assert maximum_exact_error < 1e-10
    assert max(direct.values()) < 0
    assert positive == ["frozen_adaptive", "openloop16_T48"]
    assert r["aliases"] == {"openloop16_T96": "constant16_T96"}
    total_duration = sum(p["policy"]["duration"] for p in r["policy_results"])
    assert total_duration == 549
    assert r["validation_vectorized_transitions"] == 2 * total_duration
    assert r["search_active_transitions"] == 8192 * total_duration
    assert r["confirmation_active_transitions_including_partial"] == 524288 * total_duration
    assert r["actual_vectorized_transitions_all_phases"] == (2 + 8192 + 524288) * total_duration
    assert r["outer_confirmation_slots_if_all_policies_complete"] == 9 * 524288 * 96
    assert r["source_check"]["arm_path_checks"] == 36
    assert r["source_check"]["max_abs_error_eur"] < 1e-7
    assert r["source_check"]["frozen_environment_step_calls"] == 36 * 96
    print(json.dumps(dict(status="PASS", checked_source_hashes=len(hashes), complete_policy_bounds=9,
                          maximum_bound_recomputation_error=maximum_bound_error,
                          maximum_exact_one_action_recomputation_error=maximum_exact_error,
                          exact_one_action_direct_integration=direct, positive_certificates=positive,
                          conclusion="A simple delayed constant exposure is also a certified witness; adaptive feedback necessity is not established."), indent=2))


if __name__ == "__main__":
    main()
