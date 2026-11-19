"""EXPLORATORY design-only feasibility; no learner outcomes or DP inputs read.

Frozen before execution: fees .4/1.5, DK1/DK2, standard caps/no impact,
threshold1 reference, gamma .99, alpha grid, c=.366768%, 24x256 MC batches.
No R registration or RL training is created or launched.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MODEL_FREE = HERE.parent
FEES = (.4, 1.5)
ZONES = ("DK1", "DK2")
ALPHAS = (0., .25, .5, .7, .8, .9, .95, 1.)
C = .366768
HIGH, LOW = 1.5 * C, C / 1.5
BATCHES, PATHS, BOOTSTRAP = 24, 256, 2000
GAMMA, SEED = .99, 2026101562


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=HERE / "outputs")
    ap.add_argument("--max-wall-seconds", type=float, default=300)
    a = ap.parse_args()
    out = a.out.resolve()
    if not out.is_relative_to(HERE):
        raise SystemExit("Outputs must stay inside new_fee_design.")
    out.mkdir(parents=True, exist_ok=True)
    module_spec = importlib.util.spec_from_file_location("fixed_rollout_helpers", MODEL_FREE / "exploratory_rollout_surrogate.py")
    helper = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(helper)
    helper.np, helper.pd = np, pd
    started = time.perf_counter()
    rows, groups = [], []
    complete = True
    for fee in FEES:
        for zone in ZONES:
            # Only empirical TRAIN arrays are loaded; no conditions(), DP files,
            # original outcomes, or calibration-fitting function is called.
            mag, old_cost, _, source = helper.empirical_payoffs(zone, "base")
            cost = fee * old_cost / .1
            sigma = float(np.sqrt(np.mean(mag ** 2 + cost ** 2) - np.mean(cost) ** 2))
            source["payoff_sha256"] = hashlib.sha256(mag.tobytes() + cost.tobytes()).hexdigest()
            horizon, tail = helper.horizon_for("threshold1", GAMMA, float((mag + cost).max()), sigma, .01, 6000)
            group = f"{zone}_fee{fee:g}_threshold1".replace(".", "p")
            spec = dict(group=group, zone=zone, fee=fee, costs="new_fee_standard_cap_no_impact",
                        gamma=GAMMA, reference="threshold1", paths=PATHS, seed=SEED,
                        horizon=horizon, frozen_threshold_pct=C,
                        script_sha256=helper.sha(__file__), helper_script_sha256=helper.sha(helper.__file__), **source)
            checkpoint = out / f"exploratory_checkpoint_{group}.npz"
            means, seconds, counts, steps, active = [], [], [], [], []
            if checkpoint.exists():
                with np.load(checkpoint, allow_pickle=False) as saved:
                    if json.loads(str(saved["spec"])) != spec:
                        raise SystemExit("Incompatible checkpoint; preserve it and use a new output directory.")
                    means, seconds, counts, steps, active = [saved[k].tolist() for k in
                                                            ("means", "seconds", "counts", "steps", "active")]
            group_seed = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16)
            for batch in range(len(means), BATCHES):
                if time.perf_counter() - started >= a.max_wall_seconds:
                    complete = False
                    break
                before = time.perf_counter()
                estimate, count, step, survival = helper.batch_rollout(
                    mag, cost, GAMMA, "threshold1", horizon, PATHS,
                    np.random.default_rng([SEED, group_seed, batch]))
                means.append(estimate.tolist())
                seconds.append(time.perf_counter() - before)
                counts.append(count)
                steps.append(step)
                active.append(survival)
                helper.checkpoint(checkpoint, spec, means, seconds, counts, steps, active)
                print(f"{group} batch {batch + 1}/{BATCHES}: {seconds[-1]:.2f}s, {count} transitions", flush=True)
            if len(means) >= 2:
                b = np.asarray(means)
                rng = np.random.default_rng([SEED, group_seed, 991])
                boot = b[rng.integers(0, len(b), size=(BOOTSTRAP, len(b)))].mean(axis=1)
                for alpha in ALPHAS:
                    point = helper.index_pct((1 - alpha) * b[:, 0].mean(axis=0) + alpha * b[:, 1].mean(axis=0), sigma)
                    values = [helper.index_pct((1 - alpha) * v[0] + alpha * v[1], sigma) for v in boot]
                    lo, hi = [float(v) for v in np.quantile(values, [.025, .975])]
                    combined_lo, combined_hi = max(0., lo - tail), hi + tail
                    rows.append(dict(fee=fee, zone=zone, alpha=alpha, control=alpha == 1,
                                     surrogate_index_pct=point, mc_low_pct=lo, mc_high_pct=hi,
                                     truncation_bound_pct=tail, combined_low_pct=combined_lo,
                                     combined_high_pct=combined_hi, frozen_threshold_pct=C,
                                     high_boundary_pct=HIGH, low_boundary_pct=LOW,
                                     high_confident=combined_lo >= HIGH,
                                     low_confident=combined_hi <= LOW,
                                     sigma16_empirical=sigma, n_batches=len(b),
                                     antithetic_pairs_per_batch=PATHS, horizon=horizon,
                                     shared_rollout_group=group))
            groups.append(dict(**spec, completed_batches=len(means), simulation_seconds=float(sum(seconds)),
                               simulated_branch_transitions=int(sum(counts)),
                               deterministic_tail_bound_pct=tail, tail_target_met=tail <= .01000001))
            helper.write_csv(out / "exploratory_new_fee_indices.csv", rows)
            if not complete:
                break
        if not complete:
            break
    selections = []
    for fee in FEES:
        eligible_high, eligible_low = [], []
        for alpha in ALPHAS[:-1]:
            matched = [r for r in rows if r["fee"] == fee and r["alpha"] == alpha]
            all_zones = len(matched) == len(ZONES) and all(r["n_batches"] == BATCHES for r in matched)
            if all_zones and all(r["high_confident"] for r in matched):
                eligible_high.append(alpha)
            if all_zones and all(r["low_confident"] for r in matched):
                eligible_low.append(alpha)
        # Deterministic nearest-alpha rule, fixed before reading results.
        high_alpha = max(eligible_high) if eligible_high else None
        low_alpha = min(eligible_low) if eligible_low else None
        selections.append(dict(fee=fee, viable=high_alpha is not None and low_alpha is not None,
                               eligible_high_alphas=eligible_high, eligible_low_alphas=eligible_low,
                               selected_high_alpha=high_alpha, selected_low_alpha=low_alpha,
                               matched_control_alpha=1.,
                               selection_rule="Largest high-eligible treatment alpha and smallest low-eligible treatment alpha, each satisfying the fixed interval criterion in both zones. Alpha1 is control only."))
    result = dict(label="EXPLORATORY design-only; no registration or training", status="complete" if complete else "partial_cost_cap",
                  frozen_threshold_pct=C, high_boundary_pct=HIGH, low_boundary_pct=LOW,
                  fees=FEES, zones=ZONES, alpha_grid=ALPHAS, reference="threshold1", gamma=GAMMA,
                  batches=BATCHES, independent_antithetic_pairs_per_batch=PATHS,
                  selections=selections, all_fees_viable=complete and all(s["viable"] for s in selections),
                  elapsed_wall_seconds=time.perf_counter() - started,
                  total_simulated_branch_transitions=sum(g["simulated_branch_transitions"] for g in groups), groups=groups,
                  uncertainty="Marginal95% batch-bootstrap MC intervals plus deterministic discount tail, fixed market sample. Threshold uncertainty is not absorbed into the fixed bracketing rule.",
                  forbidden_inputs="No learner outcomes, DP outputs/solvers, new fee search, weaker criterion, or reference tuning used.")
    helper.atomic_json(out / "exploratory_new_fee_summary.json", result)
    print(json.dumps({k: result[k] for k in ("status", "all_fees_viable", "selections", "elapsed_wall_seconds", "total_simulated_branch_transitions")}, indent=2), flush=True)
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
