"""Exploratory post-run MC audit; never selects a reference or refits on O/P/Q.

Read complete model-free output and batch checkpoints. Bootstrap the shared alpha
contrasts and calibration midpoint jointly. Outcomes/market panels stay fixed, so
these are MC-sensitivity intervals, not population or prospective validation CIs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROBE = np.asarray((.02, .1, .25, .5, 1, 1.5, 2))
LOG_GRID = np.linspace(np.log(.05), np.log(2), 400)


def key(zone, costs, gamma, alpha):
    return f"{zone}|{costs}|g{gamma:g}|a{alpha:g}"


def index_rows(gaps, sigma):
    return np.asarray([100 * np.maximum(np.interp(LOG_GRID, np.log(PROBE), g), 0).mean() / sigma
                       for g in gaps])


def interval(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return [float(v) for v in np.quantile(values, [.025, .5, .975])] if len(values) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=HERE / "outputs")
    ap.add_argument("--bootstrap", type=int, default=2000)
    a = ap.parse_args()
    p = a.out.resolve()
    if not p.is_relative_to(HERE):
        raise SystemExit("Audit output must remain inside model_free.")
    summary = json.loads((p / "exploratory_summary.json").read_text())
    if summary["status"] != "complete":
        raise SystemExit("Run is incomplete; interpret its coverage/cost failure before this audit.")
    d = pd.read_csv(p / "exploratory_indices.csv")
    primary = d[d.reference == "threshold1"].copy()
    draws = {}
    rng = np.random.default_rng(2026101561)
    for group, rows in primary.groupby("shared_rollout_group", sort=True):
        with np.load(p / f"exploratory_checkpoint_{group}.npz", allow_pickle=False) as f:
            b = f["means"]
        pick = rng.integers(0, len(b), size=(a.bootstrap, len(b)))
        means = b[pick].mean(axis=1)
        sigma = float(rows.sigma16_empirical.iloc[0])
        for row in rows.itertuples():
            gaps = (1 - row.alpha) * means[:, 0] + row.alpha * means[:, 1]
            draws[row.condition_id] = index_rows(gaps, sigma)
    cal = json.loads((ROOT / "ACCOUNTING_OPTIONS_2026-10-14/study_o/calibration_o.json").read_text())["rows"]
    thresholds = []
    threshold_draws = {}
    for name, selected in (("original_10_conditions_revised_FI_included", cal),
                           ("sensitivity_9_conditions_FI_excluded", [r for r in cal if r["zone"] != "FI"])):
        learned, unlearned, learned_point, unlearned_point = [], [], [], []
        for r in selected:
            costs = "H" if r["condition"] == "Study H costs" else "base"
            gamma = .99 if costs == "H" else float(r["condition"].split()[-1])
            ident = key(r["zone"], costs, gamma, 0)
            (learned if r["learned"] else unlearned).append(draws[ident])
            point = float(primary.loc[primary.condition_id == ident, "surrogate_index_pct"].iloc[0])
            (learned_point if r["learned"] else unlearned_point).append(point)
        lo_learned, hi_unlearned = np.min(learned, axis=0), np.max(unlearned, axis=0)
        valid = (hi_unlearned > 0) & (hi_unlearned < lo_learned)
        mid = np.full(a.bootstrap, np.nan)
        mid[valid] = np.sqrt(hi_unlearned[valid] * lo_learned[valid])
        threshold_draws[name] = mid
        h, l = max(unlearned_point), min(learned_point)
        thresholds.append(dict(calibration=name, n_conditions=len(selected),
                               nominal_threshold_pct=math.sqrt(h * l) if 0 < h < l else None,
                               bootstrap_positive_separation_fraction=float(valid.mean()),
                               threshold_MC_interval_pct_conditional_on_separation=interval(mid),
                               bootstrap_nonseparating_or_zero_endpoint_count=int((~valid).sum()),
                               n_bootstrap=a.bootstrap,
                               warning="Conditional interval excludes invalid calibration draws; outcome labels and market sample fixed."))
    controls = []
    original_name = thresholds[0]["calibration"]
    midpoint = threshold_draws[original_name]
    valid = np.isfinite(midpoint)
    for row in primary[(primary.alpha == 1) & primary.zone.isin(["DK1", "DK2"])].itertuples():
        v = draws[row.condition_id]
        controls.append(dict(zone=row.zone, dp_index_pct=row.dp_index_pct,
                             surrogate_index_pct=row.surrogate_index_pct,
                             mc_interval_pct=interval(v),
                             point_predicts_control_exploit=bool(row.surrogate_index_pct >= thresholds[0]["nominal_threshold_pct"])
                             if thresholds[0]["nominal_threshold_pct"] is not None else None,
                             bootstrap_control_above_threshold_fraction_conditional_on_calibration=
                             float(np.mean(v[valid] >= midpoint[valid])) if valid.any() else None,
                             interpretation="Positive Q_pi can reflect avoiding future reference-policy fees; it is not evidence of an optimal exploit in the fully booked control."))
    arm_uncertainty = []
    for alpha in (0., .25, .5, .7, .8, .9, .95):
        values = (draws[key("DK1", "base", .99, alpha)] + draws[key("DK2", "base", .99, alpha)]) / 2
        margin = values[valid] - midpoint[valid]
        ci = interval(margin)
        arm_uncertainty.append(dict(alpha=alpha,
                                    probability_above_threshold_MC_only=float(np.mean(margin >= 0)),
                                    mean_zone_index_minus_calibration_midpoint_MC_interval_pct=ci,
                                    interval_excludes_zero=bool(ci[0] > 0 or ci[2] < 0),
                                    interpretation="Joint batch bootstrap preserves shared alpha/probe/calibration randomness; excludes market/outcome uncertainty and deterministic tail allowance."))
    bias = []
    for ref, rows in d.groupby("reference"):
        err = rows.surrogate_index_pct - rows.dp_index_pct
        bias.append(dict(reference=ref, mean_signed_error_pct=float(err.mean()),
                         mean_absolute_error_pct=float(err.abs().mean()),
                         rmse_pct=float(np.sqrt(np.mean(err ** 2))),
                         median_mc_interval_width_pct=float((rows.mc_high_pct - rows.mc_low_pct).median()),
                         max_mc_interval_width_pct=float((rows.mc_high_pct - rows.mc_low_pct).max()),
                         comparison_caveat="Q_pi vs Q*, empirical vs binned noise/dynamics, and revised FI source; not pure estimator bias."))
    result = dict(label="EXPLORATORY Monte Carlo sensitivity audit", thresholds=thresholds,
                  alpha1_controls=controls, primary_O_arm_joint_MC_uncertainty=arm_uncertainty, absolute_discrepancy=bias,
                  source_manifest=d[["zone", "costs", "sigma16_empirical", "sigma_empirical_over_binned",
                                     "empirical_panel_registration_status"]].drop_duplicates().to_dict("records"),
                  caveat="No population/generalization confidence claim. Bootstrap batches preserve alpha/probe dependence; existing outcomes are fixed. No reference selected and no new study launched.")
    (p / "exploratory_uncertainty_audit.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
