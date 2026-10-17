"""EXPLORATORY diagnostics of completed Study Q; writes only beside this file.

Refuses incomplete Q. Does not launch training or aggregation, and does not
alter registered hypotheses. Greedy probe summaries cannot recover Q-value
gaps, training visitation, or a saved critic that was never serialized.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from statistical_audit import cluster_gap, o_arm_label_permutation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
Q = ROOT / "ACCOUNTING_OPTIONS_2026-10-14/study_q"
LEARNERS = ("A2C-n64", "DQN-default")
ALPHAS = (0, .5, .8, .9, 1)


def main():
    # Check the frozen verdict first: do not inspect partial outcome summaries.
    if not (Q / "VERDICTS_Q.md").is_file() or not (Q / "runs_q.csv").is_file():
        raise SystemExit("Study Q is not aggregated and complete: no diagnostics executed.")
    paths = sorted((Q / "results_q/jobs").glob("*.json"))
    if len(paths) != 100:
        raise SystemExit(f"Study Q has {len(paths)} job files; exactly 100 required. No diagnostics executed.")
    records = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    identity = {(r["learner"], r["alpha"], r["seed"], r["region"]) for r in records}
    expected = {(l, a, s, z) for l in LEARNERS for a in ALPHAS for s in range(50, 55) for z in ("DK1", "DK2")}
    if identity != expected:
        raise SystemExit("Completed job identities differ from the registered design. No diagnostics executed.")
    rows = []
    for r in records:
        lev = r["lev"]
        values = list(lev.values())
        rows.append({"learner": r["learner"], "alpha": r["alpha"], "seed": r["seed"], "region": r["region"],
            "L002": lev["0.02"], "L2": lev["2"], "mean_probe_leverage": float(np.mean(values)),
            "maximal_at_every_probe": bool(all(v == 16 for v in values)),
            "same_leverage_at_every_probe": bool(len(set(values)) == 1),
            "number_distinct_probe_leverages": len(set(values)),
            "low_exceeds_high": bool(lev["0.02"] > lev["2"]),
            "floor_bound_fraction_of_eval_draws": r["drawshare_floor_bound"],
            "evaluation_mean_abs_leverage_open": r["drawmean_mean_abs_lev_open"],
            "reported_pct": r["drawmean_reported_return_pct"], "booked_pct": r["drawmean_ledger_return_pct"]})
    D = pd.DataFrame(rows)
    summary = []
    for (learner, alpha), d in D.groupby(["learner", "alpha"]):
        control = D[(D.learner == learner) & (D.alpha == 1)]
        summary.append({"learner": learner, "alpha": alpha, "n": len(d),
            "L002_mean": float(d.L002.mean()), "L002_n16": int((d.L002 == 16).sum()),
            "L2_mean": float(d.L2.mean()), "L2_n16": int((d.L2 == 16).sum()),
            "L002_gap_over_control": float(d.L002.mean()-control.L002.mean()),
            "maximal_at_every_probe": int(d.maximal_at_every_probe.sum()),
            "same_leverage_at_every_probe": int(d.same_leverage_at_every_probe.sum()),
            "low_exceeds_high": int(d.low_exceeds_high.sum()),
            "mean_floor_bound_fraction_of_eval_draws": float(d.floor_bound_fraction_of_eval_draws.mean()),
            "mean_eval_abs_leverage_open": float(d.evaluation_mean_abs_leverage_open.mean()),
            "reported_pct": float(d.reported_pct.mean()), "booked_pct": float(d.booked_pct.mean()),
            "by_zone": {z: {"L002_mean": float(g.L002.mean()), "L2_mean": float(g.L2.mean())}
                        for z, g in d.groupby("region")}})
    keys = sorted(set().union(*(set(r) for r in records)))
    suspicious = [k for k in keys if any(w in k.lower() for w in ("visit", "occupan", "q_value", "critic", "weight", "checkpoint"))]
    registered_csv = pd.read_csv(Q / "runs_q.csv")
    sensitivity = {}
    for learner, d in registered_csv.groupby("learner"):
        t = d[d.alpha != 1]
        y = (t.L002 == 16).to_numpy()
        def auc(score):
            values = np.asarray(score)
            return float(stats.mannwhitneyu(values[y], values[~y]).statistic/(y.sum()*(~y).sum()))
        sensitivity[learner] = {
            "Q1_common_arm_label_sensitivity": o_arm_label_permutation(d),
            "Q2_paired_seed_sensitivity": cluster_gap(d, 0, 1, "alpha", "L002"),
            "treatment_only_ranking": {"n": len(t), "learned_at_max_probe": int(y.sum()),
                 "auc_index": auc(t["index"]), "auc_one_minus_alpha": auc(1-t.alpha),
                 "caveat": "Exploratory ranking only; no transferable dose threshold is calibrated."}}
    out = {"status": "EXPLORATORY, completed Q only; registered verdicts unchanged",
           "cohort": len(D), "serialized_top_level_keys": keys,
           "possible_visitation_critic_checkpoint_fields": suspicious,
           "limitations": [
             "No training-state visit counts, trajectories, Q-values, network weights or checkpoints are serialized by run_q.py:93-100.",
             "Training visitation of 2%-equity probes and DQN Q-value gaps cannot be recovered from these summaries.",
             "drawshare_floor_bound is the share of evaluation draws in which the floor bound, not the fraction of training steps at low equity.",
             "Greedy absolute leverage loses action direction and value confidence; identical probe choices do not imply identical Q-values.",
             "High control leverage can explain a small treatment-control gap, but this diagnostic does not change Q1-Q3 or identify a causal training failure."],
           "groups": summary,
           "exploratory_dependence_and_dose_sensitivity": sensitivity,
           "input_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in paths + [Q / "VERDICTS_Q.md", Q / "runs_q.csv", Q / "run_q.py"]}}
    D.to_csv(HERE / "exploratory_completed_q_probe_runs.csv", index=False)
    (HERE / "exploratory_completed_q_diagnostics.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(summary).drop(columns="by_zone").to_string(index=False))
    print(json.dumps(sensitivity, indent=2))
    print("No training visitation or Q-value gaps can be recovered: runner saves greedy probe decisions only.")


if __name__ == "__main__":
    main()
