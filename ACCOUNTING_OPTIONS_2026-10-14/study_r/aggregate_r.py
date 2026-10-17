"""Study R's single registered composite primary test; refuse partial/duplicate data.

All other contrasts, bootstrap intervals, probe profiles and training counters are
descriptive/exploratory. Nothing overwrites registered verdicts or run tables.
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def analyze(records, predictions):
    data = pd.DataFrame([dict(fee=r["fee"], alpha=r["alpha"], seed=r["seed"], region=r["region"],
                              L002=r["lev"]["0.02"], L01=r["lev"]["0.1"], L1=r["lev"]["1"],
                              seconds=r["seconds"], actual_timesteps=r["actual_timesteps"]) for r in records])
    expected = {(f, a, s, z) for f in (.4, 1.5) for a in (0, .8, 1)
                for s in range(60, 65) for z in ("DK1", "DK2")}
    actual = set(zip(data.fee, data.alpha, data.seed, data.region))
    if len(data) != 60 or actual != expected:
        raise RuntimeError("Require exactly the 60 planned unique jobs; never analyze a partial cohort")
    arms = []
    for fee in (.4, 1.5):
        control = float(data[(data.fee == fee) & (data.alpha == 1)].L002.mean())
        for alpha in (0, .8, 1):
            group = data[(data.fee == fee) & (data.alpha == alpha)]
            indices = [p["index_pct"] for p in predictions["predictions"] if p["fee"] == fee and p["alpha"] == alpha]
            gap = float(group.L002.mean() - control)
            arms.append(dict(fee=fee, alpha=alpha, n=len(group), mean_L002=float(group.L002.mean()),
                             gap_over_matched_control=gap, max16_runs=int((group.L002 == 16).sum()),
                             mean_index_pct=float(np.mean(indices)),
                             predicted_learned=bool(np.mean(indices) >= predictions["threshold_pct"]),
                             observed_learned=bool(gap >= 4)))
    table = data.pivot(index=["seed", "region"], columns=["fee", "alpha"], values="L002")
    interaction = ((table[(.4, .8)] - table[(.4, 1.)]) - (table[(1.5, .8)] - table[(1.5, 1.)]))
    clusters = interaction.groupby(level="seed").mean().reindex(range(60, 65)).to_numpy()
    positives = int((clusters > 0).sum())
    p = sum(math.comb(5, j) for j in range(positives, 6)) / 32
    low = next(a["gap_over_matched_control"] for a in arms if a["fee"] == .4 and a["alpha"] == .8)
    high = next(a["gap_over_matched_control"] for a in arms if a["fee"] == 1.5 and a["alpha"] == .8)
    components = {"low_fee_gap_at_least_4": low >= 4, "high_fee_gap_below_4": high < 4,
                  "mean_interaction_at_least_4": float(clusters.mean()) >= 4,
                  "exact_one_sided_seed_sign_p_at_most_0p05": p <= .05}
    bootstrap = np.array([clusters[list(indices)].mean() for indices in itertools.product(range(5), repeat=5)])
    result = dict(n=60, primary_R1_supported=bool(all(components.values())), components=components,
                  low_fee_G=low, high_fee_G=high, mean_interaction=float(clusters.mean()),
                  paired_seed_interactions=dict(zip(map(str, range(60, 65)), map(float, clusters))),
                  positive_seed_clusters=positives, exact_one_sided_sign_p=p,
                  descriptive_paired_seed_bootstrap_95pct=np.quantile(bootstrap, [.025, .975]).tolist(),
                  arms=arms, inferential_target="seed-wise probability of a positive interaction; fixed two zones",
                  limitations="Operational cohort thresholds do not prove population equivalence/nonlearning. Single primary composite; anchors and bootstrap are descriptive.")
    return data, result


def main():
    from run_r import verify_registration, validate_completed
    registration = verify_registration()
    if (HERE / "VERDICTS_R.md").exists() or (HERE / "runs_r.csv").exists():
        raise RuntimeError("Frozen R outputs already exist; do not overwrite")
    paths = sorted((HERE / "results_r/jobs").glob("*.json"))
    records = [json.loads(p.read_text()) for p in paths]
    if any(r["registration_sha256"] != registration for r in records):
        raise RuntimeError("Mixed or changed registrations")
    for r in records:
        validate_completed(r, r["fee"], r["alpha"], r["seed"], r["region"], registration)
    predictions = json.loads((HERE / "predictions_r.json").read_text())
    data, result = analyze(records, predictions)
    data.to_csv(HERE / "runs_r.csv", index=False)
    (HERE / "analysis_r.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# Study R verdicts", "", "One internally hash-stamped composite primary decision; all 60 planned agents included.", "",
             "| ID | Decision | Verdict |", "|---|---|---|",
             f"| R1 | G(.4,.8)={result['low_fee_G']:.3f} >=4; G(1.5,.8)={result['high_fee_G']:.3f} <4; interaction={result['mean_interaction']:.3f} >=4; exact seed sign p={result['exact_one_sided_sign_p']:.5f} <=.05 | {'SUPPORTED' if result['primary_R1_supported'] else 'NOT SUPPORTED'} |",
             "", "## Every arm (descriptive except the R1 components)", "",
             "| Fee | alpha | index (%) | mean L(.02) | gap over matched control | max16 runs |", "|---|---|---|---|---|---|"]
    for a in result["arms"]:
        lines.append(f"| {a['fee']:g} | {a['alpha']:g} | {a['mean_index_pct']:.6f} | {a['mean_L002']:.3f} | {a['gap_over_matched_control']:.3f} | {a['max16_runs']}/10 |")
    lines += ["", f"Paired seed interactions: {result['paired_seed_interactions']}", "",
              "The sign test targets the probability of a positive seed-cluster interaction under independent seed clusters, not a population mean or generalization to markets. Zeros count as nonpositive.",
              "The G cutoffs are operational sample-cohort classifications, not equivalence confidence intervals. All prior retrospective failures and calibration uncertainty remain relevant.", ""]
    (HERE / "VERDICTS_R.md").write_text("\n".join(lines), encoding="utf-8")
    from datetime import datetime, timezone
    entry = f"\n### Completion and unchanged registered aggregation\n\nUTC {datetime.now(timezone.utc).isoformat(timespec='seconds')}: all 60 jobs complete; registered aggregator executed unchanged. R1 {'SUPPORTED' if result['primary_R1_supported'] else 'NOT SUPPORTED'}, seed sign p={result['exact_one_sided_sign_p']:.5f}. No selective stopping or reruns based on outcomes.\n"
    with (HERE / "RUN_LOG_R.md").open("ab") as stream:
        stream.write(entry.encode("utf-8"))
    print("\n".join(lines), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
