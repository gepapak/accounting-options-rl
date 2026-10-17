"""Freeze Study R's exact empirical one-step diagnostic, before any R training.

The numerical heuristic was developed exploratorily after O/P/Q. The ten-condition
historical calibration and all prior failures are disclosed in the registration.
This standalone prediction code does not read any R agent outputs.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROBES = np.array([.02, .1, .25, .5, 1, 1.5, 2])
GRID = np.exp(np.linspace(np.log(.05), np.log(2), 400))
INPUTS = {}


def tracked(path):
    INPUTS[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def rewards(zone, alpha, fee=.1, high_cost=False):
    folder = ("GENERALIZATION_STUDY_2026-09-29/data_panel" if zone in ("DK1", "DK2")
              else "LIABILITY_STUDY_2026-09-29/data_panel_j")
    panel = pd.read_pickle(tracked(ROOT / folder / f"panel_{zone}.pkl"))
    panel = panel.loc[panel.split == "train"].iloc[200:]
    cap = .25 * .25 * panel.bal_mw.to_numpy(float)
    if high_cost:
        cap *= .1 / .25
    q = np.minimum(4, cap)
    z = q * np.abs(panel.imb.to_numpy(float) - panel.spot.to_numpy(float))
    cost = fee * q if not high_cost else q + 50 * q * q / np.maximum(cap, .05)
    # Direct exact integration; only 7 probes and 400 reference equities.
    losses = np.sort(np.r_[cost - z, cost + z])
    suffix = np.r_[np.cumsum(losses[::-1])[::-1], 0.]
    e = 20000 * PROBES
    i = np.searchsorted(losses, e, side="right")
    h = (suffix[i] - (len(losses) - i) * e) / len(losses)
    gap = (1 - alpha) * h - cost.mean()
    sigma = np.sqrt(np.mean(z * z + cost * cost) - cost.mean() ** 2)
    index = 100 * np.maximum(np.interp(np.log(GRID), np.log(PROBES), gap), 0).mean() / sigma
    return dict(index_pct=float(index), gap_at_2pct_eur=float(gap[0]),
                mean_cost_eur=float(cost.mean()), sigma16_eur=float(sigma),
                gaps_eur=gap.tolist(), empirical_train_rows=len(panel))


def main():
    destination = HERE / "predictions_r.json"
    if destination.exists():
        raise RuntimeError("Predictions already exist; do not overwrite frozen predictions")
    old = json.loads(tracked(ROOT / "ACCOUNTING_OPTIONS_2026-10-14/study_o/calibration_o.json").read_text())
    calibration = []
    for c in old["rows"]:
        record = rewards(c["zone"], 0, high_cost=c["condition"] == "Study H costs")
        calibration.append(dict(zone=c["zone"], condition=c["condition"],
                                learned=bool(c["learned"]), **record))
    high = max(c["index_pct"] for c in calibration if not c["learned"])
    low = min(c["index_pct"] for c in calibration if c["learned"])
    threshold = float(np.sqrt(high * low))
    assert 0 < high < low
    predictions = []
    for fee in (.4, 1.5):
        for alpha in (0, .8, 1):
            for zone in ("DK1", "DK2"):
                d = rewards(zone, alpha, fee)
                predictions.append(dict(fee=fee, alpha=alpha, zone=zone, **d,
                    predicts_learned=bool(d["index_pct"] >= threshold),
                    confident_high=bool(d["index_pct"] >= 1.5 * threshold),
                    confident_low=bool(d["index_pct"] <= threshold / 1.5)))
    same = [p for p in predictions if p["alpha"] == .8]
    assert all(p["gap_at_2pct_eur"] > 0 for p in same), "One-step exploit must remain feasible at both fees"
    assert all(p["confident_high"] if p["fee"] == .4 else p["confident_low"] for p in same)
    value = dict(threshold_pct=threshold, calibration_largest_unlearned_pct=high,
                 calibration_smallest_learned_pct=low, calibration=calibration, predictions=predictions,
                 input_sha256=INPUTS,
                 limitations=["Exact conditional on empirical TRAIN rows, not future-market certainty.",
                              "Myopic contrast, not Q-star or a generic algorithmic complexity result.",
                              "Geometric midpoint is a chosen heuristic in a wide separating interval.",
                              "FI calibration panel is disclosed REVISED; exclusion sensitivity is exploratory.",
                              "R condition design uses prior predictor calculations, never R learner outcomes."])
    destination.write_text(json.dumps(value, indent=2), encoding="utf-8")
    print(json.dumps({"threshold_pct": threshold, "predictions": same}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
