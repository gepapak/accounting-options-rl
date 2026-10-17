"""EXPLORATORY (not pre-registered): does the learnability index predict learning better than simpler predictors?

Runs: the equity-only PPO runs of the calibration conditions (Studies F, H, J, K: floor-trained runs) and the treatment
runs of Study O. Outcome: the run chooses leverage 16 at 2% equity. Predictors, all computed before training from the
dynamic program:
    index        the learnability index (dp_o.learnability_index)
    region       the optimal gambling region (largest equity below which leverage 16 is optimal)
    local_0.02   the action gap at 2% equity over the reward noise, A(0.02)/sigma16
    local_0.1    the action gap at 10% equity over the reward noise, A(0.1)/sigma16
Reported: AUC of each predictor, and the accuracy of a threshold fitted on the calibration runs and applied to Study O.

    python baselines_o.py      # writes baselines_o.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from dp_o import PROBE, learnability_index

HERE = Path(__file__).resolve().parent
LIAB = HERE.parents[1] / "LIABILITY_STUDY_2026-09-29"


def auc(score: np.ndarray, y: np.ndarray) -> float:
    pos, neg = score[y == 1], score[y == 0]
    return float(stats.mannwhitneyu(pos, neg, alternative="greater").statistic / (len(pos) * len(neg)))


def calibration_runs() -> pd.DataFrame:
    AG = pd.read_csv(LIAB / "results_k" / "action_gap.csv")
    F, K = pd.read_csv(LIAB / "results_f/runs_f.csv"), pd.read_csv(LIAB / "results_k/runs_k.csv")
    J, H = pd.read_csv(LIAB / "results_j/runs_j.csv"), pd.read_csv(LIAB / "results_h/runs_h.csv")
    sets = []
    for z in ("DK1", "DK2"):
        sets.append((z, "gamma 0.99", pd.concat([F[(F.region == z) & (F.obsset == "EQ") & (F.rule == "ZF")],
                                                 K[(K.region == z) & (K.gamma == 0.99) & (K.rule == "ZF")]])))
        for g in (0.95, 0.98, 0.995):
            sets.append((z, f"gamma {g}", K[(K.region == z) & (K.gamma == g) & (K.rule == "ZF")]))
        sets.append((z, "Study H costs", H[(H.region == z) & (H.obsset == "EQ") & (H.rule == "ZF")]))
    for z in ("FI", "NO2"):
        sets.append((z, "gamma 0.99", J[(J.region == z) & (J.obsset == "EQ") & (J.rule == "ZF")]))
    rows = []
    for z, cond, runs in sets:
        g = AG[(AG.zone == z) & (AG.condition == cond)].iloc[0]
        A = np.array([g[f"A_{e:g}"] for e in PROBE])
        for v in runs["L0.02"]:
            rows.append(dict(set="calibration", zone=z, condition=cond, learned=int(v == 16), index=learnability_index(A, g.sigma16),
                             region=g.threshold_K, local_002=g["A_0.02"] / g.sigma16, local_01=g["A_0.1"] / g.sigma16))
    return pd.DataFrame(rows)


def study_o_runs() -> pd.DataFrame:
    dp = {z: json.loads((HERE / f"dp_o_{z}.json").read_text()) for z in ("DK1", "DK2")}
    rows = []
    for p in sorted((HERE / "results_o" / "jobs").glob("*.json")):
        r = json.loads(p.read_text())
        if r["alpha"] == 1.0:
            continue
        arm = dp[r["region"]]["arms"][f"{r['alpha']:g}"]
        s = dp[r["region"]]["sigma16"]
        rows.append(dict(set="Study O", zone=r["region"], condition=f"alpha {r['alpha']:g}", learned=int(r["lev"]["0.02"] == 16),
                         index=arm["index_pct"], region=arm["region16_K"], local_002=arm["A"][0] / s, local_01=arm["A"][1] / s))
    return pd.DataFrame(rows)


def best_threshold(x: np.ndarray, y: np.ndarray) -> float:
    """Threshold maximizing accuracy of 'learned iff x >= t' (midpoints between sorted values)."""
    xs = np.unique(x)
    cands = np.concatenate([[xs[0] - 1], (xs[:-1] + xs[1:]) / 2, [xs[-1] + 1]])
    acc = [np.mean((x >= t) == y) for t in cands]
    return float(cands[int(np.argmax(acc))])


def main() -> int:
    C, O = calibration_runs(), study_o_runs()
    D = pd.concat([C, O], ignore_index=True)
    out = []
    for name in ("index", "region", "local_002", "local_01"):
        t = best_threshold(C[name].to_numpy(), C.learned.to_numpy())
        out.append(dict(predictor=name, auc_calibration=auc(C[name].to_numpy(), C.learned.to_numpy()),
                        auc_study_O=auc(O[name].to_numpy(), O.learned.to_numpy()),
                        auc_pooled=auc(D[name].to_numpy(), D.learned.to_numpy()),
                        threshold_from_calibration=t, accuracy_on_study_O=float(np.mean((O[name] >= t) == O.learned))))
    R = pd.DataFrame(out)
    R.to_csv(HERE / "baselines_o.csv", index=False)
    print(f"calibration runs {len(C)} (learned {C.learned.sum()}), Study O treatment runs {len(O)} (learned {O.learned.sum()})")
    print(R.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
