"""Study O calibration (run before registration): the learnability index of every completed PPO condition with equity as
the only input and a matched full-liability control (Studies F, H, J, K), and the threshold c* that Study O tests.

Index: dp_o.learnability_index applied to the action gaps of LIABILITY_STUDY_2026-09-29/results_k/action_gap.csv.
Outcome: learned gap G = mean |lev| at 2% equity of the floor-trained agents minus that of the full-liability agents.
Learned: G >= 4. c* = geometric mean of the largest index among unlearned conditions and the smallest among learned ones.

    python calibrate_o.py      # writes calibration_o.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dp_o import PROBE, learnability_index

HERE = Path(__file__).resolve().parent
LIAB = HERE.parents[1] / "LIABILITY_STUDY_2026-09-29"
LEARNED_GAP = 4.0


def main() -> int:
    AG = pd.read_csv(LIAB / "results_k" / "action_gap.csv")
    F = pd.read_csv(LIAB / "results_f" / "runs_f.csv")
    K = pd.read_csv(LIAB / "results_k" / "runs_k.csv")
    J = pd.read_csv(LIAB / "results_j" / "runs_j.csv")
    H = pd.read_csv(LIAB / "results_h" / "runs_h.csv")
    sets = {}
    for z in ("DK1", "DK2"):
        f, k = F[(F.region == z) & (F.obsset == "EQ")], K[(K.region == z) & (K.gamma == 0.99)]
        sets[(z, "gamma 0.99")] = (pd.concat([f[f.rule == "ZF"], k[k.rule == "ZF"]]), pd.concat([f[f.rule == "FL"], k[k.rule == "FL"]]))
        for g in (0.95, 0.995):          # Study K trained full-liability controls only at these discount factors
            k = K[(K.region == z) & (K.gamma == g)]
            sets[(z, f"gamma {g}")] = (k[k.rule == "ZF"], k[k.rule == "FL"])
        h = H[(H.region == z) & (H.obsset == "EQ")]
        sets[(z, "Study H costs")] = (h[h.rule == "ZF"], h[h.rule == "FL"])
    for z in ("FI", "NO2"):
        j = J[(J.region == z) & (J.obsset == "EQ")]
        sets[(z, "gamma 0.99")] = (j[j.rule == "ZF"], j[j.rule == "FL"])
    rows = []
    for (z, cond), (zf, fl) in sets.items():
        g = AG[(AG.zone == z) & (AG.condition == cond)].iloc[0]
        A = np.array([g[f"A_{e:g}"] for e in PROBE])
        gap = float(zf["L0.02"].mean() - fl["L0.02"].mean())
        rows.append(dict(zone=z, condition=cond, index_pct=learnability_index(A, g.sigma16), gap_002=gap,
                         learned=bool(gap >= LEARNED_GAP), n_zf=len(zf), n_fl=len(fl)))
    D = pd.DataFrame(rows)
    hi_unlearned = float(D[~D.learned].index_pct.max())
    lo_learned = float(D[D.learned].index_pct.min())
    c_star = float(np.sqrt(hi_unlearned * lo_learned))
    out = dict(rows=rows, largest_unlearned_index_pct=hi_unlearned, smallest_learned_index_pct=lo_learned, c_star_pct=c_star,
               separated=bool(hi_unlearned < lo_learned))
    (HERE / "calibration_o.json").write_text(json.dumps(out, indent=1))
    print(D.round(3).to_string(index=False))
    print(f"largest unlearned {hi_unlearned:.3f}%, smallest learned {lo_learned:.3f}%, c* = {c_star:.3f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
