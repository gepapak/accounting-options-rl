r"""Post-hoc: do the traces show the mechanisms the text gives for the two point estimates' errors?

Volume payoff: on a step with a non-zero gross and an empty account, one branch is floored and SPR's magnitude |r - r'|/2 is about half
the gross. Percent payoff: TIR's large errors should sit on steps where a position was partly closed with an accumulated loss in
its bucket, so that both branches are credited.
Writes results/mechanisms.json (from every trace in traces/); --print prints it and writes nothing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LARGE = 0.01  # points of the initial sleeve


def load(path):
    z = np.load(path, allow_pickle=False)
    c = {str(n): z["rows"][:, i] for i, n in enumerate(z["columns"])}
    own = c["access"] + c["roll"] + c["transaction"] + c["funding"] - c["battery"]
    r, rt, g = c["r"], c["r_twin"], c["gross"]
    x = g - own
    differs = np.abs(r - rt) > 1e-6
    sign = np.where(g >= 0, 1.0, -1.0)
    tir = np.where(differs, sign * (np.maximum(r, rt) + own) - own, -own)
    spr = (r - rt) / 2 - own
    return c, own, r, rt, g, x, tir, spr, 100.0 / float(z["e0"])


def main() -> int:
    vol, pct = [], []
    for p in sorted((HERE / "traces").glob("*.npz")):
        variant = p.stem.split("__")[0]
        c, own, r, rt, g, x, tir, spr, pp = load(p)
        if "_capon" in variant or variant.startswith(("L0", "L5", "L6", "L4", "L2")):
            continue
        percent = variant.startswith(("L7", "L8")) and variant != "L7_minus_percent_mtm"
        if not percent:
            empty = (c["cash_pre"] <= 1e-6) & (np.abs(g) * pp > LARGE)
            if empty.any():
                vol.append(dict(run=p.stem, steps=int(empty.sum()),
                                spr_over_gross=float(np.median(np.abs((r - rt)[empty] / 2) / np.abs(g[empty]))),
                                tir_over_gross=float(np.median((np.maximum(r, rt)[empty] + own[empty]) / np.abs(g[empty])))))
        else:
            big = np.abs(tir - x) * pp > LARGE
            bucket_fell = np.abs(c["mtm_post"] - (c["mtm_pre"] + g)) * pp > 1e-9  # part of the bucket moved to cash
            both = ((r - x) * pp > LARGE) & ((rt + g + own) * pp > LARGE)  # credit on both branches
            cl = bucket_fell & (c["n_floor_real"] > 0)
            if big.any():
                pct.append(dict(run=p.stem, large_error_steps=int(big.sum()), on_closure=float(np.mean(cl[big])), on_both_credited=float(np.mean(both[big])),
                                share_of_abs_error_on_closure=float(np.abs(tir - x)[cl].sum() / np.abs(tir - x).sum()),
                                share_of_abs_spr_error_on_closure=float(np.abs(spr - x)[cl].sum() / max(np.abs(spr - x).sum(), 1e-9))))
    out = dict(
        volume_runs=len(vol), volume_empty_steps=int(sum(v["steps"] for v in vol)),
        volume_median_spr_over_gross=float(np.median([v["spr_over_gross"] for v in vol])),
        volume_min_spr_over_gross=float(min(v["spr_over_gross"] for v in vol)), volume_max_spr_over_gross=float(max(v["spr_over_gross"] for v in vol)),
        volume_median_tir_over_gross=float(np.median([v["tir_over_gross"] for v in vol])),
        percent_runs=len(pct), percent_large_error_steps=int(sum(v["large_error_steps"] for v in pct)),
        percent_median_share_on_closure=float(np.median([v["on_closure"] for v in pct])),
        percent_median_share_both_credited=float(np.median([v["on_both_credited"] for v in pct])),
        percent_median_abs_error_share_on_closure=float(np.median([v["share_of_abs_error_on_closure"] for v in pct])),
        percent_min_abs_error_share_on_closure=float(min(v["share_of_abs_error_on_closure"] for v in pct)),
    )
    print(json.dumps(out, indent=1))
    if "--print" not in sys.argv:
        (HERE / "results" / "mechanisms.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
