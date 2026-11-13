"""Summaries of the NFL pilot (results/jobs) and stage 2 (results2/jobs). EXPLORATORY.

    python analyze_nfl.py            # both, if present
Per arm and accounting rule (seed means, with the spread across seeds):
    edge booked     booked P&L in the edge market, % of K0 per 7-day episode (legitimate performance; higher is better)
    pl reported     reported P&L in the placebo market (positive = profit where none is possible)
    pl forgiven     loss forgiven by the accounting in the placebo market
    max16 low       share of low-equity probes (equity <= 0.02 K0, or negative) at maximal leverage
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def load(folder):
    rows = []
    for p in sorted((HERE / folder / "jobs").glob("*.json")):
        d = json.loads(p.read_text())
        low = [abs(v) == 16.0 for k, v in d["lev"].items() if float(k.split("|")[0]) <= 0.1]
        hi = [v for k, v in d["lev"].items() if float(k.split("|")[0]) == 1.0]
        rows.append(dict(arm=d["arm"], acc=d["accounting"], seed=d["seed"], region=d["region"], algo=d.get("algo", "PPO"),
                         ins=d.get("p_insolvent", 0.0),
                         edge_booked=d["test_edge"]["booked_pct_K0"], edge_reported=d["test_edge"]["reported_pct_K0"],
                         pl_reported=d["test_placebo"]["reported_pct_K0"], pl_forgiven=d["test_placebo"]["forgiven_pct_K0"],
                         max16_low=float(np.mean(low)), lev_hi=" ".join(f"{v:g}" for v in hi), sec=d["seconds"]))
    return pd.DataFrame(rows)


def summary(df):
    g = df.groupby(["acc", "ins", "algo", "region", "arm"])
    out = g.agg(n=("seed", "size"), edge_booked=("edge_booked", "mean"), edge_sd=("edge_booked", "std"),
                pl_reported=("pl_reported", "mean"), pl_forgiven=("pl_forgiven", "mean"), max16_low=("max16_low", "mean"))
    return out.round(2)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    for folder in ("results", "results2"):
        df = load(folder)
        if df.empty:
            continue
        print(f"=== {folder}: {len(df)} runs")
        print(df.sort_values(["acc", "arm", "seed"]).to_string(index=False))
        print()
        print(summary(df).to_string())
        print()
