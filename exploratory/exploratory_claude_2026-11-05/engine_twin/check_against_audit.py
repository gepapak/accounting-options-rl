"""Gate: does each traced run reproduce the audit's run? Compares the evaluation JSON and the trace's own totals with the
campaign table, and, for the four runs whose step logs were released, the trace with that log step by step."""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / "github" / "accounting-options-rl"
TAB = pd.read_csv(REPO / "results" / "campaign" / "results_long.csv")
EXAMPLES = {
    "L0_strict__marl__seed7__original": "L0_strict_marl_seed7_DK1",
    "L3_price_taker_no_solvency__marl__seed7__original": "L3_marl_seed7_DK1",
    "L7_thesis_like__marl__seed7__original": "L7_marl_seed7_DK1",
    "L7_thesis_like__marl__seed42__original": "L7_marl_seed42_DK1",
}


def check(name: str) -> dict:
    variant, agent, seed, region = name.split("__")
    seed = int(seed.replace("seed", ""))
    js = sorted(glob.glob(str(HERE / "campaign_runs" / "evals" / name / "**" / "evaluation_tiers_*.json"), recursive=True))
    tier = json.load(open(js[-1], encoding="utf-8"))["tiers"]["tier1"]
    sleeve = float(tier["sleeve_metrics"]["sleeve_trading_return_pct"])
    z = np.load(HERE / "traces" / f"{name}.npz", allow_pickle=False)
    c = {str(n): z["rows"][:, i] for i, n in enumerate(z["columns"])}
    e0 = float(z["e0"])
    if variant.endswith("_capon"):  # not in the audit's table
        return dict(name=name, campaign_sleeve=float("nan"), json_sleeve=sleeve, json_minus_campaign=0.0,
                    trace_reported=float(c["r"].sum() * 100 / e0), steps=int(len(c["r"])), e0=e0, campaign_ledger=float("nan"))
    row = TAB[(TAB.variant == variant) & (TAB.agent == agent) & (TAB.seed == seed) & (TAB.region == region)].iloc[0]
    out = dict(name=name, campaign_sleeve=float(row.sleeve_return_pct), json_sleeve=sleeve,
               json_minus_campaign=sleeve - float(row.sleeve_return_pct),
               trace_reported=float(c["r"].sum() * 100 / e0), steps=int(len(c["r"])), e0=e0,
               campaign_ledger=float(row.investor_return_pct))
    if name in EXAMPLES:
        log = pd.read_csv(REPO / "results" / "ledger_examples" / f"{EXAMPLES[name]}.csv.gz")
        n = min(len(log), len(c["r"]))
        idx = log.timestep.values[:n].astype(int)
        sv = (np.maximum(c["cash_post"], 0) + c["mtm_post"])[idx]
        out["log_rows"] = int(n)
        out["log_max_abs_sleeve_diff"] = float(np.abs(sv - log.trading_sleeve_value_dkk.values[:n]).max())
        out["log_max_abs_gross_diff"] = float(np.abs(c["gross"][idx] - log.mtm_pnl.fillna(0).values[:n]).max())
        out["log_max_abs_dist_diff"] = float(np.abs(c["dist_post"][idx] - log.total_distributions_dkk.fillna(0).values[:n]).max())
    return out


def main() -> int:
    names = sys.argv[1:] or [p.stem for p in sorted((HERE / "traces").glob("*.npz"))]
    rows = [check(n) for n in names]
    worst = max(abs(r["json_minus_campaign"]) for r in rows)
    print(json.dumps(rows if len(rows) <= 6 else rows[:2], indent=1))
    print(f"{len(rows)} runs; largest |JSON - campaign| sleeve return: {worst:.3e} pp")
    return 0 if worst <= 1e-6 else 1


if __name__ == "__main__":
    raise SystemExit(main())
