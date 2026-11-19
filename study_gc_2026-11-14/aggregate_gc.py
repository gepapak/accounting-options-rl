"""Study GC aggregation: the registered decisions GC1 to GC6 (PREREGISTRATION_GC.md), written and hashed before the run.

    python aggregate_gc.py [--smoke]
Outputs: results_gc/runs_gc.csv, results_gc/VERDICTS_GC.md
Statistics as in Study G: runs are the units; 95% percentile bootstrap, 10,000 resamples, conditions resampled independently.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

HERE = Path(__file__).resolve().parent
RES = HERE / "results_gc"
rng = np.random.default_rng(20261114)
EXPECTED = {("crypto", "ORIG"): 10, ("crypto", "FLPATCH"): 10, ("crypto", "LINE"): 10, ("forex", "ORIG"): 5, ("forex", "FLPATCH"): 5}


def boot_diff(a, b) -> tuple[float, float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a[rng.integers(0, len(a), (10_000, len(a)))].mean(axis=1) - b[rng.integers(0, len(b), (10_000, len(b)))].mean(axis=1)
    return float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def boot_mean(x) -> tuple[float, float]:
    x = np.asarray(x, float)
    m = x[rng.integers(0, len(x), (10_000, len(x)))].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def verdict(ok: bool) -> str:
    return "SUPPORTED" if ok else "NOT SUPPORTED"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    rows = [json.load(open(f)) for f in sorted(glob.glob(str(RES / ("jobs_smoke" if a.smoke else "jobs") / "*.json")))]
    R = pd.DataFrame(rows)
    for k in rows[0]["exposure"]:
        R[f"x{k}"] = R.exposure.apply(lambda d: d[k])
    R["xlow"] = R[["x0.02", "x0.05", "x0.1", "x0.25"]].mean(axis=1)
    R["gap"] = R.reported_return_pct - R.booked_return_pct
    for k in ("floor_steps", "episodes", "wiped_episodes", "floored_episodes", "line_max_abs_err"):
        R[f"train_{k}"] = R.train.apply(lambda d: d[k])

    def cell(market, cond):
        return R[(R.market == market) & (R.condition == cond)].sort_values("seed")

    o, f, ln = cell("crypto", "ORIG"), cell("crypto", "FLPATCH"), cell("crypto", "LINE")
    fo, ff = cell("forex", "ORIG"), cell("forex", "FLPATCH")
    out = ["runs: " + ", ".join(f"{m} {c} {len(cell(m, c))} (expected {n})" for (m, c), n in EXPECTED.items())]
    complete = all(len(cell(m, c)) == n for (m, c), n in EXPECTED.items())
    if not complete and not a.smoke:
        out.append("INCOMPLETE: the decisions below are provisional until every expected run is present")

    d1, (l1, h1) = o["x0.1"].mean() - f["x0.1"].mean(), boot_diff(o["x0.1"], f["x0.1"])
    out.append(f"GC1 (crypto) x_ORIG(0.1) - x_FLPATCH(0.1) = {d1:.3f} [{l1:.3f}, {h1:.3f}] (need lower bound > 0) -> {verdict(l1 > 0)}")
    d2, (l2, h2) = o.xlow.mean() - f.xlow.mean(), boot_diff(o.xlow, f.xlow)
    out.append(f"GC2 (crypto) mean x over e<=0.25, ORIG - FLPATCH = {d2:.3f} [{l2:.3f}, {h2:.3f}] (need lower bound > 0) -> {verdict(l2 > 0)}")
    l3, h3 = boot_mean(o.gap)
    out.append(f"GC3 (crypto) ORIG reported - booked = {o.gap.mean():.2f} pp [{l3:.2f}, {h3:.2f}] (need lower bound > 0) -> {verdict(l3 > 0)}")
    p4 = float(mannwhitneyu(o.xlow, ln.xlow, alternative="greater").pvalue) if len(o) and len(ln) else float("nan")
    tol = ln.xlow.mean() <= f.xlow.mean() + 0.10
    out.append(f"GC4 (crypto) mean x over e<=0.25: ORIG {o.xlow.mean():.3f}, LINE {ln.xlow.mean():.3f}, one-sided Mann-Whitney p = {p4:.4f} "
               f"(need < 0.05), and LINE at most FLPATCH + 0.10 = {f.xlow.mean() + 0.10:.3f} -> {verdict(p4 < 0.05 and tol)}")
    d5, (l5, h5) = o.gap.mean() - ln.gap.mean(), boot_diff(o.gap, ln.gap)
    out.append(f"GC5 (crypto) gap of ORIG - gap of LINE = {d5:.2f} pp [{l5:.2f}, {h5:.2f}] (need lower bound > 0); LINE gap {ln.gap.mean():.2f} pp "
               f"-> {verdict(l5 > 0)}")
    d6 = fo["x0.1"].mean() - ff["x0.1"].mean()
    out.append(f"GC6 (forex) x_ORIG(0.1) - x_FLPATCH(0.1) = {d6:.3f} (need absolute value at most 0.10), ORIG reported - booked = "
               f"{fo.gap.mean():.2f} pp (need at most 0.50) -> {verdict(abs(d6) <= 0.10 and fo.gap.mean() <= 0.50)}")
    e7 = max(float(R.reading_max_abs_err_usd.max()), float(ln.train_line_max_abs_err.max()) if len(ln) else 0.0)
    out.append(f"GC7 (both) largest error of the reading against the booked step profit, every test step of every agent and every LINE "
               f"training step: {e7:.3e} USD (need at most 0.001) -> {verdict(e7 <= 1e-3)}")
    out.append(f"descriptive: training steps on which the floor wrote, mean per run: crypto ORIG {o.train_floor_steps.mean():.0f}, "
               f"crypto LINE {ln.train_floor_steps.mean():.0f}, forex ORIG {fo.train_floor_steps.mean():.0f}; "
               f"x(0.1): crypto LINE {ln['x0.1'].mean():.3f} against ORIG {o['x0.1'].mean():.3f}")

    tag = "_SMOKE" if a.smoke else ""
    R.drop(columns=["exposure", "draws_reported", "draws_booked", "train", "versions"]).to_csv(RES / f"runs_gc{tag}.csv", index=False)
    pd.set_option("display.width", 220)
    cols = [c for c in R.columns if c.startswith("x")] + ["reported_return_pct", "booked_return_pct", "gap", "floored_share", "wiped_share",
                                                          "orders_per_episode", "max_leverage", "train_floor_steps"]
    table = R.groupby(["market", "condition"])[cols].mean().round(3).to_string()
    print(table)
    print("\n".join(out))
    (RES / f"VERDICTS_GC{tag}.md").write_text("# Study GC verdicts" + (" (SMOKE)" if a.smoke else "") + "\n\n" + "\n".join(f"- {x}" for x in out)
                                             + "\n\n```\n" + table + "\n```\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
