"""Exploratory outputs of Study X (not registered; written after the verdicts): EXPLORATORY_X.md from runs_x.csv.

    python explore_x.py [--runs runs_x.csv] [--out EXPLORATORY_X.md]

Booked edge from zero equity, the placebo-market reported and forgiven returns, the TIR - SPR difference in booked edge with a bootstrap interval,
the number of gambling agents per arm, and the audit errors of gambling and non-gambling agents (Study V's single-floor values for comparison:
largest TIR error 0.385 point in the edge market and 0.129 in the placebo market; median reported-return error of the gamblers 68 points).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ARMS = ("PLAIN", "ORACLE", "SPR", "TIR")
GAMBLER = 0.25
rng = np.random.default_rng(20261103)


def boot_diff(a, b, n=10_000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = np.array([rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)])
    return float(a.mean() - b.mean()), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(HERE / "runs_x.csv"))
    ap.add_argument("--out", default=str(HERE / "EXPLORATORY_X.md"))
    a = ap.parse_args()
    df = pd.read_csv(a.runs)
    L = ["# Study X exploratory outputs (not registered)", "", f"Runs: {len(df)}.", ""]
    L += ["| arm | runs | mean X | agents X>0 | gamblers (X>=0.25) | min E | median E | max E | mean E from zero equity | placebo reported | placebo forgiven |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        x = df[df.arm == arm]
        if len(x):
            L.append(f"| {arm} | {len(x)} | {x.X.mean():.3f} | {int((x.X > 0).sum())} | {int((x.X >= GAMBLER).sum())} | {x.E.min():+.2f} | {x.E.median():+.2f} | "
                     f"{x.E.max():+.2f} | {x.Ez.mean():+.2f} | {x.placebo_reported.mean():+.2f} | {x.placebo_forgiven.mean():.2f} |")
    T, S, O = (df[df.arm == k] for k in ("TIR", "SPR", "ORACLE"))
    if len(T) and len(S):
        m, lo, hi = boot_diff(T.E, S.E)
        L += ["", f"Booked edge, TIR minus SPR: {m:+.2f} points, 95% bootstrap interval [{lo:+.2f}, {hi:+.2f}] (percent of the allocation per week; 10,000 resamples of runs)."]
    if len(T) and len(O):
        m, lo, hi = boot_diff(T.E, O.E)
        L += [f"Booked edge, TIR minus ledger: {m:+.2f} [{lo:+.2f}, {hi:+.2f}]."]
    if len(S) and len(O):
        m, lo, hi = boot_diff(S.E, O.E)
        L += [f"Booked edge, SPR minus ledger: {m:+.2f} [{lo:+.2f}, {hi:+.2f}]."]
    L += ["", "## Audit errors (absolute, points of the allocation per week; the mean over 100 test weeks of each agent)", "",
          "| group | agents | estimate | median error, edge | max error, edge | median error, placebo | max error, placebo |", "|---|---|---|---|---|---|---|"]
    groups = (("gamblers (X>=0.25)", df[df.X >= GAMBLER]), ("non-gamblers", df[df.X < GAMBLER]))
    for name, g in groups:
        for est, lab in (("reported", "reported"), ("spr", "SPR"), ("tir", "TIR")):
            if len(g):
                L.append(f"| {name} | {len(g)} | {lab} | {g[f'err_{est}_edge'].median():.3f} | {g[f'err_{est}_edge'].max():.3f} | "
                         f"{g[f'err_{est}_plac'].median():.3f} | {g[f'err_{est}_plac'].max():.3f} |")
    L += ["", "Study V's single-floor audit (128 agents): largest TIR error 0.385 point (edge market) and 0.129 (placebo market); median reported-return error 68.25 points for the 28 gamblers."]
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
