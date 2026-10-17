"""Study S analysis (PREREGISTRATION_S.md): writes runs_s_A.csv, runs_s_B.csv and VERDICTS_S.md.

    python aggregate_s.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
RES = HERE / "results_s"
B_BOOT = 10_000
rng = np.random.default_rng(20261017)
LOW_E = ("0", "0.02", "0.05", "0.1")
KL_ARMS = ("KL0.003", "KL0.03", "KL0.3")


def load_a():
    rows = []
    for p in sorted((RES / "A").glob("*.json")):
        d = json.loads(p.read_text())
        low = [abs(d["lev"][f"{e}|{z}|{rem}"]) == 16.0 for e in LOW_E for z in ("-1", "+1") for rem in ("0.05", "0.5")]
        rows.append(dict(arm=d["arm"], seed=d["seed"], X_A=float(np.mean(low)), E=d["test_edge"]["booked_pct_K0"],
                         E_reported=d["test_edge"]["reported_pct_K0"], placebo_reported=d["test_placebo"]["reported_pct_K0"],
                         placebo_forgiven=d["test_placebo"]["forgiven_pct_K0"]))
    return pd.DataFrame(rows)


def load_b():
    rows = []
    for p in sorted((RES / "B").glob("*.json")):
        d = json.loads(p.read_text())
        rows.append(dict(learner=d["learner"], arm=d["arm"], seed=d["seed"], region=d["region"], L002=d["lev"]["0.02"], L2=d["lev"]["2"],
                         X_B=d["lev"]["0.02"] - d["lev"]["2"], reported=d["drawmean_reported_return_pct"],
                         ledger=d["drawmean_ledger_return_pct"]))
    return pd.DataFrame(rows)


def mw_less(a, b):
    """One-sided Mann-Whitney p for a < b."""
    return float(stats.mannwhitneyu(a, b, alternative="less").pvalue)


def verdict(ok):
    return "SUPPORTED" if ok else "NOT SUPPORTED"


def part_a(A):
    out = ["## Part A: edge market (PPO, DK1, seeds 100-109)", "",
           "| Arm | runs | mean X_A (share of low-equity probes at leverage 16) | mean E (booked % K0 / episode) | placebo reported | placebo forgiven |",
           "|---|---|---|---|---|---|"]
    g = {arm: A[A.arm == arm] for arm in A.arm.unique()}
    for arm in ("PLAIN", "ORACLE", "SPR") + KL_ARMS:
        if arm in g:
            x = g[arm]
            out.append(f"| {arm} | {len(x)} | {x.X_A.mean():.3f} | {x.E.mean():+.2f} | {x.placebo_reported.mean():+.2f} | {x.placebo_forgiven.mean():.2f} |")
    P, O, S = g["PLAIN"], g["ORACLE"], g["SPR"]
    v = []
    p1 = mw_less(O.X_A, P.X_A)
    v.append(("SA1", f"X_A PLAIN {P.X_A.mean():.3f} > ORACLE {O.X_A.mean():.3f}: one-sided Mann-Whitney p = {p1:.4f} (need < 0.05)", p1 < 0.05))
    p2 = mw_less(S.X_A, P.X_A)
    ok2 = p2 < 0.05 and S.X_A.mean() <= O.X_A.mean() + 0.10
    v.append(("SA2", f"X_A SPR {S.X_A.mean():.3f} < PLAIN: p = {p2:.4f} (need < 0.05); SPR <= ORACLE + 0.10 = {O.X_A.mean() + 0.10:.3f}", ok2))
    eo, es = O.E.to_numpy(), S.E.to_numpy()
    ratio = es.mean() / eo.mean() if eo.mean() > 0 else float("nan")
    boot = np.array([rng.choice(es, len(es)).mean() / rng.choice(eo, len(eo)).mean() for _ in range(B_BOOT)])
    lb = float(np.quantile(boot, 0.05))
    ok3 = eo.mean() > 0 and ratio >= 0.80 and lb >= 0.60
    v.append(("SA3", f"E SPR {es.mean():+.2f} / ORACLE {eo.mean():+.2f} = {ratio:.3f} (need >= 0.80); one-sided 95% bootstrap LB {lb:.3f} (need >= 0.60)", ok3))
    both = []
    for arm in KL_ARMS:
        if arm in g:
            k = g[arm]
            both.append((arm, k.X_A.mean() <= O.X_A.mean() + 0.10 and k.E.mean() >= 0.80 * eo.mean(), k.X_A.mean(), k.E.mean()))
    ok4 = len(both) == len(KL_ARMS) and not any(b[1] for b in both)
    v.append(("SA4", "; ".join(f"{a}: X_A {x:.3f}, E {e:+.2f} -> {'both' if ok else 'not both'}" for a, ok, x, e in both)
              + f" (both = X_A <= {O.X_A.mean() + 0.10:.3f} and E >= {0.80 * eo.mean():+.2f})", ok4))
    return out, v


def part_b(B):
    out = ["## Part B: registered fair-market environment (Study O), seeds 110-119, DK1 and DK2", "",
           "| Learner | Arm | runs | runs at 16 at equity 0.02 | mean X_B = L(0.02) - L(2) |", "|---|---|---|---|---|"]
    v = []
    for ln in ("PPO", "A2C-n64"):
        b = B[B.learner == ln]
        for arm in ("ZF", "SPR", "LEDGER"):
            x = b[b.arm == arm]
            out.append(f"| {ln} | {arm} | {len(x)} | {int((x.L002 == 16).sum())} | {x.X_B.mean():+.2f} |")
        w = b.pivot_table(index=["seed", "region"], columns="arm", values="X_B").dropna()
        d1 = (w["ZF"] - w["SPR"]).to_numpy()
        p1 = float(stats.wilcoxon(d1, alternative="greater").pvalue) if np.any(d1 != 0) else 1.0
        v.append((f"SB1 {ln}", f"paired ZF - SPR (n = {len(d1)}): mean {d1.mean():+.2f} (need >= 4), one-sided Wilcoxon p = {p1:.4f} (need < 0.05)",
                  p1 < 0.05 and d1.mean() >= 4))
        d2 = (w["SPR"] - w["LEDGER"]).to_numpy()
        boot = np.array([rng.choice(d2, len(d2)).mean() for _ in range(B_BOOT)])
        ub = float(np.quantile(boot, 0.95))
        v.append((f"SB2 {ln}", f"paired SPR - LEDGER (n = {len(d2)}): mean {d2.mean():+.2f}, one-sided 95% bootstrap UB {ub:+.2f} (need < 5)", ub < 5))
    return out, v


def main():
    A, B = load_a(), load_b()
    lines = ["# Study S verdicts", "", f"Runs: Part A {len(A)} of 60, Part B {len(B)} of 120.", ""]
    verdicts = []
    if len(A):
        A.to_csv(HERE / "runs_s_A.csv", index=False)
        o, v = part_a(A)
        lines += o + [""]
        verdicts += v
    if len(B):
        B.to_csv(HERE / "runs_s_B.csv", index=False)
        o, v = part_b(B)
        lines += o + [""]
        verdicts += v
    lines += ["| ID | Result | Verdict |", "|---|---|---|"] + [f"| {i} | {r} | {verdict(ok)} |" for i, r, ok in verdicts]
    text = "\n".join(lines) + "\n"
    (HERE / "VERDICTS_S.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
