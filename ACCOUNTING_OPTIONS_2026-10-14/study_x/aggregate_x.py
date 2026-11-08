"""Study X analysis (PREREGISTRATION_X.md): writes runs_x.csv and VERDICTS_X.md.

    python aggregate_x.py                             # registered analysis of results_x/*.json
    python aggregate_x.py --results DIR --print       # descriptive summary of another folder (pilot); writes nothing
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
B_BOOT = 10_000
rng = np.random.default_rng(20261102)
LOW_E = ("0", "0.02", "0.05", "0.1")
ARMS = ("PLAIN", "ORACLE", "SPR", "TIR")
AUDIT_BOUND = 1.0          # points of the allocation per episode (XB1)
GAMBLER_X = 0.25


def load(folder: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(folder.glob("*.json")):
        d = json.loads(p.read_text())
        low = [abs(d["lev"][f"{e}|{z}|{r}"]) == 16.0 for e in LOW_E for z in ("-1", "+1") for r in ("0.05", "0.5")]
        row = dict(setting=d["setting"], arm=d["arm"], seed=d["seed"], X=float(np.mean(low)), E=d["test_edge"]["booked_pct_K0"],
                   Ez=d["test_zero"]["booked_pct_K0"], placebo_reported=d["test_placebo"]["reported_pct_K0"],
                   placebo_forgiven=d["test_placebo"]["forgiven_pct_K0"])
        for mk, key in (("edge", "audit_edge"), ("plac", "audit_placebo")):
            a = d[key]
            for est in ("reported", "spr", "tir"):
                row[f"err_{est}_{mk}"] = abs(a[est] - a["booked"])           # |estimated - booked| of the 100-episode mean
        rows.append(row)
    return pd.DataFrame(rows)


def mw_less(a, b) -> float:
    return float(stats.mannwhitneyu(a, b, alternative="less").pvalue)


def verdict(ok) -> str:
    return "NOT TESTABLE" if ok is None else ("SUPPORTED" if ok else "NOT SUPPORTED")


def edge_ratio(arm_df, oracle_df, min_ratio, min_lb):
    eo, ea = oracle_df.E.to_numpy(), arm_df.E.to_numpy()
    ratio = ea.mean() / eo.mean() if eo.mean() > 0 else float("nan")
    boot = np.array([rng.choice(ea, len(ea)).mean() / rng.choice(eo, len(eo)).mean() for _ in range(B_BOOT)])
    lb = float(np.quantile(boot, 0.05))
    ok = eo.mean() > 0 and ratio >= min_ratio and lb >= min_lb
    return ok, (f"E {ea.mean():+.2f} / ORACLE {eo.mean():+.2f} = {ratio:.3f} (need >= {min_ratio:.2f}); "
                f"one-sided 95% bootstrap LB {lb:.3f} (need >= {min_lb:.2f})")


def removes(arm_df, plain_df, oracle_df, name):
    p = mw_less(arm_df.X, plain_df.X)
    ok = p < 0.05 and arm_df.X.mean() <= oracle_df.X.mean() + 0.10
    return ok, (f"X {name} {arm_df.X.mean():.3f} < PLAIN {plain_df.X.mean():.3f}: p = {p:.4f} (need < 0.05); "
                f"{name} <= ORACLE + 0.10 = {oracle_df.X.mean() + 0.10:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results_x"))
    ap.add_argument("--print", dest="print_only", action="store_true")
    a = ap.parse_args()
    df = load(Path(a.results))
    lines = ["# Study X verdicts", "", f"Runs: {len(df)} of 32.", ""]
    verdicts = []
    if len(df):
        lines += ["| arm | runs | mean X (share of low-equity probes at leverage 16) | runs with X > 0 | mean E (booked % K0 / episode) | mean E from zero equity | placebo reported | placebo forgiven |",
                  "|---|---|---|---|---|---|---|---|"]
        for arm in ARMS:
            x = df[df.arm == arm]
            if len(x):
                lines.append(f"| {arm} | {len(x)} | {x.X.mean():.3f} | {int((x.X > 0).sum())} | {x.E.mean():+.2f} | {x.Ez.mean():+.2f} | "
                             f"{x.placebo_reported.mean():+.2f} | {x.placebo_forgiven.mean():.2f} |")
        lines.append("")
        P, O, S, T = (df[df.arm == k] for k in ("PLAIN", "ORACLE", "SPR", "TIR"))
        p1 = mw_less(O.X, P.X)
        present = p1 < 0.05
        verdicts.append(("XA1", f"X PLAIN {P.X.mean():.3f} > ORACLE {O.X.mean():.3f}: one-sided Mann-Whitney p = {p1:.4f} (need < 0.05)", present))
        ok2, r2 = removes(T, P, O, "TIR")
        ok3, r3 = edge_ratio(T, O, 0.80, 0.60)
        ok4, r4 = removes(S, P, O, "SPR")
        ok5, r5 = edge_ratio(S, O, 0.80, 0.60)
        rest = [("XA2", r2, ok2), ("XA3", "TIR " + r3, ok3), ("XA4", r4, ok4), ("XA5", "SPR " + r5, ok5)]
        if not present:                                            # not testable if the exploit was not acquired
            rest = [(i, r + "; the exploit was not acquired (XA1 failed)", None) for i, r, _ in rest]
        verdicts += rest
        h = df[df.arm.isin(["ORACLE", "SPR", "TIR"])]
        e1, e2 = float(h.err_tir_edge.max()), float(h.err_tir_plac.max())
        verdicts.append(("XB1", f"ORACLE, SPR and TIR agents ({len(h)}): max |TIR - booked| = {e1:.3f} (edge market), {e2:.3f} (placebo market) (need <= {AUDIT_BOUND})",
                         max(e1, e2) <= AUDIT_BOUND))
        g = df[df.X >= GAMBLER_X]
        lines += [f"Exploratory: agents with X >= {GAMBLER_X}: {len(g)}; median |TIR - booked| (placebo) {g.err_tir_plac.median() if len(g) else float('nan'):.3f}, "
                  f"median |reported - booked| (placebo) {g.err_reported_plac.median() if len(g) else float('nan'):.3f}.", ""]
    lines += ["| ID | Result | Verdict |", "|---|---|---|"] + [f"| {i} | {r} | {verdict(ok)} |" for i, r, ok in verdicts]
    text = "\n".join(lines) + "\n"
    if a.print_only:
        print(text)
        return
    df.to_csv(HERE / "runs_x.csv", index=False)
    (HERE / "VERDICTS_X.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
