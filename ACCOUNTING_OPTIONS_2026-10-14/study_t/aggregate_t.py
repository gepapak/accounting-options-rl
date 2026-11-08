"""Study T analysis (PREREGISTRATION_T.md): writes runs_t.csv and VERDICTS_T.md.

    python aggregate_t.py                         # registered analysis of results_t/*.json
    python aggregate_t.py --results DIR --print   # descriptive summary of another folder (pilots); writes nothing
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
rng = np.random.default_rng(20261031)
LOW_E = ("0", "0.02", "0.05", "0.1")
OM_ARMS = ("OM1", "OM0.1", "OM0.05", "OM0.02", "OM0.01")
SETTING_A = "S_DK1"
SETTINGS_B = ("T_DK2", "T_DK1M")


def load(folder: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(folder.glob("*.json")):
        d = json.loads(p.read_text())
        low = [abs(d["lev"][f"{e}|{z}|{r}"]) == 16.0 for e in LOW_E for z in ("-1", "+1") for r in ("0.05", "0.5")]
        rows.append(dict(setting=d["setting"], arm=d["arm"], seed=d["seed"], X=float(np.mean(low)), E=d["test_edge"]["booked_pct_K0"],
                         placebo_reported=d["test_placebo"]["reported_pct_K0"], placebo_forgiven=d["test_placebo"]["forgiven_pct_K0"]))
    return pd.DataFrame(rows)


def mw_less(a, b) -> float:
    return float(stats.mannwhitneyu(a, b, alternative="less").pvalue)


def verdict(ok) -> str:
    if ok is None:
        return "NOT TESTABLE"
    return "SUPPORTED" if ok else "NOT SUPPORTED"


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


def core(df: pd.DataFrame, setting: str, tag: str):
    """x1 exploit present, x2/x3 SPR removes it / keeps the edge, x4/x5 TIR removes it / keeps the edge (one setting)."""
    d = df[df.setting == setting]
    P, O, S, T = d[d.arm == "PLAIN"], d[d.arm == "ORACLE"], d[d.arm == "SPR"], d[d.arm == "TIR"]
    out = []
    p1 = mw_less(O.X, P.X)
    ok1 = p1 < 0.05
    out.append((f"{tag}1", f"X PLAIN {P.X.mean():.3f} > ORACLE {O.X.mean():.3f}: one-sided Mann-Whitney p = {p1:.4f} (need < 0.05)", ok1))
    ok2, r2 = removes(S, P, O, "SPR")
    out.append((f"{tag}2", r2, ok2))
    ok3, r3 = edge_ratio(S, O, 0.80, 0.60)
    out.append((f"{tag}3", "SPR " + r3, ok3))
    ok4, r4 = removes(T, P, O, "TIR")
    out.append((f"{tag}4", r4, ok4))
    ok5, r5 = edge_ratio(T, O, 0.90, 0.75)
    out.append((f"{tag}5", "TIR " + r5, ok5))
    return out, ok1


def table(df: pd.DataFrame, setting: str, arms) -> list[str]:
    d = df[df.setting == setting]
    lines = [f"| {setting} arm | runs | mean X (share of low-equity probes at leverage 16) | mean E (booked % K0 / episode) | placebo reported | placebo forgiven |",
             "|---|---|---|---|---|---|"]
    for arm in arms:
        x = d[d.arm == arm]
        if len(x):
            lines.append(f"| {arm} | {len(x)} | {x.X.mean():.3f} | {x.E.mean():+.2f} | {x.placebo_reported.mean():+.2f} | {x.placebo_forgiven.mean():.2f} |")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results_t"))
    ap.add_argument("--print", dest="print_only", action="store_true")
    a = ap.parse_args()
    df = load(Path(a.results))
    n_a = int((df.setting == SETTING_A).sum()) if len(df) else 0
    n_b = int(df.setting.isin(SETTINGS_B).sum()) if len(df) else 0
    lines = ["# Study T verdicts", "", f"Runs: Part A {n_a} of 90, Part B {n_b} of 80.", ""]
    verdicts = []
    if n_a:
        lines += ["## Part A: Study S's edge-market test on a new market draw (DK1, PPO), with the occupancy-measure baseline", ""]
        lines += table(df, SETTING_A, ("PLAIN", "ORACLE", "SPR", "TIR") + OM_ARMS) + [""]
        v, present = core(df, SETTING_A, "TA")
        if not present:                                            # SPR/TIR hypotheses are not testable if the exploit was not acquired
            v = [v[0]] + [(i, r + "; the exploit was not acquired (TA1 failed)", None) for i, r, _ in v[1:]]
        verdicts += v
        d = df[df.setting == SETTING_A]
        O = d[d.arm == "ORACLE"]
        both = []
        for arm in OM_ARMS:
            k = d[d.arm == arm]
            if len(k):
                both.append((arm, k.X.mean() <= O.X.mean() + 0.10 and k.E.mean() >= 0.80 * O.E.mean(), k.X.mean(), k.E.mean()))
        ok4 = len(both) == len(OM_ARMS) and not any(b[1] for b in both)
        if O.E.mean() <= 0:                                        # the edge criterion is meaningless without an edge
            ok4 = None
        verdicts.append(("TA6", "; ".join(f"{a_}: X {x:.3f}, E {e:+.2f} -> {'both' if ok else 'not both'}" for a_, ok, x, e in both)
                         + f" (both = X <= {O.X.mean() + 0.10:.3f} and E >= {0.80 * O.E.mean():+.2f})", ok4))
    for st in SETTINGS_B:
        if int((df.setting == st).sum()):
            lines += [f"## Part B: {st}", ""] + table(df, st, ("PLAIN", "ORACLE", "SPR", "TIR")) + [""]
            v, present = core(df, st, f"TB-{st}-")
            if not present:                                        # SPR hypotheses are not testable if the exploit was not acquired
                v = [v[0]] + [(i, r + "; the exploit was not acquired (first hypothesis failed)", None) for i, r, _ in v[1:]]
            verdicts += v
    lines += ["| ID | Result | Verdict |", "|---|---|---|"] + [f"| {i} | {r} | {verdict(ok)} |" for i, r, ok in verdicts]
    text = "\n".join(lines) + "\n"
    if a.print_only:
        print(text)
        return
    df.to_csv(HERE / "runs_t.csv", index=False)
    (HERE / "VERDICTS_T.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
