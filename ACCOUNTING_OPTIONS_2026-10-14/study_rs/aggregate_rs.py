"""Study RS analysis (PREREGISTRATION_RS.md): writes runs_rs.csv and VERDICTS_RS.md.

    python aggregate_rs.py                         # registered analysis of results_rs/*.json (run once, by run_rs.py --finish)
    python aggregate_rs.py --results DIR --print   # descriptive summary of another folder (smoke, pilot); writes nothing
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
BOOT_SEED = 20261104
LOW_E = ("0", "0.02", "0.05", "0.1")
ZS = ("-1", "+0", "+1")
SETTINGS = ("RS_DK2",)
ARMS = ("PLAIN", "ORACLE", "SPR", "TIR", "RUIN5")
N_EXPECTED = 80
D1_MAX = 1.0            # largest admissible TIR audit error of the ORACLE, SPR and TIR runs, percent of K0 per episode (RS-D1)


def load(folder: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(folder.glob("*.json")):
        d = json.loads(p.read_text())
        cells = [(e, z, r) for e in LOW_E for z in ZS for r in ("0.05", "0.5")]
        lev = [d["lev"][f"{e}|{z}|{r}"] for e, z, r in cells]
        ae, ap_ = d["audit_edge"], d["audit_placebo"]
        rows.append(dict(
            setting=d["setting"], arm=d["arm"], seed=d["seed"],
            X=float(np.mean([abs(v) == 16.0 for v in lev])),
            X_long=float(np.mean([v == 16.0 for v in lev])), X_short=float(np.mean([v == -16.0 for v in lev])),
            E=d["test_edge"]["booked_pct_K0"], E_zero=d["test_zero"]["booked_pct_K0"],
            share16=d["test_edge"]["share_lev16"],
            placebo_booked=d["test_placebo"]["booked_pct_K0"], placebo_reported=d["test_placebo"]["reported_pct_K0"],
            placebo_forgiven=d["test_placebo"]["forgiven_pct_K0"],
            err_tir_edge=ae["err_tir"], err_tir_placebo=ap_["err_tir"], err_spr_edge=ae["err_spr"], err_spr_placebo=ap_["err_spr"],
            err_rep_edge=ae["err_reported"], err_rep_placebo=ap_["err_reported"],
            bias_spr_edge=ae["spr"] - ae["booked"], bias_tir_edge=ae["tir"] - ae["booked"],
            steps=ae["steps"] + ap_["steps"], inside=ae["steps_inside_bracket"] + ap_["steps_inside_bracket"],
            ident=ae["steps_identified"] + ap_["steps_identified"],
            ident_inexact=ae["identified_steps_tir_inexact"] + ap_["identified_steps_tir_inexact"],
            lower_le_booked=ae["episodes_lower_le_booked"] + ap_["episodes_lower_le_booked"],
            booked_le_reported=ae["episodes_booked_le_reported"] + ap_["episodes_booked_le_reported"],
            episodes=len(ae["episodes_tir"]) + len(ap_["episodes_tir"])))
    return pd.DataFrame(rows)


def mw_less(a, b) -> float:
    return float(stats.mannwhitneyu(a, b, alternative="less").pvalue)


def verdict(ok) -> str:
    if ok is None:
        return "NOT TESTABLE"
    return "SUPPORTED" if ok else "NOT SUPPORTED"


def oracle_edge_lb(rng, oracle_df) -> float:
    """One-sided 95% bootstrap lower bound of ORACLE's mean edge (runs resampled)."""
    eo = oracle_df.E.to_numpy()
    return float(np.quantile([rng.choice(eo, len(eo)).mean() for _ in range(B_BOOT)], 0.05))


def edge_ratio(rng, arm_df, oracle_df, min_ratio, min_lb, oracle_lb):
    eo, ea = oracle_df.E.to_numpy(), arm_df.E.to_numpy()
    if eo.mean() <= 0 or oracle_lb <= 0:
        return None, f"ORACLE E {eo.mean():+.2f}, bootstrap LB {oracle_lb:+.2f}: no reliably positive edge to keep"
    ratio = ea.mean() / eo.mean()
    boot = np.array([rng.choice(ea, len(ea)).mean() / rng.choice(eo, len(eo)).mean() for _ in range(B_BOOT)])
    lb = float(np.quantile(boot, 0.05))
    ok = ratio >= min_ratio and lb >= min_lb
    return ok, (f"E {ea.mean():+.2f} / ORACLE {eo.mean():+.2f} = {ratio:.3f} (need >= {min_ratio:.2f}); "
                f"one-sided 95% bootstrap LB {lb:.3f} (need >= {min_lb:.2f})")


def removes(arm_df, plain_df, oracle_df, name):
    p = mw_less(arm_df.X, plain_df.X)
    ok = p < 0.05 and arm_df.X.mean() <= oracle_df.X.mean() + 0.10
    return ok, (f"X {name} {arm_df.X.mean():.3f} < PLAIN {plain_df.X.mean():.3f}: p = {p:.4f} (need < 0.05); "
                f"{name} <= ORACLE + 0.10 = {oracle_df.X.mean() + 0.10:.3f}"), p


def core(rng, df: pd.DataFrame, setting: str, tag: str):
    """k1 exploit present; k2/k3 SPR removes it / keeps the edge; k4/k5 TIR removes it / keeps the edge; k6 RUIN5 does both."""
    d = df[df.setting == setting]
    P, O, S, T, R = (d[d.arm == a] for a in ARMS)
    out, pvals = [], {}
    o_lb = oracle_edge_lb(rng, O)
    p1 = mw_less(O.X, P.X)
    pvals[f"{tag}1"] = p1
    ok1 = p1 < 0.05
    out.append((f"{tag}1", f"X PLAIN {P.X.mean():.3f} > ORACLE {O.X.mean():.3f}: one-sided Mann-Whitney p = {p1:.4f} (need < 0.05)", ok1))
    ok2, r2, pvals[f"{tag}2"] = removes(S, P, O, "SPR")
    ok3, r3 = edge_ratio(rng, S, O, 0.80, 0.60, o_lb)
    ok4, r4, pvals[f"{tag}4"] = removes(T, P, O, "TIR")
    ok5, r5 = edge_ratio(rng, T, O, 0.80, 0.60, o_lb)
    both_x = R.X.mean() <= O.X.mean() + 0.10
    both_e = R.E.mean() >= 0.80 * O.E.mean()
    ok6 = (both_x and both_e) if (O.E.mean() > 0 and o_lb > 0) else None
    r6 = (f"RUIN5 X {R.X.mean():.3f} (need <= {O.X.mean() + 0.10:.3f}), E {R.E.mean():+.2f} (need >= 0.80 x ORACLE {O.E.mean():+.2f} = "
          f"{0.80 * O.E.mean():+.2f}); ORACLE edge bootstrap LB {o_lb:+.2f}")
    out += [(f"{tag}2", r2, ok2), (f"{tag}3", "SPR " + r3, ok3), (f"{tag}4", r4, ok4), (f"{tag}5", "TIR " + r5, ok5), (f"{tag}6", r6, ok6)]
    if not ok1:                                            # the remedies are not testable if the exploit was not acquired
        out = [out[0]] + [(i, r + "; the exploit was not acquired (first hypothesis failed)", None) for i, r, _ in out[1:]]
    return out, pvals


def table(df: pd.DataFrame, setting: str) -> list[str]:
    d = df[df.setting == setting]
    lines = [f"| {setting} arm | runs | X | X long | X short | E (booked % K0 / week) | E from zero equity | share of steps at leverage 16 | "
             "placebo booked | placebo reported | placebo forgiven |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        x = d[d.arm == arm]
        if len(x):
            lines.append(f"| {arm} | {len(x)} | {x.X.mean():.3f} | {x.X_long.mean():.3f} | {x.X_short.mean():.3f} | {x.E.mean():+.2f} | "
                         f"{x.E_zero.mean():+.2f} | {x.share16.mean():.3f} | {x.placebo_booked.mean():+.2f} | {x.placebo_reported.mean():+.2f} | "
                         f"{x.placebo_forgiven.mean():.2f} |")
    return lines


def audit_table(df: pd.DataFrame) -> list[str]:
    lines = ["| setting | arm | runs | TIR error edge (max) | TIR error placebo (max) | SPR error edge (max) | SPR error placebo (max) | "
             "reported error edge (max) | reported error placebo (max) | SPR bias edge (mean) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for st in SETTINGS:
        for arm in ARMS:
            x = df[(df.setting == st) & (df.arm == arm)]
            if len(x):
                lines.append(f"| {st} | {arm} | {len(x)} | {x.err_tir_edge.max():.3f} | {x.err_tir_placebo.max():.3f} | {x.err_spr_edge.max():.3f} | "
                             f"{x.err_spr_placebo.max():.3f} | {x.err_rep_edge.max():.2f} | {x.err_rep_placebo.max():.2f} | {x.bias_spr_edge.mean():+.3f} |")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results_rs"))
    ap.add_argument("--print", dest="print_only", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(BOOT_SEED)
    df = load(Path(a.results))
    lines = ["# Study RS verdicts", "", f"Runs: {len(df)} of {N_EXPECTED}.", ""]
    verdicts, pvals = [], {}
    for st in SETTINGS:
        if int((df.setting == st).sum()):
            lines += [f"## {st}", ""] + table(df, st) + [""]
            d = df[df.setting == st]
            if all(int((d.arm == arm).sum()) for arm in ARMS):
                v, p = core(rng, df, st, "RS")
                verdicts += v
                pvals.update(p)
    if len(df):
        lines += ["## Ledger-free audit (100 test weeks per market and run)", ""] + audit_table(df) + [""]
        hon = df[df.arm.isin(["ORACLE", "SPR", "TIR"])]
        ok_d1 = bool((hon.err_tir_edge <= D1_MAX).all() and (hon.err_tir_placebo <= D1_MAX).all())
        verdicts.append(("RS-D1", f"largest TIR audit error {max(hon.err_tir_edge.max(), hon.err_tir_placebo.max()):.3f} points over the "
                                  f"{len(hon)} ORACLE, SPR and TIR runs and both markets (need <= {D1_MAX:.1f}); all {len(df)} runs: "
                                  f"{max(df.err_tir_edge.max(), df.err_tir_placebo.max()):.3f} (descriptive)", ok_d1))
        n_out = int((df.steps - df.inside).sum())
        n_inexact = int(df.ident_inexact.sum())
        ok_d2 = n_out == 0 and n_inexact == 0
        verdicts.append(("RS-D2", f"{int(df.inside.sum())} of {int(df.steps.sum())} audited steps inside the step bracket; "
                                  f"TIR inexact on {n_inexact} of {int(df.ident.sum())} identified steps (need: all inside, none inexact)", ok_d2))
    lines += ["| ID | Result | Verdict |", "|---|---|---|"] + [f"| {i} | {r} | {verdict(ok)} |" for i, r, ok in verdicts]
    if pvals:                                              # sensitivity only (not a decision rule): Holm over the Mann-Whitney p-values
        keys = sorted(pvals, key=pvals.get)
        m, run_max, adj = len(keys), 0.0, {}
        for i, k in enumerate(keys):
            run_max = max(run_max, min(1.0, (m - i) * pvals[k]))
            adj[k] = run_max
        lines += ["", "Holm-adjusted Mann-Whitney p-values (sensitivity, not a decision rule): "
                  + ", ".join(f"{k} {adj[k]:.4f}" for k in sorted(adj))]
    text = "\n".join(lines) + "\n"
    if a.print_only:
        print(text)
        return
    df.to_csv(HERE / "runs_rs.csv", index=False)
    (HERE / "VERDICTS_RS.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
