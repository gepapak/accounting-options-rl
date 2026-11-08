"""Study Z analysis (PREREGISTRATION_Z.md): writes runs_z.csv and VERDICTS_Z.md.

    python aggregate_z.py                             # registered analysis of results_z/*.json
    python aggregate_z.py --results DIR --print       # descriptive summary of another folder (pilot); writes nothing
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
rng = np.random.default_rng(20261104)
LOW_E = ("0", "0.02", "0.05", "0.1")
EMPTY_H = ("0.5", "2")
ZS, RS = ("-1", "+1"), ("0.05", "0.5")
SPLIT_ARMS = ("REP", "LEDGER", "RUINP5", "RUINP25", "RUINA5", "SPR", "TIR")
POOLED_ARMS = ("P_REP", "P_RUIN5")
ARMS = SPLIT_ARMS + POOLED_ARMS
N_RUNS = 72
GAMBLER_X = 0.25
MARGIN = 0.10
PARK_MIN = 0.50


def measures(d: dict) -> dict:
    p = d["probes"]
    sym = [p[f"sym|{e}|{z}|{r}"] for e in LOW_E for z in ZS for r in RS]
    x_sym = float(np.mean([max(abs(a), abs(b)) == 16.0 for a, b in sym]))
    hedge = float(np.mean([abs(a) == 16.0 and a == -b for a, b in sym]))
    double = float(np.mean([abs(a) == 16.0 and a == b for a, b in sym]))
    empty = []                                                         # leverage of the empty account's leg; the other account's leg
    for h in EMPTY_H:
        for z in ZS:
            for r in RS:
                a, b = p[f"emptyA|{h}|{z}|{r}"]
                empty.append((a, b))
                a, b = p[f"emptyB|{h}|{z}|{r}"]
                empty.append((b, a))
    x_empty = float(np.mean([abs(e) == 16.0 for e, _ in empty]))
    park = float(np.mean([abs(e) == 16.0 and abs(o) < 16.0 for e, o in empty]))
    pooled = d["structure"] == "pooled"
    x = x_sym if pooled else 0.5 * (x_sym + x_empty)
    tp, ae, ap = d["test_placebo"], d["audit_edge"], d["audit_placebo"]
    return dict(arm=d["arm"], seed=d["seed"], structure=d["structure"], X=x, X_sym=x_sym, X_empty=float("nan") if pooled else x_empty,
                hedge=hedge, double=double, park=float("nan") if pooled else park,
                E=d["test_edge"]["booked_pct_K0"], Ez=d["test_zero"]["booked_pct_K0"],
                placebo_reported=tp["reported_pct_K0"], placebo_booked=tp["booked_pct_K0"], placebo_forgiven=tp["forgiven_pct_K0"],
                ruin_portfolio=tp["ruin_portfolio"], ruin_accounts=tp["ruin_accounts"], gross=tp["gross_mwh"], net=tp["net_mwh"],
                bracket_ok=int(all(a[k] == 100 for a in (ae, ap) for k in ("steps_all_inside_bracket", "episodes_lower_le_booked",
                                                                           "episodes_booked_le_reported"))),
                err_spr_plac=abs(ap["spr"] - ap["booked"]), err_tir_plac=abs(ap["tir"] - ap["booked"]),
                err_rep_plac=abs(ap["reported"] - ap["booked"]), width_plac=ap["reported"] - ap["lower"])


def load(folder: Path) -> pd.DataFrame:
    return pd.DataFrame([measures(json.loads(p.read_text())) for p in sorted(folder.glob("*.json"))])


def mw(a, b, alternative) -> float:
    return float(stats.mannwhitneyu(a, b, alternative=alternative).pvalue)


def verdict(ok) -> str:
    return "NOT TESTABLE" if ok is None else ("SUPPORTED" if ok else "NOT SUPPORTED")


def present(arm_df, ledger_df, name, col="X"):
    p = mw(arm_df[col], ledger_df[col], "greater")
    ok = p < 0.05 and arm_df[col].mean() >= ledger_df[col].mean() + MARGIN
    return ok, (f"{col} {name} {arm_df[col].mean():.3f} > LEDGER {ledger_df[col].mean():.3f}: one-sided Mann-Whitney p = {p:.4f} "
                f"(need < 0.05) and at least LEDGER + {MARGIN:.2f}")


def removes(arm_df, ref_df, ledger_df, name, ref_name="REP", col="X"):
    p = mw(arm_df[col], ref_df[col], "less")
    ok = p < 0.05 and arm_df[col].mean() <= ledger_df[col].mean() + MARGIN
    return ok, (f"{col} {name} {arm_df[col].mean():.3f} < {ref_name} {ref_df[col].mean():.3f}: p = {p:.4f} (need < 0.05); "
                f"{name} <= LEDGER + {MARGIN:.2f} = {ledger_df[col].mean() + MARGIN:.3f}")


def edge_ratio(arm_df, ledger_df, min_ratio=0.80, min_lb=0.60):
    eo, ea = ledger_df.E.to_numpy(), arm_df.E.to_numpy()
    ratio = ea.mean() / eo.mean() if eo.mean() > 0 else float("nan")
    boot = np.array([rng.choice(ea, len(ea)).mean() / rng.choice(eo, len(eo)).mean() for _ in range(B_BOOT)])
    lb = float(np.quantile(boot, 0.05))
    ok = eo.mean() > 0 and ratio >= min_ratio and lb >= min_lb
    return ok, (f"E {ea.mean():+.2f} / LEDGER {eo.mean():+.2f} = {ratio:.3f} (need >= {min_ratio:.2f}); "
                f"one-sided 95% bootstrap LB {lb:.3f} (need >= {min_lb:.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results_z"))
    ap.add_argument("--print", dest="print_only", action="store_true")
    a = ap.parse_args()
    df = load(Path(a.results))
    lines = ["# Study Z verdicts", "", f"Runs: {len(df)} of {N_RUNS}.", ""]
    verdicts = []
    if len(df):
        lines += ["| arm | runs | X | X both low | X one empty | hedge | park | E (booked edge, % K0) | E from zero | placebo reported | "
                  "placebo booked | portfolio ruins / ep | account ruins / ep |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for arm in ARMS:
            x = df[df.arm == arm]
            if len(x):
                lines.append(f"| {arm} | {len(x)} | {x.X.mean():.3f} | {x.X_sym.mean():.3f} | {x.X_empty.mean():.3f} | {x.hedge.mean():.3f} | "
                             f"{x.park.mean():.3f} | {x.E.mean():+.2f} | {x.Ez.mean():+.2f} | {x.placebo_reported.mean():+.2f} | "
                             f"{x.placebo_booked.mean():+.2f} | {x.ruin_portfolio.mean():.2f} | {x.ruin_accounts.mean():.2f} |")
        lines.append("")
    if a.print_only:
        print("\n".join(lines))
        return
    if len(df) == 0:
        print("no runs")
        return
    g = {k: df[df.arm == k] for k in ARMS}
    L = g["LEDGER"]
    ok, r = present(g["REP"], L, "REP")
    verdicts.append(("ZA1", r, ok))
    acquired = ok
    for i, lam in (("ZA2", "5"), ("ZA3", "25")):
        ok_i, r_i = present(g[f"RUINP{lam}"], L, f"RUINP{lam}")
        verdicts.append((i, "the portfolio ruin penalty does not remove it: " + r_i, ok_i if acquired else None))
    for i, arm in (("ZA4", "RUINA5"), ("ZA5", "SPR"), ("ZA7", "TIR")):
        ok_i, r_i = removes(g[arm], g["REP"], L, arm)
        verdicts.append((i, r_i, ok_i if acquired else None))
    for i, arm in (("ZA6", "SPR"), ("ZA8", "TIR")):
        ok_i, r_i = edge_ratio(g[arm], L)
        verdicts.append((i, arm + " " + r_i, ok_i))
    ok9, r9 = present(g["P_REP"], L, "P_REP", col="X_sym")
    verdicts.append(("ZA9", "pooled control, exploit present: " + r9, ok9))
    ok10, r10 = removes(g["P_RUIN5"], g["P_REP"], L, "P_RUIN5", ref_name="P_REP", col="X_sym")
    verdicts.append(("ZA10", "pooled control, the same penalty removes it: " + r10, ok10 if ok9 else None))
    nb = int(df.bracket_ok.sum())
    verdicts.append(("ZB1", f"twin bracket contains the booked P&L of every audited step and the booked return of every audited episode (both "
                     f"markets): {nb} of {len(df)} runs (need all)", nb == len(df)))
    share = lambda d: (d.ruin_portfolio / d.ruin_accounts)[d.ruin_accounts > 0]          # noqa: E731
    rp, rr = share(df[df.arm.isin(["RUINP5", "RUINP25"])]), share(g["REP"])
    if len(rp) and len(rr):
        p2 = mw(rp, rr, "less")
        verdicts.append(("ZB2", f"portfolio ruins per account ruin (placebo test): RUINP5 and RUINP25 {rp.mean():.3f} ({len(rp)} runs) < REP "
                         f"{rr.mean():.3f} ({len(rr)} runs): one-sided Mann-Whitney p = {p2:.4f} (need < 0.05)", p2 < 0.05))
    else:
        verdicts.append(("ZB2", "no run with an account ruin in RUINP or REP", None))
    gp = df[df.arm.isin(["REP", "RUINP5", "RUINP25"]) & (df.X >= GAMBLER_X)]
    if len(gp):
        verdicts.append(("ZB3", f"the exploit's form: mean park share of the REP, RUINP5 and RUINP25 runs with X >= {GAMBLER_X} ({len(gp)} runs) "
                         f"{gp.park.mean():.3f} (need >= {PARK_MIN:.2f})", gp.park.mean() >= PARK_MIN))
    else:
        verdicts.append(("ZB3", f"no REP or RUINP run with X >= {GAMBLER_X}", None))
    gam = df[df.X >= GAMBLER_X]
    lines += [f"Exploratory: runs with X >= {GAMBLER_X}: {len(gam)}; median placebo reported {gam.placebo_reported.median():+.2f}, booked "
              f"{gam.placebo_booked.median():+.2f}; median |TIR - booked| {gam.err_tir_plac.median():.3f}, |SPR - booked| "
              f"{gam.err_spr_plac.median():.3f}, |reported - booked| {gam.err_rep_plac.median():.3f}; median bracket width "
              f"{gam.width_plac.median():.2f} points (placebo).", ""]
    verdicts.sort(key=lambda v: (v[0][:2], int(v[0][2:])))
    lines += ["| ID | Result | Verdict |", "|---|---|---|"] + [f"| {i} | {r} | {verdict(ok)} |" for i, r, ok in verdicts]
    text = "\n".join(lines) + "\n"
    df.to_csv(HERE / "runs_z.csv", index=False)
    (HERE / "VERDICTS_Z.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
