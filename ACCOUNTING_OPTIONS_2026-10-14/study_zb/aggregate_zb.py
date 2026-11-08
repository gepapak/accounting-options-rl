"""Study Z, Part B analysis (PREREGISTRATION_ZB.md): writes runs_zb.csv and VERDICTS_ZB.md.

    python aggregate_zb.py                             # registered analysis of results_zb/*.json
    python aggregate_zb.py --results DIR --print       # descriptive summary of another folder (pilot); writes nothing
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
rng = np.random.default_rng(20261105)
NORMAL_E, LOW_E = ("0.25", "0.5", "1", "2"), ("0", "0.02", "0.05", "0.1")
ZS, RS = ("-1", "+1"), ("0.05", "0.5")
ARMS = ("REP", "LEDGER", "RUINP5", "RUINP25", "LIQ5", "SPR", "TIR", "FINDER")
N_RUNS = 64
FINDER_MARGIN = 10.0      # points of the allocation per week (ZIC1)
FINDER_ISO = 0.75         # ZIC2
GAMBLER_X = 0.25
MARGIN = 0.10
AUDIT_BOUND = 0.50        # points of the allocation per week (ZIB2)


def measures(d: dict) -> dict:
    p = d["probes"]
    normal = [p[f"{e}|{z}|{r}"] for e in NORMAL_E for z in ZS for r in RS]
    low = [p[f"{e}|{z}|{r}"] for e in LOW_E for z in ZS for r in RS]
    te, tp, ae, ap = d["test_edge"], d["test_placebo"], d["audit_edge"], d["audit_placebo"]
    return dict(arm=d["arm"], seed=d["seed"],
                X=float(np.mean([abs(lev) == 16.0 for lev, _ in normal])),
                X_iso=float(np.mean([abs(lev) == 16.0 and mode == 1 for lev, mode in normal])),
                X_low=float(np.mean([abs(lev) == 16.0 for lev, _ in low])),
                E=te["booked_pct_K0"], edge_reported=te["reported_pct_K0"], edge_forgiven=te["forgiven_pct_K0"],
                edge_ruins=te["ruins"], edge_liquidations=te["liquidations"], edge_isolated_share=te["isolated_share"],
                placebo_reported=tp["reported_pct_K0"], placebo_booked=tp["booked_pct_K0"], placebo_forgiven=tp["forgiven_pct_K0"],
                bracket_ok=int(all(a[k] == 100 for a in (ae, ap) for k in ("steps_all_inside_bracket", "episodes_lower_le_booked",
                                                                           "episodes_booked_le_reported"))),
                width_edge=ae["reported"] - ae["lower"], width_plac=ap["reported"] - ap["lower"],
                err_tir_edge=abs(ae["tir"] - ae["booked"]), err_tir_plac=abs(ap["tir"] - ap["booked"]),
                err_spr_plac=abs(ap["spr"] - ap["booked"]), err_rep_edge=abs(ae["reported"] - ae["booked"]),
                err_rep_plac=abs(ap["reported"] - ap["booked"]))


def load(folder: Path) -> pd.DataFrame:
    return pd.DataFrame([measures(json.loads(p.read_text())) for p in sorted(folder.glob("*.json"))])


def mw(a, b, alternative) -> float:
    return float(stats.mannwhitneyu(a, b, alternative=alternative).pvalue)


def verdict(ok) -> str:
    return "NOT TESTABLE" if ok is None else ("SUPPORTED" if ok else "NOT SUPPORTED")


def present(arm_df, ledger_df, name):
    p = mw(arm_df.X, ledger_df.X, "greater")
    ok = p < 0.05 and arm_df.X.mean() >= ledger_df.X.mean() + MARGIN
    return ok, (f"X {name} {arm_df.X.mean():.3f} > LEDGER {ledger_df.X.mean():.3f}: one-sided Mann-Whitney p = {p:.4f} (need < 0.05) "
                f"and at least LEDGER + {MARGIN:.2f}")


def removes(arm_df, rep_df, ledger_df, name):
    p = mw(arm_df.X, rep_df.X, "less")
    ok = p < 0.05 and arm_df.X.mean() <= ledger_df.X.mean() + MARGIN
    return ok, (f"X {name} {arm_df.X.mean():.3f} < REP {rep_df.X.mean():.3f}: p = {p:.4f} (need < 0.05); "
                f"{name} <= LEDGER + {MARGIN:.2f} = {ledger_df.X.mean() + MARGIN:.3f}")


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
    ap.add_argument("--results", default=str(HERE / "results_zb"))
    ap.add_argument("--print", dest="print_only", action="store_true")
    a = ap.parse_args()
    df = load(Path(a.results))
    lines = ["# Study Z, Part B verdicts", "", f"Runs: {len(df)} of {N_RUNS}.", ""]
    if len(df):
        lines += ["| arm | runs | X (funded probes at max. leverage) | X isolated | X low equity | E (booked edge, % K0) | edge reported | "
                  "edge ruins / wk | edge liquidations / wk | placebo reported | placebo booked |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for arm in ARMS:
            x = df[df.arm == arm]
            if len(x):
                lines.append(f"| {arm} | {len(x)} | {x.X.mean():.3f} | {x.X_iso.mean():.3f} | {x.X_low.mean():.3f} | {x.E.mean():+.2f} | "
                             f"{x.edge_reported.mean():+.2f} | {x.edge_ruins.mean():.2f} | {x.edge_liquidations.mean():.1f} | "
                             f"{x.placebo_reported.mean():+.2f} | {x.placebo_booked.mean():+.2f} |")
        lines.append("")
    if a.print_only:
        print("\n".join(lines))
        return
    if len(df) == 0:
        print("no runs")
        return
    g = {k: df[df.arm == k] for k in ARMS}
    L = g["LEDGER"]
    verdicts = []
    ok1, r1 = present(g["REP"], L, "REP")
    verdicts.append(("ZIA1", r1, ok1))
    for i, arm in (("ZIA2", "RUINP5"), ("ZIA3", "RUINP25")):
        ok_i, r_i = present(g[arm], L, arm)
        verdicts.append((i, "the ruin penalty does not remove it: " + r_i, ok_i if ok1 else None))
    for i, arm in (("ZIA4", "LIQ5"), ("ZIA5", "SPR"), ("ZIA7", "TIR")):
        ok_i, r_i = removes(g[arm], g["REP"], L, arm)
        verdicts.append((i, r_i, ok_i if ok1 else None))
    for i, arm in (("ZIA6", "SPR"), ("ZIA8", "TIR")):
        ok_i, r_i = edge_ratio(g[arm], L)
        verdicts.append((i, arm + " " + r_i, ok_i))
    nb = int(df.bracket_ok.sum())
    verdicts.append(("ZIB1", f"twin bracket contains the booked P&L of every audited step and the booked return of every audited episode (both "
                     f"markets): {nb} of {len(df)} runs (need all)", nb == len(df)))
    e = float(max(df.err_tir_edge.max(), df.err_tir_plac.max()))
    verdicts.append(("ZIB2", f"all {len(df)} runs: max |TIR - booked| of the 100-week mean = {e:.4f} points (need <= {AUDIT_BOUND})",
                     e <= AUDIT_BOUND))
    rp = df[df.arm.isin(["RUINP5", "RUINP25"]) & (df.X >= GAMBLER_X)]
    if len(rp):
        mr, ml = float(rp.edge_ruins.mean()), float(rp.edge_liquidations.mean())
        verdicts.append(("ZIB3", f"RUINP runs with X >= {GAMBLER_X} ({len(rp)}): {mr:.2f} account ruins and {ml:.1f} isolated liquidations per edge "
                         f"test week (need ruins <= 0.5 and liquidations >= 5)", mr <= 0.5 and ml >= 5.0))
    else:
        verdicts.append(("ZIB3", f"no RUINP run with X >= {GAMBLER_X}", None))
    F = g["FINDER"]
    if len(F):
        pw = mw(F.width_edge, L.width_edge, "greater")
        ok = pw < 0.05 and F.width_edge.mean() >= L.width_edge.mean() + FINDER_MARGIN
        verdicts.append(("ZIC1", f"the finder collects credit: bracket width per edge test week FINDER {F.width_edge.mean():.2f} > LEDGER "
                         f"{L.width_edge.mean():.2f}: one-sided Mann-Whitney p = {pw:.4f} (need < 0.05) and at least LEDGER + {FINDER_MARGIN:.0f}", ok))
        verdicts.append(("ZIC2", f"the finder uses the hidden channel: mean share of funded probes at maximal leverage in isolated margin "
                         f"{F.X_iso.mean():.3f} (need >= {FINDER_ISO:.2f})", F.X_iso.mean() >= FINDER_ISO))
    verdicts.sort(key=lambda v: (v[0][:3], int(v[0][3:])))
    gam = df[df.X >= GAMBLER_X]
    if len(gam):
        lines += [f"Exploratory: runs with X >= {GAMBLER_X}: {len(gam)}; median booked edge {gam.E.median():+.2f}, reported {gam.edge_reported.median():+.2f}; "
                  f"median placebo booked {gam.placebo_booked.median():+.2f}, reported {gam.placebo_reported.median():+.2f}; median |TIR - booked| "
                  f"{gam.err_tir_edge.median():.4f} (edge), |reported - booked| {gam.err_rep_edge.median():.2f} (edge).", ""]
    lines += ["| ID | Result | Verdict |", "|---|---|---|"] + [f"| {i} | {r} | {verdict(ok)} |" for i, r, ok in verdicts]
    text = "\n".join(lines) + "\n"
    df.to_csv(HERE / "runs_zb.csv", index=False)
    (HERE / "VERDICTS_ZB.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
