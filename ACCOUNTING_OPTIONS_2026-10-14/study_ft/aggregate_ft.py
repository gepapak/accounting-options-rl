"""Study FT analysis (PREREGISTRATION_FT.md): writes runs_ft.csv and VERDICTS_FT.md.

    python aggregate_ft.py                             # registered analysis of results_ft/*.json
    python aggregate_ft.py --results DIR --print       # descriptive summary of another folder (pilot); writes nothing
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
rng = np.random.default_rng(20261107)
LOW_E = ("0", "0.02", "0.05", "0.1")
BUCKET = ("-0.05", "-0.5", "-2")
ZS, RS = ("-1", "+1"), ("0.05", "0.5")
ARMS = ("REP", "LEDGER", "SPR", "TIR", "TSR")
MARGIN = 0.10


def measures(d: dict) -> dict:
    p = d["probes"]
    low = [p[f"low|{e}|{z}|{r}"] for e in LOW_E for z in ZS for r in RS]
    x_low = float(np.mean([abs(v[0]) == 16.0 for v in low]))
    out = dict(part=d["part"], arm=d["arm"], seed=d["seed"], X=x_low, X_low=x_low, mean_low=float(np.mean([abs(v[0]) for v in low])),
               E=d["test_edge"]["booked_pct_K0"], Ez=d["test_zero"]["booked_pct_K0"], Ed=d["test_deficit"]["booked_pct_K0"],
               lev_d=d["test_deficit"]["mean_abs_leverage"], rep_d=d["test_deficit"]["reported_pct_K0"], placebo_reported=d["test_placebo"]["reported_pct_K0"], placebo_booked=d["test_placebo"]["booked_pct_K0"], forgiven_placebo=d["test_placebo"]["forgiven_pct_K0"], seconds=d["seconds"])
    if d["part"] == "C":                       # trained only in deficit: reported equity is 0 throughout nearly every episode, so the probes are those at 0
        out["X"] = float(np.mean([abs(p[f"low|0|{z}|{r}"][0]) == 16.0 for z in ZS for r in RS]))
    if d["part"] == "A":
        dfp = [p[f"def|{b}|{z}|{r}"] for b in BUCKET for z in ZS for r in RS]
        out["X_def"] = float(np.mean([abs(v[0]) == 16.0 for v in dfp]))
        out["R_def"] = float(np.mean([v[1] > 0.0 for v in dfp]))
        out["X"] = 0.5 * (out["X_low"] + out["X_def"])
    for name in ("audit_edge", "audit_deficit", "audit_placebo"):
        a = d[name]
        short = name.split("_")[1]
        for k in ("tsr", "tir", "spr", "reported"):
            out[f"mae_{k}_{short}"] = a[f"mae_{k}"]
        out[f"max_tsr_{short}"] = a["maxerr_tsr"]
        out[f"inside_{short}"] = a["steps_all_inside_bracket"]
        out[f"low_ok_{short}"] = a["episodes_lower_le_booked"]
    return out


def load(folder: Path) -> pd.DataFrame:
    return pd.DataFrame([measures(json.loads(p.read_text())) for p in sorted(folder.glob("FT_*.json"))])


def mw(a, b, alternative) -> float:
    return float(stats.mannwhitneyu(a, b, alternative=alternative).pvalue)


def verdict(ok) -> str:
    return "NOT TESTABLE" if ok is None else ("SUPPORTED" if ok else "NOT SUPPORTED")


def present(arm_df, ledger_df, name):
    p = mw(arm_df.X, ledger_df.X, "greater")
    return p < 0.05 and arm_df.X.mean() >= ledger_df.X.mean() + MARGIN, f"X {name} {arm_df.X.mean():.3f} > LEDGER {ledger_df.X.mean():.3f}: p = {p:.4f} (need < 0.05) and at least LEDGER + {MARGIN:.2f}"


def removes(arm_df, ref_df, ledger_df, name):
    p = mw(arm_df.X, ref_df.X, "less")
    return p < 0.05 and arm_df.X.mean() <= ledger_df.X.mean() + MARGIN, f"X {name} {arm_df.X.mean():.3f} < REP {ref_df.X.mean():.3f}: p = {p:.4f} (need < 0.05); at most LEDGER + {MARGIN:.2f} = {ledger_df.X.mean() + MARGIN:.3f}"


def ratio_boot(arm_df, ledger_df, col):
    ea, eo = arm_df[col].to_numpy(), ledger_df[col].to_numpy()
    r = ea.mean() / eo.mean() if eo.mean() > 0 else float("nan")
    boot = np.array([rng.choice(ea, len(ea)).mean() / rng.choice(eo, len(eo)).mean() for _ in range(B_BOOT)])
    return r, float(np.quantile(boot, 0.05)), float(np.quantile(boot, 0.95))


def keeps(arm_df, ledger_df, name, col="E"):
    r, lb, _ = ratio_boot(arm_df, ledger_df, col)
    return ledger_df[col].mean() > 0 and r >= 0.80 and lb >= 0.60, f"{col} {name} {arm_df[col].mean():+.2f} / LEDGER {ledger_df[col].mean():+.2f} = {r:.3f} (need >= 0.80); one-sided 95% bound {lb:.3f} (need >= 0.60)"


def loses(arm_df, ledger_df, name, col="Ed"):
    r, _, ub = ratio_boot(arm_df, ledger_df, col)
    return ledger_df[col].mean() > 0 and r < 0.50 and ub < 0.80, f"{col} {name} {arm_df[col].mean():+.2f} / LEDGER {ledger_df[col].mean():+.2f} = {r:.3f} (need < 0.50); one-sided 95% upper bound {ub:.3f} (need < 0.80)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results_ft"))
    ap.add_argument("--print", dest="print_only", action="store_true")
    a = ap.parse_args()
    df = load(Path(a.results))
    lines = ["# Study FT verdicts", "", f"Runs: {len(df)}.", "", "| part | arm | runs | X | X low | mean leverage low | E funded | E zero | E deficit | leverage in deficit | placebo reported | placebo booked | audit error (edge): TSR, TIR, SPR, reported | audit error (deficit): TSR, TIR, SPR |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for part in ("A", "B", "C"):
        for arm in ARMS:
            x = df[(df.part == part) & (df.arm == arm)]
            if len(x):
                lines.append(f"| {part} | {arm} | {len(x)} | {x.X.mean():.3f} | {x.X_low.mean():.3f} | {x.mean_low.mean():.2f} | {x.E.mean():+.2f} | {x.Ez.mean():+.2f} | {x.Ed.mean():+.2f} | {x.lev_d.mean():.2f} | {x.placebo_reported.mean():+.2f} | {x.placebo_booked.mean():+.2f} | {x.mae_tsr_edge.mean():.1e}, {x.mae_tir_edge.mean():.3f}, {x.mae_spr_edge.mean():.3f}, {x.mae_reported_edge.mean():.2f} | {x.mae_tsr_deficit.mean():.1e}, {x.mae_tir_deficit.mean():.3f}, {x.mae_spr_deficit.mean():.3f} |")
    lines.append("")
    if a.print_only:
        print("\n".join(lines))
        return
    v = []
    A = {k: df[(df.part == "A") & (df.arm == k)] for k in ARMS}
    B = {k: df[(df.part == "B") & (df.arm == k)] for k in ARMS}
    D = {k: df[(df.part == "C") & (df.arm == k)] for k in ARMS}
    ok, r = present(A["REP"], A["LEDGER"], "REP")
    v.append(("FA1", "bucket: exploit present: " + r, ok))
    acq_a = ok
    for i, j, arm in (("FA2", "FA3", "SPR"), ("FA4", "FA5", "TIR"), ("FA6", "FA7", "TSR")):
        ok_r, rr = removes(A[arm], A["REP"], A["LEDGER"], arm)
        ok_k, rk = keeps(A[arm], A["LEDGER"], arm)
        v.append((i, "bucket: removes: " + rr, ok_r if acq_a else None))
        v.append((j, "bucket: keeps: " + rk, ok_k))
    ok, r = present(B["REP"], B["LEDGER"], "REP")
    v.append(("FB1", "deficit: exploit present: " + r, ok))
    acq_b = ok
    for i, arm in (("FB2", "SPR"), ("FB3", "TIR"), ("FB4", "TSR")):
        ok_r, rr = removes(B[arm], B["REP"], B["LEDGER"], arm)
        v.append((i, "deficit: removes: " + rr, ok_r if acq_b else None))
    for i, arm in (("FB5", "TSR"), ("FB6", "SPR"), ("FB7", "TIR")):
        ok_k, rk = keeps(B[arm], B["LEDGER"], arm, col="Ed")
        v.append((i, "deficit, half of training funded: keeps the edge from a deficit start: " + rk, ok_k))
    ok, r = present(D["REP"], D["LEDGER"], "REP")
    v.append(("FD1", "deficit only: exploit present: " + r, ok))
    acq_d = ok
    for i, arm in (("FD2", "SPR"), ("FD3", "TIR"), ("FD4", "TSR")):
        ok_r, rr = removes(D[arm], D["REP"], D["LEDGER"], arm)
        v.append((i, "deficit only: removes: " + rr, ok_r if acq_d else None))
    ok_k, rk = keeps(D["TSR"], D["LEDGER"], "TSR", col="Ed")
    v.append(("FD5", "deficit only: the line reward keeps the edge: " + rk, ok_k))
    for i, arm in (("FD6", "SPR"), ("FD7", "TIR")):
        ok_l, rl = loses(D[arm], D["LEDGER"], arm)
        v.append((i, "deficit only: a sign query cannot read the edge: " + rl, ok_l))
    m = float(max(df.max_tsr_edge.max(), df.max_tsr_deficit.max(), df.max_tsr_placebo.max()))
    v.append(("FC1", f"TSR's audit: largest error over all windows of all runs {m:.2e} points (need <= 1e-6)", m <= 1e-6))
    na = int(((A["REP"].inside_edge == 100) & (A["REP"].inside_placebo == 100)).sum() + sum(((A[k].inside_edge == 100) & (A[k].inside_placebo == 100)).sum() for k in ARMS if k != "REP"))
    v.append(("FC2", f"bucket: the bracket holds at every audited step in {na} of {sum(len(A[k]) for k in ARMS)} runs (need all)", na == sum(len(A[k]) for k in ARMS)))
    lines += [f"Exploratory: Part B bracket, windows with every step inside: edge funded {B['REP'].inside_edge.mean():.1f}, deficit {B['REP'].inside_deficit.mean():.1f} of 100 for REP (the owner's claim is not forgiving, so failures are expected).", ""]
    lines += ["| ID | Result | Verdict |", "|---|---|---|"] + [f"| {i} | {r} | {verdict(ok)} |" for i, r, ok in v]
    text = "\n".join(lines) + "\n"
    df.to_csv(HERE / "runs_ft.csv", index=False)
    (HERE / "VERDICTS_FT.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
