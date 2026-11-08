"""Study EXT analysis (PREREGISTRATION_EXT.md): writes runs_ext.csv and VERDICTS_EXT.md.

    python aggregate_ext.py                             # registered analysis of results_ext/*.json
    python aggregate_ext.py --results DIR --print       # descriptive summary of another folder (pilot); writes nothing
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ARMS = ("REP", "LEDGER", "SPR", "TIR", "TSR", "RUIN5", "RUIN25", "RUIN100", "FINDER", "FINDER_DEF")
N_RUNS = 62
MARGIN = 0.10


def measures(d: dict) -> dict:
    t = d["test_placebo"]
    return dict(arm=d["arm"], seed=d["seed"], X=d["x10"], mean_abs_probe=d["mean_abs_probe"], reported=t["reported"], booked=t["booked"], booked_se=t["booked_se"], bust=t["bust"], mean_abs_position=t["mean_abs_position"],
                width=t.get("width", float("nan")), max_width=t.get("max_width", float("nan")), inside=t["episodes_steps_inside_bracket"], episodes=t["episodes"], steps=t["steps"], mae_line=t["mae_line"], max_err_line=t["max_err_line"], mae_tir=t["mae_tir"], mae_spr=t["mae_spr"], mae_reported=t["mae_reported"], seconds=d["seconds"])


def load(folder: Path) -> pd.DataFrame:
    return pd.DataFrame([measures(json.loads(p.read_text())) for p in sorted(folder.glob("EXT__*.json"))])


def mw(a, b, alternative) -> float:
    return float(stats.mannwhitneyu(a, b, alternative=alternative).pvalue)


def verdict(ok) -> str:
    return "NOT TESTABLE" if ok is None else ("SUPPORTED" if ok else "NOT SUPPORTED")


def present(arm_df, ledger_df, name):
    p = mw(arm_df.X, ledger_df.X, "greater")
    ok = p < 0.05 and arm_df.X.mean() >= ledger_df.X.mean() + MARGIN
    return ok, f"X {name} {arm_df.X.mean():.3f} > LEDGER {ledger_df.X.mean():.3f}: one-sided Mann-Whitney p = {p:.4f} (need < 0.05) and at least LEDGER + {MARGIN:.2f}"


def removes(arm_df, ref_df, ledger_df, name):
    p = mw(arm_df.X, ref_df.X, "less")
    ok = p < 0.05 and arm_df.X.mean() <= ledger_df.X.mean() + MARGIN
    return ok, f"X {name} {arm_df.X.mean():.3f} < REP {ref_df.X.mean():.3f}: p = {p:.4f} (need < 0.05); at most LEDGER + {MARGIN:.2f} = {ledger_df.X.mean() + MARGIN:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results_ext"))
    ap.add_argument("--print", dest="print_only", action="store_true")
    a = ap.parse_args()
    df = load(Path(a.results))
    lines = ["# Study EXT verdicts", "", f"Runs: {len(df)} of {N_RUNS}.", "",
             "| arm | runs | X | mean position at probes | bust share | reported | booked | mean width | audit error: reading, TIR, SPR, reported |", "|---|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        x = df[df.arm == arm]
        if len(x):
            lines.append(f"| {arm} | {len(x)} | {x.X.mean():.3f} | {x.mean_abs_probe.mean():.2f} | {x.bust.mean():.3f} | {x.reported.mean():+.2f} | {x.booked.mean():+.2f} | {x.width.mean():.2f} | {x.mae_line.mean():.2e}, {x.mae_tir.mean():.2e}, {x.mae_spr.mean():.2f}, {x.mae_reported.mean():.2f} |")
    lines.append("")
    if a.print_only:
        print("\n".join(lines))
        return
    g = {k: df[df.arm == k] for k in ARMS}
    L = g["LEDGER"]
    v = []
    ok1, r1 = present(g["REP"], L, "REP")
    v.append(("EA1", r1, ok1))
    gap = float((g["REP"].reported - g["REP"].booked).mean())
    v.append(("EA2", f"REP mean reported {g['REP'].reported.mean():+.2f} minus mean booked {g['REP'].booked.mean():+.2f} = {gap:.2f} points (need >= 10)", gap >= 10.0))
    p3 = mw(g["REP"].bust, L.bust, "greater")
    v.append(("EA3", f"bust share REP {g['REP'].bust.mean():.3f} > LEDGER {L.bust.mean():.3f}: p = {p3:.4f} (need < 0.05), difference at least 0.10", p3 < 0.05 and g["REP"].bust.mean() >= L.bust.mean() + 0.10))
    for i, arm in (("EA4", "SPR"), ("EA5", "TIR"), ("EA6", "TSR")):
        ok, r = removes(g[arm], g["REP"], L, arm)
        v.append((i, r, ok if ok1 else None))
    for i, arm in (("EA7", "RUIN5"), ("EA8", "RUIN25")):
        ok, r = present(g[arm], L, arm)
        v.append((i, "the penalty does not remove it: " + r, ok if ok1 else None))
    ok9, r9 = removes(g["RUIN100"], g["REP"], L, "RUIN100")
    v.append(("EA9", r9, ok9 if ok1 else None))
    nf = int((g["FINDER"].X >= 0.5).sum())
    v.append(("EA10", f"FINDER X at least 0.5 in {nf} of {len(g['FINDER'])} runs (need >= 7); mean X {g['FINDER'].X.mean():.3f}", nf >= 7))
    w = float(g["FINDER_DEF"].max_width.max())
    v.append(("EA11", f"default position set: largest width over the test windows {w:.2e} points in {len(g['FINDER_DEF'])} runs (need <= 1e-6)", w <= 1e-6))
    ni = int((df.inside == df.episodes).sum())
    v.append(("EA12", f"bracket holds at every audited step in {ni} of {len(df)} runs (need all)", ni == len(df)))
    m = float(df.max_err_line.max())
    v.append(("EA13", f"largest error of the tail reading's audit in any window of any run {m:.2e} points (need <= 0.01)", m <= 0.01))
    s, t = float(g["REP"].mae_spr.mean()), float(g["REP"].mae_tir.mean())
    v.append(("EA14", f"REP runs: audit error SPR {s:.2f} points (need >= 5), TIR {t:.2e} (need <= 0.01)", s >= 5.0 and t <= 0.01))
    scan = json.loads((HERE / "results_ext" / "scan_ext.json").read_text()) if (HERE / "results_ext" / "scan_ext.json").exists() else None
    if scan:
        d0, d1 = scan["default_0_1"], scan["leverage_10"]
        s0, s1 = d0["exact_within_1e_6"] / d0["states"], d1["exact_within_1e_6"] / d1["states"]
        v.append(("ES1", f"tail reading exact on {s0:.4f} of the default set and {s1:.4f} of the leverage set (need >= 0.999 each)", s0 >= 0.999 and s1 >= 0.999))
        a1, a2 = d1["jumps_at_true_location"] / max(1, d1["jumps"]), d1["jump_equals_wealth"] / max(1, d1["jumps"])
        v.append(("ES2", f"leverage set: {d1['jumps']} jumps, {a1:.4f} at the true location, {a2:.4f} of a size equal to the current wealth within 5% (need >= 0.99 each)", a1 >= 0.99 and a2 >= 0.99))
        v.append(("ES3", f"default set: width non-zero on {d0['width_nonzero']} of {d0['states']} states at t = +1, -1 (need 0)", d0["width_nonzero"] == 0))
        lines += [f"Exploratory: median distance to the jump, multiples of the realized move, by position (leverage set): {d1['median_abs_t_star_by_position']}; share within 16: {d1['share_kink_within_16_by_position']}.", ""]
    v.sort(key=lambda z: (z[0][:2], int(z[0][2:])))
    lines += ["| ID | Result | Verdict |", "|---|---|---|"] + [f"| {i} | {r} | {verdict(ok)} |" for i, r, ok in v]
    text = "\n".join(lines) + "\n"
    df.to_csv(HERE / "runs_ext.csv", index=False)
    (HERE / "VERDICTS_EXT.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
