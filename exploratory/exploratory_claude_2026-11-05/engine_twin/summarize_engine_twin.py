r"""Decisions and tables of the engine twin check, from results/engine_twin_runs.csv (rules: PROTOCOL.md).

Writes results/VERDICTS.md, results/engine_twin_summary.json and results/tab_engine_twin.tex. With --print it prints the
summary JSON and writes nothing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TOL = 1e-3
GROUPS = [
    ("floor bound, volume payoff", lambda d: d.bound & (d["mode"] == "horizon") & ~d.capon),
    ("floor bound, percent payoff", lambda d: d.bound & (d["mode"] == "percent") & ~d.capon),
    ("floor did not bind, volume payoff", lambda d: ~d.bound & (d["mode"] == "horizon") & ~d.capon),
    ("floor did not bind, percent payoff", lambda d: ~d.bound & (d["mode"] == "percent") & ~d.capon),
    ("L7 with the cap on", lambda d: d.capon),
]


def load(results: Path, jobs: Path) -> pd.DataFrame:
    d = pd.read_csv(results / "engine_twin_runs.csv")
    bound, section = set(), None
    for line in jobs.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            section = line
        elif line.strip() and section and "floor bound" in section:
            bound.add(line.strip())
    d["bound"] = d.name.isin(bound)
    d["capon"] = d.variant.str.endswith("_capon")
    return d


def decide(d: pd.DataFrame, audit_gap: float | None) -> dict:
    a = d[~d.capon]
    b, n = a[a.bound], a[~a.bound]
    h, p = b[b["mode"] == "horizon"], b[b["mode"] == "percent"]
    no_liq = a[a.liquidations == 0]
    large = int(b.shift_large_tir.sum())
    out = {
        "runs": int(len(a)), "bound": int(len(b)), "comparison": int(len(n)), "cap_on": int(d.capon.sum()),
        "G1_largest_json_minus_campaign_pp": audit_gap,
        "G2_max_parity_dkk": float(d.max_abs_parity.max()), "G2_unknown_writes": int(d.unknown_writes.sum()),
        "E1_min_real_credit_dkk": float(a.min_credit.min()), "E1": bool(a.min_credit.min() >= -TOL),
        "E2_steps_outside": int(a.steps_outside.sum()), "E2_runs_containing": int(a.contains_run.sum()),
        "E2": bool(a.steps_outside.sum() == 0 and a.contains_run.all()),
        "reconcile_flags_bound": int(b.flag_reconcile.sum()), "reconcile_flags_comparison": int(n.flag_reconcile.sum()),
        "E3_flags_bound": int(b.flag_tir_gap.sum()), "E3_flags_comparison": int(n.flag_tir_gap.sum()),
        "E3": bool(b.flag_tir_gap.sum() >= 53 and n.flag_tir_gap.sum() == 0),
        "E4_flags_bound": int(b.flag_width.sum()), "E4_flags_comparison": int(n.flag_width.sum()),
        "E4": bool(b.flag_width.sum() >= 53),
        "E5_horizon_tir_better": int((h.tir_error_pp.abs() < h.spr_error_pp.abs()).sum()), "E5_horizon_runs": int(len(h)),
        "E5_percent_spr_better": int((p.spr_error_pp.abs() < p.tir_error_pp.abs()).sum()), "E5_percent_runs": int(len(p)),
        "E5": bool((h.tir_error_pp.abs() < h.spr_error_pp.abs()).all() and (p.spr_error_pp.abs() < p.tir_error_pp.abs()).all()),
        "E6_max_direct_error_pp_no_liquidation": float(no_liq.direct_error_pp.abs().max()),
        "E6_runs_without_liquidation": int(len(no_liq)),
        "E6_other_formula_over_5pp_bound": int((b.other_formula_error_pp.abs() > 5).sum()),
        "E6": bool(no_liq.direct_error_pp.abs().max() <= 1e-3 and (b.other_formula_error_pp.abs() > 5).sum() >= 50),
        "E7_runs_flagged_bound": int((b.shift_flagged > 0).sum()),
        "E7_large_tir_steps": large, "E7_large_tir_flagged": int(b.shift_flagged_large_tir.sum()),
        "E7_share": float(b.shift_flagged_large_tir.sum() / large) if large else float("nan"),
        "E7": bool((b.shift_flagged > 0).all() and large > 0 and b.shift_flagged_large_tir.sum() / large >= 0.90),
        "comparison_runs_shift_flagged": int((n.shift_flagged > 0).sum()),
    }
    return out


def groups(d: pd.DataFrame) -> list[dict]:
    rows = []
    for label, sel in GROUPS:
        g = d[sel(d)]
        if not len(g):
            continue
        rows.append(dict(
            group=label, runs=int(len(g)),
            reported=float(g.reported_pct.median()), booked=float(g.booked_pct.median()), credit=float(g.credit_pp.median()),
            width=float(g.width_pp.median()), lower=float(g.lower_pct.median()),
            width_over_credit_min=float(g.width_over_credit.min()), width_over_credit_max=float(g.width_over_credit.max()),
            steps_outside=int(g.steps_outside.sum()), contained=int(g.contains_run.sum()),
            flag_reconcile=int(g.flag_reconcile.sum()), flag_tir_gap=int(g.flag_tir_gap.sum()), flag_width=int(g.flag_width.sum()),
            tir_error=float(g.tir_error_pp.abs().median()), tir_error_max=float(g.tir_error_pp.abs().max()),
            spr_error=float(g.spr_error_pp.abs().median()), spr_error_max=float(g.spr_error_pp.abs().max()),
            direct_error_max=float(g.direct_error_pp.abs().max()),
            other_error=float(g.other_formula_error_pp.abs().median()), other_error_min=float(g.other_formula_error_pp.abs().min()),
            other_error_max=float(g.other_formula_error_pp.abs().max()),
            shift_flagged_runs=int((g.shift_flagged > 0).sum()),
        ))
    return rows


def cap_table(d: pd.DataFrame) -> list[dict]:
    rows = []
    for variant, label in (("L7_thesis_like", "cap off (as run in the audit)"), ("L7_thesis_like_capon", "cap on")):
        for region, zone in (("original", "DK1"), ("v2", "DK2")):
            g = d[(d.variant == variant) & (d.region == region)]
            m, an = g[g.agent == "marl"], g[g.agent == "anchor"]
            if not len(m):
                continue
            rows.append(dict(label=label, zone=zone, seeds=int(len(m)), marl_reported_mean=float(m.reported_pct.mean()),
                             marl_reported_median=float(m.reported_pct.median()), marl_booked_mean=float(m.booked_pct.mean()),
                             marl_booked_median=float(m.booked_pct.median()),
                             anchor_reported=float(an.reported_pct.mean()) if len(an) else float("nan"),
                             anchor_booked=float(an.booked_pct.mean()) if len(an) else float("nan"),
                             floor_runs=int((m.floors_real > 0).sum()), flagged=int(m.flag_reconcile.sum()), max_credit=float(m.credit_pp.max())))
    return rows


def fmt(x: float) -> str:
    return f"{x:,.1f}".replace(",", "{,}").replace("-", "$-$") if abs(x) >= 0.05 else "0.0"


def write_table(rows: list[dict], path: Path) -> None:
    lines = [r"\begin{tabular}{@{}lrrrrrrrrrrr@{}}", r"\toprule",
             r"Runs & $n$ & Reported & Booked & Credit & Width & Outside & \multicolumn{3}{c}{Flagged by} & \multicolumn{2}{c}{Audit error} \\",
             r"\cmidrule(lr){8-10}\cmidrule(l){11-12}",
             r" & & & & & & & ledger & TIR gap & width & TIR & SPR \\", r"\midrule"]
    for r in rows:
        lines.append(f"{r['group'].capitalize()} & {r['runs']} & {fmt(r['reported'])} & {fmt(r['booked'])} & {fmt(r['credit'])} & {fmt(r['width'])} & "
                     f"{r['steps_outside']} & {r['flag_reconcile']} & {r['flag_tir_gap']} & {r['flag_width']} & {fmt(r['tir_error'])} & {fmt(r['spr_error'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--jobs", default=str(HERE / "JOBS.txt"))
    ap.add_argument("--audit-gap", type=float, default=None, help="largest |JSON - campaign| from check_against_audit.py")
    ap.add_argument("--print", action="store_true", dest="to_stdout")
    a = ap.parse_args()
    res = Path(a.results)
    d = load(res, Path(a.jobs))
    summary = dict(decisions=decide(d, a.audit_gap), groups=groups(d), cap=cap_table(d))
    if a.to_stdout:
        print(json.dumps(summary, indent=1))
        return 0
    (res / "engine_twin_summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    write_table(summary["groups"], res / "tab_engine_twin.tex")
    dd = summary["decisions"]
    lines = ["# Engine twin check: decisions", "",
             "Exploratory. No agent trained; frozen checkpoints of the engine audit. Rules: PROTOCOL.md.", "",
             f"Runs: {dd['runs']} ({dd['bound']} floor bound, {dd['comparison']} comparison), plus {dd['cap_on']} with the percent payoff's cap on.", "",
             "| Rule | Result | Decision |", "|---|---|---|",
             f"| G1 re-evaluation | largest difference from the audit's table {dd['G1_largest_json_minus_campaign_pp']} pp | {'met' if dd['G1_largest_json_minus_campaign_pp'] is not None and dd['G1_largest_json_minus_campaign_pp'] <= 1e-6 else 'NOT MET'} |",
             f"| G2 replay | largest parity error {dd['G2_max_parity_dkk']:.3g} DKK, {dd['G2_unknown_writes']} unknown writes | {'met' if dd['G2_max_parity_dkk'] <= TOL and dd['G2_unknown_writes'] == 0 else 'NOT MET'} |",
             f"| E1 forgiving on the real path | smallest credit {dd['E1_min_real_credit_dkk']:.3g} DKK | {'supported' if dd['E1'] else 'NOT SUPPORTED'} |",
             f"| E2 bracket contains booked P&L | {dd['E2_steps_outside']} steps outside; {dd['E2_runs_containing']} of {dd['runs']} runs contained | {'supported' if dd['E2'] else 'NOT SUPPORTED'} |",
             f"| E3 TIR gap flags | {dd['E3_flags_bound']} of {dd['bound']} bound, {dd['E3_flags_comparison']} of {dd['comparison']} comparison (ledger: {dd['reconcile_flags_bound']}, {dd['reconcile_flags_comparison']}) | {'supported' if dd['E3'] else 'NOT SUPPORTED'} |",
             f"| E4 width flags | {dd['E4_flags_bound']} of {dd['bound']} bound ({dd['E4_flags_comparison']} of {dd['comparison']} comparison, no rule) | {'supported' if dd['E4'] else 'NOT SUPPORTED'} |",
             f"| E5 estimators split by payoff | TIR closer in {dd['E5_horizon_tir_better']} of {dd['E5_horizon_runs']} volume-payoff runs, SPR closer in {dd['E5_percent_spr_better']} of {dd['E5_percent_runs']} percent-payoff runs | {'supported' if dd['E5'] else 'NOT SUPPORTED'} |",
             f"| E6 direct calculation | right formula: largest error {dd['E6_max_direct_error_pp_no_liquidation']:.3g} pp over {dd['E6_runs_without_liquidation']} runs without a liquidation; other generation's formula over 5 pp in {dd['E6_other_formula_over_5pp_bound']} of {dd['bound']} | {'supported' if dd['E6'] else 'NOT SUPPORTED'} |",
             f"| E7 cost-shift test | flags {dd['E7_runs_flagged_bound']} of {dd['bound']} bound runs and {dd['E7_large_tir_flagged']} of {dd['E7_large_tir_steps']} steps where TIR errs by over 0.01 pp ({dd['E7_share']:.3f}) | {'supported' if dd['E7'] else 'NOT SUPPORTED'} |",
             ""]
    (res / "VERDICTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
