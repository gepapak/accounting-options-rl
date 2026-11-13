"""Post-hoc look at the runs behind the decisions (exploratory; written after the results were read)."""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
d = pd.read_csv(HERE / "results" / "engine_twin_runs.csv")
bound = set()
section = ""
for line in (HERE / "JOBS.txt").read_text(encoding="utf-8").splitlines():
    if line.startswith("#"):
        section = line
    elif line.strip() and "floor bound" in section:
        bound.add(line.strip())
d["bound"] = d.name.isin(bound)
d["capon"] = d.variant.str.endswith("_capon")
a = d[~d.capon]
b, n = a[a.bound], a[~a.bound]
pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_colwidth", 60)
cols = ["name", "reported_pct", "booked_pct", "gap_pp", "credit_pp", "credit_twin_pp", "width_pp", "tir_error_pp", "spr_error_pp", "floors_real", "steps_credit", "shift_flagged", "shift_informative", "liquidations"]

print("steps, all 93:", int(a.steps.sum()), " credited real:", int(a.steps_credit.sum()), " credited twin:", int(a.steps_credit_twin.sum()))
print("identified steps:", int(a.steps_identified.sum()), " TIR wrong:", int(a.steps_tir_wrong.sum()), " TIR outside bracket:", int(a.steps_tir_outside.sum()))
print("same flagged sets, reconcile vs TIR gap:", set(b[b.flag_reconcile].name) == set(b[b.flag_tir_gap].name), " vs width:", set(b[b.flag_reconcile].name) == set(b[b.flag_width].name))
print("\nnot flagged by reconciliation:\n", b[~b.flag_reconcile][cols].round(3).to_string())
print("\nnot flagged by the shift test:\n", b[b.shift_flagged == 0][cols].round(3).to_string())
h = b[b["mode"] == "horizon"]
print("\nE5 exceptions (volume payoff, SPR closer):\n", h[h.tir_error_pp.abs() >= h.spr_error_pp.abs()][cols].round(3).to_string())
print("\nwidth flag on a comparison run:\n", n[n.flag_width][cols].round(3).to_string())
print("\nsmallest width among comparison runs with positive width:\n", n[n.width_pp > 1e-6][["name", "width_pp", "steps_credit_twin", "spr_error_pp", "tir_error_pp"]].round(3).to_string())
print("\nother formula under 5 pp, floor bound:\n", b[b.other_formula_error_pp.abs() <= 5][["name", "other_formula_error_pp", "direct_error_pp", "gross_pp"]].round(3).to_string())
print("\nliquidation runs, direct error:\n", a[a.liquidations > 0][["name", "liquidations", "direct_error_pp", "direct_max_abs_step_error_pp", "booked_pct"]].round(4).to_string())
print("\nshift-flagged comparison runs by variant and agent:\n", n.assign(fl=n.shift_flagged > 0).groupby(["variant", "agent"]).fl.agg(["sum", "size"]).to_string())
print("\nshare of informative steps flagged, comparison:", round(n.shift_flagged.sum() / n.shift_informative.sum(), 4), " floor bound:", round(b.shift_flagged.sum() / b.shift_informative.sum(), 4))
print("flagged with credit, floor bound:", int(b.shift_flagged_with_credit.sum()), "of", int(b.shift_with_credit.sum()))
for label, g in (("volume", b[b["mode"] == "horizon"]), ("percent", b[b["mode"] == "percent"])):
    print(f"\n{label}: n={len(g)}  gap median {g.gap_pp.median():.1f}  TIR |err| median {g.tir_error_pp.abs().median():.1f} range {g.tir_error_pp.abs().min():.2f}-{g.tir_error_pp.abs().max():.1f}; SPR |err| median {g.spr_error_pp.abs().median():.1f} range {g.spr_error_pp.abs().min():.2f}-{g.spr_error_pp.abs().max():.1f}; other formula median {g.other_formula_error_pp.abs().median():.1f}; width/credit {g.width_over_credit.min():.2f}-{g.width_over_credit.max():.2f}")
    print("   lower end minus booked: median", round((g.lower_pct - g.booked_pct).median(), 1), " share of TIR error over gap: median", round((g.tir_error_pp.abs() / g.gap_pp.abs()).median(), 3), " SPR:", round((g.spr_error_pp.abs() / g.gap_pp.abs()).median(), 3))
gg = a[(a.variant == "L7_minus_percent_mtm") & (a.agent == "marl")]
print("\nL7 minus percent, tracer's booked:", gg.booked_pct.round(1).tolist(), "reported:", gg.reported_pct.round(1).tolist(), "TIR err", gg.tir_error_pp.round(1).tolist(), "SPR err", gg.spr_error_pp.round(1).tolist())
print("TIR gap vs ledger gap, floor bound, largest relative difference among flagged:", round(((b.reported_pct - b.tir_pct).abs() / b.gap_pp.abs())[b.flag_reconcile].describe()[["min", "50%", "max"]], 3).to_dict())
