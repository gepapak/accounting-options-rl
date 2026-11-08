r"""Every number the manuscript quotes from the engine reward-diagram campaign, recomputed from scan_runs_with_fallback.csv, no_cost_reading.json, mechanisms_scan.json and the scans. Writes scan_campaign/scan_numbers.json."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))
from reward_diagram import read_diagram  # noqa: E402

F = HERE / "scan_campaign"
d = pd.read_csv(F / "scan_runs_with_fallback.csv")
ver = json.loads((F / "scan_verdicts.json").read_text())
nc = json.loads((F / "no_cost_reading.json").read_text())
me = json.loads((F / "mechanisms_scan.json").read_text())
V, K, N, P = (d[d.group == g] for g in "VKNP")
K7, K5 = K[K.run.str.contains("seed7__")], K[K.run.str.contains("seed5001")]
PF, P42 = P[~P.run.str.contains("seed42")], P[P.run.str.contains("seed42")]
six = V[V.run.str.startswith("L7_minus_percent")]


def span(s):
    return [float(s.min()), float(s.max())]


# the flat level's distance from minus (cash + bucket), volume runs
gaps = []
for r in V.run:
    z = dict(np.load(F / "scans" / f"{r}.npz", allow_pickle=False))
    fields = [str(f) for f in z["fields"]]
    g = z["grid"]
    o = np.argsort(g)
    ts = g[o]
    phi = z["d_line"][:, :, fields.index("r")][:, o]
    i0 = int(np.argmin(np.abs(ts)))
    for k in range(len(phi)):
        if z["d_cash_pre"][k] <= 1e3:
            continue
        di = read_diagram(ts, phi[k], rel_tol=1e-9, abs_tol=1e-3)
        s = di.slopes
        flats = [phi[k][i] for i in range(len(s) - 1) if abs(s[i]) * (ts[i + 1] - ts[i]) <= di.tol and abs(s[i + 1]) * (ts[i + 2] - ts[i + 1]) <= di.tol]
        below = [v for v in flats if v < phi[k][i0] - di.tol]
        if below:
            bal = z["d_cash_pre"][k] + z["d_mtm_pre"][k]
            gaps.append((abs(min(below) + bal), abs(min(below) + bal) / bal))
gaps = np.array(gaps)
vnc = [nc[r]["err_free"] for r in V.run]
out = dict(
    verdicts=ver, runs=int(len(d)), read_steps=int(d.reads.sum()), clones=int(round((d.reads * d.mean_clones).sum())), diagrams=int(d.n_diagrams.sum()),
    g1_dr=float(d.g1_dr.max()), g1_dcash=float(d.g1_dcash.max()), g2=float(d.g2_phi1.max()),
    volume=dict(runs=int(len(V)), clones=span(V.mean_clones), stage3=int(V.stage3.sum()), audit_err=span(V.audit_err), audit_err_tir=span(V.audit_err_tir), audit_err_spr=span(V.audit_err_spr), exact=span(V.step_exact_share), tir_exact=span(V.tir_exact_share), spr_exact=span(V.spr_exact_share), sum_err=span(V.sum_err), sum_tir=span(V.sum_err_tir), sum_spr=span(V.sum_err_spr)),
    six=dict(runs=int(len(six)), booked=span(six.booked), reported=span(six.reported), audit_err=span(six.audit_err), audit_err_tir=span(six.audit_err_tir), audit_err_spr=span(six.audit_err_spr)),
    percent_bound=dict(runs=int(len(PF)), clones=span(PF.mean_clones), exact=span(PF.step_exact_share), tir_exact=span(PF.tir_exact_share), sum_err=span(PF.sum_err), sum_tir=span(PF.sum_err_tir), sum_spr=span(PF.sum_err_spr)),
    percent_free=dict(runs=int(len(P42)), sum_err=span(P42.sum_err), sum_tir=span(P42.sum_err_tir), sum_spr=span(P42.sum_err_spr)),
    cap_floor=dict(runs=int(len(K5)), sum_err=span(K5.sum_err), sum_fb=span(K5.sum_err_fb), sum_tir=span(K5.sum_err_tir), sum_spr=span(K5.sum_err_spr), clones=span(K5.mean_clones), exact=span(K5.step_exact_share), exact_fb=span(K5.exact_fb)),
    cap_free=dict(runs=int(len(K7)), sum_err=span(K7.sum_err), clones=span(K7.mean_clones)),
    nothing=dict(runs=int(len(N)), clones=span(N.mean_clones), audit_err_max=float(N.audit_err.max())),
    cap_share_on=span(K.cap_share), cap_share_off=span(P.cap_share), cap_share_volume=span(V.cap_share),
    fallback_worst_over_best_twin=float((d.sum_err_fb - np.minimum(d.sum_err_tir, d.sum_err_spr)).max()),
    flat=dict(tried=int(V.flat_try.sum()), within_1_dkk=int(V.flat_hit.sum()), within_1000_dkk=int((gaps[:, 0] <= 1000.0).sum()), largest_dkk=float(gaps[:, 0].max()), largest_rel=float(gaps[:, 1].max()), n=int(len(gaps))),
    twin_vs_replay_group_vn_worst=float(pd.concat([V, N]).s1_twin_replay.max()), twin_replay_gap_steps_vn=int(sum(me[r]["twin_replay_gap_steps"] for r in pd.concat([V, N]).run)),
    bracket=dict(outside=int(sum(v["outside"] for v in me.values())), reads=int(sum(v["reads"] for v in me.values())), largest_dkk=float(max(v["largest_dkk"] for v in me.values())), largest_points=float(max(v["largest_points"] for v in me.values())), outside_on_cost_gap=int(sum(v["of_outside_with_cost_gap"] for v in me.values())), outside_with_each_branch_cost=int(sum(v["outside_with_each_branch_cost"] for v in me.values()))),
    no_declared_cost_volume=[float(min(vnc)), float(max(vnc))],
    funded=dict(exact=int(d.funded_exact.sum()), diagrams=int(d.n_diagrams.sum()), anchor_exact=int(d[d.run.str.contains("anchor")].funded_exact.sum()), anchor_diagrams=int(d[d.run.str.contains("anchor")].n_diagrams.sum())), zerocost_exact=int(d.zerocost_exact.sum()),
    percent_misses_position_moves=int(sum(me[r]["of_which_position_moves_within_8"] for r in PF.run)), percent_misses=int(sum(me[r]["reading_misses_sampled"] for r in PF.run)),
)
(F / "scan_numbers.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=0)[:4200])
