r"""Analysis of the engine reward-diagram campaign (PROTOCOL_SCAN.md). The estimator reads only the clones' rewards and the declared own cost; the ledger is used only to score it.
Usage: python analyze_scan.py [folder, default scan_campaign] -> <folder>/scan_runs.csv, <folder>/scan_verdicts.json and a printed table."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WORK = HERE.parent / "engine_twin_2026-11-05"
sys.path.append(str(HERE))   # the reader frozen with this protocol (a copy of twin_reward/reward_diagram.py)
from reward_diagram import read_diagram  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
E0 = 662068965.5172415
PP = 100 / E0


def group_of(name):
    v = name.split("__")[0]
    a = name.split("__")[1]
    s = name.split("__")[2]
    if v == "L7_thesis_like_capon":
        return "K"
    if v == "L7_thesis_like":
        return "P"
    if v in ("L7_minus_percent_mtm",) or (v == "L3_price_taker_no_solvency" and a == "marl"):
        return "V"
    return "N"


def estimate(z):
    """The reading of every read step: (estimate, width mask, tail gross)."""
    grid = z["read_grid"]
    phi = z["phi"]
    idx = {float(t): j for j, t in enumerate(grid)}
    r1, rm = phi[:, idx[1.0]], phi[:, idx[-1.0]]
    C = z["own_1"]
    tol = 1e-3 + 1e-9 * np.abs(z["cash_pre"] + z["mtm_pre"])
    width = r1 + rm + 2.0 * C
    est = r1.copy()
    gross = np.full(len(r1), np.nan)
    escalated = np.abs(width) > tol
    for k in np.flatnonzero(escalated):
        have = ~np.isnan(phi[k])
        order = np.argsort(grid[have])
        ts, ps = grid[have][order], phi[k][have][order]
        d = read_diagram(ts, ps, rel_tol=1e-9, abs_tol=1e-3)
        gross[k] = d.gross
        est[k] = d.gross - C[k]
    return est, escalated, gross, width, tol, C, r1, rm


def diagram_stats(z):
    """On the sampled full diagrams: cap signature, flat-level read-out, kink counts, comparators."""
    if "d_line" not in z:
        return {}
    fields = [str(f) for f in z["fields"]]
    grid = z["grid"]
    order = np.argsort(grid)
    ts = grid[order]
    phi = z["d_line"][:, :, fields.index("r")][:, order]
    cols = [str(c) for c in z["real_columns"]]
    real = {c: z["d_real"][:, i] for i, c in enumerate(cols)}
    gross = real["gross"]
    C = real["access"] + real["roll"] + real["transaction"] + real["funding"] - real["battery"]
    x = gross - C
    moved = np.abs(gross) > 1.0
    i0 = int(np.argmin(np.abs(ts)))
    cap = conv_near = 0
    flat_try = flat_hit = 0
    n_conv, n_conc = [], []
    for k in range(len(phi)):
        d = read_diagram(ts, phi[k], rel_tol=1e-9, abs_tol=1e-3)
        n_conv.append(len(d.convex_kinks))
        n_conc.append(len(d.concave_kinks))
        if moved[k]:
            cap += any(abs(t) <= 2.0 for t, _ in d.concave_kinks)
            conv_near += any(abs(t) <= 2.0 for t, _ in d.convex_kinks)
        if z["d_cash_pre"][k] > 1e3:
            s = d.slopes
            flats = [phi[k][i] for i in range(len(s) - 1) if abs(s[i]) * (ts[i + 1] - ts[i]) <= d.tol and abs(s[i + 1]) * (ts[i + 2] - ts[i + 1]) <= d.tol]
            below = [v for v in flats if v < phi[k][i0] - d.tol]
            if below:
                flat_try += 1
                flat_hit += abs(min(below) + z["d_cash_pre"][k] + z["d_mtm_pre"][k]) <= 1.0
    # comparators: zero-cost twin and funded clone (extra access)
    zf = {f: z["d_zero"][:, :, fields.index(f)] for f in fields}
    z1, zm = zf["r"][:, 0], zf["r"][:, 1]
    b1, bm = zf["battery"][:, 0], zf["battery"][:, 1]
    a0, c0 = z1 - b1, zm - bm
    u0, l0 = np.maximum(a0, c0), np.minimum(a0, c0)
    sg = np.where(z1 >= zm, 1.0, -1.0)
    common = l0 > np.maximum(-np.minimum(b1, bm), 0) + 1e-3
    tol = 1e-9 * (1 + np.abs(z["d_cash_pre"]))
    gz = np.where(np.abs(z1 - zm) > tol, np.where(common, (u0 - l0) / 2, u0), 0.0)
    zt = sg * gz - C
    fr = z["d_funded"][:, 0, fields.index("r")]
    return dict(n_diagrams=len(phi), moved=int(moved.sum()), cap_near=int(cap), convex_near=int(conv_near), cap_share=float(cap / max(1, moved.sum())), flat_try=flat_try, flat_hit=flat_hit,
                conv_median=float(np.median(n_conv)), conc_median=float(np.median(n_conc)), affine=int(sum(a == 0 and b == 0 for a, b in zip(n_conv, n_conc))),
                zerocost_sum=float(np.abs(zt - x).sum() * PP), funded_sum=float(np.abs(fr - x).sum() * PP), funded_exact=int((np.abs(fr - x) < 1.0).sum()), zerocost_exact=int((np.abs(zt - x) < 1.0).sum()))


def analyse(p):
    name = Path(p).stem
    z = dict(np.load(p, allow_pickle=False))
    g = group_of(name)
    cols = [str(c) for c in z["real_columns"]]
    real = {c: z["real"][:, i] for i, c in enumerate(cols)}
    C_real = real["access"] + real["roll"] + real["transaction"] + real["funding"] - real["battery"]
    x = real["gross"] - C_real
    est, esc, gross, width, tol, C, r1, rm = estimate(z)
    err = est - x
    tir = np.where(np.abs(r1 - rm) > tol, np.where(r1 >= rm, 1.0, -1.0) * (np.maximum(r1, rm) + C) - C, -C)
    spr = (r1 - rm) / 2 - C
    out = dict(run=name, group=g, reads=len(x), escalated=int(esc.sum()), stage3=int((z["stage"] == 3).sum()), mean_clones=float(z["n"].mean()), wall_median=float(np.median(z["wall"])), wall_total_min=float(z["wall"].sum() / 60))
    # gates against the audit's trace of the same job
    tr = WORK / "traces" / f"{name}.npz"
    this = Path(p).parent.parent / "traces" / f"{name}.npz"
    a = np.load(tr, allow_pickle=False)
    b = np.load(this, allow_pickle=False)
    ca, cb = [str(c) for c in a["columns"]], [str(c) for c in b["columns"]]
    ra, rb = a["rows"], b["rows"]
    out["g1_steps_equal"] = bool(len(ra) == len(rb))
    out["g1_dr"] = float(np.abs(ra[:, ca.index("r")] - rb[:, cb.index("r")]).max()) if len(ra) == len(rb) else float("inf")
    out["g1_dcash"] = float(np.abs(ra[:, ca.index("cash_post")] - rb[:, cb.index("cash_post")]).max()) if len(ra) == len(rb) else float("inf")
    out["g2_phi1"] = float(np.abs(r1 - real["r"]).max())
    out["s1_twin_replay"] = float(np.abs(rm - real["r_twin"]).max())
    lo = -rm - 2.0 * C
    out["s2_outside"] = int(((x < lo - 1.0) | (x > r1 + 1.0)).sum())
    out["s2_worst"] = float(max((lo - x).max(), (x - r1).max()))
    # run-level audit over the whole trace: estimate on read steps, -C elsewhere
    allC = rb[:, cb.index("access")] + rb[:, cb.index("roll")] + rb[:, cb.index("transaction")] + rb[:, cb.index("funding")] - rb[:, cb.index("battery")]
    allg = rb[:, cb.index("gross")]
    allx = allg - allC
    allt = rb[:, cb.index("t")].astype(int)
    mask = np.isin(allt, z["t"])
    offgrid_gross = float(np.abs(allg[~mask]).sum() * PP)
    booked = float(allx.sum() * PP)
    reported = float(rb[:, cb.index("r")].sum() * PP)
    base = float((-allC[~mask]).sum() * PP)
    aud = base + float(est.sum() * PP)
    aud_tir = base + float(tir.sum() * PP)
    aud_spr = base + float(spr.sum() * PP)
    out.update(booked=booked, reported=reported, offgrid_gross=offgrid_gross, audit=aud, audit_tir=aud_tir, audit_spr=aud_spr, audit_err=abs(aud - booked), audit_err_tir=abs(aud_tir - booked), audit_err_spr=abs(aud_spr - booked))
    out["step_exact"] = int((np.abs(err) < 1.0).sum())
    out["step_exact_share"] = float((np.abs(err) < 1.0).mean())
    out["tir_exact_share"] = float((np.abs(tir - x) < 1.0).mean())
    out["spr_exact_share"] = float((np.abs(spr - x) < 1.0).mean())
    out["sum_err"] = float(np.abs(err).sum() * PP)
    out["sum_err_tir"] = float(np.abs(tir - x).sum() * PP)
    out["sum_err_spr"] = float(np.abs(spr - x).sum() * PP)
    out["max_err"] = float(np.abs(err).max() * PP)
    out["width_sum"] = float(width.sum() * PP)
    out["credit_read"] = float((r1 - x).sum() * PP)
    out.update(diagram_stats(z))
    return out


def verdicts(df):
    v = {}
    V, K, N, P = (df[df.group == g] for g in "VKNP")
    PF = P[~P.run.str.contains("seed42")]
    K7 = K[K.run.str.contains("seed7__")]
    K5 = K[K.run.str.contains("seed5001")]
    v["G1"] = bool((df.g1_steps_equal & (df.g1_dr <= 1e-3) & (df.g1_dcash <= 1e-3)).all())
    v["G2"] = bool((df.g2_phi1 <= 1e-3).all())
    v["S1"] = bool((pd.concat([V, N]).s1_twin_replay <= 1.0).all())
    v["S2"] = bool((df.s2_outside == 0).all())
    v["S3"] = bool(((V.audit_err <= 0.01) & (V.audit_err < V.audit_err_tir) & (V.audit_err < V.audit_err_spr)).all())
    v["S4"] = bool((V.step_exact_share >= 0.98).all())
    v["S5"] = bool((N.audit_err <= 0.01).all() and (K7.sum_err <= 0.01).all())
    v["S6"] = bool((K5.sum_err <= 1.0).all())
    v["S7"] = bool(((PF.sum_err < PF.sum_err_tir) & (PF.sum_err <= 2.0 * PF.sum_err_spr)).all())
    v["S8"] = bool((K.cap_share >= 0.90).all() and (P.cap_share <= 0.05).all())
    v["S9"] = bool((V.flat_hit.sum() >= 0.99 * V.flat_try.sum()) and V.flat_try.sum() > 0)
    v["S10"] = bool((V.mean_clones <= 8).all() and (N.mean_clones <= 2.5).all())
    return v


def main():
    folder = HERE / (sys.argv[1] if len(sys.argv) > 1 else "scan_campaign")
    rows = [analyse(p) for p in sorted((folder / "scans").glob("*.npz"))]
    df = pd.DataFrame(rows).sort_values(["group", "run"])
    df.to_csv(folder / "scan_runs.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 60)
    show = ["run", "reads", "escalated", "stage3", "mean_clones", "booked", "reported", "audit_err", "audit_err_tir", "audit_err_spr", "step_exact_share", "tir_exact_share", "spr_exact_share", "sum_err", "sum_err_tir", "sum_err_spr", "cap_share", "flat_try", "flat_hit", "s2_outside"]
    print(df[[c for c in show if c in df]].round(4).to_string(index=False))
    if set("VKNP") <= set(df.group):
        v = verdicts(df)
        print(json.dumps(v, indent=1))
        (folder / "scan_verdicts.json").write_text(json.dumps(v, indent=1))
    else:
        print("partial campaign: groups present", sorted(set(df.group)))


if __name__ == "__main__":
    main()
