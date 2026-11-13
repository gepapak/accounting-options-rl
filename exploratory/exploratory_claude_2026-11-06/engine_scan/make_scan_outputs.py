r"""Outputs of the engine reward-diagram campaign for the paper. Exploratory, written after the rules were hashed; reads scan_campaign/scan_runs.csv and the scans, changes nothing there.
Writes tab_engine_scan.tex, scan_summary.json. Also the post hoc fallback (the pair's half-difference where no slope repeats), which was written after rule S6 failed and is reported as post hoc."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))
from reward_diagram import read_diagram  # noqa: E402

E0 = 662068965.5172415
PP = 100 / E0
FOLDER = HERE / "scan_campaign"


def fallback_sum(p):
    z = dict(np.load(p, allow_pickle=False))
    grid = z["read_grid"]
    phi = z["phi"]
    idx = {float(t): j for j, t in enumerate(grid)}
    r1, rm = phi[:, idx[1.0]], phi[:, idx[-1.0]]
    C = z["own_1"]
    tol = 1e-3 + 1e-9 * np.abs(z["cash_pre"] + z["mtm_pre"])
    cols = [str(c) for c in z["real_columns"]]
    real = {c: z["real"][:, i] for i, c in enumerate(cols)}
    x = real["gross"] - (real["access"] + real["roll"] + real["transaction"] + real["funding"] - real["battery"])
    est = r1.copy()
    fell = 0
    for k in np.flatnonzero(np.abs(r1 + rm + 2.0 * C) > tol):
        have = ~np.isnan(phi[k])
        order = np.argsort(grid[have])
        d = read_diagram(grid[have][order], phi[k][have][order], rel_tol=1e-9, abs_tol=1e-3)
        if d.s_plus == 0.0 and d.s_minus == 0.0 and abs(r1[k] - rm[k]) > tol[k]:
            est[k] = 0.5 * (r1[k] - rm[k]) - C[k]
            fell += 1
        else:
            est[k] = d.gross - C[k]
    return float(np.abs(est - x).sum() * PP), fell, float((np.abs(est - x) < 1.0).mean())


def f(v, d=1):
    return f"{v:,.{d}f}".replace(",", "{,}")


def rng(s, d=1):
    return f"{f(s.min(), d)}--{f(s.max(), d)}"


def main():
    df = pd.read_csv(FOLDER / "scan_runs.csv")
    fb = [fallback_sum(FOLDER / "scans" / f"{r}.npz") for r in df.run]
    df["sum_err_fb"], df["fell"], df["exact_fb"] = zip(*fb)
    df.to_csv(FOLDER / "scan_runs_with_fallback.csv", index=False)
    V, K, N, P = (df[df.group == g] for g in "VKNP")
    K7, K5 = K[K.run.str.contains("seed7__")], K[K.run.str.contains("seed5001")]
    PF, P42 = P[~P.run.str.contains("seed42")], P[P.run.str.contains("seed42")]
    rows = [("Volume payoff, floor bound", V), ("Percentage payoff uncapped, floor bound", PF), ("Percentage payoff uncapped, floor not bound", P42), ("Cap on, floor bound", K5), ("Cap on, floor not bound", K7), ("Nothing forgiven (L0, L1, L3 rule)", N)]
    lines = [r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule", r"Group & Runs & Clones per step & Exact (\%) & TIR exact (\%) & Error, reading & Error, TIR & Error, SPR \\", r"\midrule"]
    for name, g in rows:
        lines.append(f"{name} & {len(g)} & {rng(g.mean_clones)} & {rng(100 * g.step_exact_share, 1)} & {rng(100 * g.tir_exact_share, 1)} & {rng(g.sum_err, 3)} & {rng(g.sum_err_tir, 1)} & {rng(g.sum_err_spr, 1)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (FOLDER / "tab_engine_scan.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    groups = {}
    for name, g in rows:
        groups[name] = {c: [float(g[c].min()), float(g[c].median()), float(g[c].max())] for c in ("mean_clones", "step_exact_share", "tir_exact_share", "spr_exact_share", "sum_err", "sum_err_tir", "sum_err_spr", "sum_err_fb", "audit_err", "audit_err_tir", "audit_err_spr", "cap_share", "booked", "reported")}
        groups[name]["runs"] = int(len(g))
    out = dict(runs=int(len(df)), reads=int(df.reads.sum()), clones=float((df.reads * df.mean_clones).sum()), diagrams=int(df.n_diagrams.sum()), groups=groups,
               flat_try=int(V.flat_try.sum()), flat_hit=int(V.flat_hit.sum()), funded_exact=int(df.funded_exact.sum()), zerocost_exact=int(df.zerocost_exact.sum()), s2_outside=int(df.s2_outside.sum()), s2_worst=float(df.s2_worst.max()), s1_worst_VN=float(pd.concat([V, N]).s1_twin_replay.max()),
               g1_dr=float(df.g1_dr.max()), g1_dcash=float(df.g1_dcash.max()), g2=float(df.g2_phi1.max()))
    (FOLDER / "scan_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\n".join(lines))
    print(json.dumps(out, indent=1)[:3000])


if __name__ == "__main__":
    main()
