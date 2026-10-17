"""Study N aggregation (PREREGISTRATION_N.md): verdicts for N1-N4, T1-T3, X1 from results_n/jobs/*.json.
Bootstrap: 10,000 resamples of the 10 runs per arm (5 seeds x 2 zones), independently per arm, percentile intervals."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
TAG = sys.argv[1] if len(sys.argv) > 1 else ""
JOBS = HERE / f"results_n{TAG}" / "jobs"
rng = np.random.default_rng(20261014)
B = 10_000
ZN, ZK = (-0.5, -0.25, -0.1, -0.02), (96, 48, 12)
TL, TT = (0.02, 0.1, 0.25), (1, 4, 16, 48, 96)
TTN = (1, 4, 16, 48, 96, 192)
TNEG = (-0.25, -0.1)

def load():
    rows = []
    for f in sorted(JOBS.glob("*.json")):
        r = json.loads(f.read_text()); p = r["probes"]
        d = dict(arm=r["arm"], seed=r["seed"], region=r["region"], rep=r["drawmean_reported_pct"], book=r["drawmean_booked_pct"])
        if r["arm"][0] == "Z":
            d["zbar"] = np.mean([p[f"zombie_n{n:g}_k{k}"] for n in ZN for k in ZK])
            d["z_n01_k48"] = p["zombie_n-0.1_k48"]; d["solv_e01"] = p["solvent_e0.1"]
            d["zombie_maxlev_share"] = float(np.mean([x["zombie_maxlev_share"] for x in r["draws"]]))
            d["forgiven_pct"] = float(np.mean([x["forgiven_pct"] for x in r["draws"]]))
        else:
            d["tbar"] = np.mean([p[f"L{l:g}_tau{t}"] for l in TL for t in TT])
            d["t_L01_tau16"] = p["L0.1_tau16"]
            d["tneg"] = np.mean([p[f"L{l:g}_tau{t}"] for l in TNEG for t in TTN])
            d["t_L01_tau192"] = p["L0.1_tau192"]
        rows.append(d)
    return pd.DataFrame(rows)

def gap(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    bs = rng.choice(a, (B, len(a))).mean(1) - rng.choice(b, (B, len(b))).mean(1)
    return a.mean() - b.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

def main():
    D = load()
    g = {arm: D[D.arm == arm] for arm in ("ZFG", "ZBK", "TFG", "TBK")}
    n = {k: len(v) for k, v in g.items()}
    out = [f"# Study N verdicts{' (' + TAG + ')' if TAG else ''}", "", f"runs per arm: {n}", ""]
    V = []
    m, lo, hi = gap(g["ZFG"].zbar, g["ZBK"].zbar)
    V.append(("N1", "mean zombie-probe |lev| ZFG - ZBK >= 8, lower bound > 4", f"{m:.2f} [{lo:.2f}, {hi:.2f}]", m >= 8 and lo > 4))
    c = int((g["ZFG"].z_n01_k48 == 16).sum())
    V.append(("N2", "ZFG |lev| = 16 at (n=-0.1, k=48) in >= 8 of 10 runs", f"{c}/{n['ZFG']}", c >= 8))
    m, lo, hi = gap(g["ZFG"].solv_e01, g["ZBK"].solv_e01)
    V.append(("N3", "solvent |lev| at e=0.1: ZFG - ZBK, lower bound > 0", f"{m:.2f} [{lo:.2f}, {hi:.2f}]", lo > 0))
    c = int((g["ZBK"].zbar <= 1).sum())
    V.append(("N4", "ZBK mean zombie-probe |lev| <= 1 in >= 8 of 10 runs", f"{c}/{n['ZBK']}", c >= 8))
    m, lo, hi = gap(g["TFG"].tbar, g["TBK"].tbar)
    V.append(("T1", "mean probe |lev| (L in {0.02,0.1,0.25}, tau <= 96) TFG - TBK >= 8, lower bound > 4", f"{m:.2f} [{lo:.2f}, {hi:.2f}]", m >= 8 and lo > 4))
    c = int((g["TFG"].t_L01_tau16 == 16).sum())
    V.append(("T2", "TFG |lev| = 16 at (L=0.1, tau=16) in >= 8 of 10 runs", f"{c}/{n['TFG']}", c >= 8))
    m, lo, hi = gap(g["TFG"].tneg, g["TBK"].tneg)
    V.append(("T3", "insolvent-before-horizon |lev| (L in {-0.25,-0.1}) TFG - TBK, lower bound > 0", f"{m:.2f} [{lo:.2f}, {hi:.2f}]", lo > 0))
    c = int((g["TFG"].t_L01_tau16 > g["TFG"].t_L01_tau192).sum())
    V.append(("T4", "TFG |lev| at (L=0.1, tau=16) > at (L=0.1, tau=192) in >= 8 of 10 runs", f"{c}/{n['TFG']}", c >= 8))
    cz = int(((g["ZFG"].rep - g["ZFG"].book) > 0).sum()); ct = int(((g["TFG"].rep - g["TFG"].book) > 0).sum())
    V.append(("X1", "reported - booked > 0 in >= 8 of 10 runs, in ZFG and in TFG", f"ZFG {cz}/{n['ZFG']}, TFG {ct}/{n['TFG']}", cz >= 8 and ct >= 8))
    out += ["| ID | Decision rule | Result | Verdict |", "|---|---|---|---|"]
    out += [f"| {i} | {r} | {res} | {'SUPPORTED' if ok else 'NOT SUPPORTED'} |" for i, r, res, ok in V]
    out += ["", "## Exploratory", ""]
    for arm in ("ZFG", "ZBK"):
        x = g[arm]
        out.append(f"- {arm}: zombie-probe |lev| {x.zbar.mean():.2f}; solvent e=0.1 {x.solv_e01.mean():.2f}; "
                   f"eval zombie-step max-leverage share {x.zombie_maxlev_share.mean():.2f}; forgiven {x.forgiven_pct.mean():.1f}% of K0; "
                   f"reported {x.rep.mean():.1f}% vs booked {x.book.mean():.1f}%")
    for arm in ("TFG", "TBK"):
        x = g[arm]
        out.append(f"- {arm}: probe |lev| {x.tbar.mean():.2f}; insolvent probes {x.tneg.mean():.2f}; reported {x.rep.mean():.2f}% vs booked {x.book.mean():.2f}% per 2-day horizon")
    (HERE / f"VERDICTS_N{TAG}.md").write_text("\n".join(out) + "\n")
    D.to_csv(HERE / f"runs_n{TAG}.csv", index=False)
    print("\n".join(out))

if __name__ == "__main__":
    main()
