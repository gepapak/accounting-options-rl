"""Study O aggregation (PREREGISTRATION_O.md): verdicts for O1-O4 and the exploratory analyses.

Reads results_o/jobs/*.json, dp_o_DK1.json, dp_o_DK2.json and calibration_o.json; writes runs_o.csv and VERDICTS_O.md.

    python aggregate_o.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

HERE = Path(__file__).resolve().parent
CONTROL = 1.0
LEARNED_GAP = 4.0
BAND = 1.5           # confident arms: index >= BAND * c* (predicted learned) or <= c* / BAND (predicted not learned)
rng = np.random.default_rng(20261015)


def region_T(lev: dict) -> float:
    """Largest probed equity below which every probe chose |leverage| >= 8 (Study K's measure)."""
    T = 0.0
    for e in sorted(lev, key=float):
        if lev[e] >= 8:
            T = float(e)
        else:
            break
    return T


def main() -> int:
    dp = {z: json.loads((HERE / f"dp_o_{z}.json").read_text()) for z in ("DK1", "DK2")}
    cal = json.loads((HERE / "calibration_o.json").read_text())
    c = cal["c_star_pct"]
    runs = [json.loads(p.read_text()) for p in sorted((HERE / "results_o" / "jobs").glob("*.json"))]
    R = pd.DataFrame([dict(alpha=r["alpha"], seed=r["seed"], region=r["region"], L002=r["lev"]["0.02"], L01=r["lev"]["0.1"],
                           L1=r["lev"]["1"], T=region_T(r["lev"]), rep=r["drawmean_reported_return_pct"],
                           led=r["drawmean_ledger_return_pct"]) for r in runs])
    R["index"] = [dp[z]["arms"][f"{a:g}"]["index_pct"] for a, z in zip(R.alpha, R.region)]
    R["dp_region"] = [dp[z]["arms"][f"{a:g}"]["region16_K"] for a, z in zip(R.alpha, R.region)]
    R.to_csv(HERE / "runs_o.csv", index=False)
    ctrl = R[R.alpha == CONTROL].L002.to_numpy()
    out = [f"# Study O verdicts\n\nruns: {len(R)} ({R.groupby('alpha').size().to_dict()}); c* = {c:.3f}%; control alpha = {CONTROL:g}\n",
           "| alpha | index (%) | DP region (alloc.) | mean L(0.02) | gap G | runs at 16 | prediction | outcome |", "|---|---|---|---|---|---|---|---|"]
    arms = []
    for a, g in R[R.alpha != CONTROL].groupby("alpha"):
        idx, G = float(g["index"].mean()), float(g.L002.mean() - ctrl.mean())
        pred = "learned" if idx >= c else "not learned"
        conf = idx >= BAND * c or idx <= c / BAND
        outc = "learned" if G >= LEARNED_GAP else "not learned"
        arms.append(dict(alpha=a, index=idx, G=G, pred=pred, conf=conf, outc=outc, n16=int((g.L002 == 16).sum()), n=len(g),
                         dp_region=float(g.dp_region.mean()), T=float(g["T"].mean())))
        out.append(f"| {a:g} | {idx:.3f} | {g.dp_region.mean():.2f} | {g.L002.mean():.2f} | {G:+.2f} | {arms[-1]['n16']}/{len(g)} | "
                   f"{pred}{'' if conf else ' (transition)'} | {outc} |")
    out.append(f"| {CONTROL:g} (control) | {R[R.alpha == CONTROL]['index'].mean():.3f} | "
               f"{R[R.alpha == CONTROL].dp_region.mean():.2f} | {ctrl.mean():.2f} | | {(ctrl == 16).sum()}/{len(ctrl)} | | |\n")
    A = pd.DataFrame(arms)
    v = {}
    C = A[A.conf]
    ok = int((C.pred == C.outc).sum())
    v["O1"] = (f"confident arms classified correctly: {ok}/{len(C)} (need >= {len(C) - 1})", ok >= len(C) - 1)
    x, y = R["index"].to_numpy(), R.L002.to_numpy()
    rho = float(stats.spearmanr(x, y).statistic)
    perm = np.array([stats.spearmanr(x, rng.permutation(y)).statistic for _ in range(10000)])
    p = float((np.sum(perm >= rho) + 1) / (len(perm) + 1))
    v["O2"] = (f"run-level Spearman(index, L(0.02)) = {rho:.3f}, one-sided permutation p = {p:.4f} (need >= 0.4 and < 0.01)",
               rho >= 0.4 and p < 0.01)
    a9 = A[A.alpha == 0.9].iloc[0]
    dp_ok = all(dp[z]["arms"]["0.9"]["region16_K"] >= 0.5 for z in dp)
    v["O3"] = (f"alpha 0.9: DP region {a9.dp_region:.2f} (need >= 0.5 in both zones: {dp_ok}); gap {a9.G:+.2f} (need < 4); "
               f"runs at 16: {a9.n16}/10 (need < 5)", dp_ok and a9.G < LEARNED_GAP and a9.n16 < 5)
    a0 = R[R.alpha == 0.0].L002.to_numpy()
    boot = np.array([rng.choice(a0, len(a0)).mean() - rng.choice(ctrl, len(ctrl)).mean() for _ in range(10000)])
    G0 = float(a0.mean() - ctrl.mean())
    v["O4"] = (f"alpha 0 vs control: gap {G0:+.2f} (need >= 4), one-sided 95% bootstrap lower bound {np.quantile(boot, 0.05):+.2f} (need > 0)",
               G0 >= LEARNED_GAP and np.quantile(boot, 0.05) > 0)
    out += ["| ID | Result | Verdict |", "|---|---|---|"] + [f"| {k} | {t} | {'SUPPORTED' if s else 'NOT SUPPORTED'} |" for k, (t, s) in v.items()]
    # exploratory: logistic fit of P(L(0.02) = 16) against log10(index), treatment runs only
    T_ = R[R.alpha != CONTROL]
    lx, ly = np.log10(T_["index"].to_numpy()), (T_.L002 == 16).to_numpy(float)
    nll = lambda b: -np.sum(ly * (b[0] + b[1] * lx) - np.logaddexp(0, b[0] + b[1] * lx))  # noqa: E731
    b = optimize.minimize(nll, [0.0, 5.0], method="Nelder-Mead").x
    bs = []
    for _ in range(1000):
        i = rng.integers(0, len(lx), len(lx))
        bb = optimize.minimize(lambda q: -np.sum(ly[i] * (q[0] + q[1] * lx[i]) - np.logaddexp(0, q[0] + q[1] * lx[i])),
                               b, method="Nelder-Mead").x
        if bb[1] > 0:
            bs.append(10 ** (-bb[0] / bb[1]))
    c50 = 10 ** (-b[0] / b[1]) if b[1] > 0 else float("nan")
    out += ["\n## Exploratory",
            f"- logistic P(L(0.02) = 16) on log10(index): c50 = {c50:.3f}% (bootstrap 95% interval "
            f"[{np.quantile(bs, 0.025):.3f}, {np.quantile(bs, 0.975):.3f}]%), slope {b[1]:.2f}; registered c* = {c:.3f}%",
            "- per arm: mean learned region T vs DP region; reported vs booked return (% of K0, test draws):"]
    for a, g in R.groupby("alpha"):
        out.append(f"  - alpha {a:g}: T {g['T'].mean():.2f} (DP {g.dp_region.mean():.2f}); reported {g.rep.mean():+.1f}, booked {g.led.mean():+.1f}")
    (HERE / "VERDICTS_O.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
