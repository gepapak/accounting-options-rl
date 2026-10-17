"""Study P aggregation (PREREGISTRATION_P.md): does the learnability threshold scale with the training budget?

Joint logistic model over the treatment runs of Study O (budget 10^6) and Study P (other budgets):
    P(run learned) = sigmoid( s * [log10(index) - log10 c50(N)] ),   log10 c50(N) = log10 c0 + beta * log10(N / 10^6)
A run is learned when it chooses leverage 16 at 2% equity (L(0.02) = 16). The gap-dependent sample-complexity law
predicts beta = -1/2 (steps needed proportional to 1 / index^2). Controls (alpha = 1, index 0) are excluded from the fit.
Bootstrap: runs resampled within each (budget, alpha) cell, 2,000 replicates.

    python aggregate_p.py      # writes runs_p.csv and VERDICTS_P.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

HERE = Path(__file__).resolve().parent
STUDY_O = HERE.parent / "study_o"
CONTROL = 1.0
LEARNED_GAP = 4.0
rng = np.random.default_rng(20261016)


def load() -> pd.DataFrame:
    dp = {z: json.loads((STUDY_O / f"dp_o_{z}.json").read_text()) for z in ("DK1", "DK2")}
    rows = []
    for folder, default_budget in ((STUDY_O / "results_o" / "jobs", 1_000_000), (HERE / "results_p" / "jobs", None)):
        for p in sorted(folder.glob("*.json")):
            r = json.loads(p.read_text())
            rows.append(dict(study="O" if default_budget else "P", budget=r.get("budget", default_budget), alpha=r["alpha"],
                             seed=r["seed"], region=r["region"], L002=r["lev"]["0.02"],
                             index=dp[r["region"]]["arms"][f"{r['alpha']:g}"]["index_pct"]))
    return pd.DataFrame(rows)


def fit(D: pd.DataFrame, x0=(5.0, np.log10(0.3), -0.5)) -> np.ndarray:
    x, ln, y = np.log10(D["index"].to_numpy()), np.log10(D.budget.to_numpy() / 1e6), (D.L002 == 16).to_numpy(float)

    def nll(p):
        s, lc0, beta = p
        z = s * (x - (lc0 + beta * ln))
        return -np.sum(y * z - np.logaddexp(0, z))
    return optimize.minimize(nll, x0, method="Nelder-Mead", options=dict(maxiter=20000, xatol=1e-6, fatol=1e-8)).x


def main() -> int:
    R = load()
    R.to_csv(HERE / "runs_p.csv", index=False)
    T = R[R.alpha != CONTROL]
    s, lc0, beta = fit(T)
    cells = [g for _, g in T.groupby(["budget", "alpha"])]
    boot = []
    for _ in range(2000):
        B = pd.concat([g.sample(len(g), replace=True, random_state=int(rng.integers(1 << 31))) for g in cells])
        boot.append(fit(B, (s, lc0, beta)))
    boot = np.array(boot)
    lo, hi = np.quantile(boot[:, 2], [0.025, 0.975])
    out = ["# Study P verdicts\n", f"runs: {len(R)} ({R.groupby(['study', 'budget']).size().to_dict()})\n",
           f"joint fit: slope s = {s:.2f}, c50(10^6) = {10 ** lc0:.3f}%, beta = {beta:.3f} (95% bootstrap [{lo:.3f}, {hi:.3f}]); "
           f"law: beta = -0.5\n", "| budget | alpha | index (%) | runs at 16 | mean L(0.02) | gap over control |", "|---|---|---|---|---|---|"]
    for (n, a), g in R.groupby(["budget", "alpha"]):
        ctrl = R[(R.budget == n) & (R.alpha == CONTROL)].L002
        gap = g.L002.mean() - ctrl.mean() if len(ctrl) else float("nan")
        out.append(f"| {n:,} | {a:g} | {g['index'].mean():.3f} | {(g.L002 == 16).sum()}/{len(g)} | {g.L002.mean():.2f} | {gap:+.2f} |")
    v = {}
    v["P1"] = (f"beta = {beta:.3f}, 95% interval [{lo:.3f}, {hi:.3f}] (need upper bound < 0)", hi < 0)
    v["P2"] = (f"95% interval [{lo:.3f}, {hi:.3f}] contains -0.5", lo <= -0.5 <= hi)

    def share(n, arms):
        g = R[(R.budget == n) & (R.alpha.isin(arms))]
        return int((g.L002 == 16).sum()), len(g)
    k4, n4 = share(4_000_000, (0.8, 0.9))
    k1, n1 = share(1_000_000, (0.8, 0.9))
    p3 = stats.fisher_exact([[k4, n4 - k4], [k1, n1 - k1]], alternative="greater").pvalue if n4 and n1 else float("nan")
    v["P3"] = (f"alpha 0.8 and 0.9 at leverage 16: 4M steps {k4}/{n4} vs 1M steps {k1}/{n1}, one-sided Fisher p = {p3:.4f} "
               f"(need < 0.05)", bool(p3 < 0.05))
    k25, n25 = share(250_000, (0.5, 0.7, 0.8))
    k15, n15 = share(1_000_000, (0.5, 0.7, 0.8))
    p4 = stats.fisher_exact([[k15, n15 - k15], [k25, n25 - k25]], alternative="greater").pvalue if n25 and n15 else float("nan")
    v["P4"] = (f"alpha 0.5, 0.7 and 0.8 at leverage 16: 1M steps {k15}/{n15} vs 250k steps {k25}/{n25}, one-sided Fisher p = "
               f"{p4:.4f} (need < 0.05)", bool(p4 < 0.05))
    out += ["", "| ID | Result | Verdict |", "|---|---|---|"] + [f"| {k} | {t} | {'SUPPORTED' if ok else 'NOT SUPPORTED'} |" for k, (t, ok) in v.items()]
    (HERE / "VERDICTS_P.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
