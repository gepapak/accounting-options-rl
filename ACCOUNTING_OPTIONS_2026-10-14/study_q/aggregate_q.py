"""Study Q aggregation (PREREGISTRATION_Q.md): does the learnability index transfer from PPO to A2C and DQN?

Reads results_q/jobs/*.json and Study O's dp_o_DK1.json, dp_o_DK2.json (the index does not depend on the learner);
writes runs_q.csv and VERDICTS_Q.md. Verdicts are per learner (A2C-n64, DQN-default).

    python aggregate_q.py
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
rng = np.random.default_rng(20261017)


def logistic_c50(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    lx = np.log10(x)
    nll = lambda b: -np.sum(y * (b[0] + b[1] * lx) - np.logaddexp(0, b[0] + b[1] * lx))  # noqa: E731
    b = optimize.minimize(nll, [0.0, 5.0], method="Nelder-Mead").x
    return (10 ** (-b[0] / b[1]) if b[1] > 0 else float("nan")), float(b[1])


def main() -> int:
    dp = {z: json.loads((STUDY_O / f"dp_o_{z}.json").read_text()) for z in ("DK1", "DK2")}
    runs = [json.loads(p.read_text()) for p in sorted((HERE / "results_q" / "jobs").glob("*.json"))]
    R = pd.DataFrame([dict(learner=r["learner"], alpha=r["alpha"], seed=r["seed"], region=r["region"], L002=r["lev"]["0.02"],
                           L2=r["lev"]["2"], rep=r["drawmean_reported_return_pct"], led=r["drawmean_ledger_return_pct"]) for r in runs])
    R["index"] = [dp[z]["arms"][f"{a:g}"]["index_pct"] for a, z in zip(R.alpha, R.region)]
    R.to_csv(HERE / "runs_q.csv", index=False)
    out = [f"# Study Q verdicts\n\nruns: {len(R)} ({R.groupby(['learner', 'alpha']).size().to_dict()})\n"]
    v = {}
    for ln, G in R.groupby("learner"):
        ctrl = G[G.alpha == CONTROL].L002.to_numpy()
        out += [f"## {ln}\n", "| alpha | index (%) | runs at 16 | mean L(0.02) | gap over control | mean L(2) |", "|---|---|---|---|---|---|"]
        for a, g in G.groupby("alpha"):
            out.append(f"| {a:g} | {g['index'].mean():.3f} | {(g.L002 == 16).sum()}/{len(g)} | {g.L002.mean():.2f} | "
                       f"{g.L002.mean() - ctrl.mean():+.2f} | {g.L2.mean():.2f} |")
        x, y = G["index"].to_numpy(), G.L002.to_numpy()
        rho = float(stats.spearmanr(x, y).statistic)
        perm = np.array([stats.spearmanr(x, rng.permutation(y)).statistic for _ in range(10000)])
        p = float((np.sum(perm >= rho) + 1) / (len(perm) + 1))
        v[f"Q1 {ln}"] = (f"run-level Spearman(index, L(0.02)) = {rho:.3f}, one-sided permutation p = {p:.4f} (need > 0 and p < 0.05)",
                         rho > 0 and p < 0.05)
        a0 = G[G.alpha == 0.0].L002.to_numpy()
        boot = np.array([rng.choice(a0, len(a0)).mean() - rng.choice(ctrl, len(ctrl)).mean() for _ in range(10000)])
        g0 = float(a0.mean() - ctrl.mean())
        v[f"Q2 {ln}"] = (f"alpha 0 vs control: gap {g0:+.2f} (need >= 4), one-sided 95% bootstrap lower bound {np.quantile(boot, 0.05):+.2f} "
                         f"(need > 0)", g0 >= LEARNED_GAP and np.quantile(boot, 0.05) > 0)
        g9 = float(G[G.alpha == 0.9].L002.mean() - ctrl.mean())
        v[f"Q3 {ln}"] = (f"alpha 0.9 vs control: gap {g9:+.2f} (need < 4)", g9 < LEARNED_GAP)
        T = G[G.alpha != CONTROL]
        c50, slope = logistic_c50(T["index"].to_numpy(), (T.L002 == 16).to_numpy(float))
        out.append(f"\nexploratory: logistic c50 = {c50:.3f}% (slope {slope:.2f}); Study O (PPO) c50 = 0.429%\n")
    out += ["| ID | Result | Verdict |", "|---|---|---|"] + [f"| {k} | {t} | {'SUPPORTED' if ok else 'NOT SUPPORTED'} |" for k, (t, ok) in v.items()]
    (HERE / "VERDICTS_Q.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
