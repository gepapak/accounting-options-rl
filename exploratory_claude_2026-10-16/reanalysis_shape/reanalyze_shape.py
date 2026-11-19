"""EXPLORATORY reanalysis (17 Oct 2026, after Studies O-R): the exploit's shape L(0.02) - L(2) as an outcome.

Motivation: same-seed ledger controls (exploratory_claude_2026-10-16/nfl_training, seeds 85-89) showed that the count of runs at leverage
16 at equity 0.02 has a seed-dependent noise floor in the fair-market environment (PPO ledger control 5/10 vs 0/10 registered), because
every leverage has the same expected booked P&L there up to the fee. The shape (high leverage only at low equity) is the exploit's
signature. This script recomputes it from the registered result files of Studies O, P, Q and R; it changes no registered verdict.

    python reanalyze_shape.py
"""
import glob
import json
from pathlib import Path

import pandas as pd
from scipy import stats

ACC = Path(__file__).resolve().parents[2] / "ACCOUNTING_OPTIONS_2026-10-14"
IDX = {0.0: 1.805, 0.25: 1.35, 0.5: 0.886, 0.7: 0.52, 0.8: 0.335, 0.9: 0.154, 0.95: 0.06, 1.0: 0.0}   # Study O index by alpha (%)


def table(pattern, keys):
    rows = []
    for p in glob.glob(str(ACC / pattern)):
        d = json.loads(Path(p).read_text())
        rows.append({**{k: d.get(k) for k in keys}, "L002": d["lev"]["0.02"], "L2": d["lev"]["2"]})
    df = pd.DataFrame(rows)
    df["shape"] = df.L002 - df.L2
    df["at16"] = (df.L002 == 16).astype(int)
    return df


def show(name, df, by):
    print(f"== {name}")
    print(df.groupby(by).agg(runs=("shape", "size"), at16=("at16", "sum"), mean_L002=("L002", "mean"), mean_shape=("shape", "mean"),
                             sd_shape=("shape", "std")).round(2).to_string())


if __name__ == "__main__":
    O = table("study_o/results_o/jobs/*.json", ["alpha", "seed", "region"])
    show("Study O (PPO, 1M steps)", O, ["alpha"])
    O["index"] = O.alpha.map(IDX)
    print("Spearman(index, L(0.02)) = %.3f; Spearman(index, shape) = %.3f (p = %.1e)" % (
        stats.spearmanr(O["index"], O.L002)[0], *stats.spearmanr(O["index"], O["shape"])))
    show("Study P (PPO, budgets)", table("study_p/results_p/jobs/*.json", ["budget", "alpha"]), ["budget", "alpha"])
    Q = table("study_q/results_q/jobs/*.json", ["learner", "alpha"])
    show("Study Q (A2C-n64, DQN-default)", Q, ["learner", "alpha"])
    for ln, g in Q.groupby("learner"):
        g = g.assign(index=g.alpha.map(IDX))
        print(f"{ln}: Spearman(index, L(0.02)) = %.3f; Spearman(index, shape) = %.3f (p = %.1e)" % (
            stats.spearmanr(g["index"], g.L002)[0], *stats.spearmanr(g["index"], g["shape"])))
    show("Study R (PPO, fee contrast)", table("study_r/results_r/*/*.json", ["fee", "alpha"]), ["fee", "alpha"])
