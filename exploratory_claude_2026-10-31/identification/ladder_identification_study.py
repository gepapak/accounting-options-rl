"""EXPLORATORY (31 Oct 2026): how well do sign-twin estimators recover the booked P&L under each rule of the accounting ladder?

For random-action rollouts in the edge-market testbed (exploratory_claude_2026-10-16/nfl_training/envs_nfl2.py: rules ZF, LL, TERM, LAG; placebo
and edge markets; with and without insolvent starts) the script compares, step by step, the booked P&L b with
    SPR  = (r - r_twin)/2 - c            (conservative end of the identified interval)
    TMR  = sigma*(max(r, r_twin) + c) - c (exact when the favourable branch is uncensored)
    TIR  = TMR if r != r_twin (g identified) else -c (conservative fallback when both branches are censored)
and reports, separately for "credited steps" (r != b, where the accounting matters) and all steps:
    share exact, mean absolute error, mean signed error (percent of the allocation per quarter).
No training. Output: ladder_identification_results.csv.gz and a printed table.

    python ladder_identification_study.py
"""
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "exploratory_claude_2026-10-16" / "nfl_training"), str(ROOT / "LIABILITY_STUDY_2026-09-29")]
from envs_nfl2 import K0, LEV, NFLEnv2  # noqa: E402

N_STEPS = 80_000
CONFIGS = [("ZF", 0.0), ("LL", 0.0), ("TERM", 0.0), ("TERM", 0.5), ("LAG", 0.0), ("LAG", 0.5)]


def study(acc: str, p_ins: float, kind: str, seed: int = 11):
    env = NFLEnv2("DK1", "train", kind, "reported", acc, seed=seed, p_insolvent=p_ins)
    rng = np.random.default_rng(seed)
    o, _ = env.reset()
    rows = []
    for _ in range(N_STEPS):
        a = int(rng.integers(9))
        t = env.t
        eps = env.eps_ep[t - env.t0]
        sigma = float(np.sign(LEV[a])) * eps
        twin = copy.copy(env)
        twin.flip_next = True
        _, r_twin, _, _, _ = twin.step(a)
        o, r, done, _, info = env.step(a)
        b, c = info["pnl"] / K0 * 100.0, info["cost"] / K0 * 100.0
        spr = 0.5 * (r - r_twin) - c
        tmr = sigma * (max(r, r_twin) + c) - c
        tir = tmr if abs(r - r_twin) > 1e-12 else -c
        rows.append((acc, p_ins, kind, abs(r - b) > 1e-9, abs(spr - b), spr - b, abs(tmr - b), tmr - b, abs(r - b), abs(tir - b), tir - b))
        if done:
            o, _ = env.reset()
    return rows


if __name__ == "__main__":
    allrows = []
    for acc, p_ins in CONFIGS:
        for kind in ("edge", "placebo"):
            allrows += study(acc, p_ins, kind)
    df = pd.DataFrame(allrows, columns=["rule", "p_insolvent", "market", "credited", "spr_abs", "spr_signed", "tmr_abs", "tmr_signed", "reported_abs", "tir_abs", "tir_signed"])
    df.to_csv(HERE / "ladder_identification_results.csv.gz", index=False)
    pd.set_option("display.width", 220)
    print("Credited steps (reported reward differs from booked), errors in percent of the allocation per quarter")
    g = df[df.credited].groupby(["rule", "p_insolvent", "market"])
    t = g.agg(n=("credited", "size"), reported_err=("reported_abs", "mean"), spr_abs=("spr_abs", "mean"), spr_bias=("spr_signed", "mean"),
              tmr_abs=("tmr_abs", "mean"), tir_abs=("tir_abs", "mean"), tir_bias=("tir_signed", "mean"),
              tmr_exact=("tmr_abs", lambda x: float((x < 1e-6).mean())), tir_exact=("tir_abs", lambda x: float((x < 1e-6).mean())),
              spr_exact=("spr_abs", lambda x: float((x < 1e-6).mean())))
    print(t.round(4).to_string())
    print("\nAll steps")
    g = df.groupby(["rule", "p_insolvent", "market"])
    t2 = g.agg(credited_share=("credited", "mean"), reported_err=("reported_abs", "mean"), spr_abs=("spr_abs", "mean"), tmr_abs=("tmr_abs", "mean"),
              tir_abs=("tir_abs", "mean"))
    print(t2.round(4).to_string())
