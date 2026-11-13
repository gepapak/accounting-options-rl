"""Exploratory: accuracy of the reported reward, SPR and TIR against the booked P&L on random-action rollouts of Study N's environments
(the copy envs_tn.LagEnvT used by Study U), forgiven arms. One aligned loop per environment: at every step the twin and the real step
share the state, so the three estimates refer to the same step.

    python study_n_identification.py            # writes study_n_identification.txt next to this file

Errors are mean absolute differences from the booked P&L on credited steps (reported reward above booked), in percent of the allocation.
"""
import copy
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ACC = HERE.parents[1] / "ACCOUNTING_OPTIONS_2026-10-14"
sys.path[:0] = [str(ACC.parent / "LIABILITY_STUDY_2026-09-29"), str(ACC / "study_t")]
from envs_tn import K0, LEV, LagEnvT, load_market  # noqa: E402


def run(arm, region="DK1", n_steps=120_000, seed=11):
    m = load_market(region, "train", "NE")
    env = LagEnvT(m, arm, train=True, seed=seed)
    rng = np.random.default_rng(seed)
    env.reset()
    rows = []
    for _ in range(n_steps):
        a = int(rng.integers(0, len(LEV)))
        sigma = float(np.sign(LEV[a])) * float(np.sign(env.m.spread[env.t]))
        twin = copy.copy(env)
        twin.flip_next = True
        _, r_twin, _, _, _ = twin.step(a)
        _, r, term, trunc, info = env.step(a)
        b, c = info["pnl"] / K0 * 100.0, info["cost"] / K0 * 100.0
        spr = 0.5 * (r - r_twin) - c
        tir = sigma * (max(r, r_twin) + c) - c if abs(r - r_twin) > 1e-12 else -c
        rows.append((r - b, spr - b, tir - b, abs(r - r_twin) > 1e-12))
        if term or trunc:
            env.reset()
    x = np.array(rows, dtype=float)
    credited = x[:, 0] > 1e-9
    return credited.sum(), np.abs(x[credited, 0]).mean(), np.abs(x[credited, 1]).mean(), np.abs(x[credited, 2]).mean(), x[credited, 3].mean()


if __name__ == "__main__":
    out = ["Study N environments (envs_tn), random actions, forgiven arms: mean absolute error on credited steps (percent of the allocation)",
           "arm region credited_steps reported SPR TIR share_identified"]
    for arm in ("ZFG", "TFG"):
        for region in ("DK1", "DK2"):
            n, rep, spr, tir, ident = run(arm, region)
            out.append(f"{arm} {region} {n} {rep:.3f} {spr:.3f} {tir:.3f} {ident:.3f}")
            print(out[-1])
    (HERE / "study_n_identification.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
