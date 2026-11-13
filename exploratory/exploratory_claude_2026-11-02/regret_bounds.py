"""Regret-bound terms for reward substitution (exploratory; no training, only the shipped market panels).

For a substitute reward r~ with E[r~_t - x_t | F_t] = -beta_t, the booked regret of an eps-optimal policy under r~ is at most
eps + A(pi*) + A(pi~), with A(pi) = E_pi sum_t |beta_t| (Proposition "regret of a reward substitute" of the paper). This script evaluates the
per-step bias exactly, from the two branches of the twin pair, along the honest policy of the edge-market sleeve (leverage 4 in the direction of
the signal z) -- which is the booked-optimal policy of the sleeve -- and reports A(pi_honest) for the reported reward, SPR and TIR:

    p      probability that the position's gross P&L is positive (the signal's accuracy p_edge when following z)
    r_f, r_u   rewards of the favourable and the unfavourable branch of the step (the pair {r, r'} ordered by the booked value)
    g, c       gross magnitude and own cost (percent of K0)
    reported : beta = -(p*credit_f + (1-p)*credit_u)                 (credit = reported reward - booked P&L)
    SPR      : beta = (2p-1) * (g - (r_f - r_u)/2)
    TIR      : beta = (2p-1) * g  if r_f == r_u else 0

    python regret_bounds.py          # writes regret_bounds_results.json next to this file
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
sys.path[:0] = [str(CODE / "LIABILITY_STUDY_2026-09-29"), str(CODE / "ACCOUNTING_OPTIONS_2026-10-14" / "study_t"), str(HERE)]
from envs_t import K0, LEV, SETTINGS, EdgeEnvT  # noqa: E402

TOL = 1e-12
P_EDGE = {k: v["p_edge"] for k, v in SETTINGS.items()}


def honest_action(z: float) -> int:
    target = 4.0 * z
    idx = [i for i, v in enumerate(LEV) if abs(float(v) - target) < 1e-9]
    assert idx, "leverage 4 not in the grid"
    return idx[0]


def rollout_terms(setting: str, start_mode: str, n_episodes: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    env = EdgeEnvT(setting, "test", "edge", "reported", seed=int(rng.integers(1 << 30)))
    p = P_EDGE[setting]
    tot = dict(booked=0.0, reported=0.0, spr=0.0, tir=0.0, n_nonident=0, steps=0, n_credit=0)
    for ep in range(n_episodes):
        if start_mode == "zero_half" and ep % 2 == 0:
            e0 = 0.0
        else:
            e0 = float(np.exp(rng.uniform(np.log(0.05), np.log(2.0))))
        env.reset(options={"e0": e0})
        done = False
        while not done:
            z = float(env.z_ep[env.t - env.t0])
            a = honest_action(z)
            sign_env = float(np.sign(LEV[a])) * float(env.eps_ep[env.t - env.t0])     # realized sign of the gross P&L
            twin = copy.copy(env)
            twin.flip_next = True
            _, r_twin, _, _, _ = twin.step(a)
            _, r, done, _, info = env.step(a)
            c = info["cost"] / K0 * 100.0
            x = info["pnl"] / K0 * 100.0
            g = (x + c) * sign_env if sign_env != 0 else 0.0                              # gross magnitude, percent of K0
            g = abs(x + c)
            r_f, r_u = (r, r_twin) if sign_env > 0 else (r_twin, r)                       # favourable / unfavourable branch rewards
            credit_f = r_f - (g - c)
            credit_u = r_u - (-g - c)
            tot["booked"] += x
            tot["reported"] += abs(p * credit_f + (1 - p) * credit_u)
            tot["spr"] += abs((2 * p - 1) * (g - (r_f - r_u) / 2.0))
            nonident = abs(r_f - r_u) <= TOL
            tot["tir"] += abs((2 * p - 1) * g) if nonident else 0.0
            tot["n_nonident"] += int(nonident and g > 0)
            tot["n_credit"] += int(info["forgiven"] > 1e-12)
            tot["steps"] += 1
    n = n_episodes
    return dict(setting=setting, start=start_mode, episodes=n, steps=tot["steps"],
                booked_edge_per_episode=tot["booked"] / n, A_reported=tot["reported"] / n, A_SPR=tot["spr"] / n, A_TIR=tot["tir"] / n,
                credited_step_share=tot["n_credit"] / tot["steps"], nonidentified_step_share=tot["n_nonident"] / tot["steps"])


def main():
    out = []
    for setting in ("S_DK1", "T_DK2", "T_DK1M"):
        for start in ("log_uniform", "zero_half"):
            row = rollout_terms(setting, start, n_episodes=200, seed=17)
            out.append(row)
            e = row["booked_edge_per_episode"]
            print(f"{setting:7s} {start:12s} booked {e:7.2f}  A_rep {row['A_reported']:7.3f}  A_SPR {row['A_SPR']:7.3f} ({100 * row['A_SPR'] / e:5.2f}% of edge)  "
                  f"A_TIR {row['A_TIR']:7.4f} ({100 * row['A_TIR'] / e:5.3f}% of edge)  credited {100 * row['credited_step_share']:.2f}%  "
                  f"non-identified {100 * row['nonidentified_step_share']:.3f}%")
    (HERE / "regret_bounds_results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
