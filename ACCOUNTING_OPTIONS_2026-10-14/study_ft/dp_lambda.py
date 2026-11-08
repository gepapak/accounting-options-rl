"""Prediction of Part B before any learning: the leverage that maximizes reported return under partial forgiveness, by dynamic programming on the sleeve's real magnitudes.

Fair market, stationary draw of the magnitude and the liquidity cap from the training split, discount 0.99 as the learner's, reported equity on a grid, linear interpolation.
The value does not depend on the sign of the position in a fair market. Output: dp_lambda.json, the optimal leverage at each probe equity for each lam, and the one-step answer.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from envs_ft import FEE, K0, LEV, Q1, SETTING, SETTINGS, market_arrays

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
LAMS = (0.0, 0.01, 0.03, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)
GAMMA = 0.99
GRID = np.concatenate([-np.geomspace(5.0, 0.002, 60), [0.0], np.geomspace(0.002, 5.0, 90)]) * K0
LEVS = np.array([0.0, 0.25, 1.0, 4.0, 16.0])
N_DRAW = 1500


def solve(lam: float, abs_s, cap, kappa):
    n = len(GRID)
    V = np.zeros(n)
    q = np.minimum(LEVS[:, None] * Q1, cap[None, :])
    cost = FEE * q + kappa * q * q
    gross = q * abs_s[None, :]
    pol = np.zeros(n, dtype=int)
    for it in range(3000):
        Q = np.zeros((len(LEVS), n))
        for j, e in enumerate(GRID):
            total = 0.0
            for sg in (1.0, -1.0):
                pnl = sg * gross - cost
                raw = e + pnl
                new_def = np.maximum(np.maximum(-raw, 0.0) - max(-e, 0.0), 0.0)
                nxt = raw + lam * new_def
                r = (nxt - e) / K0 * 100.0
                total = total + r + GAMMA * np.interp(nxt, GRID, V)
            Q[:, j] = (0.5 * total).mean(axis=1)
        new = Q.max(axis=0)
        d = float(np.abs(new - V).max())
        V = new
        pol = Q.argmax(axis=0)
        if d < 1e-5:
            break
    return V, pol, it


def main():
    abs_s, _, _, cap = market_arrays(SETTING, "train", "placebo")
    kappa = SETTINGS[SETTING]["kappa"]
    rng = np.random.default_rng(61109)
    idx = rng.choice(len(abs_s), N_DRAW, replace=False)
    s, c = abs_s[idx], cap[idx]
    out = {"setting": SETTING, "gamma": GAMMA, "levs": LEVS.tolist(), "draws": N_DRAW, "fee": FEE, "kappa": kappa, "Q1": Q1, "K0": K0, "lam": {}}
    probes = (-0.5, -0.1, 0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
    for lam in LAMS:
        V, pol, it = solve(lam, s, c, kappa)
        row = {f"{e:g}": float(LEVS[pol[int(np.argmin(np.abs(GRID - e * K0)))]]) for e in probes}
        # one step at an empty account
        q = np.minimum(LEVS[:, None] * Q1, c[None, :])
        cost = FEE * q + kappa * q * q
        g = q * s[None, :]
        m = sum(sg * g - cost + lam * np.maximum(-(sg * g - cost), 0.0) for sg in (1.0, -1.0)) / 2.0
        one = float(LEVS[int(m.mean(axis=1).argmax())])
        out["lam"][f"{lam:g}"] = dict(iterations=it, policy=row, one_step_empty=one, value_at_zero=float(V[int(np.argmin(np.abs(GRID)))]))
        print(f"lam {lam:5.2f}: one step {one:5.2f}; dynamic program by equity: " + " ".join(f"{k}:{v:g}" for k, v in row.items()) + f" ({it} iterations)", flush=True)
    (HERE / "dp_lambda.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
