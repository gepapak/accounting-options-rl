"""Study RS myopic witness (no training): the one-step-optimal leverage under each training reward, from the training split only.

For each setting, signal value z and start equity e (multiples of K0), every leverage of the grid is scored by the mean over all training-split
quarters with that signal (t >= 200) of the one-step reward that each arm would receive at equity e:
    booked    pnl = q*s - c                                            (ORACLE)
    reported  max(e*K0 + pnl, 0) - e*K0                                (PLAIN: the zero floor forgives the loss beyond e)
    SPR       (r - r_twin)/2 - c        r_twin: the reported reward with the sign of s reversed
    TIR       sign(q*s) * g_hat - c,    g_hat = max(r, r_twin) + c if r != r_twin else 0
    RUIN5     reported - 5 (percent of K0) if e > 0 and e*K0 + pnl <= 0
all in percent of K0. The witness is myopic (one step, no equity dynamics); it shows what each reward pays for, not what PPO will learn.

    python witness_rs.py        # prints the table and writes witness_rs.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from envs_rs import FEE, K0, LEV, Q1, SETTINGS, market_arrays

HERE = Path(__file__).resolve().parent
E_GRID = (0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
LOW_E = (0.0, 0.02, 0.05, 0.1)
ARMS = ("ORACLE", "PLAIN", "SPR", "TIR", "RUIN5")


def scores(setting: str, z: int, e: float) -> dict:
    abs_s, eps, zz, cap = market_arrays(setting, "train")
    idx = np.arange(200, len(abs_s))
    idx = idx[zz[idx] == z]
    s, c_ = abs_s[idx] * eps[idx], cap[idx]
    kappa, E = SETTINGS[setting]["kappa"], e * K0
    out = {a: [] for a in ARMS}
    for lev in LEV:
        q = np.clip(lev * Q1, -c_, c_)
        cost = FEE * np.abs(q) + kappa * q * q
        pnl, pnl_t = q * s - cost, -q * s - cost
        r, r_t = np.maximum(E + pnl, 0.0) - E, np.maximum(E + pnl_t, 0.0) - E
        g_hat = np.where(np.abs(r - r_t) > 1e-9, np.maximum(r, r_t) + cost, 0.0)
        ruin = (E > 0) & (E + pnl <= 0)
        vals = {"ORACLE": pnl, "PLAIN": r, "SPR": 0.5 * (r - r_t) - cost, "TIR": np.sign(q * s) * g_hat - cost,
                "RUIN5": r - np.where(ruin, 5.0 * K0 / 100.0, 0.0)}
        for a in ARMS:
            out[a].append(float(vals[a].mean() / K0 * 100.0))
    return out


def main():
    res = {}
    for st in SETTINGS:
        res[st] = {}
        print(f"\n{st}: myopic optimal leverage (training split)")
        print("   z     e  " + "  ".join(f"{a:>6}" for a in ARMS))
        for z in (-1, 0, 1):
            for e in E_GRID:
                sc = scores(st, z, e)
                best = {a: float(LEV[int(np.argmax(sc[a]))]) for a in ARMS}
                res[st][f"{e:g}|{z:+d}"] = dict(best=best, scores=sc)
                print(f"  {z:+d}  {e:5.2f}  " + "  ".join(f"{best[a]:+6g}" for a in ARMS))
        x = {a: float(np.mean([abs(res[st][f"{e:g}|{z:+d}"]["best"][a]) == 16.0 for e in LOW_E for z in (-1, 0, 1)])) for a in ARMS}
        res[st]["X_myopic"] = x
        print("  myopic X (share of the 12 low-equity (e, z) cells at leverage 16): " + ", ".join(f"{a} {v:.2f}" for a, v in x.items()))
    (HERE / "witness_rs.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
