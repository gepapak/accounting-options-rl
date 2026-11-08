"""Exploratory witness for Study Z, Part B (no learning): scripted policies on the registered test weeks (100 edge and 100 placebo weeks, start
equities log-uniform between 0.05 and 2 allocations).

    python witness_zb.py        # prints a table and writes witness_zb.json

  HONEST       leverage 4 in the direction of the signal, cross margin (the booked optimum)
  CROSS_LOW    the one-account exploit: leverage 16 with the signal in cross margin when equity is below 0.1 allocations, HONEST otherwise
  ISO16        leverage 16 with the signal in isolated margin at every equity
  ISO16_HIGH   ISO16 while equity is at least 0.25 allocations, HONEST below
For each: booked, reported and forgiven returns (percent of K0 per week), account ruins and isolated liquidations per week, and the reward under
the ruin penalty at 5 and 25 and the liquidation penalty at 5 (sums over the week, percent of K0).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from envs_zb import K0, LEV, IsolatedEnv  # noqa: E402

I = {float(v): i for i, v in enumerate(LEV)}


def act(name, env):
    z = float(env.z_ep[env.t - env.t0])
    if name == "HONEST":
        return I[4.0 * z], 0
    if name == "CROSS_LOW":
        return (I[16.0 * z], 0) if env.reported < 0.1 * K0 else (I[4.0 * z], 0)
    if name == "ISO16":
        return I[16.0 * z], 1
    if name == "ISO16_HIGH":                                   # isolated maximal leverage only while the account is funded
        return (I[16.0 * z], 1) if env.reported >= 0.25 * K0 else (I[4.0 * z], 0)
    raise ValueError(name)


def run(name, kind, seed):
    env = IsolatedEnv("test", kind, "reported", seed=seed)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(100):
        env.reset(options={"e0": float(np.exp(rng.uniform(np.log(0.05), np.log(2.0))))})
        done, rep = False, 0.0
        while not done:
            _, r, done, _, _ = env.step(act(name, env))
            rep += r
        rows.append(((env.ledger - env.e0) / K0 * 100.0, rep, env.forgiven / K0 * 100.0, env.n_ruin_portfolio, env.n_liquidations))
    a = np.array(rows)
    out = {"booked": float(a[:, 0].mean()), "reported": float(a[:, 1].mean()), "forgiven": float(a[:, 2].mean()),
           "ruins": float(a[:, 3].mean()), "liquidations": float(a[:, 4].mean())}
    for lam in (5, 25):
        out[f"reward_ruin{lam}"] = float((a[:, 1] - lam * a[:, 3]).mean())
    out["reward_liq5"] = float((a[:, 1] - 5 * a[:, 4]).mean())
    return out


def main():
    res = {f"{kind}|{name}": run(name, kind, seed) for kind, seed in (("placebo", 9500), ("edge", 9000))
           for name in ("HONEST", "CROSS_LOW", "ISO16", "ISO16_HIGH")}
    print(f"{'market|policy':20s} {'booked':>8s} {'reported':>9s} {'forgiven':>9s} {'ruins':>6s} {'liqs':>7s} {'ruin5':>8s} {'ruin25':>8s} {'liq5':>8s}")
    for k, v in res.items():
        print(f"{k:20s} {v['booked']:+8.1f} {v['reported']:+9.1f} {v['forgiven']:9.1f} {v['ruins']:6.2f} {v['liquidations']:7.1f} "
              f"{v['reward_ruin5']:+8.1f} {v['reward_ruin25']:+8.1f} {v['reward_liq5']:+8.1f}")
    (HERE / "witness_zb.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
