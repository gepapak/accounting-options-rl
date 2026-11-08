"""Exploratory witness for Study Z (no learning): scripted policies on the 100 placebo and edge test weeks of the registered evaluation.

    python witness_z.py        # prints a table and writes witness_z.json

Policies (leverage per account, A and B):
  FLAT            (0, 0)
  HONEST          both legs at leverage 4 in the direction of the signal (the booked optimum of the sleeve, per leg)
  GAMBLE_BOTH     both legs at maximal leverage in the direction of the signal whenever the portfolio's reported equity is below 0.1 allocations
                  (the one-account exploit, run in both accounts), HONEST otherwise
  PARK_GAMBLE     from two empty accounts a hedge (+16 in A, -16 in B); otherwise the account with the larger equity stays flat (it is
                  "parked") and the other trades maximal leverage in the direction of the signal. The parked account never trades, so the
                  portfolio can never return to zero (Proposition, part 3).
Start regimes: "zero" (every account at zero, the regime in which half of the training weeks start) and "registered" (the registered test
weeks' start equities, log-uniform between 0.05 and 2 allocations, split equally).
For each: booked, reported and forgiven returns (percent of K0 per week), portfolio and account ruin events per week, and the reward under
the portfolio ruin penalty and the account ruin penalty at strengths 5 and 25 (sum over the week, percent of K0).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from envs_z import K0, LEV, SplitEnv  # noqa: E402

I = {float(v): i for i, v in enumerate(LEV)}
LOW = 0.1 * K0
N_TEST = 100


def act(name: str, env) -> tuple[int, int]:
    z = float(env.z_ep[env.t - env.t0])
    honest = I[4.0 * z]
    if name == "FLAT":
        return I[0.0], I[0.0]
    if name == "HONEST":
        return honest, honest
    if name == "GAMBLE_BOTH":
        g = I[16.0 * z] if env.reported < LOW else honest
        return g, g
    if name == "PARK_GAMBLE":
        if env.ea <= 0.0 and env.eb <= 0.0:
            return I[16.0], I[-16.0]
        if env.ea >= env.eb:
            return I[0.0], I[16.0 * z]
        return I[16.0 * z], I[0.0]
    raise ValueError(name)


def run(name: str, kind: str, seed: int, start: str) -> dict:
    env = SplitEnv("test", kind, "reported", seed=seed)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(N_TEST):
        e0 = float(np.exp(rng.uniform(np.log(0.05), np.log(2.0))))
        env.reset(options={"e0": 0.0 if start == "zero" else e0})
        done, rep = False, 0.0
        while not done:
            _, r, done, _, _ = env.step(act(name, env))
            rep += r
        rows.append(((env.ledger - env.e0) / K0 * 100.0, rep, env.forgiven / K0 * 100.0, env.n_ruin_portfolio, env.n_ruin_accounts))
    a = np.array(rows)
    out = {"booked": float(a[:, 0].mean()), "reported": float(a[:, 1].mean()), "forgiven": float(a[:, 2].mean()),
           "ruin_portfolio": float(a[:, 3].mean()), "ruin_accounts": float(a[:, 4].mean())}
    for lam in (5, 25):
        out[f"reward_ruinP{lam}"] = float((a[:, 1] - lam * a[:, 3]).mean())
        out[f"reward_ruinA{lam}"] = float((a[:, 1] - lam * a[:, 4]).mean())
    return out


def main() -> None:
    res = {}
    for start in ("zero", "registered"):
        for kind, seed in (("placebo", 9500), ("edge", 9000)):        # the registered evaluation's test weeks
            for name in ("FLAT", "HONEST", "GAMBLE_BOTH", "PARK_GAMBLE"):
                res[f"{start}|{kind}|{name}"] = run(name, kind, seed, start)
    print(f"{'start|market|policy':34s} {'booked':>8s} {'reported':>9s} {'forgiven':>9s} {'ruinP':>6s} {'ruinA':>6s} "
          f"{'rew P5':>8s} {'rew P25':>8s} {'rew A5':>8s} {'rew A25':>8s}")
    for k, v in res.items():
        print(f"{k:34s} {v['booked']:+8.1f} {v['reported']:+9.1f} {v['forgiven']:9.1f} {v['ruin_portfolio']:6.2f} {v['ruin_accounts']:6.2f} "
              f"{v['reward_ruinP5']:+8.1f} {v['reward_ruinP25']:+8.1f} {v['reward_ruinA5']:+8.1f} {v['reward_ruinA25']:+8.1f}")
    (HERE / "witness_z.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
