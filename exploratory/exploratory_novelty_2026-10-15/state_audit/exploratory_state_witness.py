"""EXPLORATORY: exact one-action incentive witnesses in Study N ledger/clock states.

Read README.md for the fixed analysis and its limited estimand. No DP or training.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STUDY = ROOT / "ACCOUNTING_OPTIONS_2026-10-14/study_n"
LIAB = ROOT / "LIABILITY_STUDY_2026-09-29"
K0, GAMMA, FEE = 20000.0, 0.99, 1.0
OUT = HERE / "outputs"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pathwise_environment_check():
    """Verify actual reward timing, negative starts and both reward arms."""
    sys.path[:0] = [str(LIAB), str(STUDY)]
    from envs_b import LEV
    from envs_n import LagEnv

    max_action = int(np.flatnonzero(LEV == 16)[0])
    flat_action = int(np.flatnonzero(LEV == 0)[0])
    cases, max_error = 0, 0.0
    for n in (-10000.0, -2000.0, 400.0, 10000.0):
        for spread in (-4000.0, -10.0, 10.0, 4000.0):
            for tau in (1, 4, 48, 192):
                market = SimpleNamespace(n=1000, spread=np.full(1000, spread), cap_mwh=np.full(1000, 4.0))
                x = 4 * spread - FEE * 4
                for arm in ("TFG", "TBK"):
                    for delayed in (False, True):
                        values = []
                        for trade in (True, False):
                            env = LagEnv(market, arm, train=False)
                            env.reset(options={"e0": n / K0, "tau0": tau, "t0": 200})
                            total = 0.0
                            for t in range(tau):
                                target = tau - 1 if delayed else 0
                                action = max_action if trade and t == target else flat_action
                                _, r, terminated, truncated, _ = env.step(action)
                                total += GAMMA ** t * r * K0 / 100
                                if terminated or truncated:
                                    break
                            values.append(total)
                        if arm == "TBK":
                            expected = x * (GAMMA ** (tau - 1) if delayed else 1)
                        else:
                            delta_d = max(-n - x, 0) - max(-n, 0)
                            expected = (GAMMA ** (tau - 1) * (x + delta_d) if delayed
                                        else x + GAMMA ** (tau - 1) * delta_d)
                        error = abs(values[0] - values[1] - expected)
                        max_error = max(max_error, error)
                        assert error < 1e-7, (arm, n, spread, tau, delayed, error)
                        cases += 1
                if n < 0:
                    for arm in ("ZFG", "ZBK"):
                        env = LagEnv(market, arm, train=False)
                        env.reset(options={"e0": n / K0, "t0": 200})
                        env.k = min(tau, 96)
                        _, r, _, _, _ = env.step(max_action)
                        expected = max(n + x, 0) - max(n, 0) if arm == "ZFG" else x
                        error = abs(r * K0 / 100 - expected)
                        max_error = max(max_error, error)
                        assert error < 1e-7
                        cases += 1
    return {"cases": cases, "max_abs_error_eur": max_error}


def main():
    started = time.perf_counter()
    check = pathwise_environment_check()
    rows, sources = [], {}
    for zone in ("DK1", "DK2"):
        panel_path = ROOT / f"GENERALIZATION_STUDY_2026-09-29/data_panel/panel_{zone}.pkl"
        panel = pd.read_pickle(panel_path)
        panel = panel.loc[panel["split"] == "train"].iloc[200:]
        q = np.minimum(4.0, 0.25 * 0.25 * panel["bal_mw"].to_numpy(float))
        z = q * np.abs(panel["imb"].to_numpy(float) - panel["spot"].to_numpy(float))
        cost = FEE * q
        payoffs = np.concatenate([z - cost, -z - cost])
        mean_x = -float(cost.mean())
        assert abs(payoffs.mean() - mean_x) < 1e-8
        sigma = float(payoffs.std())
        source = STUDY / f"dp_n2_{zone}.json"
        dp = json.loads(source.read_text())
        sources[str(panel_path.relative_to(ROOT))] = sha(panel_path)
        sources[str(source.relative_to(ROOT))] = sha(source)
        agents = {}
        for arm in ("ZFG", "ZBK", "TFG", "TBK"):
            files = sorted((STUDY / "results_n/jobs").glob(f"{arm}__PPO__s*__{zone}.json"))
            assert len(files) == 5, (zone, arm, len(files))
            agents[arm] = [json.loads(p.read_text())["probes"] for p in files]
            for p in files:
                sources[str(p.relative_to(ROOT))] = sha(p)
        for arm in ("ZFG", "ZBK", "TFG", "TBK"):
            terminal = arm.startswith("T")
            ledgers = (-0.25, -0.1, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0) if terminal else (-0.5, -0.25, -0.1, -0.02)
            clocks = (1, 4, 16, 48, 96, 192) if terminal else (96, 48, 12)
            for ledger in ledgers:
                n = ledger * K0
                delta_d = float(np.maximum(-n - payoffs, 0).mean() - max(-n, 0))
                for clock in clocks:
                    key = f"L{ledger:g}_tau{clock}" if terminal else f"n{ledger:g}_k{clock}"
                    probe = key if terminal else f"zombie_{key}"
                    choices = ("immediate", "deadline") if terminal else ("immediate",)
                    for when in choices:
                        factor = GAMMA ** (clock - 1) if when == "deadline" else 1.0
                        booked = factor * mean_x
                        if arm.endswith("BK"):
                            witness = booked
                        elif terminal:
                            witness = (factor * (mean_x + delta_d) if when == "deadline"
                                       else mean_x + GAMMA ** (clock - 1) * delta_d)
                        else:
                            witness = mean_x + delta_d
                        dp_lev = dp[arm]["lev" if terminal else "zombie_lev"][key]
                        observed = [a[probe] for a in agents[arm]]
                        rows.append(dict(zone=zone, arm=arm, ledger_multiple=ledger, clock=clock,
                                         witness_policy=when, advantage_eur=witness,
                                         expected_booked_advantage_eur=booked,
                                         one_step_sigma_eur=sigma,
                                         positive_witness=bool(witness > 1e-10),
                                         dp_first_action_abs_leverage=dp_lev,
                                         observed_mean_abs_leverage=float(np.mean(observed)),
                                         observed_max16_runs=sum(v == 16 for v in observed),
                                         independent_agent_runs=len(observed),
                                         empirical_train_rows=len(q)))
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "exploratory_state_witness.csv", index=False)
    summary = {"status": "exploratory_complete", "estimand": "one-action policy reward minus always-flat under same account",
               "population_claim": False, "dp_solved": False, "new_agents_trained": 0,
               "environment_pathwise_check": check, "elapsed_seconds": time.perf_counter() - started,
               "input_sha256": sources, "by_arm_policy": [],
               "interpretation": "A positive witness proves an empirical-model incentive for this policy, not optimal first action or finite-training acquisition. A nonpositive witness does not certify safety. Delayed and immediate terminal values estimate different policies."}
    for (arm, policy), d in frame.groupby(["arm", "witness_policy"], sort=False):
        summary["by_arm_policy"].append(dict(arm=arm, policy=policy, state_zone_probes=len(d),
            positive_witnesses=int(d.positive_witness.sum()),
            min_eur=float(d.advantage_eur.min()), max_eur=float(d.advantage_eur.max()),
            dp_maximal_states_with_nonpositive_witness=int(((d.dp_first_action_abs_leverage == 16) & ~d.positive_witness).sum()),
            positive_witnesses_with_dp_flat_first_action=int((d.positive_witness & (d.dp_first_action_abs_leverage == 0)).sum())))
    (OUT / "exploratory_state_witness_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "input_sha256"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
