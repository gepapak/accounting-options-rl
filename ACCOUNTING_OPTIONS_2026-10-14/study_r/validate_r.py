"""Pre-registration software validation only; no RL training or TEST evaluation."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.dont_write_bytecode = True
sys.path[:0] = [str(HERE), str(ROOT / "LIABILITY_STUDY_2026-09-29"),
               str(ROOT / "ACCOUNTING_OPTIONS_2026-10-14/study_o")]


def main():
    import importlib.metadata as metadata
    from envs_b import load_market
    from envs_o import PartialEnv
    from envs_r import CostPartialEnv
    from aggregate_r import analyze

    transitions, max_error = 0, 0.0
    for zone in ("DK1", "DK2"):
        market = load_market(zone, "train", "NE")
        for alpha in (0, .8, 1):
            old, new = PartialEnv(market, alpha, True, 6000), CostPartialEnv(market, alpha, .1, True, 6000)
            old_obs, _ = old.reset()
            new_obs, _ = new.reset()
            assert np.array_equal(old_obs, new_obs)
            actions = np.random.default_rng(20261015).integers(0, 9, 2016)
            for action in actions:
                a, b = old.step(action), new.step(action)
                assert np.array_equal(a[0], b[0]) and a[2:4] == b[2:4]
                error = abs(a[1] - b[1])
                max_error = max(max_error, error)
                assert error < 1e-12 and old.ledger == new.ledger and old.reported == new.reported
                transitions += 1
                if a[2]:
                    x, _ = old.reset()
                    y, _ = new.reset()
                    assert np.array_equal(x, y)
            assert new.visit_counts.sum() == len(actions)
    predictions = json.loads((HERE / "predictions_r.json").read_text())
    records = []
    for seed in range(60, 65):
        for fee in (.4, 1.5):
            for alpha in (0, .8, 1):
                for zone in ("DK1", "DK2"):
                    leverage = 0 if alpha == 1 or (fee == 1.5 and alpha == .8) else 16
                    records.append(dict(fee=fee, alpha=alpha, seed=seed, region=zone, seconds=0,
                                        actual_timesteps=1007616, lev={"0.02": leverage, "0.1": leverage, "1": 0}))
    _, good = analyze(records, predictions)
    assert good["primary_R1_supported"] and good["exact_one_sided_sign_p"] == 1 / 32
    for record in records:
        if record["fee"] == .4 and record["alpha"] == .8:
            record["lev"]["0.02"] = 0
    _, bad = analyze(records, predictions)
    assert not bad["primary_R1_supported"] and bad["exact_one_sided_sign_p"] == 1
    incomplete_refused = False
    try:
        analyze(records[:-1], predictions)
    except RuntimeError:
        incomplete_refused = True
    assert incomplete_refused
    result = dict(no_RL_training=True, no_TEST_evaluation=True, matching_transitions=transitions,
                  max_reward_abs_error=max_error, same_states_dynamics_observations=True,
                  visitation_counter_check=True, synthetic_primary_supported_and_failed_checks=True,
                  incomplete_cohort_refused=True)
    result["runtime_versions"] = {package: metadata.version(package) for package in
                                  ("numpy", "pandas", "torch", "gymnasium", "stable-baselines3")}
    (HERE / "validation_r.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
