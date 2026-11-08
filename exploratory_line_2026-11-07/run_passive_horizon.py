"""The passive horizon on unmodified gym-mtsim (PROTOCOL_HORIZON.md).

For the 100 held positions of run_formula_gap.py (same seed, same construction) the floor's distance lambda* in return units is
read from the diagram (equity before the step over the exposure's slope). The package's own EURUSD series (88 daily bars) gives
an empirical return distribution; Gaussian and Student-t(3) laws matched to its standard deviation give two tail classes. For each,
the per-step probability p of a move past the floor and the number of logged transitions n_50 = ln 2 / p at which a passive log
has a 50% chance of containing one informative move are reported, at the daily scale and at an hourly scale (sd / sqrt(24)).

Simulator check (positions 0-9, 1,000 draws each, seed 20261109): rewards of the public simulator and of the control without the
two floor lines are compared on the same draws, for (a) a bootstrap of the package's own daily returns and (b) Student-t(3) returns
at the same sd. The number of draws on which the two rewards differ must equal the number of draws past lambda*, on every position.

Range by path (positions 0-9): sixteen consecutive steps of -1% of the entry price each (linear in the entry, as on the line) against
one step of -16%: the summed public reward and the single-step public reward are compared (the floor's level is minus the equity in
both); for the control, the margin stop-out closes the order on the path, which the line does not show, and the sum is reported.

    .venv-mtsim\\Scripts\\python.exe run_passive_horizon.py      # writes results_horizon.json
"""
from __future__ import annotations

import copy
import json
import math
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from gym_mtsim import FOREX_DATA_PATH, MtEnv

import run_formula_gap as fg

SEED = 20261109
N_CHECK = 10
N_DRAWS = 1000
N_PATH = 16
OUT = Path(__file__).with_name("results_horizon.json")


def empirical_returns() -> np.ndarray:
    _, data = pickle.load(open(FOREX_DATA_PATH, "rb"))
    close = data[fg.SYMBOL]["Close"].astype(float)
    return np.diff(np.log(close.to_numpy()))


def tails(lam: float, sd: float) -> dict:
    """Per-step probability of a move past -lam under three laws matched to sd, and n_50 = ln 2 / p."""
    out = {}
    out["gaussian"] = float(stats.norm.cdf(-lam / sd))
    scale3 = sd / math.sqrt(3.0)  # Student-t(3) has variance 3 scale^2
    out["student_t3"] = float(stats.t.cdf(-lam / scale3, df=3))
    return out


def n50(p: float) -> float:
    return float("inf") if p <= 0 else math.log(2.0) / p


def long_prices(entry: float, n: int) -> pd.DataFrame:
    idx = pd.date_range(datetime(2030, 1, 1), periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"Open": entry, "Close": entry, "Low": entry, "High": entry, "Volume": 1.0}, index=idx)


def prepared_long(entry: float, floor: bool, n: int):
    spec = fg.MtSimulator(symbols_filename=FOREX_DATA_PATH)
    cls = fg.MtSimulator if floor else fg.NoFloor
    sim = cls(unit="USD", balance=1000.0, leverage=100.0, stop_out_level=0.2, hedge=True)
    sim.symbols_info = {fg.SYMBOL: copy.deepcopy(spec.symbols_info[fg.SYMBOL])}
    sim.symbols_data = {fg.SYMBOL: long_prices(entry, n)}
    env = MtEnv(original_simulator=sim, trading_symbols=[fg.SYMBOL], window_size=1, symbol_max_orders=1)
    env.reset()
    volume = fg.open_volume(env)
    env.step(fg.open_action(env, volume))
    assert len(env.simulator.orders) == 1
    return env


def path_reward(entry: float, floor: bool, k: int) -> dict:
    env = prepared_long(entry, floor, k + 4)
    entry_price = float(env.simulator.orders[0].entry_price)
    e0 = float(env.simulator.equity)
    total = 0.0
    closed_at = None
    for j in range(1, k + 1):
        fg.set_next_close(env, entry_price * (1.0 - fg.DP_REL * j))
        _, r, _, _, info = env.step(fg.hold_action(env))
        total += float(r)
        if closed_at is None and len(env.simulator.orders) == 0:
            closed_at = j
    return {"sum": total, "equity_before": e0, "closed_at_step": closed_at, "balance_end": float(env.simulator.balance)}


def main() -> None:
    res = json.loads(fg.OUT.read_text())
    pub = res["public"]
    lam = np.array([p["equity_before"] / (p["truth_slope"] * p["entry"]) for p in pub])  # return that exhausts the equity
    r_emp = empirical_returns()
    sd_d = float(r_emp.std(ddof=1))
    sd_h = sd_d / math.sqrt(24.0)
    summary = {
        "n_positions": len(pub),
        "lambda_star_return": {"min": float(lam.min()), "median": float(np.median(lam)), "max": float(lam.max())},
        "empirical": {"n_returns": int(len(r_emp)), "sd": sd_d, "min": float(r_emp.min()), "max": float(r_emp.max()),
                      "share_past_floor_min_lambda": float((r_emp <= -lam.min()).mean())},
        "horizon": {},
    }
    lam_med = float(np.median(lam))
    for scale, sd in (("daily", sd_d), ("hourly", sd_h)):
        t = tails(lam_med, sd)
        emp_p = float((r_emp <= -lam_med).mean()) if scale == "daily" else None
        summary["horizon"][scale] = {
            "sd": sd, "lambda_over_sd": lam_med / sd,
            "p_gaussian": t["gaussian"], "n50_gaussian": n50(t["gaussian"]),
            "p_student_t3": t["student_t3"], "n50_student_t3": n50(t["student_t3"]),
            "p_empirical_bootstrap": emp_p, "n50_empirical_bootstrap": (n50(emp_p) if emp_p is not None else None),
        }
        print(f"{scale}: sd {sd:.5f}, lambda*/sd {lam_med / sd:.1f}; Gaussian p {t['gaussian']:.3e} n50 {n50(t['gaussian']):.3g}; "
              f"t3 p {t['student_t3']:.3e} n50 {n50(t['student_t3']):.3g}; empirical p {emp_p}")

    rng = np.random.default_rng(SEED)
    checks = []
    for i in range(N_CHECK):
        entry = float(pub[i]["entry"])
        env_pub, _ = fg.prepared(entry, True)
        env_ctl, _ = fg.prepared(entry, False)
        entry_price = float(env_pub.simulator.orders[0].entry_price)
        lam_i = float(lam[i])
        row = {"position": i, "lambda_star": lam_i}
        for name, draws in (("bootstrap_daily", rng.choice(r_emp, size=N_DRAWS, replace=True)),
                            ("student_t3_daily", stats.t.rvs(df=3, scale=sd_d / math.sqrt(3.0), size=N_DRAWS, random_state=rng))):
            returns = np.asarray(draws, dtype=float)
            differ = 0
            for r in returns:
                price = entry_price * (1.0 + float(r))
                a = fg.query(env_pub, price)["reward"]
                b = fg.query(env_ctl, price)["reward"]
                differ += int(abs(a - b) > 1e-9)
            crossed = int((returns <= -lam_i).sum())
            row[name] = {"draws": N_DRAWS, "rewards_differ": differ, "draws_past_floor": crossed,
                         "max_adverse_return": float(returns.min())}
            print(f"position {i} {name}: differ {differ}, past floor {crossed}, worst {returns.min():.4f}")
        checks.append(row)

    paths = []
    for i in range(N_CHECK):
        entry = float(pub[i]["entry"])
        p_pub = path_reward(entry, True, N_PATH)
        p_ctl = path_reward(entry, False, N_PATH)
        single_pub = float(pub[i]["worst_reward"])
        single_ctl = float(res["control"][i]["worst_reward"])
        paths.append({"position": i, "public_path_sum": p_pub["sum"], "public_single": single_pub,
                      "public_equity_before": p_pub["equity_before"], "public_closed_at_step": p_pub["closed_at_step"],
                      "control_path_sum": p_ctl["sum"], "control_single": single_ctl,
                      "control_closed_at_step": p_ctl["closed_at_step"], "control_balance_end": p_ctl["balance_end"]})
        print(f"position {i} path: public sum {p_pub['sum']:.2f} vs single {single_pub:.2f} (equity {p_pub['equity_before']:.2f}); "
              f"control sum {p_ctl['sum']:.2f} vs single {single_ctl:.2f}, closed at step {p_ctl['closed_at_step']}")

    identity_holds = all(row[k]["rewards_differ"] == row[k]["draws_past_floor"] for row in checks
                         for k in ("bootstrap_daily", "student_t3_daily"))
    bootstrap_all_identical = all(row["bootstrap_daily"]["rewards_differ"] == 0 for row in checks)
    public_path_equals_single = all(abs(p["public_path_sum"] - p["public_single"]) <= 1e-6 for p in paths)
    summary["checks"] = {"crossing_identity_holds_on_all": identity_holds,
                         "bootstrap_logs_all_identical": bootstrap_all_identical,
                         "public_path_sum_equals_single_step": public_path_equals_single}
    OUT.write_text(json.dumps({"summary": summary, "positions": checks, "paths": paths}, indent=1))
    print(json.dumps(summary["checks"], indent=1))


if __name__ == "__main__":
    main()
