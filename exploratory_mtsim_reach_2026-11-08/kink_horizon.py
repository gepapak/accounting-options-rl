"""Locate the gym-mtsim kink and the passive horizon at Study G's volatility.

The one-bar value table in results.json, seed 20261108, had a zero gap at
0.15% per hour. This script asks where the gap starts.
"""
from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from gym_mtsim import FOREX_DATA_PATH, MtEnv, MtSimulator

SYMBOL = "EURUSD"
ENTRY = 1.10
VOL = 0.0015
OUT = Path(__file__).with_name("kink.json")


class NoFloor(MtSimulator):
    def tick(self, delta_time: timedelta = timedelta()) -> None:
        self._check_current_time()
        self.current_time += delta_time
        self.equity = self.balance
        for order in self.orders:
            order.exit_time = self.current_time
            order.exit_price = self.price_at(order.symbol, order.exit_time)["Close"]
            self._update_order_profit(order)
            self.equity += order.profit
        while self.margin_level < self.stop_out_level and len(self.orders) > 0:
            worst = min(self.orders, key=lambda order: order.profit)
            self.close_order(worst)


def prices() -> pd.DataFrame:
    idx = pd.date_range(datetime(2030, 1, 1), periods=4, freq="h", tz="UTC")
    return pd.DataFrame(
        {"Open": ENTRY, "Close": ENTRY, "Low": ENTRY, "High": ENTRY, "Volume": 1.0},
        index=idx,
    )


def make_env(equity: float, floor: bool) -> MtEnv:
    spec = MtSimulator(symbols_filename=FOREX_DATA_PATH)
    cls = MtSimulator if floor else NoFloor
    sim = cls(unit="USD", balance=equity, leverage=100.0, stop_out_level=0.2, hedge=True)
    sim.symbols_info = {SYMBOL: copy.deepcopy(spec.symbols_info[SYMBOL])}
    sim.symbols_data = {SYMBOL: prices()}
    return MtEnv(original_simulator=sim, trading_symbols=[SYMBOL], window_size=1, symbol_max_orders=1)


def open_action(env: MtEnv, volume: float) -> np.ndarray:
    action = np.zeros(env.action_space.shape, dtype=np.float64)
    action[-2] = -20.0
    action[-1] = volume
    action[:-2] = -20.0
    return action


def hold_action(env: MtEnv) -> np.ndarray:
    action = np.zeros(env.action_space.shape, dtype=np.float64)
    action[-2] = 20.0
    action[:-2] = -20.0
    return action


def set_next_close(env: MtEnv, price: float) -> None:
    tick = env.unwrapped._current_tick
    when = env.unwrapped.time_points[tick + 1]
    df = env.unwrapped.simulator.symbols_data[SYMBOL]
    df.loc[when, "Close"] = price
    df.loc[when, "High"] = max(float(df.loc[when, "Open"]), price)
    df.loc[when, "Low"] = min(float(df.loc[when, "Open"]), price)


def opened(equity: float, floor: bool, volume: float) -> MtEnv:
    env = make_env(equity, floor)
    env.reset()
    _, _, _, _, info = env.step(open_action(env, volume))
    if len(env.simulator.orders) != 1:
        raise RuntimeError(info)
    return env


def reward_at(template: MtEnv, log_ret: float) -> float:
    env = copy.deepcopy(template)
    entry = float(env.simulator.orders[0].entry_price)
    set_next_close(env, entry * float(np.exp(log_ret)))
    _, reward, _, _, _ = env.step(hold_action(env))
    return float(reward)


def kink(equity: float, volume: float) -> dict:
    floored = opened(equity, True, volume)
    plain = opened(equity, False, volume)
    # adverse log-returns from 0 to -40%
    agree, differ = 0.0, -0.40
    gap_at_zero = reward_at(floored, agree) - reward_at(plain, agree)
    gap_at_tail = reward_at(floored, differ) - reward_at(plain, differ)
    if abs(gap_at_tail) <= 1e-4:
        return {
            "equity": equity,
            "volume": volume,
            "notional": volume * 100000.0 * ENTRY,
            "effective_leverage": volume * 100000.0 * ENTRY / equity,
            "kink_found_in_window": False,
            "gap_at_minus_40pct": gap_at_tail,
            "gap_at_zero": gap_at_zero,
            "one_bar_p_studyG": 0.0,
            "n50_bars_studyG": None,
        }
    for _ in range(40):
        mid = 0.5 * (agree + differ)
        gap = reward_at(floored, mid) - reward_at(plain, mid)
        if abs(gap) > 1e-4:
            differ = mid
        else:
            agree = mid
    lo = agree
    # unit-variance t(3): Study G draws standard_t(3)/sqrt(3)*VOL
    z = lo / VOL
    # survival of the unit-variance t: raw t_3 = z * sqrt(3)
    p = float(student_t.cdf(z * np.sqrt(3.0), df=3))
    n50 = float(np.log(2) / p) if p > 0 else None
    return {
        "equity": equity,
        "volume": volume,
        "notional": volume * 100000.0 * ENTRY,
        "effective_leverage": volume * 100000.0 * ENTRY / equity,
        "kink_found_in_window": True,
        "kink_log_return_inside": lo,
        "kink_log_return_outside": differ,
        "gap_at_minus_40pct": gap_at_tail,
        "gap_at_zero": gap_at_zero,
        "one_bar_p_studyG": p,
        "n50_bars_studyG": n50,
        "reward_inside_floor": reward_at(floored, lo),
        "reward_inside_plain": reward_at(plain, lo),
        "reward_outside_floor": reward_at(floored, differ),
        "reward_outside_plain": reward_at(plain, differ),
        "hidden_at_kink": reward_at(floored, differ) - reward_at(plain, differ),
    }


def main() -> None:
    rows = []
    for equity in (500.0, 10_000.0):
        env = make_env(equity, True)
        env.reset()
        info = env.simulator.symbols_info[SYMBOL]
        raw = 0.8 * env.simulator.free_margin * env.simulator.leverage / (info.trade_contract_size * ENTRY)
        mx = float(np.clip(np.floor(raw / info.volume_step) * info.volume_step, info.volume_min, info.volume_max))
        for name, volume in (("min", float(info.volume_min)), ("max", mx)):
            row = kink(equity, volume)
            row["lot"] = name
            rows.append(row)
            print(name, equity, "kink", row.get("kink_log_return_outside"), "p", row["one_bar_p_studyG"], "n50", row["n50_bars_studyG"], "hidden", row.get("hidden_at_kink"), "gap40", row["gap_at_minus_40pct"])
    OUT.write_text(json.dumps({"vol": VOL, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
