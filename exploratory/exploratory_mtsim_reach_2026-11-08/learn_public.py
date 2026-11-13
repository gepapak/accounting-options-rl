"""Two-action sample-mean learner on published gym-mtsim.

Rules are in PROTOCOL_LEARN.md, written before this run.
"""
from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from gym_mtsim import FOREX_DATA_PATH, MtEnv, MtSimulator

SYMBOL = "EURUSD"
ENTRY = 1.10
EQUITY = 10_000.0
N = 1000
N_SEEDS = 20
SEED = 20261108
SCALES = (0.0015, 0.012)
OUT = Path(__file__).with_name("learn_public.json")


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


def make_env(floor: bool) -> MtEnv:
    spec = MtSimulator(symbols_filename=FOREX_DATA_PATH)
    cls = MtSimulator if floor else NoFloor
    sim = cls(unit="USD", balance=EQUITY, leverage=100.0, stop_out_level=0.2, hedge=True)
    sim.symbols_info = {SYMBOL: copy.deepcopy(spec.symbols_info[SYMBOL])}
    sim.symbols_data = {SYMBOL: prices()}
    return MtEnv(original_simulator=sim, trading_symbols=[SYMBOL], window_size=1, symbol_max_orders=1)


def open_max(floor: bool):
    env = make_env(floor)
    env.reset()
    info = env.simulator.symbols_info[SYMBOL]
    raw = 0.8 * env.simulator.free_margin * env.simulator.leverage / (info.trade_contract_size * ENTRY)
    volume = float(np.clip(np.floor(raw / info.volume_step) * info.volume_step, info.volume_min, info.volume_max))
    action = np.zeros(env.action_space.shape, dtype=np.float64)
    action[-2] = -20.0
    action[-1] = volume
    action[:-2] = -20.0
    _, open_reward, _, _, opened = env.step(action)
    if len(env.simulator.orders) != 1:
        raise RuntimeError(opened)
    return env, float(open_reward), volume


def set_next_close(env: MtEnv, price: float) -> None:
    tick = env.unwrapped._current_tick
    when = env.unwrapped.time_points[tick + 1]
    df = env.unwrapped.simulator.symbols_data[SYMBOL]
    df.loc[when, "Close"] = price
    df.loc[when, "High"] = max(float(df.loc[when, "Open"]), price)
    df.loc[when, "Low"] = min(float(df.loc[when, "Open"]), price)


def hold_reward(template: MtEnv, log_ret: float) -> float:
    env = copy.deepcopy(template)
    entry = float(env.simulator.orders[0].entry_price)
    set_next_close(env, entry * float(np.exp(log_ret)))
    action = np.zeros(env.action_space.shape, dtype=np.float64)
    action[-2] = 20.0
    action[:-2] = -20.0
    _, reward, _, _, _ = env.step(action)
    return float(reward)


def line_model(template: MtEnv) -> tuple[float, float, float]:
    entry = float(template.simulator.orders[0].entry_price)
    r0 = hold_reward(template, 0.0)
    plus = hold_reward(template, np.log(1.001))
    minus = hold_reward(template, np.log(0.999))
    slope = (plus - minus) / (entry * 0.002)
    return entry, r0, slope


def main() -> None:
    floored, open_f, volume = open_max(True)
    plain, open_u, volume_u = open_max(False)
    if volume != volume_u or abs(open_f - open_u) > 1e-6:
        raise RuntimeError((volume, volume_u, open_f, open_u))
    entry, r0, slope = line_model(floored)
    checks = []
    for log_ret in np.linspace(np.log(0.8), np.log(1.2), 200):
        price = entry * float(np.exp(log_ret))
        line = r0 + slope * (price - entry)
        booked = hold_reward(plain, float(log_ret))
        checks.append(abs(line - booked))
    line_err = float(np.max(checks))

    rng = np.random.default_rng(SEED)
    blocks = {scale: rng.standard_t(3, size=(N_SEEDS, N)) / np.sqrt(3.0) * scale for scale in SCALES}
    rows = []
    for scale, draws in blocks.items():
        # cache unique returns would not help; evaluate per seed
        for seed in range(N_SEEDS):
            pub, unf, line = [], [], []
            for log_ret in draws[seed]:
                price = entry * float(np.exp(log_ret))
                pub.append(open_f + hold_reward(floored, float(log_ret)))
                unf.append(open_u + hold_reward(plain, float(log_ret)))
                line.append(open_f + r0 + slope * (price - entry))
            pub_m, unf_m, line_m = float(np.mean(pub)), float(np.mean(unf)), float(np.mean(line))
            rows.append({
                "scale": scale,
                "seed": seed,
                "mean_published": pub_m,
                "mean_nofloor": unf_m,
                "mean_line": line_m,
                "commit_published": "max" if pub_m > 0 else "flat",
                "commit_nofloor": "max" if unf_m > 0 else "flat",
                "commit_line": "max" if line_m > 0 else "flat",
                "n_reward_differs": int(np.sum(np.abs(np.array(pub) - np.array(unf)) > 0.05)),
            })
            print(scale, seed, rows[-1]["commit_published"], rows[-1]["commit_nofloor"], rows[-1]["commit_line"], pub_m, unf_m, flush=True)
    def count(scale, key, label):
        return sum(r[key] == label for r in rows if r["scale"] == scale)
    summary = {
        "volume": volume,
        "open_fee": open_f,
        "line_max_abs_err_vs_nofloor": line_err,
        "slope_usd_per_price": slope,
        "commits": {
            str(scale): {
                "published_max": count(scale, "commit_published", "max"),
                "nofloor_flat": count(scale, "commit_nofloor", "flat"),
                "line_flat": count(scale, "commit_line", "flat"),
            }
            for scale in SCALES
        },
    }
    OUT.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
