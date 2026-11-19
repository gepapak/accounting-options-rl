"""When the published gym-mtsim fee still leaves a reason to take the floor.

Kink from kink.json: about -1.22% at the maximum lot, either equity.
Open fee at equity 10000, max lot, was -363.5 USD.
Stopping rule, fixed before this run: among scales 0.003, 0.006, 0.012, 0.02,
take the smallest at which 1000 Student-t(3) bars give a positive mean
floored episode return and a negative mean unfloored episode return.
If none does, do not train. If one does, train two-action REINFORCE
at that scale, equity 10000, seeds 0, 1, 2, 600 episodes, reward divided
by equity, baseline the running mean.
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
SCALES = (0.003, 0.006, 0.012, 0.02)
N = 1000
SEED = 20261108
OUT = Path(__file__).with_name("scale_learn.json")


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


def actions(env: MtEnv, volume: float):
    hold = np.zeros(env.action_space.shape, dtype=np.float64)
    hold[-2] = 20.0
    hold[:-2] = -20.0
    open_ = np.zeros(env.action_space.shape, dtype=np.float64)
    open_[-2] = -20.0
    open_[-1] = volume
    open_[:-2] = -20.0
    return hold, open_


def max_volume(env: MtEnv) -> float:
    info = env.simulator.symbols_info[SYMBOL]
    raw = 0.8 * env.simulator.free_margin * env.simulator.leverage / (info.trade_contract_size * ENTRY)
    return float(np.clip(np.floor(raw / info.volume_step) * info.volume_step, info.volume_min, info.volume_max))


def set_next_close(env: MtEnv, price: float) -> None:
    tick = env.unwrapped._current_tick
    when = env.unwrapped.time_points[tick + 1]
    df = env.unwrapped.simulator.symbols_data[SYMBOL]
    df.loc[when, "Close"] = price
    df.loc[when, "High"] = max(float(df.loc[when, "Open"]), price)
    df.loc[when, "Low"] = min(float(df.loc[when, "Open"]), price)


def prepare(floor: bool, volume: float):
    env = make_env(floor)
    env.reset()
    _, open_reward, _, _, info = env.step(actions(env, volume)[1])
    if len(env.simulator.orders) != 1:
        raise RuntimeError(info)
    return env, float(open_reward)


def episode_return(template, open_reward: float, log_ret: float) -> float:
    env = copy.deepcopy(template)
    entry = float(env.simulator.orders[0].entry_price)
    set_next_close(env, entry * float(np.exp(log_ret)))
    hold, _ = actions(env, 0.0)
    _, reward, _, _, _ = env.step(hold)
    return float(open_reward) + float(reward)


def sweep(scale: float, volume: float, rng: np.random.Generator) -> dict:
    draws = rng.standard_t(3, size=N) / np.sqrt(3.0) * scale
    out = {"scale": scale}
    for floor in (True, False):
        env, open_reward = prepare(floor, volume)
        vals = np.array([episode_return(env, open_reward, float(r)) for r in draws])
        key = "floor" if floor else "nofloor"
        out[key] = {
            "mean": float(vals.mean()),
            "mean_over_equity": float(vals.mean() / EQUITY),
            "p_differs_from_other": None,
            "open_reward": open_reward,
        }
        out[key + "_vals_sum"] = float(vals.sum())
    # gap
    env_f, _ = prepare(True, volume)
    env_u, _ = prepare(False, volume)
    # recompute paired gap from stored means only if open rewards match
    out["mean_gap"] = out["floor"]["mean"] - out["nofloor"]["mean"]
    return out


def learn(scale: float, floor: bool, volume: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    env, open_reward = prepare(floor, volume)
    theta = 0.0
    baseline = 0.0
    seen = 0
    last = []
    for episode in range(600):
        p = 1.0 / (1.0 + np.exp(-np.clip(theta, -20, 20)))
        take = bool(rng.random() < p)
        if take:
            ret = float(rng.standard_t(3) / np.sqrt(3.0) * scale)
            reward = episode_return(env, open_reward, ret) / EQUITY
            chosen = 1.0
        else:
            reward = 0.0
            chosen = 0.0
        seen += 1
        baseline += (reward - baseline) / seen
        theta += 0.5 * (chosen - p) * (reward - baseline)
        if episode % 100 == 99:
            last.append(float(1.0 / (1.0 + np.exp(-np.clip(theta, -20, 20)))))
    final = float(1.0 / (1.0 + np.exp(-np.clip(theta, -20, 20))))
    return {"scale": scale, "floor": floor, "seed": seed, "final_p_max": final, "trace": last}


def main() -> None:
    rng = np.random.default_rng(SEED)
    probe = make_env(True)
    probe.reset()
    volume = max_volume(probe)
    rows = [sweep(scale, volume, rng) for scale in SCALES]
    chosen = None
    for row in rows:
        if row["floor"]["mean"] > 0 and row["nofloor"]["mean"] < 0:
            chosen = row["scale"]
            break
    learners = []
    if chosen is not None:
        for floor in (True, False):
            for seed in (0, 1, 2):
                learners.append(learn(chosen, floor, volume, SEED + seed + 17 * int(floor)))
                print("learn", learners[-1], flush=True)
    payload = {"volume": volume, "sweep": rows, "chosen_scale": chosen, "learners": learners}
    OUT.write_text(json.dumps(payload, indent=2))
    for row in rows:
        print(row["scale"], "floor", row["floor"]["mean"], "nofloor", row["nofloor"]["mean"], "gap", row["mean_gap"])
    print("chosen", chosen)


if __name__ == "__main__":
    main()
