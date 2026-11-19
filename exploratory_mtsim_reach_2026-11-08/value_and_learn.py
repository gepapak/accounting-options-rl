"""Reach of the published gym-mtsim floor at min lot and at max lot.

Predictions are in PROTOCOL.md, written before this run.
"""
from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from gym_mtsim import FOREX_DATA_PATH, MtEnv, MtSimulator, OrderType  # noqa: F401

SYMBOL = "EURUSD"
SEED = 20261108
ENTRY = 1.10
EQUITIES = (500.0, 10_000.0)
N_DRAWS = 500
N_EP = 800
LR = 0.05
OUT = Path(__file__).with_name("results.json")


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


def prices(entry: float) -> pd.DataFrame:
    idx = pd.date_range(datetime(2030, 1, 1), periods=4, freq="h", tz="UTC")
    return pd.DataFrame(
        {"Open": entry, "Close": entry, "Low": entry, "High": entry, "Volume": 1.0},
        index=idx,
    )


def make_env(equity: float, floor: bool) -> MtEnv:
    spec = MtSimulator(symbols_filename=FOREX_DATA_PATH)
    cls = MtSimulator if floor else NoFloor
    sim = cls(unit="USD", balance=equity, leverage=100.0, stop_out_level=0.2, hedge=True)
    sim.symbols_info = {SYMBOL: copy.deepcopy(spec.symbols_info[SYMBOL])}
    sim.symbols_data = {SYMBOL: prices(ENTRY)}
    return MtEnv(original_simulator=sim, trading_symbols=[SYMBOL], window_size=1, symbol_max_orders=1)


def hold_action(env: MtEnv) -> np.ndarray:
    action = np.zeros(env.action_space.shape, dtype=np.float64)
    action[-2] = 20.0
    action[-1] = 0.0
    action[:-2] = -20.0
    return action


def open_action(env: MtEnv, volume: float) -> np.ndarray:
    action = np.zeros(env.action_space.shape, dtype=np.float64)
    action[-2] = -20.0
    action[-1] = volume
    action[:-2] = -20.0
    return action


def max_volume(env: MtEnv) -> float:
    sim = env.simulator
    info = sim.symbols_info[SYMBOL]
    price = float(sim.price_at(SYMBOL, sim.current_time)["Close"])
    raw = 0.8 * sim.free_margin * sim.leverage / (info.trade_contract_size * price)
    steps = np.floor(raw / info.volume_step + 1e-9)
    return float(np.clip(steps * info.volume_step, info.volume_min, info.volume_max))


def min_volume(env: MtEnv) -> float:
    return float(env.simulator.symbols_info[SYMBOL].volume_min)


def set_next_close(env: MtEnv, price: float) -> None:
    tick = env.unwrapped._current_tick
    when = env.unwrapped.time_points[tick + 1]
    df = env.unwrapped.simulator.symbols_data[SYMBOL]
    df.loc[when, "Close"] = price
    df.loc[when, "High"] = max(float(df.loc[when, "Open"]), price)
    df.loc[when, "Low"] = min(float(df.loc[when, "Open"]), price)


def obs_tuple(obs) -> tuple:
    return (
        round(float(obs["balance"][0]), 6),
        round(float(obs["equity"][0]), 6),
        round(float(obs["margin"][0]), 6),
        round(float(obs["orders"][0, 0, 1]), 6),
        round(float(obs["orders"][0, 0, 2]), 6),
    )


def opened(equity: float, floor: bool, volume: float):
    env = make_env(equity, floor)
    env.reset()
    _, reward, _, _, info = env.step(open_action(env, volume))
    if len(env.simulator.orders) != 1:
        raise RuntimeError(f"open failed equity={equity} floor={floor} vol={volume} info={info}")
    return env, float(reward)


def one_bar(template: MtEnv, ret: float) -> tuple[float, tuple]:
    env = copy.deepcopy(template)
    entry = float(env.simulator.orders[0].entry_price)
    set_next_close(env, entry * float(np.exp(ret)))
    obs, reward, _, _, _ = env.step(hold_action(env))
    return float(reward), obs_tuple(obs)


def value_table(returns: np.ndarray) -> dict:
    rows = []
    for equity in EQUITIES:
        probe = make_env(equity, floor=True)
        probe.reset()
        volumes = {"min": min_volume(probe), "max": max_volume(probe)}
        info = probe.simulator.symbols_info[SYMBOL]
        for name, volume in volumes.items():
            held = {}
            open_reward = {}
            for floor in (True, False):
                env, r0 = opened(equity, floor, volume)
                open_reward[floor] = r0
                rewards = []
                observations = []
                for ret in returns:
                    reward, obs = one_bar(env, float(ret))
                    rewards.append(reward)
                    observations.append(obs)
                held[floor] = (np.asarray(rewards), observations)
            rf, of_ = held[True]
            ru, ou = held[False]
            gap = rf - ru
            obs_mismatch = sum(a != b for a, b in zip(of_, ou))
            reward_mismatch = int(np.sum(np.abs(gap) > 1e-6))
            both_same = sum(a == b and abs(g) <= 1e-6 for a, b, g in zip(of_, ou, gap))
            reward_diff_obs_same = sum(a == b and abs(g) > 1e-6 for a, b, g in zip(of_, ou, gap))
            rows.append({
                "equity": equity,
                "lot": name,
                "volume": volume,
                "contract": float(info.trade_contract_size),
                "volume_min": float(info.volume_min),
                "volume_step": float(info.volume_step),
                "open_reward_floor": open_reward[True],
                "open_reward_nofloor": open_reward[False],
                "mean_hold_floor": float(rf.mean()),
                "mean_hold_nofloor": float(ru.mean()),
                "mean_gap": float(gap.mean()),
                "gap_over_equity": float(gap.mean() / equity),
                "reward_mismatch": reward_mismatch,
                "obs_mismatch": obs_mismatch,
                "n": int(len(returns)),
                "agree_both": both_same,
                "reward_diff_but_obs_same": reward_diff_obs_same,
            })
    return {"rows": rows}


def reinforce(equity: float, floor: bool, returns: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    probe = make_env(equity, True)
    probe.reset()
    volume = max_volume(probe)
    env, open_reward = opened(equity, floor, volume)
    theta = 0.0
    probs = []
    for episode in range(N_EP):
        p = 1.0 / (1.0 + np.exp(-theta))
        take = bool(rng.random() < p)
        if take:
            ret = float(returns[int(rng.integers(0, len(returns)))])
            reward, _ = one_bar(env, ret)
            reward = reward + open_reward
            chosen = 1.0
        else:
            reward = 0.0
            chosen = 0.0
        # baseline: running mean of rewards is unnecessary; use the logistic score
        grad = (chosen - p) * reward
        theta += LR * grad
        if episode % 50 == 49:
            probs.append(float(1.0 / (1.0 + np.exp(-theta))))
    final = float(1.0 / (1.0 + np.exp(-theta)))
    return {"equity": equity, "floor": floor, "seed": seed, "final_p_max": final, "trace": probs}


def main() -> None:
    rng = np.random.default_rng(SEED)
    returns = rng.standard_t(3, size=N_DRAWS) / np.sqrt(3.0) * 0.0015
    table = value_table(returns)
    learners = []
    for equity in EQUITIES:
        for floor in (True, False):
            for seed in (0, 1, 2):
                learners.append(reinforce(equity, floor, returns, seed + 1000 * int(equity) + int(floor)))
                print(learners[-1]["equity"], learners[-1]["floor"], learners[-1]["seed"], learners[-1]["final_p_max"], flush=True)
    payload = {"seed": SEED, "n_draws": N_DRAWS, "table": table, "learners": learners}
    OUT.write_text(json.dumps(payload, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
