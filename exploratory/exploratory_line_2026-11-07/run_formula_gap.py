"""Reward-shape audit of unmodified gym-mtsim 2.0.0.

The reader sees only (t, reward). See PROTOCOL.md, which was written before this run.
"""
from __future__ import annotations

import copy
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from gym_mtsim import FOREX_DATA_PATH, MtEnv, MtSimulator, OrderType

SYMBOL = "EURUSD"
SEED = 20261107
N = 100
GRID = np.array([-16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16], dtype=float)
DP_REL = 0.01
OUT = Path(__file__).with_name("results.json")


class NoFloor(MtSimulator):
    """Public tick() with the two balance-floor lines removed. Control only."""

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
    idx = pd.date_range(datetime(2030, 1, 1), periods=3, freq="h", tz="UTC")
    return pd.DataFrame(
        {"Open": [entry, entry, entry], "Close": [entry, entry, entry],
         "Low": [entry, entry, entry], "High": [entry, entry, entry],
         "Volume": [1.0, 1.0, 1.0]},
        index=idx,
    )


def make_env(entry: float, floor: bool, balance: float = 1000.0) -> MtEnv:
    spec = MtSimulator(symbols_filename=FOREX_DATA_PATH)
    cls = MtSimulator if floor else NoFloor
    sim = cls(unit="USD", balance=balance, leverage=100.0, stop_out_level=0.2, hedge=True)
    sim.symbols_info = {SYMBOL: copy.deepcopy(spec.symbols_info[SYMBOL])}
    sim.symbols_data = {SYMBOL: prices(entry)}
    return MtEnv(original_simulator=sim, trading_symbols=[SYMBOL], window_size=1, symbol_max_orders=1)


def open_volume(env: MtEnv) -> float:
    sim = env.simulator
    info = sim.symbols_info[SYMBOL]
    price = float(sim.price_at(SYMBOL, sim.current_time)["Close"])
    # margin = volume * contract * price / leverage <= free margin
    raw = 0.8 * sim.free_margin * sim.leverage / (info.trade_contract_size * price)
    steps = np.floor(raw / info.volume_step + 1e-9)
    volume = float(np.clip(steps * info.volume_step, info.volume_min, info.volume_max))
    return volume


def hold_action(env: MtEnv) -> np.ndarray:
    action = np.zeros(env.action_space.shape, dtype=np.float64)
    action[-2] = 20.0  # hold
    action[-1] = 0.0
    action[:-2] = -20.0
    return action


def open_action(env: MtEnv, volume: float) -> np.ndarray:
    action = np.zeros(env.action_space.shape, dtype=np.float64)
    action[-2] = -20.0  # do not hold
    action[-1] = volume
    action[:-2] = -20.0
    return action


def prepared(entry: float, floor: bool):
    env = make_env(entry, floor=floor)
    env.reset()
    volume = open_volume(env)
    _, reward, _, _, info = env.step(open_action(env, volume))
    if len(env.simulator.orders) != 1:
        raise RuntimeError(f"open failed at entry {entry}: reward {reward}, info {info}")
    return env, volume


def set_next_close(env: MtEnv, price: float) -> None:
    tick = env.unwrapped._current_tick
    when = env.unwrapped.time_points[tick + 1]
    df = env.unwrapped.simulator.symbols_data[SYMBOL]
    df.loc[when, "Close"] = price
    df.loc[when, "High"] = max(float(df.loc[when, "Open"]), price)
    df.loc[when, "Low"] = min(float(df.loc[when, "Open"]), price)


def query(env: MtEnv, price: float):
    clone = copy.deepcopy(env)
    equity_before = float(clone.simulator.equity)
    profit_before = float(clone.simulator.orders[0].profit)
    set_next_close(clone, price)
    obs, reward, _, _, info = clone.step(hold_action(clone))
    order = clone.simulator.closed_orders[-1] if clone.simulator.closed_orders else clone.simulator.orders[-1]
    return {
        "reward": float(reward),
        "equity_before": equity_before,
        "equity_after": float(obs["equity"][0]),
        "balance_after": float(obs["balance"][0]),
        "open_volume": float(obs["orders"][0, 0, 1]),
        "open_profit": float(obs["orders"][0, 0, 2]),
        "info_closed": len(info.get("closed_orders", {}).get(SYMBOL, [])),
        "economic": float(order.profit) - profit_before,
        "n_open": len(clone.simulator.orders),
    }


def slopes_agree(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-6 * max(1.0, abs(a), abs(b))


def read_reward(ts: np.ndarray, phi: np.ndarray):
    s = np.diff(phi) / np.diff(ts)
    i0 = int(np.where(ts == 0)[0][0])
    left, right = s[i0 - 1], s[i0]
    solvent_t = float(right) if slopes_agree(left, right) else None
    flat = bool(abs(s[0]) < 1e-6 and abs(s[1]) < 1e-6 and phi[0] < phi[i0] - 1.0)
    return solvent_t, flat, s


def one(entry: float, floor: bool) -> dict:
    env, volume = prepared(entry, floor=floor)
    contract = float(env.simulator.symbols_info[SYMBOL].trade_contract_size)
    entry_price = float(env.simulator.orders[0].entry_price)
    truth_slope = volume * contract  # buy, USD profit, unit ratio 1
    rows = []
    for t in GRID:
        price = entry_price * (1.0 + DP_REL * float(t))
        if price <= 0:
            raise RuntimeError("non-positive price")
        row = query(env, price)
        row["t"] = float(t)
        row["price"] = price
        rows.append(row)
    ts = np.array([r["t"] for r in rows])
    phi = np.array([r["reward"] for r in rows])
    solvent_t, flat, _ = read_reward(ts, phi)
    i0 = int(np.where(ts == 0)[0][0])
    booked = None if solvent_t is None else solvent_t * ts + phi[i0]
    dprice_dt = entry_price * DP_REL
    solvent = None if solvent_t is None else solvent_t / dprice_dt
    gaps = []
    if booked is not None:
        for r, b in zip(rows, booked):
            gaps.append({
                "t": r["t"],
                "formula_gap": abs(r["reward"] - (r["equity_after"] - r["equity_before"])),
                "book_err": abs(float(b) - r["economic"]),
                "shortfall": r["reward"] - r["economic"],
                "forgiven_err": abs((r["reward"] - float(b)) - (r["reward"] - r["economic"])),
            })
    worst = rows[0]
    return {
        "entry": entry,
        "floor": floor,
        "volume": volume,
        "contract": contract,
        "truth_slope": truth_slope,
        "solvent_slope": solvent,
        "slope_rel_err": None if solvent is None else abs(solvent - truth_slope) / truth_slope,
        "flat_tail": flat,
        "phi0": float(phi[i0]),
        "worst_reward": worst["reward"],
        "worst_economic": worst["economic"],
        "worst_shortfall": worst["reward"] - worst["economic"],
        "worst_open_volume": worst["open_volume"],
        "worst_equity": worst["equity_after"],
        "worst_balance": worst["balance_after"],
        "worst_info_closed": worst["info_closed"],
        "max_formula_gap": max(g["formula_gap"] for g in gaps) if gaps else None,
        "max_book_err": max(g["book_err"] for g in gaps) if gaps else None,
        "max_forgiven_err": max(g["forgiven_err"] for g in gaps) if gaps else None,
        "equity_before": rows[0]["equity_before"],
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    entries = rng.uniform(0.8, 1.6, size=N)
    public = [one(float(e), True) for e in entries]
    control = [one(float(e), False) for e in entries]

    def ok_public(r):
        return (
            r["max_formula_gap"] is not None and r["max_formula_gap"] <= 1e-6
            and r["worst_open_volume"] == 0.0
            and abs(r["worst_equity"] - r["worst_balance"]) <= 1e-6
            and r["worst_shortfall"] >= 1.0
            and r["worst_info_closed"] == 0
            and r["solvent_slope"] is not None and r["slope_rel_err"] <= 1e-4
            and r["flat_tail"]
            and r["max_book_err"] <= 0.05
            and r["max_forgiven_err"] <= 0.05
        )

    def ok_control(r):
        return (not r["flat_tail"]) and r["max_book_err"] is not None and r["max_book_err"] <= 0.05 and r["max_formula_gap"] <= 1e-6

    summary = {
        "package": "gym-mtsim==2.0.0",
        "n": N,
        "seed": SEED,
        "grid": GRID.tolist(),
        "public_pass": int(sum(ok_public(r) for r in public)),
        "control_pass": int(sum(ok_control(r) for r in control)),
        "public_formula_gap_max": max(r["max_formula_gap"] for r in public),
        "public_slope_rel_err_max": max(r["slope_rel_err"] for r in public),
        "public_book_err_max": max(r["max_book_err"] for r in public),
        "public_shortfall_min": min(r["worst_shortfall"] for r in public),
        "public_shortfall_max": max(r["worst_shortfall"] for r in public),
        "public_flat": int(sum(r["flat_tail"] for r in public)),
        "control_flat": int(sum(r["flat_tail"] for r in control)),
        "control_book_err_max": max(r["max_book_err"] for r in control),
        "example_public": public[0],
        "example_control": control[0],
    }
    OUT.write_text(json.dumps({"summary": summary, "public": public, "control": control}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("FAILED", type(exc).__name__, exc, file=sys.stderr)
        raise
