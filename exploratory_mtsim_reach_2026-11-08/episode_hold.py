"""Hold a fixed lot for Study G's episode length on the published simulator.

One bar at 0.15% did not reach the floor. Study G trained on 500-hour episodes.
This measures the paired gap over that horizon, minimum lot and maximum lot.
Prices and starts are fixed before the comparison: seed 20261108 for prices,
seed 20261109 for the 60 start indices. Both accountings see the same bars.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "LIABILITY_STUDY_2026-09-29" / "mtsim_study"))
from gym_mtsim import MtEnv  # noqa: E402
from mtsim_common import SYMBOL, make_simulator, placebo_prices  # noqa: E402

HOURS = 500
N_EP = 60
PRICE_HOURS = 20_000
EQUITIES = (500.0, 10_000.0)
OUT = Path(__file__).with_name("episode_hold.json")


def hold_action(env: MtEnv) -> np.ndarray:
    action = np.zeros(env.action_space.shape, dtype=np.float64)
    action[-2] = 20.0
    action[:-2] = -20.0
    return action


def open_action(env: MtEnv, volume: float) -> np.ndarray:
    action = np.zeros(env.action_space.shape, dtype=np.float64)
    action[-2] = -20.0
    action[-1] = volume
    action[:-2] = -20.0
    return action


def lot_volume(env: MtEnv, lot: str) -> float:
    info = env.simulator.symbols_info[SYMBOL]
    if lot == "min":
        return float(info.volume_min)
    price = float(env.simulator.price_at(SYMBOL, env.simulator.current_time)["Close"])
    raw = 0.8 * env.simulator.free_margin * env.simulator.leverage / (info.trade_contract_size * price)
    steps = np.floor(raw / info.volume_step + 1e-9)
    return float(np.clip(steps * info.volume_step, info.volume_min, info.volume_max))


def one_episode(env: MtEnv, start: int, equity: float, lot: str) -> dict:
    raw = env.unwrapped
    raw.original_simulator.balance = equity
    raw.original_simulator.equity = equity
    raw._start_tick = int(start)
    raw._end_tick = int(start) + HOURS
    raw.reset()
    volume = lot_volume(raw, lot)
    total = 0.0
    opened = False
    wiped = False
    for _ in range(HOURS):
        action = open_action(raw, volume) if not opened else hold_action(raw)
        _, reward, _, truncated, _ = raw.step(action)
        total += float(reward)
        opened = True
        if raw.simulator.balance <= 0.0 and len(raw.simulator.orders) == 0:
            wiped = True
        if truncated:
            break
    return {"reward": total, "volume": volume, "wiped": wiped, "final_balance": float(raw.simulator.balance)}


def summarize(pairs: list[dict]) -> dict:
    gap = np.array([p["floor"] - p["nofloor"] for p in pairs], dtype=float)
    differ = np.abs(gap) > 0.01
    boots = []
    rng = np.random.default_rng(0)
    idx = np.arange(len(gap))
    for _ in range(2000):
        take = gap[rng.choice(idx, size=len(gap), replace=True)]
        boots.append(float(take.mean()))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {
        "n": len(pairs),
        "mean_floor": float(np.mean([p["floor"] for p in pairs])),
        "mean_nofloor": float(np.mean([p["nofloor"] for p in pairs])),
        "mean_gap": float(gap.mean()),
        "gap_ci95": [float(lo), float(hi)],
        "n_differ": int(differ.sum()),
        "mean_gap_when_differ": float(gap[differ].mean()) if differ.any() else 0.0,
        "n_wiped_floor": int(sum(p["wiped_floor"] for p in pairs)),
        "volume": pairs[0]["volume"],
    }


def main() -> None:
    prices = placebo_prices(20261108, hours=PRICE_HOURS)
    starts = np.random.default_rng(20261109).integers(2, PRICE_HOURS - HOURS - 2, size=N_EP)
    rows = []
    detail = []
    for equity in EQUITIES:
        envs = {}
        for condition in ("ORIG", "FLPATCH"):
            sim = make_simulator(condition, prices, balance=equity)
            envs[condition] = MtEnv(original_simulator=sim, trading_symbols=[SYMBOL], window_size=2, symbol_max_orders=1)
        for lot in ("min", "max"):
            pairs = []
            for i, start in enumerate(starts):
                floored = one_episode(envs["ORIG"], int(start), equity, lot)
                plain = one_episode(envs["FLPATCH"], int(start), equity, lot)
                pairs.append({
                    "floor": floored["reward"],
                    "nofloor": plain["reward"],
                    "wiped_floor": floored["wiped"],
                    "volume": floored["volume"],
                })
                if i % 10 == 9:
                    print(equity, lot, i + 1, "gap", floored["reward"] - plain["reward"], flush=True)
            summary = summarize(pairs)
            summary["equity"] = equity
            summary["lot"] = lot
            rows.append(summary)
            detail.append({"equity": equity, "lot": lot, "gaps": [p["floor"] - p["nofloor"] for p in pairs]})
            print("SUMMARY", summary, flush=True)
    OUT.write_text(json.dumps({"hours": HOURS, "n_episodes": N_EP, "rows": rows, "detail": detail}, indent=2))


if __name__ == "__main__":
    main()
