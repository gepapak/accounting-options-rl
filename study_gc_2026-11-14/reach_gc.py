"""Study GC, before any training: reach and credit of gym-mtsim's zero-balance floor in the two registered markets. Writes reach_gc.json.

1. One bar at the largest position. From 10,000 USD at the bundle's first close, the largest buy the margin check accepts, held one bar,
   over every bundled magnitude with both signs: the share of bars on which the floor writes, and the mean published reward and mean
   booked profit (percent of the balance).
2. Reach in training. Training episodes of the study's wrapper (balance log-uniform in 0.05 to 2 times 10,000 USD, fresh placebo paths)
   under two fixed policies: PPO's initial policy (each action dimension standard normal, clipped to the box) and the largest-position
   policy (open the largest accepted volume whenever flat). Counted: episodes in which the floor writes, scaled to the 1,007,616 steps of
   a run, and the forgiven amount per such episode.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np

from gym_mtsim import OrderType

import envs_gc as G

STEPS_PER_RUN = 1_007_616
N_EPISODES = 2000


def one_bar(market: str) -> dict:
    m, b = G.MARKETS[market], G.bundle(market)
    sym, p0, idx = m["symbol"], b["p0"], b["index"]
    v = G.vmax_closed(market, G.BALANCE, p0)
    rep, book, floor = [], [], []
    for mag in b["mags"]:
        for s in (-1.0, 1.0):
            sim = G.make_simulator(market, "ORIG")
            sim.current_time = idx[0]
            sim.create_order(OrderType.Buy, sym, round(v, 6), m["fee"])
            frame = sim.symbols_data[sym].copy()
            frame.iloc[1, frame.columns.get_loc("Close")] = p0 * (1.0 + s * mag)
            sim.symbols_data[sym] = frame
            sim.tick(idx[1] - idx[0])
            led = G.ledger(sim, G.BALANCE)
            rep.append(float(sim.equity) - G.BALANCE)
            book.append(led - G.BALANCE)
            floor.append(float(sim.equity) - led > 1e-6)
    notional = v * b["info"].trade_contract_size * p0
    return dict(volume=v, position_over_balance=notional / G.BALANCE, adverse_move_that_empties_the_account_pct=100 * G.BALANCE / notional,
                bars=2 * len(b["mags"]), floor_share=float(np.mean(floor)), published_reward_pct=float(np.mean(rep)) / 100,
                booked_profit_pct=float(np.mean(book)) / 100, placebo_return_sd_pct=100 * float(np.sqrt(np.mean(b["mags"] ** 2))))


def reach(market: str, policy: str) -> dict:
    m, info = G.MARKETS[market], G.bundle(market)["info"]
    env = G.GCEnv(market, "ORIG", train=True, seed=20261114)
    rng = np.random.default_rng(20261115)
    floored, forgiven, steps = 0, 0.0, 0
    for _ in range(N_EPISODES):
        env.reset()
        done = False
        while not done:
            if policy == "initial":
                a = np.clip(rng.standard_normal(3), -100.0, 100.0)
            else:
                e = env.env.unwrapped
                price = float(e.prices[m["symbol"]][e._current_tick, 0])
                vm = max(float(e.simulator.free_margin), 0.0) / (info.trade_contract_size * (price / G.LEVERAGE + m["fee"]))
                vm = float(np.floor(vm / info.volume_step + 1e-9) * info.volume_step)
                a = np.array([-20.0, -20.0 if vm >= info.volume_min else 20.0, max(vm, info.volume_min)])
            _, _, te, tr, out = env.step(a)
            steps += 1
            done = te or tr
        st = out["stats"]
        floored += st["floored"]
        forgiven += st["reported"] - st["booked"]
    per_run = STEPS_PER_RUN / (steps / N_EPISODES)
    return dict(episodes=N_EPISODES, floored_share=floored / N_EPISODES, expected_floored_episodes_per_run=floored / N_EPISODES * per_run,
                forgiven_per_floored_episode_usd=forgiven / max(floored, 1), episodes_per_run=per_run)


def main() -> None:
    out = {}
    for market in G.MARKETS:
        out[market] = dict(one_bar_largest_position=one_bar(market), reach_initial_policy=reach(market, "initial"),
                           reach_largest_position_policy=reach(market, "largest"))
        print(market, json.dumps(out[market], indent=1), flush=True)
    Path(__file__).with_name("reach_gc.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
