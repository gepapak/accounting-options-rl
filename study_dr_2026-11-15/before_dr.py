"""Study DR, before any training: the published edge, the credit and the reach of gym-mtsim's zero-balance floor in every arm, from which
the registered predictions are fixed (PREREGISTRATION_DR.md). Writes before_dr.json. Uses Study GC's wrapper and placebo read-only.

1. One bar at the largest position. From 10,000 USD at the bundle's first close and the arm's leverage, the largest buy the margin check
   accepts, held one bar, over every bundled magnitude times k (capped at 0.9) with both signs: the share of bars on which the floor
   writes, the mean published reward and the mean booked profit (percent of the balance). In V arms this is averaged over the arm's
   k ~ log-uniform[1, k_hi] (trapezoid rule on 33 points in log k, the pilot's grid); in L arms k = 1. The published edge C of an arm is
   its mean published reward; the credit is published minus booked.
2. Reach in training. 1,000 training episodes of the arm's wrapper (fresh scaled placebo paths, balance log-uniform in 0.05 to 2 times
   10,000 USD) under PPO's initial policy (each action dimension standard normal, clipped to the box) and under the largest-position
   policy (open the largest accepted volume whenever flat, at the arm's leverage): episodes in which the floor writes, scaled to a run.
3. Why the leverage arms are fixed, not randomized. For leverage ~ log-uniform[lo, hi] hidden from the agent (the arms first proposed),
   a request sized for leverage L* is refused whenever the drawn leverage is below L*. The best single request earns
   max over L* of P(L >= L*) x (published reward of the largest position at L*), against the mean over L of the largest position's
   published reward, which only an agent that knows L can collect.

    <venv>\\Scripts\\python before_dr.py
"""
from __future__ import annotations

import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np

import envs_dr as D
from envs_dr import G

HERE = Path(__file__).resolve().parent
STEPS_PER_RUN = 1_007_616
N_EPISODES = 1000


def one_bar(k: float, leverage: float) -> dict:
    from gym_mtsim import OrderType
    m, b = G.MARKETS[D.MARKET], G.bundle(D.MARKET)
    sym, p0, idx = m["symbol"], b["p0"], b["index"]
    v = min(D.vmax(D.BALANCE, p0, leverage), b["info"].volume_max)
    rep, book, floor = [], [], []
    for mag in b["mags"]:
        for s in (-1.0, 1.0):
            sim = G.make_simulator(D.MARKET, "ORIG")
            sim.leverage = leverage
            sim.current_time = idx[0]
            sim.create_order(OrderType.Buy, sym, round(v, 6), m["fee"])
            frame = sim.symbols_data[sym].copy()
            frame.iloc[1, frame.columns.get_loc("Close")] = p0 * (1.0 + s * min(k * mag, D.K_CAP))
            sim.symbols_data[sym] = frame
            sim.tick(idx[1] - idx[0])
            led = G.ledger(sim, D.BALANCE)
            rep.append(float(sim.equity) - D.BALANCE)
            book.append(led - D.BALANCE)
            floor.append(float(sim.equity) - led > 1e-6)
    notional = v * b["info"].trade_contract_size * p0
    return dict(k=k, leverage=leverage, volume=v, position_over_balance=notional / D.BALANCE, floor_share=float(np.mean(floor)),
                published_pct=float(np.mean(rep)) / 100, booked_pct=float(np.mean(book)) / 100)


def arm_edge(arm: str) -> dict:
    spec = D.ARMS[arm]
    if spec["k_hi"] > 1.0:
        lk = np.linspace(0.0, np.log(spec["k_hi"]), 33)
        rows = [one_bar(float(np.exp(x)), spec["leverage"]) for x in lk]
        avg = lambda key: float(np.trapz([r[key] for r in rows], lk) / np.log(spec["k_hi"]))  # noqa: E731
        out = dict(published_pct=avg("published_pct"), booked_pct=avg("booked_pct"), floor_share=avg("floor_share"), grid=rows)
    else:
        r = one_bar(1.0, spec["leverage"])
        out = dict(published_pct=r["published_pct"], booked_pct=r["booked_pct"], floor_share=r["floor_share"], grid=[r])
    out["credit_pts"] = out["published_pct"] - out["booked_pct"]
    out["position_over_balance"] = out["grid"][0]["position_over_balance"]
    return out


def reach(arm: str, policy: str) -> dict:
    m, info = G.MARKETS[D.MARKET], G.bundle(D.MARKET)["info"]
    lev = D.ARMS[arm]["leverage"]
    env = D.DREnv(arm, "ORIG", train=True, seed=20261115)
    rng = np.random.default_rng(20261116)
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
                vm = max(float(e.simulator.free_margin), 0.0) / (info.trade_contract_size * (price / lev + m["fee"]))
                vm = min(float(np.floor(vm / info.volume_step + 1e-9) * info.volume_step), info.volume_max)
                a = np.array([-20.0, -20.0 if vm >= info.volume_min else 20.0, max(vm, info.volume_min)])
            _, _, te, tr, out = env.step(a)
            steps += 1
            done = te or tr
        st = out["stats"]
        floored += st["floored"]
        forgiven += st["reported"] - st["booked"]
    per_run = STEPS_PER_RUN / (steps / N_EPISODES)
    return dict(arm=arm, policy=policy, episodes=N_EPISODES, floored_share=floored / N_EPISODES,
                expected_floored_episodes_per_run=floored / N_EPISODES * per_run, forgiven_per_floored_episode_usd=forgiven / max(floored, 1))


def hidden_leverage(lo: float, hi: float) -> dict:
    lk = np.linspace(np.log(lo), np.log(hi), 25)                       # the pilot's grid for the leverage-aware mean
    aware = [one_bar(1.0, float(np.exp(x)))["published_pct"] for x in lk]
    grid = np.exp(np.linspace(np.log(lo), np.log(hi), 49))
    best, best_at = 0.0, None                                          # holding (no request) earns 0
    rows = []
    for ls in grid:
        pub = one_bar(1.0, float(ls))["published_pct"]
        p_acc = float(np.log(hi / ls) / np.log(hi / lo))
        rows.append(dict(sized_for=float(ls), accepted_share=p_acc, published_pct_if_accepted=pub, expected_pct=p_acc * pub))
        if p_acc * pub > best:
            best, best_at = p_acc * pub, float(ls)
    return dict(lo=lo, hi=hi, leverage_aware_mean_published_pct=float(np.trapz(aware, lk) / (lk[-1] - lk[0])),
                best_hidden_leverage_request_pct=best, best_request_sized_for_leverage=best_at, grid=rows)


def _init_worker():
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"


def main() -> None:
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=8, initializer=_init_worker) as ex:
        f_reach = {(a, p): ex.submit(reach, a, p) for a in D.ARMS for p in ("initial", "largest")}
        f_edge = {a: ex.submit(arm_edge, a) for a in D.ARMS}
        f_nom = ex.submit(arm_edge_nominal)
        f_hid = {(lo, hi): ex.submit(hidden_leverage, lo, hi) for lo, hi in ((100.0, 1000.0), (30.0, 500.0))}
        edge = {a: f.result() for a, f in f_edge.items()}
        rch = {f"{a}/{p}": f.result() for (a, p), f in f_reach.items()}
        hid = {f"{lo:g}-{hi:g}": f.result() for (lo, hi), f in f_hid.items()}
        nominal = f_nom.result()
    out = dict(edge=edge, nominal_gc=nominal, reach=rch, hidden_leverage=hid, seconds=round(time.time() - t0, 1))
    (HERE / "before_dr.json").write_text(json.dumps(out, indent=1))
    print(f"nominal (Study GC's forex arm, k = 1, leverage 100): published {nominal['published_pct']:+.2f}%  booked "
          f"{nominal['booked_pct']:+.2f}%  floor share {nominal['floor_share']:.4f}")
    for a in D.ARMS:
        e, ri, rl = edge[a], rch[f"{a}/initial"], rch[f"{a}/largest"]
        print(f"{a:5s} position {e['position_over_balance']:6.1f}x  published C {e['published_pct']:+7.2f}%  booked {e['booked_pct']:+6.2f}%  "
              f"credit {e['credit_pts']:+7.2f} pts  floor share {e['floor_share']:.3f} | reach initial {ri['floored_share']:.3f} "
              f"({ri['expected_floored_episodes_per_run']:.0f}/run, {ri['forgiven_per_floored_episode_usd']:.0f} USD each), largest "
              f"{rl['floored_share']:.3f}")
    for key, h in hid.items():
        print(f"hidden leverage ~ logU[{key}]: leverage-aware mean {h['leverage_aware_mean_published_pct']:+.2f}%, best single request "
              f"{h['best_hidden_leverage_request_pct']:+.2f}% (sized for leverage {h['best_request_sized_for_leverage']})")
    print(f"{out['seconds']} s")


def arm_edge_nominal() -> dict:
    r = one_bar(1.0, 100.0)
    return dict(published_pct=r["published_pct"], booked_pct=r["booked_pct"], floor_share=r["floor_share"],
                position_over_balance=r["position_over_balance"])


if __name__ == "__main__":
    main()
