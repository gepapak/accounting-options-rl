"""Study GC exact checks (PREREGISTRATION_GC.md). Run from this folder with the study's environment:

    <venv>\\Scripts\\python test_envs_gc.py

Exit code 0 = every check passes. What is checked:
  1. the installed gym-mtsim is version 2.0.0 and every file equals the wheel's RECORD (the package is unmodified);
  2. the control (FLPATCH) is the published tick() without exactly its three balance-floor lines;
  3. the placebo paths: the bundle's length, its first close, its magnitudes only, fair signs; the shared test bank is fixed;
  4. the margin limit used by the probe is the largest volume the simulator's own margin check accepts;
  5. the exposure probe equals the order the environment actually opens;
  6. on random trajectories that reach the floor: the booked ledger taken from the order records equals the control's equity under the
     same prices and actions, and the control's reward equals the booked step profit; the forgiven amount is non-negative and never
     falls; the reading equals the booked step profit on every step;
  7. training episodes have the bundle's length.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import importlib.metadata
import inspect
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np

from gym_mtsim import MtSimulator, OrderType

import envs_gc as G

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(("PASS  " if ok else "FAIL  ") + name + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


def package_record() -> None:
    dist = importlib.metadata.distribution("gym-mtsim")
    good = bad = 0
    for f in dist.files or []:
        if f.hash is None or f.hash.mode != "sha256":
            continue
        digest = base64.urlsafe_b64encode(hashlib.sha256(dist.locate_file(f).read_bytes()).digest()).rstrip(b"=").decode()
        good, bad = good + (digest == f.hash.value), bad + (digest != f.hash.value)
    check("gym-mtsim is 2.0.0 and every installed file equals the wheel's RECORD", dist.version == "2.0.0" and bad == 0 and good >= 10,
          f"{good} files match, {bad} differ")


def control_source() -> None:
    def lines(fn):
        out = [ln.rstrip() for ln in inspect.getsource(fn).splitlines()]
        while out and not out[-1]:
            out.pop()
        return out
    pub, fl = lines(MtSimulator.tick), lines(G.MtSimulatorFL.tick)
    floor, head = [ln.strip() for ln in pub[-3:]], pub[:-3]
    while head and not head[-1]:
        head.pop()
    check("FLPATCH's tick() is the published tick() without its last three lines, the balance floor",
          floor == ["if self.balance < 0.:", "self.balance = 0.", "self.equity = self.balance"] and head == fl)


def placebo() -> None:
    for market in G.MARKETS:
        b = G.bundle(market)
        rng = np.random.default_rng(1)
        paths = np.stack([G.placebo_closes(rng, market) for _ in range(4000)])
        r = (paths[:, 1:] / paths[:, :-1] - 1.0).ravel()
        mags = np.sort(b["mags"])
        i = np.searchsorted(mags, np.abs(r)).clip(1, len(mags) - 1)
        dist = np.minimum(np.abs(mags[i] - np.abs(r)), np.abs(mags[i - 1] - np.abs(r)))
        up = float(np.mean(r > 0))
        se = 0.5 / np.sqrt(r.size)
        ok = (paths.shape[1] == len(b["index"]) and np.allclose(paths[:, 0], b["p0"]) and float(dist.max()) < 1e-9 and (paths > 0).all()
              and abs(up - 0.5) < 4 * se and abs(float(r.mean())) < 4 * float(r.std()) / np.sqrt(r.size))
        check(f"{market}: placebo paths have the bundle's length and first close, its magnitudes only and fair signs", ok,
              f"{paths.shape[0]} paths, share up {up:.4f}, mean return {r.mean():+.2e}")
        bank1, bank2 = G.test_bank(market), G.test_bank(market)
        check(f"{market}: the shared test bank is fixed and has {G.N_TEST} distinct paths",
              np.array_equal(bank1, bank2) and bank1.shape[0] == G.N_TEST and len({p.tobytes() for p in bank1}) == G.N_TEST)


def margin_limit_and_probe() -> None:
    for market in G.MARKETS:
        m, b = G.MARKETS[market], G.bundle(market)
        info, p0, sym = b["info"], b["p0"], m["symbol"]
        ok = True
        for bal in np.r_[np.geomspace(0.02, 2.0, 25) * G.BALANCE, 123.45, 9876.5]:
            v = G.vmax_closed(market, float(bal), p0)
            sim = G.make_simulator(market, "ORIG")
            sim.balance = sim.equity = float(bal)
            sim.current_time = b["index"][0]

            def accepts(vol: float) -> bool:
                s = copy.deepcopy(sim)
                try:
                    s.create_order(OrderType.Buy, sym, round(vol, 6), m["fee"])
                    return True
                except ValueError:
                    return False
            ok &= (accepts(v) and not accepts(v + info.volume_step)) if v >= info.volume_min else not accepts(info.volume_min)
        check(f"{market}: the probe's margin limit is the largest volume the simulator accepts (27 balances)", bool(ok))

        env = G.GCEnv(market, "ORIG", train=False, seed=3)
        rng = np.random.default_rng(5)
        n_open = n_zero = bad = 0
        for e in G.EGRID:
            for _ in range(40):
                a = np.array([rng.normal(0, 2), rng.normal(0, 2), rng.normal(0, 1) * rng.choice([0.3, 3.0, 30.0])])
                x = G.exposure(lambda obs, a=a: a, market, e)
                path = G.placebo_closes(rng, market)
                path[1] = path[0]                      # the first action is taken at the bundle's first close
                env.reset(options={"path": path, "balance": e * G.BALANCE})
                env.step(a)
                sim = env.env.unwrapped.simulator
                vols = [o.volume for o in sim.orders] + [o.volume for o in sim.closed_orders]
                if x > 0:
                    n_open += 1
                    bad += not (len(vols) == 1 and abs(vols[0] - x * G.vmax_closed(market, e * G.BALANCE, p0)) < 1e-6)
                else:
                    n_zero += 1
                    bad += len(vols) != 0
        check(f"{market}: the exposure probe equals the order the environment opens", bad == 0 and n_open >= 20 and n_zero >= 20,
              f"{n_open} probes opened, {n_zero} held or refused, {bad} mismatches")


def trajectories() -> None:
    for market in G.MARKETS:
        m, info = G.MARKETS[market], G.bundle(market)["info"]
        rng = np.random.default_rng(11)
        orig = G.GCEnv(market, "ORIG", train=False, seed=1, audit=True)
        flp = G.GCEnv(market, "FLPATCH", train=False, seed=1)
        worst_ledger = worst_reward = 0.0
        forgiven_ok, same_orders, steps = True, True, 0
        for _ in range(300):
            path = G.placebo_closes(rng, market)
            bal = G.BALANCE * float(np.exp(rng.uniform(np.log(0.05), np.log(2.0))))
            orig.reset(options={"path": path, "balance": bal})
            flp.reset(options={"path": path, "balance": bal})
            prev_led, prev_forgiven, done = bal, 0.0, False
            while not done:
                e = orig.env.unwrapped
                price = float(e.prices[m["symbol"]][e._current_tick, 0])
                vm = max(float(e.simulator.free_margin), 0.0) / (info.trade_contract_size * (price / G.LEVERAGE + m["fee"]))
                a = np.array([rng.normal(-1.0, 2.0), rng.normal(0.0, 1.5), rng.choice([-1.0, 1.0]) * rng.uniform(0.0, 1.2) * max(vm, 0.01)])
                _, _, t1, tr1, _ = orig.step(a)
                _, r2, _, _, _ = flp.step(a)
                so, sf = orig.env.unwrapped.simulator, flp.env.unwrapped.simulator
                led = G.ledger(so, bal)
                worst_ledger = max(worst_ledger, abs(float(sf.equity) - led))
                worst_reward = max(worst_reward, abs(100.0 * r2 - (led - prev_led)))
                forgiven = float(so.equity) - led
                forgiven_ok &= forgiven >= -1e-6 and forgiven >= prev_forgiven - 1e-6
                same_orders &= ([o.volume for o in so.orders + so.closed_orders] == [o.volume for o in sf.orders + sf.closed_orders])
                prev_led, prev_forgiven = led, forgiven
                steps += 1
                done = t1 or tr1
        c = orig.count
        check(f"{market}: booked ledger from the order records equals the control's equity, and the control's reward the booked step "
              f"profit", worst_ledger < 1e-6 and worst_reward < 1e-6 and same_orders,
              f"{steps} steps, largest gaps {worst_ledger:.2e} and {worst_reward:.2e} USD")
        check(f"{market}: the forgiven amount is non-negative and never falls", bool(forgiven_ok))
        need_floor = c["floor_steps"] >= 50 if market == "crypto" else True
        check(f"{market}: the reading equals the booked step profit on every step", c["line_steps"] == steps and c["line_max_abs_err"] < 1e-4
              and need_floor, f"{c['line_steps']} steps, {c['floor_steps']} with the floor writing, largest error {c['line_max_abs_err']:.2e} USD")


def episode_length() -> None:
    for market in G.MARKETS:
        env = G.GCEnv(market, "ORIG", train=True, seed=2)
        lengths = set()
        for _ in range(3):
            env.reset()
            n, done = 0, False
            while not done:
                _, _, te, tr, _ = env.step(np.array([0.0, 5.0, 0.0]))
                n, done = n + 1, te or tr
            lengths.add(n)
        check(f"{market}: training episodes run from the second bar of the bundle to its last", lengths == {len(G.bundle(market)["index"]) - 2},
              f"length {sorted(lengths)}")


if __name__ == "__main__":
    package_record()
    control_source()
    placebo()
    margin_limit_and_probe()
    trajectories()
    episode_length()
    print("ALL CHECKS PASS" if not FAILS else f"{len(FAILS)} CHECK(S) FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)
