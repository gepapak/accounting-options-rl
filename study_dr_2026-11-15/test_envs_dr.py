"""Study DR exact checks (PREREGISTRATION_DR.md). Run from this folder with Study GC's environment:

    <venv>\\Scripts\\python test_envs_dr.py

Exit code 0 = every check passes. What is checked:
  1. Study GC's package checks: gym-mtsim 2.0.0 equals the wheel's RECORD; FLPATCH is the published tick() without its three floor lines;
  2. scaled placebo paths: every bar's magnitude is a bundled magnitude times k (capped at 0.9), fair signs; k = 1 draws exactly
     Study GC's path; the nominal and stressed test banks are fixed;
  3. the training distribution: in V arms log k is uniform on [0, log k_hi] and the installed path is scaled by that k; in L arms
     the training paths and balances are Study GC's, draw for draw;
  4. every arm's simulator has the arm's leverage after reset, and the margin of an opened order is volume x contract x price / leverage;
  5. at leverages 30, 100 and 500 the probe's margin limit is the largest volume the simulator accepts;
  6. the exposure probe equals the order the environment opens, in every arm; at leverage 100 it equals Study GC's probe;
  7. on random trajectories that reach the floor (V16 and L500 paths): the booked ledger from the order records equals the control's
     equity under the same prices and actions, the control's reward equals the booked step profit, and the forgiven amount is
     non-negative and never falls;
  8. training episodes have the bundle's length in every arm.
"""
from __future__ import annotations

import copy
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np

import envs_dr as D
from envs_dr import G
from gym_mtsim import OrderType

import test_envs_gc as TG  # noqa: E402  (Study GC's checks, imported read-only)

FAILS: list[str] = []
B = G.bundle(D.MARKET)
M = G.MARKETS[D.MARKET]
INFO, P0, SYM = B["info"], B["p0"], M["symbol"]


def check(name: str, ok: bool, detail: str = "") -> None:
    print(("PASS  " if ok else "FAIL  ") + name + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


def gc_package_checks() -> None:
    TG.package_record()
    TG.control_source()
    for name in TG.FAILS:
        FAILS.append("Study GC check: " + name)


def placebo() -> None:
    mags = np.sort(B["mags"])
    for k in (1.0, 2.5, 16.0):
        rng = np.random.default_rng(7)
        paths = np.stack([D.scaled_placebo(rng, k) for _ in range(3000)])
        r = (paths[:, 1:] / paths[:, :-1] - 1.0).ravel()
        target = np.minimum(k * mags, D.K_CAP)
        i = np.searchsorted(target, np.abs(r)).clip(1, len(target) - 1)
        dist = np.minimum(np.abs(target[i] - np.abs(r)), np.abs(target[i - 1] - np.abs(r)))
        up, se = float(np.mean(r > 0)), 0.5 / np.sqrt(r.size)
        ok = (paths.shape[1] == len(B["index"]) and np.allclose(paths[:, 0], P0) and float(dist.max()) < 1e-9 and (paths > 0).all()
              and abs(up - 0.5) < 4 * se)
        check(f"scaled placebo k={k:g}: bundled magnitudes times k (cap 0.9), the bundle's length and first close, fair signs", ok,
              f"share up {up:.4f}, largest distance {dist.max():.1e}")
    r1, r2 = np.random.default_rng(3), np.random.default_rng(3)
    same = all(np.array_equal(D.scaled_placebo(r1, 1.0), G.placebo_closes(r2, D.MARKET)) for _ in range(200))
    check("k = 1 draws exactly Study GC's placebo path from the same generator state (200 paths)", same)
    n1, n2, s1, s2 = D.nominal_bank(), D.nominal_bank(), D.stressed_bank(), D.stressed_bank()
    check("the nominal test bank is Study GC's and fixed; the stressed bank is fixed; 300 distinct paths each",
          np.array_equal(n1, n2) and np.array_equal(n1, G.test_bank(D.MARKET)) and np.array_equal(s1, s2)
          and len({p.tobytes() for p in n1}) == 300 and len({p.tobytes() for p in s1}) == 300)
    rs = np.abs(s1[:, 1:] / s1[:, :-1] - 1.0).ravel()
    t4 = np.minimum(4.0 * mags, D.K_CAP)
    j = np.searchsorted(t4, rs).clip(1, len(t4) - 1)
    d4 = np.minimum(np.abs(t4[j] - rs), np.abs(t4[j - 1] - rs))
    check("the stressed bank's magnitudes are four times the bundled ones", float(d4.max()) < 1e-9, f"largest distance {d4.max():.1e}")


def training_distribution() -> None:
    for arm in ("V2", "V4", "V8", "V16"):
        env = D.DREnv(arm, "ORIG", train=True, seed=21)
        k_hi, ks, ok_path = D.ARMS[arm]["k_hi"], [], True
        mags = np.sort(B["mags"])
        for _ in range(1000):
            env.reset()
            k = env.last_k
            ks.append(k)
            close = env.env.unwrapped.prices[SYM][:, 0]
            r = np.abs(close[1:] / close[:-1] - 1.0)
            t = np.minimum(k * mags, D.K_CAP)
            i = np.searchsorted(t, r).clip(1, len(t) - 1)
            ok_path &= float(np.minimum(np.abs(t[i] - r), np.abs(t[i - 1] - r)).max()) < 1e-9
        u = np.log(np.array(ks)) / np.log(k_hi)
        se = (1 / np.sqrt(12)) / np.sqrt(len(u))
        check(f"{arm}: log k is uniform on [0, log {k_hi:g}] and the installed path is scaled by that k (1,000 resets)",
              bool(ok_path) and u.min() >= 0 and u.max() <= 1 and abs(u.mean() - 0.5) < 4 * se and abs(u.std() - 1 / np.sqrt(12)) < 0.03,
              f"mean of log k / log k_hi {u.mean():.3f}, sd {u.std():.3f}")
    for arm in ("L30", "L500"):
        dr, gc, same = D.DREnv(arm, "ORIG", train=True, seed=33), G.GCEnv(D.MARKET, "ORIG", train=True, seed=33), True
        for _ in range(50):
            dr.reset()
            gc.reset()
            same &= (np.array_equal(dr.env.unwrapped.prices[SYM], gc.env.unwrapped.prices[SYM]) and dr.e0 == gc.e0 and dr.last_k == 1.0)
        check(f"{arm}: training paths and balances are Study GC's, draw for draw (50 episodes)", bool(same))


def leverage_and_margin() -> None:
    for arm, spec in D.ARMS.items():
        env = D.DREnv(arm, "ORIG", train=False, seed=4)
        path = G.placebo_closes(np.random.default_rng(9), D.MARKET)
        path[1] = path[0]
        env.reset(options={"path": path, "balance": D.BALANCE})
        sim = env.env.unwrapped.simulator
        v = 0.5 * D.vmax(D.BALANCE, P0, spec["leverage"])
        v = round(v / INFO.volume_step) * INFO.volume_step
        env.step(np.array([-5.0, -5.0, v]))
        sim = env.env.unwrapped.simulator
        orders = sim.orders + sim.closed_orders
        want = v * INFO.trade_contract_size * P0 / spec["leverage"]
        check(f"{arm}: the simulator has leverage {spec['leverage']:g} after reset and margins an order at that leverage",
              sim.leverage == spec["leverage"] and len(orders) == 1 and abs(orders[0].margin - want) < 1e-9 * max(want, 1.0),
              f"margin {orders[0].margin:.4f} USD, expected {want:.4f}" if orders else "no order opened")


def margin_limit_and_probe() -> None:
    for lev in (30.0, 100.0, 500.0):
        ok = True
        for bal in np.r_[np.geomspace(0.02, 2.0, 25) * D.BALANCE, 123.45, 9876.5]:
            v = D.vmax(float(bal), P0, lev)
            sim = G.make_simulator(D.MARKET, "ORIG")
            sim.leverage = lev
            sim.balance = sim.equity = float(bal)
            sim.current_time = B["index"][0]

            def accepts(vol: float) -> bool:
                s = copy.deepcopy(sim)
                try:
                    s.create_order(OrderType.Buy, SYM, round(vol, 6), M["fee"])
                    return True
                except ValueError:
                    return False
            ok &= (accepts(v) and not accepts(v + INFO.volume_step)) if v >= INFO.volume_min else not accepts(INFO.volume_min)
        check(f"leverage {lev:g}: the probe's margin limit is the largest volume the simulator accepts (27 balances)", bool(ok))
    for arm in D.ARMS:
        lev = D.ARMS[arm]["leverage"]
        env = D.DREnv(arm, "ORIG", train=False, seed=3)
        rng = np.random.default_rng(5)
        n_open = n_zero = bad = gc_diff = 0
        for e in D.EGRID:
            for _ in range(40):
                a = np.array([rng.normal(0, 2), rng.normal(0, 2), rng.normal(0, 1) * rng.choice([0.3, 3.0, 30.0])])
                x = D.exposure(lambda obs, a=a: a, arm, e)
                if lev == 100.0:
                    gc_diff += x != G.exposure(lambda obs, a=a: a, D.MARKET, e)
                path = G.placebo_closes(rng, D.MARKET)
                path[1] = path[0]                      # the first action is taken at the bundle's first close
                env.reset(options={"path": path, "balance": e * D.BALANCE})
                env.step(a)
                sim = env.env.unwrapped.simulator
                vols = [o.volume for o in sim.orders] + [o.volume for o in sim.closed_orders]
                if x > 0:
                    n_open += 1
                    bad += not (len(vols) == 1 and abs(vols[0] - x * D.vmax(e * D.BALANCE, P0, lev)) < 1e-6)
                else:
                    n_zero += 1
                    bad += len(vols) != 0
        extra = f", {gc_diff} differ from Study GC's probe" if lev == 100.0 else ""
        check(f"{arm}: the exposure probe equals the order the environment opens" + (" and Study GC's probe" if lev == 100.0 else ""),
              bad == 0 and gc_diff == 0 and n_open >= 20 and n_zero >= 20, f"{n_open} opened, {n_zero} held or refused, {bad} mismatches{extra}")


def trajectories() -> None:
    for arm in ("V16", "L500", "L30"):
        lev = D.ARMS[arm]["leverage"]
        rng = np.random.default_rng(11)
        orig, flp = D.DREnv(arm, "ORIG", train=False, seed=1), D.DREnv(arm, "FLPATCH", train=False, seed=1)
        worst_ledger = worst_reward = 0.0
        forgiven_ok, same_orders, steps = True, True, 0
        for _ in range(150):
            k = float(np.exp(rng.uniform(0.0, np.log(D.ARMS[arm]["k_hi"])))) if D.ARMS[arm]["k_hi"] > 1 else 1.0
            path = D.scaled_placebo(rng, k)
            bal = D.BALANCE * float(np.exp(rng.uniform(np.log(0.05), np.log(2.0))))
            orig.reset(options={"path": path, "balance": bal})
            flp.reset(options={"path": path, "balance": bal})
            prev_led, prev_forgiven, done = bal, 0.0, False
            while not done:
                e = orig.env.unwrapped
                price = float(e.prices[SYM][e._current_tick, 0])
                vm = max(float(e.simulator.free_margin), 0.0) / (INFO.trade_contract_size * (price / lev + M["fee"]))
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
        need = c["floor_steps"] >= 20 if arm != "L30" else True
        check(f"{arm}: booked ledger equals the control's equity, the control's reward the booked step profit; forgiven amount "
              f"non-negative and never falling", worst_ledger < 1e-6 and worst_reward < 1e-6 and same_orders and bool(forgiven_ok) and need,
              f"{steps} steps, {c['floor_steps']} with the floor writing, largest gaps {worst_ledger:.2e} and {worst_reward:.2e} USD")


def episode_length() -> None:
    for arm in D.ARMS:
        env = D.DREnv(arm, "ORIG", train=True, seed=2)
        lengths = set()
        for _ in range(3):
            env.reset()
            n, done = 0, False
            while not done:
                _, _, te, tr, _ = env.step(np.array([0.0, 5.0, 0.0]))
                n, done = n + 1, te or tr
            lengths.add(n)
        check(f"{arm}: training episodes run from the second bar of the bundle to its last", lengths == {len(B["index"]) - 2},
              f"length {sorted(lengths)}")


if __name__ == "__main__":
    gc_package_checks()
    placebo()
    training_distribution()
    leverage_and_margin()
    margin_limit_and_probe()
    trajectories()
    episode_length()
    print("ALL CHECKS PASS" if not FAILS else f"{len(FAILS)} CHECK(S) FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)
