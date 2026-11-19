"""Study DR, POST HOC and exploratory: written after the full run and its registered verdicts; not registered, not hashed.

Why. The registered predictor, the published edge C of the LARGEST position (before_dr.py), ranked the arms wrongly: L500 (C +13.9%)
had 2/10 learners and V2 (C -1.8%) 3/6. One guess, made before this was computed (run log, 21:30): at leverage 500 the floor pays
only near the largest size, while with randomized volatility it still pays at smaller sizes, where gradient learning can find it.

What. For each arm, the mean one-bar published reward and booked profit (percent of the balance) when the order is a fraction f of the
largest accepted volume, f in FRACS, computed exactly as before_dr.one_bar (same placebo magnitudes, both signs, k averaged on the same
33-point log-k grid). f = 1 reproduces before_dr.json. No training. Writes posthoc_partial_size.json and prints the table.

    <venv>\\Scripts\\python posthoc_partial_size.py
"""
from __future__ import annotations

import csv
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
FRACS = (0.1, 0.25, 0.5, 0.75, 1.0)


def one_bar_frac(k: float, leverage: float, frac: float) -> dict:
    from gym_mtsim import OrderType
    m, b = G.MARKETS[D.MARKET], G.bundle(D.MARKET)
    sym, p0, idx, info = m["symbol"], b["p0"], b["index"], b["info"]
    vfull = min(D.vmax(D.BALANCE, p0, leverage), info.volume_max)
    if frac >= 1.0:
        v = vfull
    else:
        v = max(float(np.floor(frac * vfull / info.volume_step + 1e-9) * info.volume_step), info.volume_min)
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
    return dict(k=k, frac=frac, volume=v, published_pct=float(np.mean(rep)) / 100, booked_pct=float(np.mean(book)) / 100,
                floor_share=float(np.mean(floor)))


def _init_worker():
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"


def learners() -> dict:
    out = {}
    with open(HERE / "results_dr" / "runs_dr.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            key = (r["arm"], r["condition"])
            n, l = out.get(key, (0, 0))
            out[key] = (n + 1, l + int(float(r["xlow"]) >= 0.10))
    return out


def main() -> None:
    t0 = time.time()
    tasks = {}
    with ProcessPoolExecutor(max_workers=6, initializer=_init_worker) as ex:
        for arm, spec in D.ARMS.items():
            ks = np.exp(np.linspace(0.0, np.log(spec["k_hi"]), 33)) if spec["k_hi"] > 1.0 else np.array([1.0])
            for f in FRACS:
                for k in ks:
                    tasks[(arm, f, float(k))] = ex.submit(one_bar_frac, float(k), spec["leverage"], f)
        res = {key: fut.result() for key, fut in tasks.items()}
    table = {}
    for arm, spec in D.ARMS.items():
        table[arm] = {}
        for f in FRACS:
            rows = [res[key] for key in res if key[0] == arm and key[1] == f]
            rows.sort(key=lambda r: r["k"])
            if spec["k_hi"] > 1.0:
                lk = np.log([r["k"] for r in rows])
                avg = lambda name: float(np.trapz([r[name] for r in rows], lk) / np.log(spec["k_hi"]))  # noqa: E731
                cell = dict(published_pct=avg("published_pct"), booked_pct=avg("booked_pct"), floor_share=avg("floor_share"))
            else:
                cell = {name: rows[0][name] for name in ("published_pct", "booked_pct", "floor_share")}
            cell["volume_lots"] = rows[0]["volume"]
            table[arm][str(f)] = cell
    before = json.loads((HERE / "before_dr.json").read_text())["edge"]
    check = {arm: table[arm]["1.0"]["published_pct"] - before[arm]["published_pct"] for arm in D.ARMS}
    ln = learners()
    out = dict(note="post hoc, exploratory, not registered", fracs=FRACS, table=table, check_f1_minus_before_dr=check,
               learners={f"{a}/{c}": v for (a, c), v in ln.items()}, seconds=round(time.time() - t0, 1))
    (HERE / "posthoc_partial_size.json").write_text(json.dumps(out, indent=1))
    print("POST HOC (not registered). Mean one-bar published reward, percent of the balance, at a fraction f of the largest accepted order")
    print("arm    learners | " + " | ".join(f"f={f:<4g} pub (booked)" for f in FRACS))
    for arm in D.ARMS:
        n, l = ln.get((arm, "ORIG"), (0, 0))
        cells = " | ".join(f"{table[arm][str(f)]['published_pct']:+7.2f} ({table[arm][str(f)]['booked_pct']:+6.2f})" for f in FRACS)
        print(f"{arm:5s}  {l:2d}/{n:<2d}    | {cells}")
    print("f = 1 minus before_dr.json (must be 0): " + ", ".join(f"{a} {v:+.2e}" for a, v in check.items()))
    print(f"{out['seconds']} s")


if __name__ == "__main__":
    main()
