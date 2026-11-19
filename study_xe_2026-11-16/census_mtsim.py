"""Census, gym-mtsim's own simulator (exploratory): the pilot's trade on the pilot's paths in MtSimulator as published (zero-balance
floor, stop-out at a margin level of 0.2), and in the floor-deleted subclass used by Studies G, GC and DR. Buy the largest volume the
margin check accepts at leverage L on bar 1 (the package's fee per unit), hold to bar 86 unless stopped out. Reported: the simulator's
equity at bar 86. Booked: its order records. Uses ../study_gc_2026-11-14/envs_gc.py read-only. Writes census_mtsim.csv."""
import sys
import warnings

warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE.parent / "study_gc_2026-11-14"))
import envs_gc as G  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 500
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 20261116
OUT_CSV = sys.argv[3] if len(sys.argv) > 3 else "census_mtsim.csv"
LEVS = [1, 2, 5, 10, 20, 50, 100, 200]


def paths(n, seed=None, nbars=88):
    seed = SEED if seed is None else seed
    mags = np.load(HERE / "eurusd_mags.npy")
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        m = mags[rng.integers(0, len(mags), size=nbars - 1)]
        s = rng.choice([-1.0, 1.0], size=nbars - 1)
        out.append(1.20036 * np.cumprod(np.r_[1.0, 1.0 + s * m]))
    return out


def run(close, L, condition):
    from gym_mtsim import OrderType
    m, b = G.MARKETS["forex"], G.bundle("forex")
    sym, idx = m["symbol"], b["index"]
    sim = G.make_simulator("forex", condition)
    sim.leverage = float(L)
    frame = sim.symbols_data[sym].copy()
    col = frame.columns.get_loc("Close")
    for t in range(len(close)):
        frame.iloc[t, col] = close[t]
    sim.symbols_data[sym] = frame
    sim.current_time = idx[0]
    sim.tick(idx[1] - idx[0])                                         # to bar 1
    per_unit = b["info"].trade_contract_size * (close[1] / L + m["fee"])
    v = float(np.floor(G.BALANCE / per_unit / b["info"].volume_step + 1e-9) * b["info"].volume_step)
    v = min(v, b["info"].volume_max)
    if v >= b["info"].volume_min:
        sim.create_order(OrderType.Buy, sym, round(v, 6), m["fee"])
    zero = False
    for t in range(2, 87):                                            # through bar 86
        sim.tick(idx[t] - idx[t - 1])
        zero |= sim.equity <= 1e-9
    led = G.ledger(sim, G.BALANCE)
    return float(sim.equity) / G.BALANCE - 1.0, led / G.BALANCE - 1.0, bool(zero), v


def main():
    rows = []
    for i, close in enumerate(paths(N)):
        for L in LEVS:
            for cond in ("ORIG", "FLPATCH"):
                rep, book, zero, v = run(close, L, cond)
                rows.append(dict(L=L, path=i, cond=cond, rep=rep, book=book, zero=zero, volume=v))
    df = pd.DataFrame(rows)
    df.to_csv(HERE / OUT_CSV, index=False)
    for (L, cond), g in df.groupby(["L", "cond"]):
        print(f"L={L:>3} {cond:<7}: reported {100 * g.rep.mean():+8.2f}  booked {100 * g.book.mean():+8.2f}  "
              f"forgiven {100 * (g.rep - g.book).mean():+7.2f}  equity hit zero {g.zero.mean():6.1%}  volume {g.volume.mean():8.2f}")


if __name__ == "__main__":
    main()
