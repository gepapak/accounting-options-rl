"""Post hoc, exploratory, no trained agent used (16 Nov 2026; not registered; chosen after all results were known): the
forgiveness PPO's INITIAL policy meets. Result: too noisy to read (see RUN_LOG_DR.md, 16 November).

For each arm and starting equity e0 (x 10,000 USD), 300 training-distribution paths. On each path two rollouts in gym-mtsim as
published at the arm's leverage, with common random numbers:
  A  step 1 opens an order (volume and side from the initial policy's draw; refused if above the margin maximum),
  B  step 1 holds,
and every later action is the same N(0, I) draw (PPO's initial Gaussian policy, clipped to the box) in both.
The paired difference of the published episode return (A - B) is the initial policy's advantage of opening at a flat state: its
sign is the direction of PPO's first gradient on the hold decision there. Also: the same difference in booked terms, and the
share of A rollouts in which the floor wrote.
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np

STUDY = Path(__file__).resolve().parents[1]  # the study folder; this script sits in its posthoc_2026-11-16/
sys.path.insert(0, str(STUDY))
HERE = Path(__file__).resolve().parent
PSEUDO = {"N100": ("V2", 1.0), "V2": ("V2", 2.0), "V4": ("V4", 4.0), "V8": ("V8", 8.0), "V16": ("V16", 16.0), "L30": ("L30", 1.0),
          "L500": ("L500", 1.0)}
LEARNERS = {"N100": "0/5 (GC forex)", "V2": "3/6", "V4": "4/10", "V8": "8/10", "V16": "8/10", "L30": "0/6", "L500": "2/10"}
E0S = (0.05, 0.1, 0.25, 0.5, 1.0)
N = 300


def job(args):
    name, e0 = args
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"
    import warnings
    warnings.filterwarnings("ignore")
    import envs_dr as D
    env_arm, k_hi = PSEUDO[name]
    env = D.DREnv(env_arm, "ORIG", train=False, seed=0)
    b0 = e0 * D.BALANCE
    d_pub, d_book, flo_a, acc = [], [], [], []
    for i in range(N):
        rng = np.random.default_rng([20261116, list(PSEUDO).index(name), int(e0 * 1000), i])
        k = float(np.exp(rng.uniform(0.0, np.log(k_hi)))) if k_hi > 1.0 else 1.0
        path = D.scaled_placebo(rng, k)
        draws = np.clip(rng.standard_normal((len(path) + 1, 3)), -100.0, 100.0)
        out = []
        for first in ("open", "hold"):
            o, _ = env.reset(options={"path": path, "balance": b0})
            t, done = 0, False
            while not done:
                a = draws[t].copy()
                if t == 0:
                    a[1] = -20.0 if first == "open" else 20.0
                o, _, te, tr, inf = env.step(a)
                if t == 0 and first == "open":
                    acc.append(abs(float(o[3])) > 0)
                t += 1
                done = te or tr
            st = inf["stats"]
            out.append((st["reported"], st["booked"], st["floored"]))
        d_pub.append(out[0][0] - out[1][0])
        d_book.append(out[0][1] - out[1][1])
        flo_a.append(out[0][2])
    d_pub, d_book = np.array(d_pub), np.array(d_book)
    return dict(arm=name, e0=e0, adv_pub_usd=float(d_pub.mean()), se_pub_usd=float(d_pub.std(ddof=1) / np.sqrt(N)),
                adv_pub_pct=float(100 * d_pub.mean() / b0), adv_book_pct=float(100 * d_book.mean() / b0),
                accepted=float(np.mean(acc)), floored_open=float(np.mean(flo_a)))


def main():
    jobs = [(a, e) for a in PSEUDO for e in E0S]
    with ProcessPoolExecutor(6) as ex:
        res = list(ex.map(job, jobs))
    (HERE / "initial_gradient_dr.json").write_text(json.dumps(res, indent=1))
    print("initial-policy advantage of opening at a flat state, published (booked) in % of the starting balance; [z]; accepted share")
    print("arm   learners          " + "  ".join(f"e0={e:<5g}                 " for e in E0S))
    for a in PSEUDO:
        cells = []
        for e in E0S:
            r = next(x for x in res if x["arm"] == a and x["e0"] == e)
            z = r["adv_pub_usd"] / r["se_pub_usd"] if r["se_pub_usd"] > 0 else 0.0
            cells.append(f"{r['adv_pub_pct']:+6.2f} ({r['adv_book_pct']:+6.2f}) [{z:+5.1f}] {r['accepted']:.2f}")
        print(f"{a:<5} {LEARNERS[a]:<17} " + "  ".join(cells))


if __name__ == "__main__":
    main()
