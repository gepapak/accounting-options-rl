"""Post hoc, read-only look at Study DR's 64 saved policies (16 Nov 2026; not registered; written after all results were known).

1. Fine probe: from a flat account at equity e x 10,000 USD (e on a finer grid than EGRID), what the deterministic policy does:
   hold, open an order of volume v (lots), or request an order the margin check refuses. Checks that the stored EGRID exposure
   is reproduced exactly.
2. Low-equity episodes: 100 paths from the arm's own training distribution (k ~ log-uniform[1, k_hi]; k = 1 in L arms), started
   at 0.1 and 0.25 x 10,000 USD, in gym-mtsim as published at the arm's leverage. Reported (equity) and booked (order records)
   returns, the credit (reported minus booked) in percent of the starting balance, the share of episodes the floor wrote in.
Writes probe_actions_dr.json next to itself and changes nothing else (run with -B, so importing envs_dr writes no bytecode).

    <venv>\\Scripts\\python -B probe_actions_dr.py > probe_actions_dr_out.txt     (about 5 minutes with 6 workers)
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
E_FINE = [0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0]
E_START = (0.1, 0.25)
N_PATHS = 100
ARM_SEED = {"V2": 1, "V4": 2, "V8": 3, "V16": 4, "L30": 5, "L500": 6}


def paths_for(arm: str):
    import envs_dr as D
    rng = np.random.default_rng(20261116 + ARM_SEED[arm])
    k_hi, out, ks = D.ARMS[arm]["k_hi"], [], []
    for _ in range(N_PATHS):
        k = float(np.exp(rng.uniform(0.0, np.log(k_hi)))) if k_hi > 1.0 else 1.0
        out.append(D.scaled_placebo(rng, k))
        ks.append(k)
    return np.stack(out), np.array(ks)


def job(args):
    arm, cond, seed = args
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"
    import warnings
    warnings.filterwarnings("ignore")
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    import envs_dr as D
    G = D.G
    stem = f"{arm}__{cond}__PPO__s{seed}"
    pol = STUDY / "results_dr" / "policies"
    model = PPO.load(str(pol / (stem + ".zip")), device="cpu")
    venv = VecNormalize.load(str(pol / (stem + "__vecnormalize.pkl")), DummyVecEnv([lambda: D.DREnv(arm, cond, train=False, seed=0)]))
    venv.training, venv.norm_reward = False, False

    def predict(x):
        return model.predict(venv.normalize_obs(x), deterministic=True)[0][0]

    stored = json.loads((STUDY / "results_dr" / "jobs" / (stem + ".json")).read_text())
    check = max(abs(D.exposure(predict, arm, e) - stored["exposure"][f"{e:g}"]) for e in D.EGRID)
    info, p0, lev = G.bundle(D.MARKET)["info"], G.bundle(D.MARKET)["p0"], D.ARMS[arm]["leverage"]
    probe = []
    for e in E_FINE:
        a = np.asarray(predict(np.array([[e, e, 0.0, 0.0, 0.0]], dtype=np.float32)), dtype=float).ravel()
        hold = bool(1.0 / (1.0 + np.exp(-a[1])) > 0.5)
        v = float(np.clip(abs(a[2]), info.volume_min, info.volume_max))
        v = round(v / info.volume_step) * info.volume_step
        feas = D.vmax(e * D.BALANCE, p0, lev)
        refused = (not hold) and (feas < info.volume_min or v > feas + 1e-9)
        x = 0.0 if (hold or refused) else v / feas
        probe.append(dict(e=e, hold=hold, refused=bool(refused), volume=v, feasible=feas, x=x))
    env = D.DREnv(arm, "ORIG", train=False, seed=0)
    bank, ks = paths_for(arm)
    low = {}
    for e0 in E_START:
        rows = []
        for close in bank:
            o, _ = env.reset(options={"path": close, "balance": e0 * D.BALANCE})
            done, open_steps = False, 0
            while not done:
                act = model.predict(venv.normalize_obs(o[None, :]), deterministic=True)[0][0]
                o, _, te, tr, inf = env.step(act)
                open_steps += int(abs(float(o[3])) > 0)
                done = te or tr
            st = inf["stats"]
            rows.append((st["reported"], st["booked"], st["floored"], st["wiped"], st["orders"], open_steps))
        r, b0 = np.array(rows, dtype=float), e0 * D.BALANCE
        low[f"{e0:g}"] = dict(reported_pct=float(100 * r[:, 0].mean() / b0), booked_pct=float(100 * r[:, 1].mean() / b0),
                              credit_pct=float(100 * (r[:, 0] - r[:, 1]).mean() / b0), floored_share=float(r[:, 2].mean()),
                              wiped_share=float(r[:, 3].mean()), orders=float(r[:, 4].mean()), open_steps=float(r[:, 5].mean()))
    return dict(arm=arm, condition=cond, seed=seed, xlow=stored["xlow"], check_max_abs_diff=float(check), probe=probe, low=low)


def main():
    import pandas as pd
    runs = pd.read_csv(STUDY / "results_dr" / "runs_dr.csv")
    jobs = [(r.arm, r.condition, int(r.seed)) for r in runs.itertuples()]
    with ProcessPoolExecutor(6) as ex:
        res = list(ex.map(job, jobs))
    (HERE / "probe_actions_dr.json").write_text(json.dumps(res, indent=1))
    print("largest difference from the stored EGRID exposure:", max(r["check_max_abs_diff"] for r in res))
    print("probe cells: '-' holds, 'R' requests an order the margin check refuses, else the volume opened (lots)")
    print("feasible maximum at leverage 100 (lots):", " ".join(f"{e:g}:{p['feasible']:.2f}" for e, p in
                                                          zip(E_FINE, next(r for r in res if r['arm'] == 'V2')['probe'])))
    hdr = "arm  cond     seed  xlow  | " + " ".join(f"{e:>5g}" for e in E_FINE) + " | e0=0.1: rep  book credit floored open | e0=0.25: credit floored"
    print(hdr)
    for r in sorted(res, key=lambda z: (z["arm"], z["condition"], -z["xlow"])):
        cells = []
        for p in r["probe"]:
            cells.append("    -" if p["hold"] else ("    R" if p["refused"] else f"{p['volume']:5.2f}"))
        a, b = r["low"]["0.1"], r["low"]["0.25"]
        print(f"{r['arm']:<4} {r['condition']:<7} {r['seed']:>4} {r['xlow']:5.3f} | " + " ".join(cells)
              + f" | {a['reported_pct']:+7.2f} {a['booked_pct']:+7.2f} {a['credit_pct']:+6.2f} {a['floored_share']:5.2f} {a['open_steps']:5.1f}"
              + f" | {b['credit_pct']:+6.2f} {b['floored_share']:5.2f}")
    print()
    print("per arm (e0 = 0.1, own training distribution, published simulator): mean credit, mean booked, runs with credit > 1 pt")
    for (arm, cond) in sorted({(r["arm"], r["condition"]) for r in res}):
        g = [r for r in res if r["arm"] == arm and r["condition"] == cond]
        c = np.array([r["low"]["0.1"]["credit_pct"] for r in g])
        bk = np.array([r["low"]["0.1"]["booked_pct"] for r in g])
        lrn = [r for r in g if r["xlow"] >= 0.10]
        cl = np.array([r["low"]["0.1"]["credit_pct"] for r in lrn]) if lrn else np.array([np.nan])
        print(f"{arm:<4} {cond:<7} n {len(g):>2}: credit {c.mean():+6.2f} (learners {np.nanmean(cl):+6.2f}, n {len(lrn)}), "
              f"booked {bk.mean():+7.2f}, runs with credit > 1 pt: {int((c > 1).sum())}")


if __name__ == "__main__":
    main()
