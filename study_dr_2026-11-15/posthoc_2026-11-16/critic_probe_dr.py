"""Exploratory, read-only: does a trained PPO agent's own critic reveal the accounting it was trained on?
For every saved Study DR policy, evaluate the critic V(o) at a flat account (no order) with equity e x 10,000 USD on a grid, through
the run's frozen observation normalization. Under a zero floor, low-equity states carry the forgiveness option, so the critic should
value poorer states more; under full liability (FLPATCH) the critic should be flat. Values are in the run's normalized-reward units,
so the comparison is within a run: S = V(0.1) - V(1) (positive = poorer states valued more)."""
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
STUDY = Path(__file__).resolve().parents[1]  # the study folder; this script sits in its posthoc_2026-11-16/
sys.path.insert(0, str(STUDY))
GRID = [0.02, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0, 1.5, 2.0]


def job(stem):
    import os
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    import pickle
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    pol = STUDY / "results_dr" / "policies"
    model = PPO.load(str(pol / (stem + ".zip")), device="cpu")
    with open(pol / (stem + "__vecnormalize.pkl"), "rb") as f:
        vn = pickle.load(f)
    obs = np.array([[e, e, 0.0, 0.0, 0.0] for e in GRID], dtype=np.float32)
    nobs = np.clip((obs - vn.obs_rms.mean) / np.sqrt(vn.obs_rms.var + vn.epsilon), -vn.clip_obs, vn.clip_obs).astype(np.float32)
    with torch.no_grad():
        v = model.policy.predict_values(torch.as_tensor(nobs)).numpy().ravel()
    return stem, [float(x) for x in v]


def main():
    import pandas as pd
    runs = pd.read_csv(STUDY / "results_dr" / "runs_dr.csv")
    stems = [f"{r.arm}__{r.condition}__PPO__s{int(r.seed)}" for r in runs.itertuples()]
    with ProcessPoolExecutor(6) as ex:
        res = dict(ex.map(job, stems))
    probe = {}
    p = Path(__file__).resolve().parent / "probe_actions_dr.json"
    if p.exists():
        for r in json.loads(p.read_text()):
            probe[f"{r['arm']}__{r['condition']}__PPO__s{r['seed']}"] = max(r["low"]["0.1"]["credit_pct"], r["low"]["0.25"]["credit_pct"])
    rows = []
    for r in runs.itertuples():
        stem = f"{r.arm}__{r.condition}__PPO__s{int(r.seed)}"
        v = res[stem]
        S = v[GRID.index(0.1)] - v[GRID.index(1.0)]
        rows.append(dict(arm=r.arm, cond=r.condition, seed=int(r.seed), learner=bool(r.xlow >= 0.10), xlow=r.xlow, credit=probe.get(stem, np.nan),
                         S=S, v=v))
    df = pd.DataFrame(rows)
    (Path(__file__).resolve().parent / "critic_probe_dr.json").write_text(df.to_json(orient="records", indent=1))
    print("critic V at a flat account, equity grid " + " ".join(f"{e:g}" for e in GRID) + "; S = V(0.1) - V(1)")
    for (arm, cond), g in df.groupby(["arm", "cond"], sort=False):
        for r in g.sort_values("S", ascending=False).itertuples():
            print(f"{arm:<4} {cond:<7} s{r.seed} {'L' if r.learner else '.'} credit {r.credit:6.2f}  S {r.S:+7.3f} | " + " ".join(f"{x:+6.2f}" for x in r.v))
    print()
    print("summary of S by arm and condition (mean, min, max; learners vs non-learners):")
    for (arm, cond), g in df.groupby(["arm", "cond"], sort=False):
        L, N = g[g.learner].S, g[~g.learner].S
        print(f"  {arm:<4} {cond:<7} n {len(g):>2}: S mean {g.S.mean():+.3f} [{g.S.min():+.3f}, {g.S.max():+.3f}]; learners {L.mean() if len(L) else float('nan'):+.3f} (n {len(L)}), "
              f"non-learners {N.mean() if len(N) else float('nan'):+.3f} (n {len(N)})")
    orig = df[df.cond == "ORIG"]
    ctrl = df[df.cond == "FLPATCH"]
    try:
        from scipy.stats import mannwhitneyu, spearmanr
        print("V16: ORIG vs FLPATCH, S, one-sided MW p =", round(mannwhitneyu(df[(df.arm == "V16") & (df.cond == "ORIG")].S, df[(df.arm == "V16") & (df.cond == "FLPATCH")].S, alternative="greater").pvalue, 4))
        print("L500: ORIG vs FLPATCH, S, one-sided MW p =", round(mannwhitneyu(df[(df.arm == "L500") & (df.cond == "ORIG")].S, df[(df.arm == "L500") & (df.cond == "FLPATCH")].S, alternative="greater").pvalue, 4))
        ok = orig.dropna(subset=["credit"])
        print("all ORIG runs: Spearman(S, collected credit) =", round(spearmanr(ok.S, ok.credit).correlation, 3), "n", len(ok))
        # AUC of S for floor-trained (all ORIG with reachable floor, i.e. not L30) vs all FLPATCH controls
        a = orig[orig.arm != "L30"].S.values
        b = ctrl.S.values
        auc = np.mean([(x > y) + 0.5 * (x == y) for x in a for y in b])
        print(f"AUC, S of floor-trained runs (reachable floor, n {len(a)}) vs floor-deleted controls (n {len(b)}): {auc:.3f}")
        al = orig[(orig.arm != "L30") & (orig.xlow >= 0.10)].S.values
        an = orig[(orig.arm != "L30") & (orig.xlow < 0.10)].S.values
        print(f"AUC, learners (n {len(al)}) vs controls: {np.mean([(x > y) + 0.5 * (x == y) for x in al for y in b]):.3f}; "
              f"non-learners (n {len(an)}) vs controls: {np.mean([(x > y) + 0.5 * (x == y) for x in an for y in b]):.3f}")
    except Exception as ex:
        print("stats failed:", ex)


if __name__ == "__main__":
    main()
