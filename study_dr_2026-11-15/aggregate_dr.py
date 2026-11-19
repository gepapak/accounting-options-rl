"""Study DR aggregation: the registered decisions DR1 to DR4 (PREREGISTRATION_DR.md), written and hashed before the full run.

    <venv>\\Scripts\\python aggregate_dr.py [--smoke]
Outputs: results_dr/runs_dr.csv, results_dr/VERDICTS_DR.md
Runs are the units. Bootstrap as in Studies G and GC: 95% percentile interval of a difference of means, 10,000 resamples, groups
resampled independently. Jonckheere-Terpstra: exact statistic, p-value from 100,000 random permutations of the arm labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results_dr"
EXPECTED = {("V2", "ORIG"): 6, ("V4", "ORIG"): 10, ("V8", "ORIG"): 10, ("V16", "ORIG"): 10, ("L30", "ORIG"): 6, ("L500", "ORIG"): 10,
            ("V16", "FLPATCH"): 6, ("L500", "FLPATCH"): 6}
LOW = ("0.02", "0.05", "0.1", "0.25")
LEARNER = 0.10                       # a floor-trained agent with low-equity exposure of at least 0.10 is counted as a learner
ARM_ORDER = ["V2", "V4", "V8", "V16", "L30", "L500"]


def boot_diff(a, b, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng([20261115, seed])
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a[rng.integers(0, len(a), (10_000, len(a)))].mean(axis=1) - b[rng.integers(0, len(b), (10_000, len(b)))].mean(axis=1)
    return float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def jt_stat(groups: list[np.ndarray]) -> float:
    s = 0.0
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            a, b = groups[i][:, None], groups[j][None, :]
            s += float((a < b).sum()) + 0.5 * float((a == b).sum())
    return s


def jt_test(groups: list[np.ndarray], n_perm: int = 100_000, seed: int = 4) -> tuple[float, float]:
    """One-sided Jonckheere-Terpstra test for values increasing in the order of `groups`; permutation p-value."""
    rng = np.random.default_rng([20261115, seed])
    sizes = np.cumsum([len(g) for g in groups])[:-1]
    pooled = np.concatenate(groups).astype(float)
    obs = jt_stat([np.asarray(g, float) for g in groups])
    hits = 0
    for _ in range(n_perm):
        hits += jt_stat(np.split(rng.permutation(pooled), sizes)) >= obs - 1e-9
    return obs, (hits + 1) / (n_perm + 1)


def verdict(ok: bool) -> str:
    return "SUPPORTED" if ok else "NOT SUPPORTED"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--perms", type=int, default=100_000)
    a = ap.parse_args()
    rows = [json.loads(p.read_text()) for p in sorted((RES / ("jobs_smoke" if a.smoke else "jobs")).glob("*.json"))]
    R = pd.DataFrame(rows)
    for k in rows[0]["exposure"]:
        R[f"x{k}"] = R.exposure.apply(lambda d: d[k])
    R["xlow"] = R[[f"x{k}" for k in LOW]].mean(axis=1)
    R["learner"] = R.xlow >= LEARNER
    for bank in ("test_nominal", "test_stressed_k4"):
        tag = "nom" if bank == "test_nominal" else "k4"
        for k in ("reported_return_pct", "booked_return_pct", "gap", "floored_share", "wiped_share", "orders_per_episode"):
            R[f"{tag}_{k}"] = R[bank].apply(lambda d: d[k])
    for k in ("floor_steps", "episodes", "floored_episodes", "wiped_episodes"):
        R[f"train_{k}"] = R.train.apply(lambda d: d[k])
    R["train_floored_share"] = R.train_floored_episodes / R.train_episodes
    before = json.loads((HERE / "before_dr.json").read_text())

    def cell(arm, cond):
        return R[(R.arm == arm) & (R.condition == cond)].sort_values("seed")

    out = ["runs: " + ", ".join(f"{arm} {c} {len(cell(arm, c))} (expected {n})" for (arm, c), n in EXPECTED.items())]
    if not all(len(cell(arm, c)) == n for (arm, c), n in EXPECTED.items()) and not a.smoke:
        out.append("INCOMPLETE: the decisions below are provisional until every expected run is present")
    f_vol, f_lev = cell("V16", "FLPATCH"), cell("L500", "FLPATCH")

    # DR1: volatility randomization switches the floor on (V8 and V16, against the volatility lever's controls)
    parts, ok1 = [], True
    for i, arm in enumerate(("V8", "V16")):
        o = cell(arm, "ORIG")
        d, (lo, hi) = o.xlow.mean() - f_vol.xlow.mean(), boot_diff(o.xlow, f_vol.xlow, seed=10 + i)
        ok1 &= lo > 0
        parts.append(f"{arm}: {o.xlow.mean():.3f} - {f_vol.xlow.mean():.3f} = {d:.3f} [{lo:.3f}, {hi:.3f}], learners {int(o.learner.sum())}/{len(o)}")
    out.append("DR1 (volatility randomization switches the floor on) low-equity exposure, floor-trained minus V16 floor-deleted controls; "
               + "; ".join(parts) + f" (need both lower bounds > 0) -> {verdict(ok1)}")

    # DR2: fixed leverage 500 switches it on at nominal volatility (against the leverage lever's controls)
    o = cell("L500", "ORIG")
    d, (lo, hi) = o.xlow.mean() - f_lev.xlow.mean(), boot_diff(o.xlow, f_lev.xlow, seed=20)
    out.append(f"DR2 (leverage 500 switches the floor on at nominal volatility) L500: {o.xlow.mean():.3f} - {f_lev.xlow.mean():.3f} = {d:.3f} "
               f"[{lo:.3f}, {hi:.3f}], learners {int(o.learner.sum())}/{len(o)} (need lower bound > 0) -> {verdict(lo > 0)}")

    # DR3: dormant where the published edge is not positive
    parts, ok3 = [], True
    for arm in ("V2", "L30"):
        o = cell(arm, "ORIG")
        n_l = int(o.learner.sum())
        ok3 &= n_l <= 1 and len(o) > 0
        parts.append(f"{arm}: learners {n_l}/{len(o)}, mean low-equity exposure {o.xlow.mean():.3f}")
    out.append("DR3 (dormant where the published edge is not positive) " + "; ".join(parts) + f" (need at most 1 in each) -> {verdict(ok3)}")

    # DR4: dose-response across the volatility arms, ordered by the published edge
    groups = [cell(arm, "ORIG").xlow.to_numpy() for arm in ("V2", "V4", "V8", "V16")]
    if all(len(g) for g in groups):
        jt, p4 = jt_test(groups, n_perm=a.perms)
        n_pairs = sum(len(groups[i]) * len(groups[j]) for i in range(4) for j in range(i + 1, 4))
        means = ", ".join(f"{arm} {g.mean():.3f}" for arm, g in zip(("V2", "V4", "V8", "V16"), groups))
        out.append(f"DR4 (dose-response) Jonckheere-Terpstra over V2 < V4 < V8 < V16 on low-equity exposure: J = {jt:.1f} of {n_pairs} pairs, "
                   f"one-sided permutation p = {p4:.5f} (need < 0.05); means {means} -> {verdict(p4 < 0.05)}")
    else:
        out.append("DR4 (dose-response) not computable: an arm has no runs")

    # descriptive
    desc = []
    for arm in ARM_ORDER:
        for cond in ("ORIG", "FLPATCH"):
            c = cell(arm, cond)
            if not len(c):
                continue
            e = before["edge"][arm]
            ri = before["reach"][f"{arm}/initial"]
            desc.append(f"{arm} {cond}: n {len(c)}, learners {int(c.learner.sum())}, low-equity exposure {c.xlow.mean():.3f} "
                        f"(max {c.xlow.max():.3f}), x(1) {c['x1'].mean():.3f}; training episodes reaching the floor {c.train_floored_share.mean():.3f} "
                        f"(before training, initial policy {ri['floored_share']:.3f}); published edge C {e['published_pct']:+.2f}%; "
                        f"nominal test reported {c.nom_reported_return_pct.mean():+.2f} booked {c.nom_booked_return_pct.mean():+.2f}; "
                        f"stressed test reported {c.k4_reported_return_pct.mean():+.2f} booked {c.k4_booked_return_pct.mean():+.2f} "
                        f"wiped {c.k4_wiped_share.mean():.3f}")
    ctrl = pd.concat([f_vol, f_lev])
    desc.append(f"floor-deleted controls with low-equity exposure >= {LEARNER}: {int(ctrl.learner.sum())} of {len(ctrl)}")

    tag = "_SMOKE" if a.smoke else ""
    R.drop(columns=["exposure", "test_nominal", "test_stressed_k4", "train", "versions"]).to_csv(RES / f"runs_dr{tag}.csv", index=False)
    pd.set_option("display.width", 250)
    cols = [f"x{k}" for k in rows[0]["exposure"]] + ["xlow", "learner", "train_floored_share", "nom_reported_return_pct",
                                                   "nom_booked_return_pct", "k4_reported_return_pct", "k4_booked_return_pct", "k4_wiped_share"]
    table = R.groupby(["arm", "condition"])[cols].mean().round(3).to_string()
    print(table)
    print("\n".join(out))
    print("\n".join(desc))
    (RES / f"VERDICTS_DR{tag}.md").write_text("# Study DR verdicts" + (" (SMOKE)" if a.smoke else "") + "\n\n"
                                             + "\n".join(f"- {x}" for x in out) + "\n\n## Descriptive\n\n" + "\n".join(f"- {x}" for x in desc)
                                             + "\n\n```\n" + table + "\n```\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
