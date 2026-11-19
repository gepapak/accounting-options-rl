"""EXPLORATORY: independent counts, endpoint checks and seed sensitivity.

Reads frozen studies without importing/running their aggregators. Writes only in
this new folder. No training, DP, network access, or Study Q access. The seed
bootstrap retains both fixed zones and every arm of each selected seed. It is
a sensitivity analysis, not a replacement for registered run-level inference.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import time

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LIAB = ROOT / "LIABILITY_STUDY_2026-09-29"
ACC = ROOT / "ACCOUNTING_OPTIONS_2026-10-14"
INPUTS: set[Path] = set()


def read_csv(path: Path) -> pd.DataFrame:
    INPUTS.add(path)
    return pd.read_csv(path)


def read_json(path: Path):
    INPUTS.add(path)
    return json.loads(path.read_text(encoding="utf-8"))


def fit(D: pd.DataFrame, joint=False):
    """Same unpenalized likelihood as registered fits, convex coefficients.

    z = b0 + b1*log10(index) + b2*log10(N/1M), beta=-b2/b1.
    Analytic gradient avoids slow per-bootstrap Nelder-Mead optimization.
    """
    cols = [np.ones(len(D)), np.log10(D["index"].to_numpy())]
    if joint:
        cols.append(np.log10(D.budget.to_numpy() / 1e6))
    X = np.column_stack(cols)
    y = (D.L002.to_numpy() == 16).astype(float)

    def objective(b):
        z = X @ b
        return float(np.sum(np.logaddexp(0, z) - y * z)), X.T @ (special.expit(z) - y)

    res = optimize.minimize(objective, np.zeros(X.shape[1]), jac=True,
                            method="BFGS", options={"gtol": 1e-7, "maxiter": 300})
    grad = float(np.max(np.abs(objective(res.x)[1])))
    if grad > 1e-4 or not np.isfinite(res.x).all() or res.x[1] <= 0:
        return {"valid": False, "gradient_max": grad}
    b = res.x
    out = {"valid": True, "slope": float(b[1]), "c50_pct": float(10 ** (-b[0] / b[1])),
           "nll": float(res.fun), "gradient_max": grad}
    if joint:
        out["beta"] = float(-b[2] / b[1])
    return out


def counts():
    result = {}
    for s, rel in {"B": "results/runs_b.csv", "E": "results_e/runs_e.csv",
                   "F": "results_f/runs_f.csv", "G": "mtsim_study/results_g/runs_g.csv",
                   "H": "results_h/runs_h.csv", "I": "results_i/runs_i.csv",
                   "J": "results_j/runs_j.csv", "K": "results_k/runs_k.csv"}.items():
        D = read_csv(LIAB / rel)
        result[s] = len(D)
    C = read_csv(LIAB / "scaling/results/snapshots_c.csv")
    result["C"] = len(C.drop_duplicates(["liability", "algo", "seed", "region"]))
    result["L"] = len(read_csv(ROOT / "revision_experiments/2026-10-10/continuation_experiment/agents.csv"))
    result["N"] = len(read_csv(ACC / "study_n/runs_n.csv"))
    result["O"] = len(read_csv(ACC / "study_o/runs_o.csv"))
    P = read_csv(ACC / "study_p/runs_p.csv")
    result["P"] = int((P.study == "P").sum())
    A = list((ROOT / "GENERALIZATION_STUDY_2026-09-29/results/jobs").glob("*.json"))
    # A jobs are separately trained agents; record filenames and content hashes.
    INPUTS.update(A)
    return {"unique_agents_by_study": result, "completed_table1_cohort_excluding_Q": sum(result.values()),
            "original_B_to_K": sum(result[s] for s in "BCEFGHIJK"), "Study_A_separate": len(A),
            "C_snapshot_rows": len(C), "M_new_agents": 0, "P_csv_rows_including_reused_O": len(P),
            "Q_not_read_or_counted": True}


def supports():
    out = {}
    for s, rel in {"B": "results", "C": "scaling/results", "E": "results_e", "F": "results_f",
                   "G": "mtsim_study/results_g", "H": "results_h", "I": "results_i",
                   "J": "results_j", "K": "results_k"}.items():
        p = LIAB / rel / f"VERDICTS_{s}.md"
        INPUTS.add(p)
        # The first arrow on B2 is its registered primary; the second is secondary.
        vals = [re.search(r"-> (NOT SUPPORTED|SUPPORTED)", line).group(1)
                for line in p.read_text().splitlines() if re.match(rf"- {s}[1-9]", line)]
        out[s] = {"supported": vals.count("SUPPORTED"), "total": len(vals)}
    for s in "NOP":
        p = ACC / f"study_{s.lower()}" / f"VERDICTS_{s}.md"
        INPUTS.add(p)
        vals = [re.search(r"\| (NOT SUPPORTED|SUPPORTED) \|$", line).group(1)
                for line in p.read_text().splitlines()
                if re.search(r"\| (NOT SUPPORTED|SUPPORTED) \|$", line)]
        out[s] = {"supported": vals.count("SUPPORTED"), "total": len(vals)}
    return out


def cluster_gap(D, treatment, control, arm, value):
    """Exact empirical paired seed bootstrap (5^5=3125 possible draws)."""
    piv = D.groupby(["seed", arm])[value].mean().unstack()
    diffs = (piv[treatment] - piv[control]).to_numpy()
    boot = np.array([diffs[list(i)].mean() for i in itertools.product(range(len(diffs)), repeat=len(diffs))])
    return {"mean": float(diffs.mean()), "seed_differences": diffs.tolist(),
            "seed_clusters": len(diffs), "exact_empirical_seed_bootstrap_95_pct": np.quantile(boot, [.025, .975]).tolist(),
            "one_sided_95_lower": float(np.quantile(boot, .05))}


def o_arm_label_permutation(O):
    """Exact 8! common arm-label permutation, retains seeds and zones.

    Stronger dependence sensitivity: one alpha relabeling is applied jointly to
    every seed and zone, treating alpha conditions as the eight exchangeable
    labels. This reference distribution assumes arm-label exchangeability under
    a global null. It does not create eight independent market replications.
    """
    arms, zones = sorted(O.alpha.unique()), sorted(O.region.unique())
    rx = stats.rankdata(O["index"])
    ry = stats.rankdata(O.L002)
    # Sum centered outcome ranks per (alpha,zone); index identical across seeds.
    tab = np.array([[np.sum((ry - ry.mean())[(O.alpha == a) & (O.region == z)])
                     for z in zones] for a in arms])
    rank_x = np.array([[rx[(O.alpha == a) & (O.region == z)][0] - rx.mean()
                        for z in zones] for a in arms])
    observed = float(np.sum(tab * rank_x))
    vals = np.array([np.sum(tab * rank_x[list(p)]) for p in itertools.permutations(range(len(arms)))])
    return {"observed_spearman": float(stats.spearmanr(O["index"], O.L002).statistic),
            "exact_common_arm_label_permutations": len(vals),
            "one_sided_p": float(np.mean(vals >= observed - 1e-10)),
            "caveat": "Exploratory arm-label exchangeability reference, conditional on two fixed zones."}


def seed_fit_bootstrap(P, n, rng):
    treatment = P[P.alpha != 1]
    clusters = {s: [g for _, g in d.groupby("seed")] for s, d in treatment.groupby("study")}
    fits = []
    for _ in range(n):
        parts = [groups[i] for groups in clusters.values()
                 for i in rng.integers(0, len(groups), len(groups))]
        f = fit(pd.concat(parts, ignore_index=True), joint=True)
        if f["valid"]:
            fits.append(f)
    betas = [f["beta"] for f in fits]
    return {"requested": n, "valid": len(fits), "cluster_unit": "seed within study; both zones/all arms/budgets kept together",
            "beta_percentile_95": np.quantile(betas, [.025, .975]).tolist() if betas else None,
            "caveat": "Exploratory sensitivity with only five seeds per study, not registered inference."}


def recompute_baselines(O):
    """Rebuild the exploratory baseline comparisons without calling their writer."""
    AG = read_csv(LIAB / "results_k/action_gap.csv")
    F, K = read_csv(LIAB / "results_f/runs_f.csv"), read_csv(LIAB / "results_k/runs_k.csv")
    H, J = read_csv(LIAB / "results_h/runs_h.csv"), read_csv(LIAB / "results_j/runs_j.csv")
    sets = []
    for z in ("DK1", "DK2"):
        sets.append((z, "gamma 0.99", pd.concat([F[(F.region == z) & (F.obsset == "EQ") & (F.rule == "ZF")],
                                                K[(K.region == z) & (K.gamma == .99) & (K.rule == "ZF")]])))
        for g in (.95, .98, .995):
            sets.append((z, f"gamma {g}", K[(K.region == z) & (K.gamma == g) & (K.rule == "ZF")]))
        sets.append((z, "Study H costs", H[(H.region == z) & (H.obsset == "EQ") & (H.rule == "ZF")]))
    for z in ("FI", "NO2"):
        sets.append((z, "gamma 0.99", J[(J.region == z) & (J.obsset == "EQ") & (J.rule == "ZF")]))
    probes = np.array([.02, .1, .25, .5, 1, 1.5, 2])
    starts = np.exp(np.linspace(np.log(.05), np.log(2), 400))
    rows = []
    for z, condition, d in sets:
        ag = AG[(AG.zone == z) & (AG.condition == condition)].iloc[0]
        A = np.array([ag[f"A_{e:g}"] for e in probes])
        index = 100 * np.maximum(np.interp(np.log(starts), np.log(probes), A), 0).mean()/ag.sigma16
        rows.extend({"learned": int(v == 16), "index": index, "region": ag.threshold_K,
                     "local_002": A[0]/ag.sigma16, "local_01": A[1]/ag.sigma16} for v in d["L0.02"])
    C = pd.DataFrame(rows)
    dp = {z: read_json(ACC / f"study_o/dp_o_{z}.json") for z in ("DK1", "DK2")}
    rows = []
    for r in O[O.alpha != 1].itertuples():
        arm, noise = dp[r.region]["arms"][f"{r.alpha:g}"], dp[r.region]["sigma16"]
        rows.append({"learned": int(r.L002 == 16), "index": arm["index_pct"], "region": arm["region16_K"],
                     "local_002": arm["A"][0]/noise, "local_01": arm["A"][1]/noise})
    T = pd.DataFrame(rows)

    def auc(d, name):
        pos, neg = d[d.learned == 1][name], d[d.learned == 0][name]
        return float(stats.mannwhitneyu(pos, neg).statistic/(len(pos)*len(neg)))

    out = []
    for name in ("index", "region", "local_002", "local_01"):
        unique = np.unique(C[name])
        candidates = np.r_[unique[0]-1, (unique[:-1]+unique[1:])/2, unique[-1]+1]
        threshold = candidates[np.argmax([np.mean((C[name] >= t) == C.learned) for t in candidates])]
        out.append({"predictor": name, "auc_calibration": auc(C, name), "auc_study_O": auc(T, name),
            "auc_pooled": auc(pd.concat([C, T]), name), "threshold_from_calibration": float(threshold),
            "accuracy_on_study_O": float(np.mean((T[name] >= threshold) == T.learned))})
    return {"calibration_runs": len(C), "calibration_conditions": len(sets), "prospective_O_treatment_runs": len(T),
            "recomputed": out}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-fit-bootstrap", type=int, default=199)
    args = parser.parse_args()
    start = time.perf_counter()
    O = read_csv(ACC / "study_o/runs_o.csv")
    P = read_csv(ACC / "study_p/runs_p.csv")
    N = read_csv(ACC / "study_n/runs_n.csv")
    cal = read_json(ACC / "study_o/calibration_o.json")
    baseline = read_csv(ACC / "study_o/baselines_o.csv")
    report = {"status": "EXPLORATORY independent audit; frozen registered records remain authoritative",
              "counts": counts(), "registered_support_counts_read_from_verdicts": supports(),
              "threshold_provenance": {"c_star_pct": cal["c_star_pct"],
                  "conditions": len(cal["rows"]), "initial_equity_proxy_not_occupancy": True},
              "O_logistic_single_fit": fit(O[O.alpha != 1]),
              "OP_joint_logistic_single_fit": fit(P[P.alpha != 1], joint=True),
              "baselines_readonly": baseline.to_dict(orient="records"),
              "baselines_independent_recompute": recompute_baselines(O),
              "O_arm_label_permutation": o_arm_label_permutation(O),
              "O4_paired_seed_sensitivity": cluster_gap(O, 0, 1, "alpha", "L002"),
              "N1_paired_seed_sensitivity": cluster_gap(N, "ZFG", "ZBK", "arm", "zbar"),
              "T1_paired_seed_sensitivity": cluster_gap(N, "TFG", "TBK", "arm", "tbar")}
    treatment = O[O.alpha != 1]
    learned = (treatment.L002 == 16).to_numpy()
    dose = (1-treatment.alpha).to_numpy()
    report["O_treatment_only_dose_comparator"] = {
        "n": len(treatment), "positive": int(learned.sum()), "negative": int((~learned).sum()),
        "auc_one_minus_alpha": float(stats.mannwhitneyu(dose[learned],dose[~learned]).statistic/(learned.sum()*(~learned).sum())),
        "auc_index": float(stats.mannwhitneyu(treatment["index"][learned],treatment["index"][~learned]).statistic/(learned.sum()*(~learned).sum())),
        "caveat": "Exploratory ranking-only comparator. Calibration has alpha=0, so dose does not supply a transferable cutoff; shared dose family cannot establish that DP is needed."}
    for label, b1, b0, arms in (("P3", 4_000_000, 1_000_000, [.8, .9]),
                                ("P4", 1_000_000, 250_000, [.5, .7, .8])):
        rows = [P[(P.budget == b) & P.alpha.isin(arms)] for b in [b1, b0]]
        kn = [(int((d.L002 == 16).sum()), len(d)) for d in rows]
        p = float(stats.fisher_exact([[k, n-k] for k, n in kn], alternative="greater").pvalue)
        report[label + "_fisher_independent_recompute"] = {"successes_totals": kn, "one_sided_p": p}
    # At fixed alpha, arm contrasts are replicated over five seeds, not over
    # quarter-hours or the 20 evaluation sign draws.
    report["OP_seed_bootstrap_joint_fit"] = seed_fit_bootstrap(P, args.seed_fit_bootstrap,
                                                             np.random.default_rng(2026101501))
    report["P_beta_leave_one_seed_out"] = {
        study: [{"omitted_seed": int(s), **fit(P[(P.alpha != 1) & ~((P.study == study) & (P.seed == s))], joint=True)}
                for s in sorted(P[P.study == study].seed.unique())] for study in ["O", "P"]}
    report["seconds"] = time.perf_counter() - start
    report["input_sha256"] = {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(INPUTS)}
    target = HERE / "statistical_audit.json"
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("counts", "O_logistic_single_fit", "OP_joint_logistic_single_fit",
          "O_arm_label_permutation", "OP_seed_bootstrap_joint_fit", "seconds")}, indent=2))
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
