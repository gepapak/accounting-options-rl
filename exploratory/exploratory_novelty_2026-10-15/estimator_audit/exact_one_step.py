"""EXPLORATORY exact empirical one-step diagnostic; no DP, training or MC."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LIAB = ROOT / "LIABILITY_STUDY_2026-09-29"
ACC = ROOT / "ACCOUNTING_OPTIONS_2026-10-14"
OLD = ROOT / "exploratory_codex_2026-10-15/model_free"
PROBES = np.array([.02, .1, .25, .5, 1, 1.5, 2])
STARTS = np.exp(np.linspace(np.log(.05), np.log(2), 400))
ALPHAS = (0, .25, .5, .7, .8, .9, .95, 1)
INPUTS = set()


def csv(path):
    INPUTS.add(path)
    return pd.read_csv(path)


def jread(path):
    INPUTS.add(path)
    return json.loads(path.read_text(encoding="utf-8"))


def key(zone, costs, gamma, alpha):
    return f"{zone}|{costs}|g{gamma:g}|a{alpha:g}"


def empirical(zone, costs="base", fee=None):
    folder = ROOT / "GENERALIZATION_STUDY_2026-09-29/data_panel" if zone in ("DK1", "DK2") else LIAB / "data_panel_j"
    path = folder / f"panel_{zone}.pkl"
    INPUTS.add(path)
    panel = pd.read_pickle(path)
    panel = panel.loc[panel.split == "train"].iloc[200:]
    cap = .25*.25*panel.bal_mw.to_numpy(float)
    if costs == "H":
        cap *= .10/.25
    q = np.minimum(4, cap)
    mag = q*np.abs(panel.imb.to_numpy(float)-panel.spot.to_numpy(float))
    cost = (.10 if fee is None else fee)*q
    if costs == "H":
        cost = q + 50*q*q/np.maximum(cap, .05)
    sigma = float(np.sqrt(np.mean(mag*mag+cost*cost)-np.mean(cost)**2))
    losses = np.sort(np.r_[cost-mag, cost+mag])
    suffix = np.r_[np.cumsum(losses[::-1])[::-1], 0.]

    def h(e):
        e = np.asarray(e)
        i = np.searchsorted(losses, e, side="right")
        return (suffix[i]-(len(losses)-i)*e)/len(losses)

    direct = np.array([np.maximum(losses-e, 0).mean() for e in 20000*PROBES])
    error = float(np.max(np.abs(h(20000*PROBES)-direct)))
    if error > 1e-7:
        raise ValueError(f"Suffix-sum/direct integration mismatch: {error}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    registered = []
    if zone in ("FI", "NO2"):
        registry = LIAB / "PREREGISTRATION_HASHES.txt"
        INPUTS.add(registry)
        for line in registry.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[-1] == f"*data_panel_j/panel_{zone}.pkl":
                registered.append(parts[-2])
    noise_diagnosis = []
    for e in PROBES:
        injection = .5*(np.maximum(cost+mag-20000*e,0)+np.maximum(cost-mag-20000*e,0))
        row_reward = injection-cost
        count = max(1,int(np.ceil(.001*len(injection))))
        top = np.partition(injection,len(injection)-count)[-count:]
        mu = float(row_reward.mean())
        noise_diagnosis.append({"equity_allocations":float(e),"mean_reward_alpha0_eur":mu,
            "sd_of_antithetic_row_reward_eur":float(row_reward.std()),
            "MCSE_mean_with_old_6144_pairs_eur":float(row_reward.std()/np.sqrt(6144)),
            "fraction_rows_with_any_injection":float((injection>0).mean()),
            "top_0p1pct_row_share_expected_injection":float(top.sum()/injection.sum()) if injection.sum()>0 else None,
            "iid_rows_for_10pct_relative_MCSE":float(row_reward.var()/(.1*mu)**2) if mu != 0 else None})
    return h, float(cost.mean()), sigma, {"panel": str(path.relative_to(ROOT)), "panel_sha256": digest,
        "registered_panel_sha256": registered, "revised_J_panel": bool(registered and digest not in registered),
        "rows": len(panel), "suffix_vs_direct_max_error_eur": error,
        "alpha0_old_flat_row_sampling_noise":noise_diagnosis}


def functional(gaps, sigma):
    interpolated = np.interp(np.log(STARTS), np.log(PROBES), gaps)
    return float(100*np.maximum(interpolated, 0).mean()/sigma)


def main():
    start = time.perf_counter()
    definitions = {}
    raw = csv(LIAB / "results_k/action_gap.csv")
    for r in raw.itertuples():
        ident = key(r.zone, r.costs, r.gamma, 0)
        v = raw[(raw.zone == r.zone)&(raw.condition == r.condition)].iloc[0]
        definitions[ident] = {"condition_id": ident, "zone": r.zone, "costs": r.costs,
            "gamma": r.gamma, "alpha": 0., "legacy": True,
            "dp_index_pct": functional(np.array([v[f"A_{e:g}"] for e in PROBES]), r.sigma16)}
    for zone in ("DK1", "DK2"):
        dp = jread(ACC / f"study_o/dp_o_{zone}.json")
        for alpha, arm in dp["arms"].items():
            ident = key(zone, "base", .99, float(alpha))
            if ident not in definitions:
                definitions[ident] = {"condition_id": ident, "zone": zone, "costs": "base", "gamma": .99,
                    "alpha": float(alpha), "legacy": False, "dp_index_pct": arm["index_pct"]}
    cache, sources, records = {}, {}, []
    for d in definitions.values():
        setting = (d["zone"],d["costs"])
        if setting not in cache:
            cache[setting] = empirical(*setting)
        h, cost, sigma, source = cache[setting]
        sources["|".join(setting)] = source
        gap = (1-d["alpha"])*h(20000*PROBES)-cost
        dense = (1-d["alpha"])*h(20000*STARTS)-cost
        records.append({**d, "exact_index_pct": functional(gap,sigma),
            "dense_direct_index_pct": float(100*np.maximum(dense,0).mean()/sigma),
            "mean_cost_eur": cost, "sigma16_empirical": sigma, "revised_J_panel": source["revised_J_panel"],
            **{f"gap_e{e:g}_eur": float(g) for e,g in zip(PROBES,gap)}})
    # All predictions, including exact zero controls, are persisted before labels.
    D = pd.DataFrame(records)
    D.to_csv(HERE / "exploratory_exact_condition_predictions.csv",index=False)
    cal = jread(ACC / "study_o/calibration_o.json")["rows"]
    calibration = []
    for exclude_FI in (False, True):
        labeled = []
        for c in cal:
            if exclude_FI and c["zone"] == "FI":
                continue
            costs = "H" if c["condition"] == "Study H costs" else "base"
            gamma = .99 if costs == "H" else float(c["condition"].split()[-1])
            row = D[D.condition_id == key(c["zone"],costs,gamma,0)].iloc[0]
            labeled.append({"zone":c["zone"],"learned":c["learned"],"index":float(row.exact_index_pct)})
        high = max(r["index"] for r in labeled if not r["learned"])
        low = min(r["index"] for r in labeled if r["learned"])
        threshold = float(np.sqrt(high*low)) if 0 < high < low else None
        calibration.append({"excludes_revised_FI":exclude_FI,"n":len(labeled),"largest_unlearned_pct":high,
            "smallest_learned_pct":low,"threshold_pct":threshold,"rows":labeled,
            "calibration_consistent_separating_threshold_set_pct":[high,low],
            "separating_set_lower_endpoint_is_open":True})
    threshold = calibration[0]["threshold_pct"]
    transfer = []
    cohorts = [("O","PPO",1000000,csv(ACC / "study_o/runs_o.csv"))]
    P = csv(ACC / "study_p/runs_p.csv")
    cohorts.extend(("P","PPO",int(b),g) for b,g in P[P.study == "P"].groupby("budget"))
    Q = csv(ACC / "study_q/runs_q.csv")
    cohorts.extend(("Q",str(l),1000000,g) for l,g in Q.groupby("learner"))
    for study,learner,budget,runs in cohorts:
        control = runs[runs.alpha == 1].L002.mean()
        treatment = runs[runs.alpha != 1]
        correct = []
        for r in treatment.itertuples():
            value = D.loc[D.condition_id == key(r.region,"base",.99,r.alpha),"exact_index_pct"].iloc[0]
            correct.append((value >= threshold)==(r.L002 == 16))
        for alpha,g in treatment.groupby("alpha"):
            value = D[D.condition_id.isin([key(z,"base",.99,alpha) for z in g.region.unique()])].exact_index_pct.mean()
            gap = g.L002.mean()-control
            transfer.append({"study":study,"learner":learner,"budget":budget,"alpha":alpha,"mean_zone_index_pct":value,
                "threshold_pct":threshold,"predicted_learned":bool(value>=threshold),"observed_gap":float(gap),
                "observed_learned":bool(gap>=4),"arm_correct":bool((value>=threshold)==(gap>=4)),
                "cohort_runs_correct":int(sum(correct)),"cohort_runs_total":len(correct)})
    pd.DataFrame(transfer).to_csv(HERE / "exploratory_exact_transfer.csv",index=False)
    newfees = []
    for fee in (.4,1.5):
        for zone in ("DK1","DK2"):
            h,cost,sigma,source = empirical(zone,fee=fee)
            sources[f"{zone}|fee{fee}"] = source
            for alpha in ALPHAS:
                gap = (1-alpha)*h(20000*PROBES)-cost
                value = functional(gap,sigma)
                newfees.append({"fee":fee,"zone":zone,"alpha":alpha,"exact_index_pct":value,
                    "gap_at_2pct_eur":float(gap[0]),
                    "dense_direct_index_pct":float(100*np.maximum((1-alpha)*h(20000*STARTS)-cost,0).mean()/sigma),
                    "threshold_pct":threshold,"predicts_learned":value>=threshold,
                    "confident_high":alpha<1 and value>=1.5*threshold,
                    "confident_low":alpha<1 and value<=threshold/1.5,
                    "label_identified_by_all_calibration_separating_cutoffs":
                        value<calibration[0]["largest_unlearned_pct"] or value>=calibration[0]["smallest_learned_pct"]})
    F = pd.DataFrame(newfees)
    F.to_csv(HERE / "exploratory_exact_new_fee_predictions.csv",index=False)
    new_design = []
    for fee,g in F.groupby("fee"):
        high = [float(a) for a,s in g[g.alpha<1].groupby("alpha") if s.confident_high.all()]
        low = [float(a) for a,s in g[g.alpha<1].groupby("alpha") if s.confident_low.all()]
        new_design.append({"fee":fee,"eligible_high_alphas":high,"eligible_low_alphas":low,
            "closest_high":max(high) if high else None,"closest_low":min(low) if low else None})
    cross_fee = []
    for alpha,g in F[F.alpha<1].groupby("alpha"):
        byfee = {str(f): {"mean_index_pct":float(s.exact_index_pct.mean()),"both_high":bool(s.confident_high.all()),
                          "both_low":bool(s.confident_low.all())} for f,s in g.groupby("fee")}
        if byfee["0.4"]["both_high"] and byfee["1.5"]["both_low"]:
            cross_fee.append({"alpha":float(alpha),"by_fee":byfee})
    old = csv(OLD / "outputs/exploratory_indices.csv")
    compare = D.merge(old[old.reference == "flat"][["condition_id","surrogate_index_pct","mc_low_pct","mc_high_pct"]],on="condition_id")
    compare["old_MC_interval_covers_exact"] = (compare.mc_low_pct<=compare.exact_index_pct)&(compare.mc_high_pct>=compare.exact_index_pct)
    compare.to_csv(HERE / "exploratory_prior_flat_MC_comparison.csv",index=False)
    ranks = []
    for name,s in (("all26",D),("legacy12",D[D.legacy]),("exclude_revised_FI",D[~D.revised_J_panel])):
        ranks.append({"subset":name,"n":len(s),"spearman_DP":float(stats.spearmanr(s.dp_index_pct,s.exact_index_pct).statistic),
                      "MAE_DP_pct":float(np.mean(abs(s.dp_index_pct-s.exact_index_pct)))})
    result = {"status":"EXPLORATORY exact empirical conditional-sign one-step diagnostic", "calibration":calibration,
        "ranks":ranks,"transfer":transfer,"new_fee_design":new_design,"same_alpha_confident_cross_fee_disagreements":cross_fee,
        "prior_flat_MC": {"n":len(compare),"intervals_covering_exact":int(compare.old_MC_interval_covers_exact.sum()),
            "mean_abs_error_pct":float(abs(compare.surrogate_index_pct-compare.exact_index_pct).mean())},
        "sources":sources,"controls":D[D.alpha==1][["condition_id","exact_index_pct"]].to_dict("records"),
        "max_interpolation_discrepancy_pct":float(abs(D.exact_index_pct-D.dense_direct_index_pct).max()),
        "seconds":time.perf_counter()-start,"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(INPUTS)},
        "limitations":["Known empirical distribution, not a black-box generative-simulator guarantee.",
            "Exact one-step reward differs from optimal continuation; absence is not safety.",
            "No MC uncertainty, but market/model/calibration/learner uncertainty remains.",
            "Existing outcomes known; candidate new-fee conditions were previously inspected, not confirmatory."]}
    (HERE / "exploratory_exact_summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({k:result[k] for k in ("calibration","ranks","new_fee_design","same_alpha_confident_cross_fee_disagreements","prior_flat_MC","seconds")},indent=2))


if __name__ == "__main__":
    main()
