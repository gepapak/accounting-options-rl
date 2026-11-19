"""EXPLORATORY completed-study cost baselines; reads no Study R artifact."""
from __future__ import annotations

import hashlib
import itertools
import json
import os
from pathlib import Path
import time

for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[name]="1"

import numpy as np
import pandas as pd
from scipy import special, stats

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
LIAB=ROOT/"LIABILITY_STUDY_2026-09-29"
ACC=ROOT/"ACCOUNTING_OPTIONS_2026-10-14"
PROBES=np.array([.02,.1,.25,.5,1,1.5,2])
GRID=np.exp(np.linspace(np.log(.05),np.log(2),400))
SCORES=("dose","negative_nominal_fee","negative_mean_cost","negative_cost_over_sigma",
        "dose_sigma_over_cost","gross_mean_over_sigma","cost_adjusted_gross_dose",
        "signed_local_g002","signed_local_g01","integrated_exact","dp_index","dp_region","gaussian_moment_index")
INPUTS=set()
FROZEN={"dp_index":.5133360550525173,"integrated_exact":.08966882375371477}


def read(path):
    INPUTS.add(path)
    return json.loads(path.read_text()) if path.suffix==".json" else pd.read_csv(path)


def ident(z,c,g,a):
    return f"{z}|{c}|g{g:g}|a{a:g}"


def index(gaps,sigma):
    return float(100*np.maximum(np.interp(np.log(GRID),np.log(PROBES),gaps),0).mean()/sigma)


def kernel(zone,costs):
    folder=ROOT/"GENERALIZATION_STUDY_2026-09-29/data_panel" if zone in ("DK1","DK2") else LIAB/"data_panel_j"
    path=folder/f"panel_{zone}.pkl";INPUTS.add(path)
    p=pd.read_pickle(path);p=p.loc[p.split=="train"].iloc[200:]
    cap=.25*.25*p.bal_mw.to_numpy(float)
    if costs=="H":cap*=.1/.25
    q=np.minimum(4,cap)
    m=q*np.abs(p.imb.to_numpy(float)-p.spot.to_numpy(float))
    fee=1. if costs=="H" else .1
    c=fee*q
    if costs=="H":c+=50*q*q/np.maximum(cap,.05)
    mean_cost=float(c.mean());sigma=float(np.sqrt(np.mean(m*m+c*c)-mean_cost**2))
    losses=np.sort(np.r_[c-m,c+m]);tail=np.r_[np.cumsum(losses[::-1])[::-1],0.]
    e=20000*PROBES;i=np.searchsorted(losses,e,side="right")
    h=(tail[i]-(len(losses)-i)*e)/len(losses)
    digest=hashlib.sha256(path.read_bytes()).hexdigest();registered=[]
    if zone in ("FI","NO2"):
        registry=LIAB/"PREREGISTRATION_HASHES.txt";INPUTS.add(registry)
        for line in registry.read_text().splitlines():
            parts=line.split()
            if len(parts)>=3 and parts[-1]==f"*data_panel_j/panel_{zone}.pkl":registered.append(parts[-2])
    revised=bool(registered and digest not in registered)
    return {"fee":fee,"cost":mean_cost,"sigma":sigma,"mean_magnitude":float(m.mean()),"h":h,
            "source_panel":str(path.relative_to(ROOT)),"source_hash":digest,"registered_J_hashes":registered,
            "revised_J_panel":revised,"revised_FI":zone=="FI" and revised}


def score_row(zone,costs,gamma,alpha,dp,region,k):
    cost,sigma,m=k["cost"],k["sigma"],k["mean_magnitude"]
    gaps=(1-alpha)*k["h"]-cost
    z=(cost-20000*PROBES)/sigma
    hg=sigma*np.exp(-z*z/2)/np.sqrt(2*np.pi)+(cost-20000*PROBES)*special.ndtr(z)
    return {"condition_id":ident(zone,costs,gamma,alpha),"zone":zone,"costs":costs,"gamma":gamma,"alpha":alpha,
        "family":zone+"|"+costs,"revised_FI":k["revised_FI"],"dose":1-alpha,
        "negative_nominal_fee":-k["fee"],"negative_mean_cost":-cost,
        "negative_cost_over_sigma":-cost/sigma,"dose_sigma_over_cost":(1-alpha)*sigma/cost,
        "gross_mean_over_sigma":m/sigma,"cost_adjusted_gross_dose":((1-alpha)*m/2-cost)/sigma,
        "signed_local_g002":float(gaps[0]/sigma),"signed_local_g01":float(gaps[1]/sigma),
        "integrated_exact":index(gaps,sigma),"dp_index":float(dp),"dp_region":float(region),
        "gaussian_moment_index":index((1-alpha)*hg-cost,sigma)}


def metric(y,pred):
    y=np.asarray(y,bool);pred=np.asarray(pred,bool)
    return {"accuracy":float((y==pred).mean()),
            "balanced_accuracy":float(((pred[y]).mean()+(~pred[~y]).mean())/2) if y.any() and (~y).any() else None}


def auc(y,x):
    y=np.asarray(y,bool);x=np.asarray(x,float)
    return float(stats.mannwhitneyu(x[y],x[~y]).statistic/(y.sum()*(~y).sum())) if y.any() and (~y).any() else None


def threshold_fit(y,x):
    y=np.asarray(y,bool);x=np.asarray(x,float);u=np.unique(x)
    if not y.any() or not (~y).any():return {"status":"one_class_calibration","canonical":None,"intervals":[]}
    intervals=[]
    span=max(1.,float(u[-1]-u[0]))
    for i in range(len(u)+1):
        lo=None if i==0 else float(u[i-1]);hi=None if i==len(u) else float(u[i])
        t=(lo+hi)/2 if lo is not None and hi is not None else (hi-span if lo is None else lo+span)
        intervals.append({"lower_open":lo,"upper_closed":hi,"balanced_accuracy":metric(y,x>=t)["balanced_accuracy"]})
    best=max(i["balanced_accuracy"] for i in intervals)
    optimum=[i for i in intervals if abs(i["balanced_accuracy"]-best)<1e-12]
    finite=[i for i in optimum if i["lower_open"] is not None and i["upper_closed"] is not None]
    canonical=(finite[0]["lower_open"]+finite[0]["upper_closed"])/2 if finite else None
    return {"status":"constant_unidentified" if len(u)==1 else "fitted", "canonical":canonical,
        "calibration_balanced_accuracy":best,"distinct_scores":len(u),"intervals":optimum}


def possible_thresholds(fit,cal_x,test_x):
    u=np.unique(np.r_[cal_x,test_x]);span=max(1.,float(u[-1]-u[0]))
    candidates=np.r_[u[0]-span,u,(u[:-1]+u[1:])/2,u[-1]+span]
    def allowed(t):
        return any((v["lower_open"] is None or t>v["lower_open"]) and
                   (v["upper_closed"] is None or t<=v["upper_closed"]) for v in fit["intervals"])
    return [float(t) for t in candidates if allowed(t)]


def evaluate(cal,test,score,fit=None,cutoff=None):
    x=test[score].to_numpy(float);y=test.learned.to_numpy(bool)
    out={"score":score,"n_arms":len(test),"learned_arms":int(y.sum()),"auc":auc(y,x),
         "distinct_scores":int(len(np.unique(x))),"cohort":str(test.cohort.iloc[0])}
    if cutoff is None:
        fit=fit or threshold_fit(cal.learned,cal[score]);out["calibration_fit"]=fit
        ts=possible_thresholds(fit,cal[score].to_numpy(float),x)
        patterns=np.unique(np.array([x>=t for t in ts]),axis=0) if ts else np.empty((0,len(test)),bool)
        outcomes=[metric(y,p) for p in patterns]
        out["admissible_prediction_patterns"]=len(patterns)
        out["accuracy_range"]=[min(o["accuracy"] for o in outcomes),max(o["accuracy"] for o in outcomes)] if outcomes else None
        bas=[o["balanced_accuracy"] for o in outcomes if o["balanced_accuracy"] is not None]
        out["balanced_accuracy_range"]=[min(bas),max(bas)] if bas else None
        out["unidentified_arm_ids"]=[str(test.arm_id.iloc[i]) for i in range(len(test))
                                     if len(patterns) and not np.all(patterns[:,i]==patterns[0,i])]
        cutoff=fit["canonical"]
    out["canonical_or_frozen_cutoff"]=cutoff
    if cutoff is not None:
        pred=x>=cutoff;out.update(metric(y,pred))
        out["correct_arms"]=int((pred==y).sum())
        out["misclassified_arm_ids"]=test.loc[pred!=y,"arm_id"].tolist()
    else:
        out.update({"accuracy":None,"balanced_accuracy":None,"correct_arms":None,"misclassified_arm_ids":None})
    return out


def legacy_gap(c,F,K,H,J):
    z,condition=c["zone"],c["condition"]
    if condition=="Study H costs":d=H[(H.region==z)&(H.obsset=="EQ")]
    elif z in ("FI","NO2"):d=J[(J.region==z)&(J.obsset=="EQ")]
    else:
        g=float(condition.split()[-1]);d=K[(K.region==z)&(K.gamma==g)]
        if g==.99:d=pd.concat([F[(F.region==z)&(F.obsset=="EQ")],d])
    return float(d[d.rule=="ZF"]["L0.02"].mean()-d[d.rule=="FL"]["L0.02"].mean())


def main():
    started=time.perf_counter()
    ag=read(LIAB/"results_k/action_gap.csv")
    cache={};rows=[]
    for r in ag.to_dict("records"):
        k=cache.setdefault((r["zone"],r["costs"]),None)
        if k is None:k=kernel(r["zone"],r["costs"]);cache[(r["zone"],r["costs"])]=k
        gaps=np.array([r[f"A_{e:g}"] for e in PROBES])
        rows.append(score_row(r["zone"],r["costs"],r["gamma"],0,index(gaps,r["sigma16"]),r["threshold_K"],k))
    for z in ("DK1","DK2"):
        dp=read(ACC/f"study_o/dp_o_{z}.json")
        for a,v in dp["arms"].items():
            if float(a)==0:continue
            rows.append(score_row(z,"base",.99,float(a),v["index_pct"],v["region16_K"],cache[(z,"base")]))
    D=pd.DataFrame(rows)
    D.to_csv(HERE/"exploratory_condition_scores.csv",index=False) # predictions before labels
    historical=read(ACC/"study_o/calibration_o.json")["rows"]
    F=read(LIAB/"results_f/runs_f.csv");K=read(LIAB/"results_k/runs_k.csv")
    H=read(LIAB/"results_h/runs_h.csv");J=read(LIAB/"results_j/runs_j.csv")
    records=[]
    for c in historical:
        costs="H" if c["condition"]=="Study H costs" else "base"
        gamma=.99 if costs=="H" else float(c["condition"].split()[-1])
        r=D.loc[D.condition_id==ident(c["zone"],costs,gamma,0)].iloc[0].to_dict()
        gap=legacy_gap(c,F,K,H,J)
        records.append({**r,"cohort":"legacy10","arm_id":r["condition_id"],"learned":bool(c["learned"]),
            "original_G":c["gap_002"],"recomputed_G":gap,"label_disagreement":bool(c["learned"]!=(gap>=4))})
    cal=pd.DataFrame(records)
    cal.to_csv(HERE/"exploratory_legacy_labels_and_scores.csv",index=False)
    cohorts=[("O_PPO_1M",read(ACC/"study_o/runs_o.csv"))]
    P=read(ACC/"study_p/runs_p.csv")
    cohorts.extend((f"P_PPO_{int(b)}",d) for b,d in P[P.study=="P"].groupby("budget"))
    Q=read(ACC/"study_q/runs_q.csv")
    cohorts.extend((f"Q_{l}_1M",d) for l,d in Q.groupby("learner"))
    arms=[]
    for name,data in cohorts:
        control=data[data.alpha==1].L002.mean()
        for a,runs in data[data.alpha!=1].groupby("alpha"):
            predictors=D[D.condition_id.isin([ident(z,"base",.99,a) for z in runs.region.unique()])]
            score={s:float(predictors[s].mean()) for s in SCORES}
            gap=float(runs.L002.mean()-control)
            arms.append({**score,"cohort":name,"arm_id":f"{name}|a{a:g}","alpha":a,"learned":gap>=4,
                "G":gap,"zones":len(runs.region.unique()),"trained_runs":len(runs)})
    tests=pd.DataFrame(arms);tests.to_csv(HERE/"exploratory_transfer_arm_scores.csv",index=False)
    fits={s:threshold_fit(cal.learned,cal[s]) for s in SCORES}
    evaluations=[];frozen=[]
    for _,test in tests.groupby("cohort",sort=False):
        evaluations.extend(evaluate(cal,test,s,fits[s]) for s in SCORES)
        frozen.extend(evaluate(cal,test,s,cutoff=c) for s,c in FROZEN.items())
    # Sensitivities use six shared-feature families, not random run rows.
    collapsed=cal[(cal.costs=="H")|(cal.gamma==.99)].copy();collapsed["cohort"]="legacy6_settings"
    calibration_summaries=[]
    for name,c in (("legacy10",cal),("legacy6_settings",collapsed),("legacy_without_revised_FI",cal[~cal.revised_FI])):
        for s in SCORES:
            fit=threshold_fit(c.learned,c[s]);calibration_summaries.append({"calibration":name,"score":s,"n":len(c),
                "auc":auc(c.learned,c[s]),"distinct_scores":len(c[s].unique()),"fit":fit})
    leaveout=[]
    for variant,c in (("legacy10",cal),("legacy6_settings",collapsed)):
        for family,held in c.groupby("family"):
            train=c[c.family!=family]
            held=held.copy();held["cohort"]="heldout_"+family
            leaveout.extend({"calibration_variant":variant,"heldout_family":family,**evaluate(train,held,s)} for s in SCORES)
    cost_subset=cal[cal.zone.isin(["DK1","DK2"])&((cal.costs=="H")|(cal.gamma==.99))].copy()
    cost_scores=[{"score":s,"n_conditions":len(cost_subset),"auc":auc(cost_subset.learned,cost_subset[s]),
        "base_vs_H_direction_correct_zones":sum(float(cost_subset[(cost_subset.zone==z)&(cost_subset.costs=="base")][s].iloc[0])>
             float(cost_subset[(cost_subset.zone==z)&(cost_subset.costs=="H")][s].iloc[0]) for z in ("DK1","DK2"))}
        for s in SCORES]
    result={"status":"EXPLORATORY completed studies only; no R access", "scores":list(SCORES),
        "calibration_summaries":calibration_summaries,"frozen_threshold_transfers":frozen,
        "fair_calibration_transfers":evaluations,"leave_one_family_out":leaveout,
        "completed_bundled_cost_intervention":cost_scores,
        "legacy_label_disagreements":cal.loc[cal.label_disagreement,["arm_id","original_G","recomputed_G"]].to_dict("records"),
        "sources":{"|".join(k):{n:v for n,v in value.items() if n!="h"} for k,value in cache.items()},
        "seconds":time.perf_counter()-started,"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "plan_sha256":hashlib.sha256((HERE/"EXPLORATORY_PLAN.md").read_bytes()).hexdigest(),
        "input_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(INPUTS)}}
    (HERE/"exploratory_cost_baseline_summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    flattened=[]
    for e in evaluations:
        flattened.append({k:v for k,v in e.items() if not isinstance(v,(dict,list))}|{
            "BA_min":e["balanced_accuracy_range"][0] if e["balanced_accuracy_range"] else None,
            "BA_max":e["balanced_accuracy_range"][1] if e["balanced_accuracy_range"] else None,
            "unidentified_arms":len(e["unidentified_arm_ids"]),
            "misclassified_arm_ids":json.dumps(e["misclassified_arm_ids"]),
            "unidentified_arm_ids":json.dumps(e["unidentified_arm_ids"]),
            "calibration_optimal_intervals":json.dumps(e["calibration_fit"]["intervals"])})
    pd.DataFrame(flattened).to_csv(HERE/"exploratory_transfer_metrics.csv",index=False)
    primary=[]
    for cohort,test in tests.groupby("cohort",sort=False):
        selected={e["score"]:e for e in frozen if e["cohort"]==cohort}
        simple=next(e for e in evaluations if e["cohort"]==cohort and e["score"]=="dose_sigma_over_cost")
        primary.append({"cohort":cohort,"n_arms":len(test),"learned_arms":int(test.learned.sum()),
            "frozen_DP_correct":selected["dp_index"]["correct_arms"],
            "frozen_exact_correct":selected["integrated_exact"]["correct_arms"],
            "dose_sigma_cost_canonical_correct":simple["correct_arms"],
            "dose_sigma_cost_optimal_cutoff_correct_min":int(round(simple["accuracy_range"][0]*len(test))),
            "dose_sigma_cost_optimal_cutoff_correct_max":int(round(simple["accuracy_range"][1]*len(test))),
            "arm_AUC_dose_sigma_cost":simple["auc"]})
    pd.DataFrame(primary).to_csv(HERE/"exploratory_primary_comparison.csv",index=False)
    pd.DataFrame(cost_scores).to_csv(HERE/"exploratory_completed_bundled_cost_intervention.csv",index=False)
    cv_summary=[]
    for variant in ("legacy10","legacy6_settings"):
        for score in SCORES:
            found=[v for v in leaveout if v["calibration_variant"]==variant and v["score"]==score]
            available=[v for v in found if v["correct_arms"] is not None]
            cv_summary.append({"calibration_variant":variant,"score":score,"heldout_families":len(found),
                "canonical_identified_families":len(available),"correct_conditions":sum(v["correct_arms"] for v in available),
                "scored_conditions":sum(v["n_arms"] for v in available),
                "total_conditions":sum(v["n_arms"] for v in found)})
    pd.DataFrame(cv_summary).to_csv(HERE/"exploratory_leave_family_out_summary.csv",index=False)
    print("FROZEN",json.dumps(frozen,indent=2))
    print(pd.DataFrame(flattened).query("score=='dose_sigma_over_cost'").to_string(index=False))
    print("Runtime",result["seconds"],"seconds; label disagreements",len(result["legacy_label_disagreements"]))


if __name__=="__main__":main()
