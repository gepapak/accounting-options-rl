"""Verify pre-registration hashes of Studies A-K and N-R and recompute the paper's headline numbers from the shipped
results (Studies B-R when present, exploratory certificates, and the settlement analysis).

Needs only this repository (no training, no network). Run from the repository root:

    python tools/verify_studies.py

Part 1 checks every file listed in the two PREREGISTRATION_HASHES.txt files and in study_n/PREREGISTRATION_HASHES_N.txt
against its recorded SHA-256. A file hashed more than once (a registration amended before its run) must match one of its
recorded hashes; the eSett-derived panels of Study J are not shipped and are reported as "not shipped" until rebuilt
(LIABILITY_STUDY_2026-09-29/get_esett_j.py and build_panel_j.py). A rebuilt panel that differs because eSett revised its
source data after the study is reported as REVISED, not as a failure. Part 2 recomputes headline numbers from the aggregated
result files. Studies L and M have their own frozen manifests and checks (revision_experiments/).
Optional exploratory novelty checks independently integrate shipped exact paths, verify fixed-n certificates and
snapshot hashes, and recompute the selected-case one-action references without new simulation draws or training.
Exit code 0 = every check passes, 1 = at least one check fails.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
LIAB = REPO / "LIABILITY_STUDY_2026-09-29"
GEN = REPO / "GENERALIZATION_STUDY_2026-09-29"
REV = REPO / "revision_experiments"
ACC = REPO / "ACCOUNTING_OPTIONS_2026-10-14"
STUDY_N = ACC / "study_n"
STUDY_O = ACC / "study_o"
STUDY_P = ACC / "study_p"
STUDY_Q = ACC / "study_q"
STUDY_S = ACC / "study_s"
NOT_SHIPPED = {"data_panel_j/panel_FI.pkl", "data_panel_j/panel_NO2.pkl"}   # derived from eSett data (not redistributed)
FAILS: list[str] = []
REVISED: list[str] = []


def check_hashes(folder: Path) -> None:
    recorded: dict[str, set[str]] = defaultdict(set)
    for line in (folder / "PREREGISTRATION_HASHES.txt").read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[-1].startswith("*"):
            recorded[parts[-1][1:]].add(parts[-2])
    for rel, hashes in sorted(recorded.items()):
        p = folder / rel
        if not p.exists():
            status = "not shipped (rebuild to verify)" if rel in NOT_SHIPPED else "MISSING"
            if status == "MISSING":
                FAILS.append(f"{folder.name}/{rel}")
            print(f"[{'info' if rel in NOT_SHIPPED else 'FAIL'}] {folder.name}/{rel}: {status}")
            continue
        ok = hashlib.sha256(p.read_bytes()).hexdigest() in hashes
        if not ok and rel in NOT_SHIPPED:
            # rebuilt locally from a later eSett download; eSett revised these source data after the study
            print(f"[REVISED] {folder.name}/{rel}: rebuilt from revised eSett data, differs from the registered panel")
            REVISED.append(f"{folder.name}/{rel}")
            continue
        print(f"[{'ok' if ok else 'FAIL'}] {folder.name}/{rel}")
        if not ok:
            FAILS.append(f"{folder.name}/{rel}")


def check(name: str, value: float, expected: float, tol: float) -> None:
    ok = bool(np.isfinite(value) and abs(value - expected) <= tol)
    print(f"[{'ok' if ok else 'FAIL'}] {name}: {value:,.2f} (expected {expected:,.3f} +/- {tol})")
    if not ok:
        FAILS.append(name)


def headlines() -> None:
    c = pd.read_csv(LIAB / "results" / "cells_b.csv")
    z = c[(c.market == "NE") & (c.liability == "ZF")]
    check("B3 zero floor, placebo: lowest reported return (%)", z.rep.min(), 38.44, 0.01)
    check("B3 zero floor, placebo: highest reported return (%)", z.rep.max(), 147.83, 0.01)
    check("B3 zero floor, placebo: highest booked return (%)", z.led.max(), -15.08, 0.01)
    check("B3 zero floor, placebo: lowest booked return (%)", z.led.min(), -57.35, 0.01)
    cc = pd.read_csv(LIAB / "scaling" / "results" / "cells_c.csv")
    zc = cc[(cc.liability == "ZF") & (cc.step == 1_600_000)]
    check("C zero floor at 1.6M steps: lowest reported return (%)", zc.rep.min(), 124.72, 0.01)
    check("C zero floor at 1.6M steps: highest reported return (%)", zc.rep.max(), 182.82, 0.01)
    F = pd.read_csv(LIAB / "results_f" / "runs_f.csv")
    F = F[F.obsset == "EQ"]
    zf, fl = F[F.rule == "ZF"]["L0.1"], F[F.rule == "FL"]["L0.1"]
    check("F3 gap L_ZF(0.1) - L_FL(0.1)", zf.mean() - fl.mean(), 11.10, 0.005)
    check("F1 floor-trained agents at leverage 16 (of 10)", float((zf == 16).sum()), 10, 0)
    I = pd.read_csv(LIAB / "results_i" / "runs_i.csv")
    for cfg, exp in (("A2C-n256", 8.53), ("A2C-n64", 11.95), ("DQN-default", 7.15)):
        g = I[I.config == cfg]
        check(f"I1 gap {cfg}", g[g.rule == "ZF"]["L0.1"].mean() - g[g.rule == "FL"]["L0.1"].mean(), exp, 0.005)
    check("I pooled gap (exploratory)", I[I.rule == "ZF"]["L0.1"].mean() - I[I.rule == "FL"]["L0.1"].mean(), 9.21, 0.005)
    J = pd.read_csv(LIAB / "results_j" / "runs_j.csv")

    def L(zone: str, rule: str, e: str) -> pd.Series:
        return J[(J.region == zone) & (J.rule == rule)].sort_values("seed")[f"L{e}"]
    check("J1 Finland gap at 0.1", L("FI", "ZF", "0.1").mean() - L("FI", "FL", "0.1").mean(), 8.18, 0.005)
    check("J2 Finland L_ZF(0.1) > L_ZF(2) (runs of 10)", float((L("FI", "ZF", "0.1").values > L("FI", "ZF", "2").values).sum()), 9, 0)
    check("J4 Finland gap at 0.25", L("FI", "ZF", "0.25").mean() - L("FI", "FL", "0.25").mean(), 8.47, 0.005)
    check("J4 Norway gap at 0.25", L("NO2", "ZF", "0.25").mean() - L("NO2", "FL", "0.25").mean(), 0.625, 0.005)
    H = pd.read_csv(LIAB / "results_h" / "runs_h.csv")
    h = H[(H.obsset == "EQ") & (H.rule == "FL")]
    cols = [col for col in H.columns if col.startswith("L")]
    check("H2 full-liability agents at leverage <= 0.25 everywhere (of 10)", float((h[cols].max(axis=1) <= 0.25).sum()), 10, 0)
    G = pd.read_csv(LIAB / "mtsim_study" / "results_g" / "runs_g.csv")
    check("G gym-mtsim runs", float(len(G)), 20, 0)
    check("G largest exposure of any agent", float(G[[col for col in G.columns if col.startswith("x")]].abs().max().max()), 0.0, 0)
    K = pd.read_csv(LIAB / "results_k" / "runs_k.csv")
    for g, exp in ((0.95, 1.35), (0.98, 1.475), (0.99, 1.85), (0.995, 1.425)):
        check(f"K mean gambling region, zero floor, gamma {g}", K[(K.rule == "ZF") & (K.gamma == g)]["T"].mean(), exp, 0.005)
    check("K full-liability agents with no gambling region (of 20)", float((K[K.rule == "FL"]["T"] == 0).sum()), 17, 0)
    BM = pd.read_csv(LIAB / "results_k" / "best_moment.csv")
    check("Theorem 1: largest KS distance, reported vs running maximum (four zones)", BM.ks_D.max(), 0.0785, 0.0005)
    check("Theorem 1: DK1 mean reported equity (allocations)", BM[BM.zone == "DK1"].mean_reported_K.item(), 2.551, 0.001)
    D = pd.read_csv(LIAB / "env_audit" / "coding_d.csv")
    d = D[D.included == "yes"]
    check("D included environments", float(len(d)), 18, 0)
    check("D environments meeting both conditions", float((d.gambling_prone.str.upper() == "YES").sum()), 1, 0)


def check_hashes_n(folder: Path = None, tag: str = "N") -> None:
    """Studies N and O: '<sha256>  <file>' lines. RUN_LOG_<tag>.md is append-only; its registered version is the prefix
    that ends before the line '## After hashing'."""
    folder = folder or STUDY_N
    for line in (folder / f"PREREGISTRATION_HASHES_{tag}.txt").read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 2 or line.startswith("#"):
            continue
        digest, rel = parts
        data = (folder / rel).read_bytes()
        log = rel == f"RUN_LOG_{tag}.md"
        if log:
            data = data[: data.index(b"\n## After hashing")]   # the registered file ended with one newline
        ok = hashlib.sha256(data).hexdigest() == digest
        print(f"[{'ok' if ok else 'FAIL'}] {folder.name}/{rel}{' (registered prefix)' if log else ''}")
        if not ok:
            FAILS.append(f"{folder.name}/{rel}")


def study_s_headlines() -> None:
    """Study S (sign-paired reward): headline numbers recomputed from the registered run files."""
    A = pd.read_csv(STUDY_S / "runs_s_A.csv")
    B = pd.read_csv(STUDY_S / "runs_s_B.csv")
    check("S runs (Part A, of 60)", float(len(A)), 60, 0)
    check("S runs (Part B, of 120)", float(len(B)), 120, 0)
    m = A.groupby("arm").mean(numeric_only=True)
    check("S Part A: X_A PLAIN", m.loc["PLAIN", "X_A"], 0.850, 0.0005)
    check("S Part A: X_A SPR", m.loc["SPR", "X_A"], 0.0, 0)
    check("S Part A: booked edge PLAIN (% K0)", m.loc["PLAIN", "E"], -6.52, 0.005)
    check("S Part A: booked edge ORACLE (% K0)", m.loc["ORACLE", "E"], 18.05, 0.005)
    check("S Part A: booked edge SPR (% K0)", m.loc["SPR", "E"], 22.58, 0.005)
    for ln, zf, spr, led in (("PPO", 12.15, 2.725, 1.3875), ("A2C-n64", 7.7625, 0.725, 3.0)):
        b = B[B.learner == ln].groupby("arm").X_B.mean()
        check(f"S Part B {ln}: shape ZF", b["ZF"], zf, 0.0005)
        check(f"S Part B {ln}: shape SPR", b["SPR"], spr, 0.0005)
        check(f"S Part B {ln}: shape LEDGER", b["LEDGER"], led, 0.0005)
    supported = (STUDY_S / "VERDICTS_S.md").read_text(encoding="utf-8").count("| SUPPORTED |")
    check("S supported hypotheses (of 8)", float(supported), 8, 0)


def headlines_new() -> None:
    """Studies L, M and N and the settlement analysis (reported separately from the 46-hypothesis scorecard)."""
    L_ = json.loads((REV / "2026-10-10" / "continuation_experiment" / "summary.json").read_text(encoding="utf-8"))
    check("L primary gap, low-equity leverage (zero charge - full charge)", L_["primary_gap"], 9.6, 0.05)
    M_ = json.loads((REV / "2026-10-10-selection" / "frontier_certification" / "summary.json").read_text(encoding="utf-8"))
    check("M DK1 discovery winner, certification reported mean (%)", M_["zones"]["DK1"]["winner"]["reported_mean"], 108.8, 0.05)
    check("M DK2 discovery winner, certification reported mean (%)", M_["zones"]["DK2"]["winner"]["reported_mean"], 104.3, 0.05)
    N = pd.read_csv(STUDY_N / "runs_n.csv")

    def arm(a: str, col: str) -> pd.Series:
        return N[N.arm == a][col]
    check("N1 zombie-probe gap ZFG - ZBK", arm("ZFG", "zbar").mean() - arm("ZBK", "zbar").mean(), 11.91, 0.005)
    check("N2 ZFG runs at leverage 16 when insolvent but undetected (of 10)", float((arm("ZFG", "z_n01_k48") == 16).sum()), 10, 0)
    check("T1 terminal probe gap TFG - TBK", arm("TFG", "tbar").mean() - arm("TBK", "tbar").mean(), 9.125, 0.005)
    O = pd.read_csv(STUDY_O / "runs_o.csv")
    o9, oc = O[O.alpha == 0.9].L002, O[O.alpha == 1.0].L002
    check("O3 alpha 0.9 runs at leverage 16 (of 10)", float((o9 == 16).sum()), 0, 0)
    check("O3 alpha 0.9 gap over control at 2% equity", o9.mean() - oc.mean(), -0.10, 0.005)
    check("O4 alpha 0 gap over control at 2% equity", O[O.alpha == 0.0].L002.mean() - oc.mean(), 14.40, 0.005)
    check("O2 run-level Spearman(index, L(0.02))", float(pd.Series(O["index"]).corr(O.L002, method="spearman")), 0.746, 0.0005)
    P = pd.read_csv(STUDY_P / "runs_p.csv")
    p4m = P[(P.budget == 4_000_000) & (P.alpha.isin((0.8, 0.9)))]
    check("P3 runs at leverage 16, alpha 0.8 and 0.9, 4M steps (of 20)", float((p4m.L002 == 16).sum()), 14, 0)
    check("P3 runs at leverage 16, alpha 0.9, 4M steps (of 10)", float((P[(P.budget == 4_000_000) & (P.alpha == 0.9)].L002 == 16).sum()), 8, 0)
    F12 = pd.read_csv(ACC / "frontier12_fee0.1.csv")
    lv = F12[(F12.zones == "FI+SE+NO") & (F12.rule.str.startswith("LEVEL"))]
    check("Settlement: volume term not deterring, before go-live (%)", 100 * lv[~lv.post].fail.item(), 8.6, 0.05)
    check("Settlement: volume term not deterring, after go-live (%)", 100 * lv[lv.post].fail.item(), 36.8, 0.05)
    RK = LIAB / "results_k"
    AG = pd.read_csv(RK / "action_gap.csv").set_index(["zone", "condition"])
    check("S11 action gap DK1, gamma 0.99, at 0.1 allocations (EUR)", AG.loc[("DK1", "gamma 0.99"), "A_0.1"], 28.42, 0.005)
    PL = pd.read_csv(RK / "partial_liability.csv").set_index(["zone", "alpha"]).threshold_K
    check("Corollary 3: DK1 gambling region with nothing booked (allocations)", PL[("DK1", 0.0)], 1.75, 0)
    check("Corollary 3: DK1 gambling region with 99.8% booked (allocations)", PL[("DK1", 0.998)], 0.0, 0)
    BZ = pd.read_csv(RK / "best_moment_zones.csv").set_index("zone")
    check("S11 Theorem 1, DK1 predicted mean reported equity (allocations)", BZ.loc["DK1", "predicted_K"], 2.722, 0.001)
    FB = pd.read_csv(RK / "false_profit_bound.csv").set_index("zone")
    check("S11 Theorem 2, DK1 cap B_T (allocations)", FB.loc["DK1", "cap_B_K"], 3.431, 0.001)
    lad = pd.read_csv(ACC / "ladder12.csv")
    check("Factor two: smallest fee-free ZF/TERM ratio, 24 zone-years", lad.ZF_over_TERM_feefree.min(), 1.62, 0.005)
    check("Factor two: largest fee-free ZF/TERM ratio, 24 zone-years", lad.ZF_over_TERM_feefree.max(), 1.99, 0.005)


def counts_and_learnability() -> None:
    """Independent counting and inexpensive O/P point fits; never rerun frozen aggregators."""
    from scipy import optimize, special, stats

    expected = {"B": 120, "C": 24, "E": 120, "F": 60, "G": 20, "H": 40,
                "I": 90, "J": 60, "K": 60, "L": 20, "N": 40, "O": 80, "P": 90}
    paths = {"B": LIAB / "results/runs_b.csv", "E": LIAB / "results_e/runs_e.csv",
             "F": LIAB / "results_f/runs_f.csv", "G": LIAB / "mtsim_study/results_g/runs_g.csv",
             "H": LIAB / "results_h/runs_h.csv", "I": LIAB / "results_i/runs_i.csv",
             "J": LIAB / "results_j/runs_j.csv", "K": LIAB / "results_k/runs_k.csv",
             "L": REV / "2026-10-10/continuation_experiment/agents.csv", "N": STUDY_N / "runs_n.csv",
             "O": STUDY_O / "runs_o.csv"}
    actual = {s: len(pd.read_csv(p)) for s, p in paths.items()}
    C = pd.read_csv(LIAB / "scaling/results/snapshots_c.csv")
    actual["C"] = len(C.drop_duplicates(["liability", "algo", "seed", "region"]))
    P = pd.read_csv(STUDY_P / "runs_p.csv")
    actual["P"] = int((P.study == "P").sum())  # P's CSV also contains O's 80 reused runs.
    for study, n in expected.items():
        check(f"Unique trained agents, Study {study}", actual[study], n, 0)
    check("Completed main-study cohort B-P (M reuses L; excludes Q and separate A)", sum(actual.values()), 824, 0)
    check("Separate Study A trained agents", len(list((GEN / "results/jobs").glob("*.json"))), 280, 0)

    # Count registered primary verdicts only. B2's second arrow is a secondary outcome.
    totals = []
    for s, rel, supported, total in [("B", "results", 3, 5), ("C", "scaling/results", 4, 5),
            ("E", "results_e", 5, 5), ("F", "results_f", 5, 6), ("G", "mtsim_study/results_g", 0, 3),
            ("H", "results_h", 2, 5), ("I", "results_i", 2, 9), ("J", "results_j", 3, 5), ("K", "results_k", 0, 3)]:
        text = (LIAB / rel / f"VERDICTS_{s}.md").read_text(encoding="utf-8")
        vals = [re.search(r"-> (NOT SUPPORTED|SUPPORTED)", line).group(1)
                for line in text.splitlines() if re.match(rf"- {s}[1-9]", line)]
        check(f"{s} supported registered primary verdicts", vals.count("SUPPORTED"), supported, 0)
        check(f"{s} registered primary verdicts", len(vals), total, 0)
        totals.append((vals.count("SUPPORTED"), len(vals)))
    check("Original scorecard supported verdicts", sum(x[0] for x in totals), 24, 0)
    check("Original scorecard total verdicts", sum(x[1] for x in totals), 46, 0)
    for s, folder, supported, total in [("N", STUDY_N, 6, 9), ("O", STUDY_O, 4, 4), ("P", STUDY_P, 3, 4)]:
        vals = re.findall(r"\| (NOT SUPPORTED|SUPPORTED) \|$",
                          (folder / f"VERDICTS_{s}.md").read_text(encoding="utf-8"), flags=re.MULTILINE)
        check(f"{s} supported registered verdicts", vals.count("SUPPORTED"), supported, 0)
        check(f"{s} total registered verdicts", len(vals), total, 0)

    O = pd.read_csv(STUDY_O / "runs_o.csv")
    cal = json.loads((STUDY_O / "calibration_o.json").read_text(encoding="utf-8"))
    check("O registered threshold c* (%)", cal["c_star_pct"], 0.513, 0.0005)
    check("O threshold geometric-midpoint identity", cal["c_star_pct"],
          np.sqrt(cal["largest_unlearned_index_pct"] * cal["smallest_learned_index_pct"]), 1e-12)

    def fit(D, joint=False):
        cols = [np.ones(len(D)), np.log10(D["index"].to_numpy())]
        if joint:
            cols.append(np.log10(D.budget.to_numpy() / 1e6))
        X = np.column_stack(cols)
        y = (D.L002.to_numpy() == 16).astype(float)
        def nll(b):
            z = X @ b
            return np.sum(np.logaddexp(0, z) - y*z), X.T @ (special.expit(z)-y)
        result = optimize.minimize(nll, np.zeros(X.shape[1]), jac=True, method="BFGS")
        check("Logistic point fit gradient", np.max(np.abs(nll(result.x)[1])), 0, 1e-3)
        return result.x
    b = fit(O[O.alpha != 1])
    check("O exploratory logistic c50 (%)", 10 ** (-b[0] / b[1]), 0.429, 0.0005)
    b = fit(P[P.alpha != 1], joint=True)
    check("P joint logistic c50 at 1M (%)", 10 ** (-b[0] / b[1]), 0.330, 0.0005)
    check("P joint logistic beta", -b[2] / b[1], -0.430, 0.0005)
    for label, n1, n0, arms, expected_p in [("P3", 4_000_000, 1_000_000, (.8, .9), .0018),
                                          ("P4", 1_000_000, 250_000, (.5, .7, .8), .3029)]:
        table = []
        for budget in (n1, n0):
            d = P[(P.budget == budget) & P.alpha.isin(arms)]
            k = int((d.L002 == 16).sum())
            table.append([k, len(d)-k])
        check(f"{label} one-sided Fisher p", stats.fisher_exact(table, alternative="greater").pvalue, expected_p, .00005)
    baseline = pd.read_csv(STUDY_O / "baselines_o.csv").set_index("predictor")
    check("Exploratory index pooled AUC", baseline.loc["index", "auc_pooled"], .9626225490196079, 1e-12)
    check("Exploratory index prospective O AUC", baseline.loc["index", "auc_study_O"], .915296052631579, 1e-12)
    check("Exploratory index prospective O accuracy", baseline.loc["index", "accuracy_on_study_O"], 58/70, 1e-12)
    treatment = O[O.alpha != 1]
    learned = (treatment.L002 == 16).to_numpy()
    dose = (1-treatment.alpha).to_numpy()
    dose_auc = stats.mannwhitneyu(dose[learned], dose[~learned]).statistic/(learned.sum()*(~learned).sum())
    check("Exploratory O treatment-only AUC(1-alpha)", dose_auc, .9194078947368421, 1e-12)


def headlines_q() -> None:
    """Check Q only after frozen verdicts and all 100 run summaries exist."""
    verdict = STUDY_Q / "VERDICTS_Q.md"
    if not verdict.exists():
        print("[info] Study Q headline checks pending: no VERDICTS_Q.md; no partial outcomes promoted")
        return
    Q = pd.read_csv(STUDY_Q / "runs_q.csv")
    check("Q complete registered cohort", len(Q), 100, 0)
    check("Q unique learner/alpha/seed/zone runs", len(Q.drop_duplicates(["learner", "alpha", "seed", "region"])), 100, 0)
    check("Q runs per learner/alpha cell, minimum", Q.groupby(["learner", "alpha"]).size().min(), 10, 0)
    check("Q runs per learner/alpha cell, maximum", Q.groupby(["learner", "alpha"]).size().max(), 10, 0)
    rows = [line for line in verdict.read_text(encoding="utf-8").splitlines() if line.startswith("| Q")]
    check("Q reports all six registered verdicts", len(rows), 6, 0)
    check("Q supported registered verdicts", sum("| SUPPORTED |" in row for row in rows), 6, 0)
    for learner, d in Q.groupby("learner"):
        control = d[d.alpha == 1].L002.mean()
        for hyp, alpha in [("Q2", 0), ("Q3", .9)]:
            row = next(line for line in rows if line.startswith(f"| {hyp} {learner} |"))
            reported_gap = float(re.search(r"gap ([+-]?[0-9.]+)", row).group(1))
            check(f"{hyp} {learner} gap matches frozen verdict", d[d.alpha == alpha].L002.mean()-control, reported_gap, .005)
        row = next(line for line in rows if line.startswith(f"| Q1 {learner} |"))
        rho = float(re.search(r"= ([+-]?[0-9.]+),", row).group(1))
        check(f"Q1 {learner} correlation matches frozen verdict", d["index"].corr(d.L002, method="spearman"), rho, .0005)
    dqn = Q[Q.learner == "DQN-default"]
    check("Q exploratory DQN control low-equity leverage", dqn[dqn.alpha == 1].L002.mean(), 9.225, 1e-10)
    check("Q DQN control runs at maximal low-equity leverage", (dqn[dqn.alpha == 1].L002 == 16).sum(), 5, 0)
    check("Q DQN alpha0.9 runs at maximal low-equity leverage", (dqn[dqn.alpha == .9].L002 == 16).sum(), 6, 0)
    check("Completed main-study cohort including Q", 824 + len(Q), 924, 0)


def cost_baseline_headlines() -> None:
    """Read-only checks of explicitly exploratory comparison and toy outputs."""
    folder = REPO / "exploratory_novelty_2026-10-15/cost_baselines"
    summary = folder / "exploratory_cost_baseline_summary.json"
    if not summary.exists():
        print("[info] Exploratory cost-baseline snapshot not present")
        return
    data = json.loads(summary.read_text(encoding="utf-8"))
    check("Exploratory cost comparison reports every fixed score", len(set(data["scores"])), 13, 0)
    check("Exploratory historical labels reproduce original controls", len(data["legacy_label_disagreements"]), 0, 0)
    for name, filename in [("script_sha256", "compare_completed_cost_baselines.py"),
                           ("plan_sha256", "EXPLORATORY_PLAN.md")]:
        check(f"Exploratory cost snapshot matches current {filename}",
              int(hashlib.sha256((folder / filename).read_bytes()).hexdigest() == data[name]), 1, 0)
    frozen = {(r["cohort"], r["score"]): r for r in data["frozen_threshold_transfers"]}
    fair = {(r["cohort"], r["score"]): r for r in data["fair_calibration_transfers"]}
    expected = {
        "O_PPO_1M": (7, 6, 6, 4, 3, 7),
        "P_PPO_250000": (5, 4, 3, 4, 3, 5),
        "P_PPO_4000000": (2, 0, 2, 0, 0, 1),
        "Q_A2C-n64_1M": (4, 3, 3, 2, 2, 4),
        "Q_DQN-default_1M": (4, 3, 1, 4, 2, 4),
    }
    for cohort, (n, dp, exact, simple, low, high) in expected.items():
        check(f"Exploratory {cohort} arm count", frozen[(cohort, "dp_index")]["n_arms"], n, 0)
        check(f"Exploratory {cohort} fixed DP correct", frozen[(cohort, "dp_index")]["correct_arms"], dp, 0)
        check(f"Exploratory {cohort} fixed exact correct", frozen[(cohort, "integrated_exact")]["correct_arms"], exact, 0)
        r = fair[(cohort, "dose_sigma_over_cost")]
        check(f"Exploratory {cohort} simple representative correct", r["correct_arms"], simple, 0)
        check(f"Exploratory {cohort} simple admissible minimum", n*r["accuracy_range"][0], low, 1e-10)
        check(f"Exploratory {cohort} simple admissible maximum", n*r["accuracy_range"][1], high, 1e-10)
        if cohort != "P_PPO_4000000":
            for score in ("dose", "dose_sigma_over_cost", "cost_adjusted_gross_dose",
                          "integrated_exact", "dp_index", "gaussian_moment_index"):
                check(f"Exploratory {cohort} arm AUC {score}", fair[(cohort, score)]["auc"], 1, 0)
    costs = data["completed_bundled_cost_intervention"]
    check("Exploratory bundled H comparison reports all scores", len(costs), 13, 0)
    check("Exploratory bundled H nominal-fee AUC", next(r["auc"] for r in costs if r["score"] == "negative_nominal_fee"), 1, 0)
    toy_folder = REPO / "exploratory_novelty_2026-10-15/matched_moments"
    toy = json.loads((toy_folder / "matched_moments.json").read_text(encoding="utf-8"))
    a, b = toy["markets"]["A"], toy["markets"]["B"]
    for field in toy["matched_fields"]:
        check(f"Exploratory matched-moment pair {field}", a[field]["decimal"], b[field]["decimal"], 0)
    check("Exploratory matched-moment A charged reward", a["charged_one_step_reward"]["decimal"], -.001, 0)
    check("Exploratory matched-moment B charged reward", b["charged_one_step_reward"]["decimal"], .001505, 0)
    check("Exploratory matched-moment B booked harm", b["net_mean"]["decimal"], -.001, 0)
    for label, row in [("A", a), ("B", b)]:
        check(f"Exploratory matched-moment {label} full-booking gate", row["fully_booked_gated_witness_value"]["decimal"], 0, 0)


def study_r_integrity() -> None:
    """Verify R without inspecting interim outcomes or editing frozen artifacts."""
    folder = ACC / "study_r"
    manifest = folder / "PREREGISTRATION_HASHES_R.txt"
    if not manifest.exists():
        return
    lines = manifest.read_text(encoding="utf-8").splitlines()
    prefix_bytes = int(next(line.split()[-1] for line in lines if line.startswith("# RUN_LOG_PREFIX_BYTES ")))
    for line in lines:
        if not line or line.startswith("#"):
            continue
        expected, relative = line.split("  ", 1)
        path = REPO / relative
        if not path.exists() and relative in {
                "LIABILITY_STUDY_2026-09-29/data_panel_j/panel_FI.pkl",
                "LIABILITY_STUDY_2026-09-29/data_panel_j/panel_NO2.pkl"}:
            print(f"[info] R calibration panel not shipped: {relative}; exact predictions retained")
            continue
        content = path.read_bytes() if path.exists() else b""
        if path.name == "RUN_LOG_R.md":
            content = content[:prefix_bytes]
        ok = hashlib.sha256(content).hexdigest() == expected
        print(f"[{'ok' if ok else 'FAIL'}] R registered bytes: {relative}")
        if not ok:
            FAILS.append(f"R hash {relative}")
    verdict = folder / "VERDICTS_R.md"
    if not verdict.exists():
        print("[info] R outcome pending; interim jobs are not analyzed or included in the completed headline")
        return
    data = pd.read_csv(folder / "runs_r.csv")
    check("R complete registered cohort", len(data), 60, 0)
    check("R unique fee/alpha/seed/zone runs", len(data.drop_duplicates(["fee", "alpha", "seed", "region"])), 60, 0)
    text = verdict.read_text(encoding="utf-8")
    rows = [line for line in text.splitlines() if line.startswith("| R1 |")]
    check("R reports exactly one composite primary", len(rows), 1, 0)
    summary = json.loads((folder / "analysis_r.json").read_text(encoding="utf-8"))
    positive = summary["positive_seed_clusters"]
    from math import comb
    p = sum(comb(5, j) for j in range(positive, 6)) / 32
    check("R exact one-sided sign-test identity", summary["exact_one_sided_sign_p"], p, 1e-15)
    expected_support = (summary["low_fee_G"] >= 4 and summary["high_fee_G"] < 4
                        and summary["mean_interaction"] >= 4 and p <= .05)
    check("R primary decision matches all frozen component rules", int(summary["primary_R1_supported"]), int(expected_support), 0)
    recorded_support = "| SUPPORTED |" in rows[0]
    check("R verdict matches frozen analysis", int(recorded_support), int(expected_support), 0)
    check("Completed main-study cohort through R (once complete)", 924 + len(data), 984, 0)
    study_r_completed_headlines(folder, data, summary)
    study_r_prediction_checks(folder)
    study_r_job_checks(folder, data)


def study_r_completed_headlines(folder: Path, data: pd.DataFrame, summary: dict) -> None:
    """Derive completed R endpoints from raw arm rows, not the stored verdict components."""
    import itertools
    from math import comb

    expected_keys = {(f, a, s, z) for f in (.4, 1.5) for a in (0, .8, 1)
                     for s in range(60, 65) for z in ("DK1", "DK2")}
    actual_keys = set(zip(data.fee, data.alpha, data.seed, data.region))
    check("R exactly covers every planned fee/alpha/seed/zone", int(actual_keys == expected_keys), 1, 0)
    check("R all agents reached the rounded full training budget", int((data.actual_timesteps == 1007616).sum()), 60, 0)
    table = data.pivot(index=["seed", "region"], columns=["fee", "alpha"], values="L002")
    interactions = ((table[(.4, .8)]-table[(.4, 1)])-(table[(1.5, .8)]-table[(1.5, 1)]))
    clusters = interactions.groupby(level="seed").mean().reindex(range(60, 65))
    expected_clusters = [-1.5, .125, 9., -7.5, 0.]
    for seed, value, expected in zip(clusters.index, clusters, expected_clusters):
        check(f"R independently recomputed seed {seed} interaction", value, expected, 1e-12)
        check(f"R seed {seed} interaction matches frozen analysis", value, summary["paired_seed_interactions"][str(seed)], 1e-12)
    positives = int((clusters > 0).sum())
    p = sum(comb(5, k) for k in range(positives, 6))/32
    check("R positive seed clusters (zeros nonpositive)", positives, 2, 0)
    check("R positive seed count matches frozen analysis", positives, summary["positive_seed_clusters"], 0)
    check("R zero interaction seed clusters", int((clusters == 0).sum()), 1, 0)
    check("R independently recomputed exact sign p", p, .8125, 0)
    check("R independent sign p matches frozen analysis", p, summary["exact_one_sided_sign_p"], 0)
    bootstrap = np.fromiter((np.mean(values) for values in itertools.product(clusters.to_numpy(), repeat=5)), float)
    interval = np.quantile(bootstrap, [.025, .975])
    check("R exact descriptive paired-seed bootstrap replicates", len(bootstrap), 3125, 0)
    for i, expected in enumerate((-4.475, 5.125)):
        check(f"R descriptive paired-seed bootstrap endpoint {i}", interval[i], expected, 1e-12)
        check(f"R descriptive interval endpoint {i} matches frozen analysis", interval[i], summary["descriptive_paired_seed_bootstrap_95pct"][i], 1e-12)
    expected_arms = {(.4, 0): (16., 12.9, 10), (.4, .8): (7.6, 4.5, 4), (.4, 1): (3.1, 0., 1),
                     (1.5, 0): (16., 14.875, 10), (1.5, .8): (5.6, 4.475, 3), (1.5, 1): (1.125, 0., 0)}
    stored_arms = {(r["fee"], r["alpha"]): r for r in summary["arms"]}
    gaps = {}
    for (fee, alpha), (mean, gap, max16) in expected_arms.items():
        group = data[(data.fee == fee) & (data.alpha == alpha)]
        control = data[(data.fee == fee) & (data.alpha == 1)].L002.mean()
        observed_gap = group.L002.mean()-control
        gaps[(fee, alpha)] = observed_gap
        check(f"R fee {fee} alpha {alpha} arm size", len(group), 10, 0)
        check(f"R fee {fee} alpha {alpha} absolute probe leverage", group.L002.mean(), mean, 1e-12)
        check(f"R fee {fee} alpha {alpha} matched-control gap", observed_gap, gap, 1e-12)
        check(f"R fee {fee} alpha {alpha} maximal-leverage runs", int((group.L002 == 16).sum()), max16, 0)
        check(f"R fee {fee} alpha {alpha} gap matches analysis", observed_gap, stored_arms[(fee, alpha)]["gap_over_matched_control"], 1e-12)
    check("R independently recomputed low-fee primary gap matches summary", gaps[(.4, .8)], summary["low_fee_G"], 1e-12)
    check("R independently recomputed high-fee primary gap matches summary", gaps[(1.5, .8)], summary["high_fee_G"], 1e-12)
    check("R independently recomputed mean interaction matches summary", clusters.mean(), summary["mean_interaction"], 1e-12)
    components = {"low_fee_gap_at_least_4": gaps[(.4, .8)] >= 4,
                  "high_fee_gap_below_4": gaps[(1.5, .8)] < 4,
                  "mean_interaction_at_least_4": clusters.mean() >= 4,
                  "exact_one_sided_seed_sign_p_at_most_0p05": p <= .05}
    check("R all four independently recomputed primary component flags agree", int(components == summary["components"]), 1, 0)
    check("R exactly one of four primary component gates passes", sum(components.values()), 1, 0)
    check("R composite remains unsupported from independent rows", int(all(components.values())), 0, 0)
    check("R low-minus-high matched interaction", clusters.mean(), .025, 1e-12)
    check("R interaction equals contrast of cohort gaps", clusters.mean(), gaps[(.4, .8)]-gaps[(1.5, .8)], 1e-12)
    check("R high-fee alpha0.8 exceeds the registered operational acquisition cutoff", int(gaps[(1.5, .8)] >= 4), 1, 0)


def study_r_prediction_checks(folder: Path) -> None:
    """Direct sign integration validates frozen predictions, without calling the predictor."""
    prediction = json.loads((folder / "predictions_r.json").read_text(encoding="utf-8"))
    rows = prediction["predictions"]
    check("R frozen prediction grid", len(rows), 12, 0)
    check("R original heuristic threshold (%)", prediction["threshold_pct"], .08966882375371477, 1e-14)
    high = max(r["index_pct"] for r in prediction["calibration"] if not r["learned"])
    low = min(r["index_pct"] for r in prediction["calibration"] if r["learned"])
    check("R threshold retains original ten-condition geometric calibration", prediction["threshold_pct"], np.sqrt(high*low), 1e-14)
    check("R frozen calibration records", len(prediction["calibration"]), 10, 0)
    bad_hashes = 0
    for rel, digest in prediction["input_sha256"].items():
        path = REPO / rel
        if not path.exists() and rel in {"LIABILITY_STUDY_2026-09-29/data_panel_j/panel_FI.pkl",
                                        "LIABILITY_STUDY_2026-09-29/data_panel_j/panel_NO2.pkl"}:
            print(f"[info] R calibration input not shipped: {rel}; frozen prediction values retained")
            continue
        bad_hashes += int(not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest)
    check("R frozen prediction input hash mismatches", bad_hashes, 0, 0)
    probes = np.array([.02, .1, .25, .5, 1, 1.5, 2])
    grid = np.exp(np.linspace(np.log(.05), np.log(2), 400))
    errors = []
    for zone in ("DK1", "DK2"):
        panel_path = GEN / f"data_panel/panel_{zone}.pkl"
        if not panel_path.exists():
            print(f"[info] R TRAIN panel not shipped: {zone}; frozen predictions retained")
            continue
        panel = pd.read_pickle(panel_path)
        panel = panel.loc[panel.split == "train"].iloc[200:]
        q = np.minimum(4, .25*.25*panel.bal_mw.to_numpy(float))
        magnitude = q*np.abs(panel.imb.to_numpy(float)-panel.spot.to_numpy(float))
        for row in (r for r in rows if r["zone"] == zone):
            cost = row["fee"]*q
            h = np.array([.5*np.mean(np.maximum(cost-magnitude-20000*e, 0)
                                     +np.maximum(cost+magnitude-20000*e, 0)) for e in probes])
            gaps = (1-row["alpha"])*h-cost.mean()
            sigma = np.sqrt(np.mean(magnitude*magnitude+cost*cost)-cost.mean()**2)
            score = 100*np.maximum(np.interp(np.log(grid), np.log(probes), gaps), 0).mean()/sigma
            errors += [abs(score-row["index_pct"]), abs(gaps[0]-row["gap_at_2pct_eur"]),
                       float(np.max(np.abs(gaps-np.asarray(row["gaps_eur"])))),
                       abs(sigma-row["sigma16_eur"]), abs(cost.mean()-row["mean_cost_eur"])]
    if errors:
        check("R direct empirical integration maximum prediction discrepancy", max(errors), 0, 1e-9)
    same = [r for r in rows if r["alpha"] == .8]
    check("R same-alpha one-action incentives remain positive at both fees", sum(r["gap_at_2pct_eur"] > 0 for r in same), 4, 0)
    check("R frozen same-alpha low-fee learned predictions", sum(r["predicts_learned"] for r in same if r["fee"] == .4), 2, 0)
    check("R frozen same-alpha high-fee nonlearned predictions", sum(not r["predicts_learned"] for r in same if r["fee"] == 1.5), 2, 0)
    check("R fully booked control indices remain exactly zero", sum(r["index_pct"] == 0 for r in rows if r["alpha"] == 1), 4, 0)


def study_r_job_checks(folder: Path, data: pd.DataFrame) -> None:
    """Audit saved jobs/counters/model hashes and NumPy probes without learner imports."""
    paths = sorted((folder / "results_r/jobs").glob("*.json"))
    if not paths:
        print("[info] R individual job/model artifacts not shipped; completed endpoint table still checked")
        return
    check("R full completed job summaries (development smoke excluded)", len(paths), 60, 0)
    registration = hashlib.sha256((folder / "PREREGISTRATION_HASHES_R.txt").read_bytes()).hexdigest()
    expected_probes = {f"{e:g}" for e in (0, .02, .05, .1, .25, .5, .75, 1, 1.25, 1.5, 2, 2.5, 3)}
    records = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    row_keys = set(zip(data.fee, data.alpha, data.seed, data.region))
    job_keys = {(r["fee"], r["alpha"], r["seed"], r["region"]) for r in records}
    check("R job keys exactly match the complete endpoint cohort", int(job_keys == row_keys), 1, 0)
    settings_errors = counter_errors = 0
    reward_errors, artifacts = [], []
    csv = data.set_index(["fee", "alpha", "seed", "region"])
    endpoint_error = 0.
    for path, r in zip(paths, records):
        if (r.get("registration_sha256") != registration or r.get("steps") != 1000000
                or r.get("actual_timesteps") != 1007616 or r.get("algo") != "PPO"
                or r.get("gamma") != .99 or r.get("obsset") != "EQ"
                or set(r.get("lev", {})) != expected_probes
                or any(v not in (0, .25, 1, 4, 16) for v in r.get("lev", {}).values())):
            settings_errors += 1
        visits = np.asarray(r.get("visit_action_counts", []))
        if visits.shape != (8, 9) or visits.sum() != 1007616 or (visits < 0).any():
            counter_errors += 1
        t = r["training_sums_eur"]
        reward_errors.append(abs(t["reward"]-t["pnl"]-(1-r["alpha"])*t["injections"]))
        key = (r["fee"], r["alpha"], r["seed"], r["region"])
        for column, probe in (("L002", "0.02"), ("L01", "0.1"), ("L1", "1")):
            endpoint_error = max(endpoint_error, abs(r["lev"][probe]-csv.loc[key, column]))
        expected_names = {path.stem+".zip", path.stem+"__normalization.pkl", path.stem+"__policy_layers.npz"}
        if set(r.get("model_artifact_sha256", {})) != expected_names:
            settings_errors += 1
        artifacts.extend((folder / "results_r/models" / name, digest) for name, digest in r["model_artifact_sha256"].items())
    check("R full job settings/probes/registration mismatches", settings_errors, 0, 0)
    check("R full training visitation-counter mismatches", counter_errors, 0, 0)
    check("R full job probes agree with frozen CSV", endpoint_error, 0, 0)
    check("R cumulative charged-reward accounting identity error", max(reward_errors), 0, 1e-5)
    check("R expected local model artifacts recorded", len(artifacts), 180, 0)
    available = sum(p.exists() for p, digest in artifacts)
    if not available:
        print("[info] R trained model files stay local and are not shipped; recorded digests retained")
        return
    check("R all recorded local model artifacts present", available, 180, 0)
    check("R local model artifact hash mismatches", sum(not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != digest
                                                     for p, digest in artifacts), 0, 0)
    leverage = np.array([-16, -4, -1, -.25, 0, .25, 1, 4, 16])
    probe_errors = 0
    for r in records:
        name = next(n for n in r["model_artifact_sha256"] if n.endswith("__policy_layers.npz"))
        path = folder / "results_r/models" / name
        if not path.exists():
            continue
        try:
            with np.load(path, allow_pickle=False) as network:
                x = np.asarray([[min(float(e), 5)] for e in r["lev"]], dtype=np.float32).astype(float)
                for i in sorted(int(k.split("_")[1]) for k in network.files if k.startswith("kind_")):
                    kind = str(network[f"kind_{i}"].item())
                    if kind == "lin":
                        x = x @ network[f"weights_{i}"].T+network[f"bias_{i}"]
                    elif kind == "tanh":
                        x = np.tanh(x)
                    elif kind == "relu":
                        x = np.maximum(x, 0)
                    else:
                        raise ValueError("unrecognized saved policy layer")
                if not np.array_equal(np.abs(leverage[x.argmax(axis=1)]), list(r["lev"].values())):
                    probe_errors += 1
        except (ValueError, OSError, KeyError):
            probe_errors += 1
    check("R saved policy forward passes reproduce all thirteen probes", probe_errors, 0, 0)


def adaptive_witness_headlines() -> None:
    """Read-only exact identities, fixed-n certificates and hashes; no new draws or training."""
    novelty = REPO / "exploratory_novelty_2026-10-15"
    exact_dir = novelty / "adaptive_witness"
    empirical = novelty / "adaptive_witness_empirical"
    heavy_dir = novelty / "adaptive_witness_heavytail"
    paths = [exact_dir / "exact_adaptive_benchmark.json", empirical / "full_clock_summary.json",
             empirical / "full_clock_certificates.csv", empirical / "selected_case_followup.json",
             heavy_dir / "exploratory_heavytail_certificate.json"]
    if not any(p.exists() for p in paths):
        print("[info] Exploratory adaptive-witness snapshots not present")
        return
    check("Exploratory adaptive-witness snapshot files present", sum(p.exists() for p in paths), len(paths), 0)
    if not all(p.exists() for p in paths):
        return

    def snapshot_hashes(label: str, folder: Path, data: dict, script: str, plan: str | None = None) -> None:
        expected = [(folder / script, data["script_sha256"])]
        if plan is not None:
            expected.append((folder / plan, data["plan_sha256"]))
        expected.extend((REPO / rel, digest) for rel, digest in data.get("input_sha256", {}).items())
        if "kernel_script_sha256" in data:
            expected.append((empirical / "benchmark_adaptive_witness.py", data["kernel_script_sha256"]))
        if "guard_script_sha256" in data:
            expected.append((empirical / "full_clock_benchmark.py", data["guard_script_sha256"]))
        bad = [str(p.relative_to(REPO)) for p, digest in expected
               if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != digest]
        check(f"Exploratory {label} script/plan/input hash mismatches", len(bad), 0, 0)
        if bad:
            print(f"[info] {label} mismatched artifacts: {bad}")

    def eb_radius(variance: float, n: int, lower: float, upper: float, delta: float) -> float:
        # One-sided M&P Thm4; an opposite endpoint is not a joint two-sided interval.
        return float(np.sqrt(2 * variance * np.log(2 / delta) / n)
                     + 7 * (upper - lower) * np.log(2 / delta) / (3 * (n - 1)))

    exact = json.loads(paths[0].read_text(encoding="utf-8"))
    snapshot_hashes("exact adaptive", exact_dir, exact, "exact_adaptive_benchmark.py")
    scenarios = exact["scenarios"]
    check("Exploratory exact adaptive scenarios", len(scenarios), 9, 0)
    check("Exploratory exact adaptive witness scenarios", sum(s["library_decision"] == "WITNESS" for s in scenarios), 5, 0)
    exact_errors = []
    for scenario in scenarios:
        alpha = Fraction(scenario["alpha"]["exact"])
        for policy in scenario["policies"]:
            outcomes = policy["paths"]
            probabilities = [Fraction(r["probability"]["exact"]) for r in outcomes]
            if sum(probabilities) != 1 or len(outcomes) != 2 ** scenario["horizon"]:
                exact_errors.append("path probabilities")
            for field, recorded in policy["expected"].items():
                integrated = sum(p * Fraction(r[field]["exact"]) for p, r in zip(probabilities, outcomes))
                if integrated != Fraction(recorded["exact"]) or abs(float(integrated)-recorded["decimal"]) > 1e-12:
                    exact_errors.append("integrated expectation")
            for field, recorded in policy["variance"].items():
                mean = Fraction(policy["expected"][field]["exact"])
                variance = sum(p * (Fraction(r[field]["exact"])-mean)**2
                               for p, r in zip(probabilities, outcomes))
                if variance != Fraction(recorded["exact"]) or abs(float(variance)-recorded["decimal"]) > 1e-12:
                    exact_errors.append("integrated variance")
            for r in outcomes:
                v = {k: Fraction(r[k]["exact"]) for k in ("charged", "booked", "gross", "cost", "floor", "residual")}
                if (v["booked"] != v["gross"]-v["cost"]
                        or v["charged"] != v["booked"]+(1-alpha)*v["floor"]
                        or v["residual"] != (1-alpha)*v["floor"]-v["cost"]):
                    exact_errors.append("path accounting identity")
            exp = policy["expected"]
            flag = exp["charged"]["decimal"] > 0 and exp["booked"]["decimal"] < 0
            if flag != policy["positive_charged_negative_booked"] or Fraction(exp["gross"]["exact"]) != 0:
                exact_errors.append("witness or no-edge identity")
        if (scenario["library_decision"] == "WITNESS") != any(p["positive_charged_negative_booked"] for p in scenario["policies"]):
            exact_errors.append("library verdict")
    check("Exploratory exact adaptive rational integration/accounting mismatches", len(exact_errors), 0, 0)
    by_name = {s["name"]: {p["policy"]: p for p in s["policies"]} for s in scenarios}
    first = by_name["floor_prefix_H2"]["forced_prefix_then_floor_gate"]
    check("Exploratory negative-prefix H2 charged witness", first["expected"]["charged"]["decimal"], .0985, 1e-15)
    check("Exploratory negative-prefix H2 booked harm", first["expected"]["booked"]["decimal"], -.598, 1e-15)
    check("Exploratory negative-prefix raw variance", first["variance"]["charged"]["decimal"], .29560675, 1e-15)
    check("Exploratory negative-prefix residual variance increases", first["variance"]["residual"]["decimal"], .48862675, 1e-15)
    check("Exploratory solvent-prefix H2 charged witness", by_name["solvent_prefix_H2"]["forced_prefix_then_positive_gate"]["expected"]["charged"]["decimal"], .02375, 1e-15)
    clock = exact["terminal_clock_compiler"]
    gamma = exact["gamma"]["decimal"]
    clock_error = max(abs(r["late_expected_charged"]["decimal"]
                          - gamma ** (r["outer_horizon"]-clock["local_episode_length"])
                          * clock["local_expected_charged"]["decimal"])
                      for r in clock["outer_horizon_results"])
    check("Exploratory exact deadline compiler scaling error", clock_error, 0, 1e-12)
    check("Exploratory exact compiler pathwise flags", sum(
        r["residual_late_dominates_early_on_every_coupled_path"]
        and r["latest_raw_and_booked_equal_scaled_local_pathwise"]
        and r["new_active_transition_evaluations_for_placement_compilation"] == 0
        for r in clock["outer_horizon_results"]), len(clock["outer_horizon_results"]), 0)

    full = json.loads(paths[1].read_text(encoding="utf-8"))
    snapshot_hashes("full-clock adaptive", empirical, full, "full_clock_benchmark.py", "EXPLORATORY_FULL_CLOCK_PLAN.md")
    check("Exploratory full-clock benchmark completed", int(full["status"] == "EXPLORATORY_complete"), 1, 0)
    F = pd.read_csv(paths[2])
    check("Exploratory full-clock selected state/method pairs", len(F), 70, 0)
    check("Exploratory full-clock summary selected pairs", full["state_method_pairs"], len(F), 0)
    check("Exploratory full-clock unique state/method pairs", len(F.drop_duplicates(["state_id", "method"])), 70, 0)
    check("Exploratory full-clock distinct market/ledger states", len(F.drop_duplicates(["zone", "equity_multiple"])), 14, 0)
    check("Exploratory full-clock planned confidence events", full["planned_bounds"], 420, 0)
    check("Exploratory full-clock per-event delta", full["delta_per_bound"], .05/420, 1e-15)
    for field, expected in [("certified_raw", 5), ("certified_residual", 11),
                            ("certified_clipped_raw", 5), ("certified_clipped", 38)]:
        check(f"Exploratory full-clock total {field}", int(F[field].sum()), expected, 0)
    check("Exploratory full-clock new certificates outside exact one-action positives",
          int((F.certified_clipped & ~F.exact_one_action_positive).sum()), 0, 0)
    check("Exploratory full-clock certificate flags follow lower bounds", sum(
        int((F[f"certified_{rep}"] != (F[f"{rep}_lower"] > 0)).sum())
        for rep in ("raw", "residual", "clipped_raw", "clipped")), 0, 0)
    expected_methods = {"one_action": (3,3,3,6), "forced_prefix_myopic": (1,3,1,8),
                        "fixed_barrier": (1,3,0,8), "structured_feedback": (0,0,0,8),
                        "random_parameter_search": (0,2,1,8)}
    flags = ("certified_raw", "certified_residual", "certified_clipped_raw", "certified_clipped")
    summaries = {r["method"]: r for r in full["summary_by_method"]}
    for method, expected in expected_methods.items():
        d = F[F.method == method]
        check(f"Exploratory full-clock {method} states", len(d), 14, 0)
        for field, n in zip(flags, expected):
            check(f"Exploratory full-clock {method} {field}", int(d[field].sum()), n, 0)
        check(f"Exploratory full-clock {method} summary agrees", summaries[method]["clipped_residual_certified"], expected[-1], 0)

    selected = json.loads(paths[3].read_text(encoding="utf-8"))
    snapshot_hashes("selected-case adaptive", empirical, selected, "selected_case_followup.py", "SELECTED_CASE_FOLLOWUP_PLAN.md")
    check("Exploratory selected-case confirmation complete", int(selected["status"] == "EXPLORATORY_selected_case_COMPLETE"), 1, 0)
    check("Exploratory selected-case planned paths", selected["planned_confirmation_paths"], 524288, 0)
    check("Exploratory selected-case completed paths", selected["completed_confirmation_paths"], 524288, 0)
    check("Exploratory selected-case cap", selected["selected_cap"], 32768, 0)
    check("Exploratory selected-case single new delta", selected["delta_single_new_test"], .01, 0)
    b = selected["certificate"]
    check("Exploratory selected-case certificate sample size", b["n"], 524288, 0)
    check("Exploratory selected-case certificate delta", b["delta"], .01, 0)
    check("Exploratory selected-case upper endpoint is the frozen cap", b["range_upper"], selected["selected_cap"], 0)
    radius = eb_radius(b["variance"], b["n"], b["range_lower"], b["range_upper"], .01)
    check("Exploratory selected-case independently recomputed EB radius", b["radius"], radius, 1e-12)
    check("Exploratory selected-case independently recomputed lower bound", b["lower"], b["mean"]-radius, 1e-12)
    check("Exploratory selected-case 99% one-sided lower bound (EUR)", b["lower"], 11.092925658493954, 1e-10)
    check("Exploratory selected-case positive verdict follows lower bound", int(selected["positive_witness_confirmed"]), int(b["lower"] > 0), 0)
    duration = selected["duration"]
    cost_max = 4 * np.sum(.99 ** np.arange(duration))
    check("Exploratory selected-case cost-only residual lower endpoint", b["range_lower"], -cost_max, 1e-10)
    panel = pd.read_pickle(GEN / f"data_panel/panel_{selected['zone']}.pkl")
    panel = panel.loc[panel.split == "train"].iloc[200:]
    cap = .25*.25*panel.bal_mw.to_numpy(float)
    price = np.abs(panel.imb.to_numpy(float)-panel.spot.to_numpy(float))
    initial = selected["initial_ledger_eur"]
    one_action = []
    for lev in (.25, 1, 4, 16):
        q = np.minimum(lev*.25, cap)
        cost, magnitude = q, q*price
        h = .5 * np.mean(np.maximum(cost-magnitude-initial, 0)+np.maximum(cost+magnitude-initial, 0))
        one_action.append(.99**(duration-1)*(h-cost.mean()))
    check("Exploratory selected-case all four nonflat isolated actions nonpositive", sum(v <= 0 for v in one_action), 4, 0)
    check("Exploratory selected-case directly integrated best nonflat isolated action", selected["exact_best_nonflat_one_action_value"], max(one_action), 1e-10)
    check("Exploratory selected-case flat-inclusive isolated reference", selected["exact_all_level_one_action_value"], max(0, *one_action), 1e-10)
    first_fee = np.minimum(4, cap).mean()
    check("Exploratory selected-case exact mandatory-action booked bound", selected["expected_booked_return_upper"], -first_fee, 1e-12)
    check("Exploratory selected-case same-action full charging has no positive witness", int(selected["full_charge_same_action_positive"]), 0, 0)
    ranking = pd.read_csv(empirical / "selected_case_selection.csv")
    check("Exploratory selected-case complete eligible development ranking", len(ranking), 25, 0)
    scores = ranking.residual_mean-np.sqrt(2*ranking.residual_variance*np.log(2/.01)/524288)
    check("Exploratory selected-case ranking score identity", np.max(np.abs(scores-ranking.selection_projected_score)), 0, 1e-10)
    check("Exploratory selected-case freezes maximum development score", ranking.selection_projected_score.iloc[0], scores.max(), 1e-12)
    check("Exploratory selected-case policy matches frozen ranking", int(json.loads(ranking.frozen_policy_json.iloc[0]) == selected["frozen_policy"]), 1, 0)
    source_policy = next(r for r in full["selected_policies"]
                         if r["state_id"] == int(ranking.state_id.iloc[0]) and r["method"] == ranking.method.iloc[0])
    check("Exploratory selected-case policy matches original full-clock controller",
          int(source_policy["selected"]["policy"] == selected["frozen_policy"]), 1, 0)
    check("Exploratory selected-case duration matches original full-clock controller",
          duration, source_policy["selected"]["duration"], 0)
    check("Exploratory selected-case search cap maximizes frozen projected score", selected["selected_cap"],
          max(selected["cap_candidates"], key=lambda r:r["projected_confirmation_lower"])["cap"], 0)

    heavy = json.loads(paths[4].read_text(encoding="utf-8"))
    snapshot_hashes("unbounded-price adaptive", heavy_dir, heavy, "bounded_cost_certificate.py", "EXPLORATORY_PLAN.md")
    check("Exploratory unbounded-price laws retained", len(heavy["rows"]), 2, 0)
    check("Exploratory unbounded-price confirmation paths per law", heavy["n_confirmation_per_law"], 262144, 0)
    check("Exploratory unbounded-price separate-family error budget", heavy["joint_error_budget"], .05, 0)
    for row in heavy["rows"]:
        shape = row["pareto_shape"]
        h = .5 * 1.9 ** (1-shape) / (shape-1)
        cap = row["selected_forgiveness_cap"]
        capped = .5 * (1.9 ** (1-shape)-(1.9+cap) ** (1-shape)) / (shape-1)
        check(f"Exploratory Pareto {shape} exact forgiveness mean", row["true_forgiveness_mean"], h, 1e-12)
        check(f"Exploratory Pareto {shape} exact capped mean", row["true_capped_forgiveness_mean"], capped, 1e-12)
        v = row["confirmation"]
        check(f"Exploratory Pareto {shape} complete confirmation sample", v["n"], 262144, 0)
        r = eb_radius(v["sample_variance"], v["n"], v["range"][0], v["range"][1], .025)
        check(f"Exploratory Pareto {shape} bound radius", v["radius"], r, 1e-12)
        check(f"Exploratory Pareto {shape} lower mean identity", v["lower_mean"], v["mean"]-r, 1e-12)
        check(f"Exploratory Pareto {shape} realized lower bound covers capped truth", int(v["lower_mean"] <= capped), 1, 0)
        check(f"Exploratory Pareto {shape} exact target proxy mean", row["target_true_proxy_mean"], .2*h-.1, 1e-12)
        check(f"Exploratory Pareto {shape} fixed target alpha", row["target_alpha"], .8, 0)
        check(f"Exploratory Pareto {shape} target lower-bound identity", row["target_proxy_lower"], .2*v["lower_mean"]-.1, 1e-12)
        check(f"Exploratory Pareto {shape} exact booked harm", row["expected_booked_return"], -.1, 0)
        check(f"Exploratory Pareto {shape} full-booking control", int(row["full_booking_positive_witness"]), 0, 0)
        check(f"Exploratory Pareto {shape} cap transformation", cap, (row["selected_cap"]+.1)/.2, 1e-10)
        check(f"Exploratory Pareto {shape} cap selected on projected search score", row["selected_cap"],
              max(row["search_candidates"], key=lambda r:r["projected_target_lower"])["residual_cap"], 0)
        check(f"Exploratory Pareto {shape} charge interval endpoint", row["certified_alpha_interval"][1], 1-.1/v["lower_mean"], 1e-12)
        check(f"Exploratory Pareto {shape} certified charge endpoint open", int(row["certified_alpha_upper_is_open"]), 1, 0)
        check(f"Exploratory Pareto {shape} one-sided law delta", v["delta"], .025, 0)
        check(f"Exploratory Pareto {shape} unbounded support declared", int(row["price_support_upper"] is None), 1, 0)
        check(f"Exploratory Pareto {shape} variance classification", int(row["gross_variance_finite"]), int(shape > 2), 0)
        check(f"Exploratory Pareto {shape} target witness follows lower bound", int(row["target_positive_witness_confirmed"]), int(row["target_proxy_lower"] > 0), 0)
    rows = {r["pareto_shape"]: r for r in heavy["rows"]}
    check("Exploratory infinite-variance target witness confirmed", int(rows[1.5]["target_positive_witness_confirmed"]), 1, 0)
    check("Exploratory negative-target control not declared witness", int(rows[2.5]["target_positive_witness_confirmed"]), 0, 0)
    check("Exploratory negative-target control has negative exact value", int(rows[2.5]["target_true_proxy_mean"] < 0), 1, 0)


def predictable_accounting_checks() -> None:
    """Read-only independent accounting/certificate checks; no new paths or training."""
    folder = REPO / "exploratory_novelty_2026-10-16/predictable_verification"
    source = folder / "verify_predictable.py"
    if not source.exists():
        print("[info] optional predictable-accounting verification is not shipped")
        return
    import runpy
    try:
        module = runpy.run_path(str(source), run_name="read_only_predictable_verification")
        result = module["audit_all"]()
    except Exception as exc:
        name = f"Exploratory predictable-accounting independent verification: {type(exc).__name__}: {exc}"
        print(f"[FAIL] {name}")
        FAILS.append(name)
        return
    check("Exploratory predictable-accounting independent audit passes", int(result["status"] == "PASS"), 1, 0)
    check("Exploratory predictable-accounting independent assertions at least 18000", int(result["assertions"] >= 18000), 1, 0)
    check("Exploratory predictable-accounting source/plan/data hash checks", result["source_hash_checks"], 25, 0)
    heldout, unbounded, fixed, matched = (result[k] for k in ("heldout", "unbounded", "fixed_policy", "matched_multistep"))
    check("Exploratory heldout all lower bounds checked", heldout["complete_bounds"], 720, 0)
    check("Exploratory heldout distinct declared cases", heldout["distinct_cases"], 24, 0)
    check("Exploratory heldout direct exact-one-action positive cases", heldout["positive_exact_one_action_cases"], 12, 0)
    for cohort, expected in (("selected", (7, 17, 24, 24, 36)), ("neutral", (5, 18, 24, 24, 33))):
        for (method, actual), target in zip(heldout["cohorts"][cohort]["counts"].items(), expected):
            check(f"Exploratory heldout {cohort} {method} certificates of72", actual, target, 0)
        paired = heldout["cohorts"][cohort]["paired_vs_predictable"]["clipped_mean_cost_residual"]
        check(f"Exploratory heldout {cohort} predictable-only versus mean-cost residual", paired["predictable_only"], 12 if cohort == "selected" else 9, 0)
        check(f"Exploratory heldout {cohort} mean-cost-only versus predictable", paired["comparator_only"], 0, 0)
    check("Exploratory heldout shared unique simulation transitions", heldout["actual_unique_transitions"], 473948288, 0)
    check("Exploratory heldout conditional queries including outside rollouts", heldout["conditional_oracle_points"], 533725901, 0)
    check("Exploratory heldout queries outside recorded rollouts", heldout["outside_rollout_oracle_points"], 285, 0)
    check("Exploratory heldout unchanged numerical core and plan across retry", int(heldout["numerical_kernel_and_plan_identical_across_retry"]), 1, 0)
    check("Exploratory unbounded-law all lower bounds checked", unbounded["complete_bounds"], 32, 0)
    check("Exploratory unbounded-law declared cases", unbounded["cases"], 16, 0)
    check("Exploratory unbounded-law both methods positive cases", len(unbounded["positive_case_ids"]), 3, 0)
    check("Exploratory unbounded-law identical sign discoveries", int(unbounded["methods_same_discoveries"]), 1, 0)
    check("Exploratory unbounded-law analytic negative controls", unbounded["negative_controls"], 8, 0)
    check("Exploratory unbounded-law predictable variance larger counterexamples", len(unbounded["compensator_variance_worse_case_ids"]), 4, 0)
    check("Exploratory unbounded-law logical transitions", unbounded["logical_transitions"], 35651584, 0)
    check("Exploratory unbounded-law active transitions", unbounded["active_transitions"], 5695585, 0)
    check("Exploratory unbounded-law scalar-only software replay", int(unbounded["identical_stream_scalar_only_replay"]), 1, 0)
    check("Exploratory fixed-policy all bounds checked", fixed["complete_bounds"], 45, 0)
    check("Exploratory fixed-policy confirmation whole paths", fixed["n_confirmation"], 32768, 0)
    for method, actual in fixed["counts"].items():
        check(f"Exploratory fixed-policy {method} certificates of9", actual, 3 if method == "predictable" else 0, 0)
    for positive in fixed["positive_certificates"]:
        expected = {"adaptive": 8.064497671686272, "final16_T16": .3259685988713632, "final16_T48": 4.542794676456023}[positive["policy"]]
        check(f"Exploratory fixed-policy {positive['policy']} predictable lower bound", positive["lower"], expected, 1e-10)
    check("Exploratory fixed-policy active transitions", fixed["actual_unique_active_transitions"], 22487040, 0)
    check("Exploratory fixed-policy conditional queries", fixed["conditional_oracle_points"], 20829763, 0)
    check("Exploratory fixed-policy confirmation separate-family delta", fixed["confirmation_delta"], .05/45, 0)
    check("Exploratory fixed-policy search-only heuristic definition-time delta", fixed["search_heuristic_delta"], .05/720, 0)
    check("Exploratory matched multi-step complete policy bounds", matched["complete_policy_bounds"], 9, 0)
    check("Exploratory matched multi-step source hashes", matched["checked_source_hashes"], 7, 0)
    check("Exploratory matched multi-step adaptive and simple delayed witness", int(matched["positive_certificates"] == ["frozen_adaptive", "openloop16_T48"]), 1, 0)
    check("Exploratory independent accounting audit uses no new simulation", int(result["no_new_simulation"]), 1, 0)


def sample_only_native_checks() -> None:
    """Reproduce exported learned-policy bounds; no native calls or model fitting."""
    import runpy
    folder = REPO / "exploratory_novelty_2026-10-16/sample_only_learned_audit"
    if not folder.exists():
        print("[info] optional sample-trajectory learned-policy audit is not shipped")
        return
    primary = folder / "results/run_001"
    auxiliary = folder / "results/run_001_new_contract"
    if not (primary / "MANIFEST.json").exists() or not (auxiliary / "MANIFEST.json").exists():
        print("[info] exploratory native audit not yet fully exported; no partial endpoint checked")
        return
    try:
        first = runpy.run_path(str(primary / "reproduce_public_native_audit.py"),
                              run_name="read_only_native_public_reproduction")["verify_directory"](primary)
        second = runpy.run_path(str(auxiliary / "reproduce_public_new_contract.py"),
                               run_name="read_only_new_contract_public_reproduction")["verify_directory"](auxiliary)
        exact = json.loads((folder / "exact_sample_only_checks.json").read_text(encoding="utf-8"))
        for filename, key in (("exact_sample_only_checks.py", "script_sha256"),
                              ("SAMPLE_ONLY_THEORY.md", "theory_sha256")):
            check(f"Exploratory sample-only exact-check source {filename}",
                  int(hashlib.sha256((folder / filename).read_bytes()).hexdigest() == exact[key]), 1, 0)
    except Exception as exc:
        name = f"Exploratory native public reproduction: {type(exc).__name__}: {exc}"
        print(f"[FAIL] {name}")
        FAILS.append(name)
        return
    check("Exploratory sample-only exact floor cases", exact["floor_certificate_cases"], 864, 0)
    check("Exploratory sample-only exact terminal cases", exact["terminal_certificate_cases"], 81, 0)
    for label, report, bounds, paths in (("primary", first, 520, 12800),
                                        ("new-claim component", second, 240, 10240)):
        check(f"Exploratory native {label} independent public arithmetic passes", int(report["status"] == "PASS"), 1, 0)
        check(f"Exploratory native {label} complete confidence limits", report["bounds_recomputed"], bounds, 0)
        check(f"Exploratory native {label} existing policies", report["policies"], 20, 0)
        check(f"Exploratory native {label} policy/profile cases", report["cases"], 40, 0)
        check(f"Exploratory native {label} independent whole paths", report["whole_paths"], paths, 0)
        for field in ("native_simulator_calls", "actor_inference", "model_fitting"):
            check(f"Exploratory native {label} verification {field}", report[field], 0, 0)
    check("Exploratory native all cases have analytic strict financial booked harm",
          first["strict_booked_harm_cases"], 40, 0)
    methods = ("raw", "capped_raw", "capped_residual", "paired", "ridge_control", "mlp_control")
    for alpha in ("0.0", "1.0"):
        for method in methods:
            check(f"Exploratory native primary alpha{alpha} {method} certificates of40",
                  first["method_counts"][alpha][method], 0, 0)
            for profile in ("strict", "vulnerable"):
                check(f"Exploratory native primary {profile} alpha{alpha} {method} certificates of20",
                      first["profile_counts"][profile][alpha][method], 0, 0)
    for method in methods:
        check(f"Exploratory native prepaid-excluded {method} certificates of40",
              second["method_counts"][method], 0, 0)
        for profile in ("strict", "vulnerable"):
            check(f"Exploratory native prepaid-excluded {profile} {method} certificates of20",
                  second["profile_counts"][profile][method], 0, 0)
    check("Exploratory native positive cost lower limits", first["positive_cost_limits"], 40, 0)
    selection = json.loads((primary / "selection.json").read_text(encoding="utf-8"))
    learned = [row for row in selection if row["method"] in ("ridge_control", "mlp_control")]
    check("Exploratory native learned choices all select zero scale",
          int(len(learned) == 160 and all(row["choice"]["beta"] == 0 for row in learned)), 1, 0)
    check("Exploratory native learned choices all select zero budget",
          int(len(learned) == 160 and all(row["choice"]["budget"] == 0 for row in learned)), 1, 0)
    import gzip
    with gzip.open(primary / "paths.jsonl.gz", "rt", encoding="utf-8") as stream:
        paths = [json.loads(line) for line in stream if line.strip()]
    for field in ("forgiveness", "paired_forgiveness"):
        check(f"Exploratory native all 12800 primary paths have zero {field}",
              int(len(paths) == 12800 and all(row[field] == 0 for row in paths)), 1, 0)

    cash_dir = folder / "native_validation/cash_inflow_budget"
    cash = json.loads((cash_dir / "exact_cash_inflow_budget_checks.json").read_text(encoding="utf-8"))
    check("Exploratory cash-inflow exact-check source hash",
          int(hashlib.sha256((cash_dir / "exact_cash_inflow_budget_checks.py").read_bytes()).hexdigest()
              == cash["source_sha256"]), 1, 0)
    check("Exploratory cash-inflow exact consistency checks pass",
          int(cash["status"] == "PASS EXACT TOY CHECKS"), 1, 0)
    for field, expected in (("nonnegative_step_paths", 4914), ("capped_step_checks", 9828),
                            ("event_credit_paths", 800), ("event_paths_with_negative_actual_cash", 116),
                            ("ordered_fair_floor_laws", 36), ("partial_charge_counterexamples", 9)):
        check(f"Exploratory cash-inflow {field}", cash["counts"][field], expected, 0)
    for field in ("native_queries", "endpoint_reads", "new_confidence_limits"):
        check(f"Exploratory cash-inflow exact checks {field}", cash[field], 0, 0)


def main() -> int:
    print("== Part 1: pre-registration hashes")
    check_hashes(GEN)
    check_hashes(LIAB)
    check_hashes_n()
    check_hashes_n(STUDY_O, "O")
    check_hashes_n(STUDY_P, "P")
    check_hashes_n(STUDY_Q, "Q")
    check_hashes_n(STUDY_S, "S")
    print("\n== Part 2: headline numbers")
    headlines()
    study_s_headlines()
    headlines_new()
    counts_and_learnability()
    headlines_q()
    cost_baseline_headlines()
    adaptive_witness_headlines()
    study_r_integrity()
    predictable_accounting_checks()
    sample_only_native_checks()
    if REVISED:
        print(f"\nnote: {len(REVISED)} rebuilt panel(s) differ from the registered ones because eSett revised the source data "
              f"after the study (see THIRD_PARTY_LICENSES.md, 'eSett data'): {REVISED}")
    print("\nall checks pass" if not FAILS else f"\n{len(FAILS)} check(s) FAILED: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
