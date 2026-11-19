"""Verify pre-registration hashes of Studies A-K, N-Z, RS, EXT and FT and recompute the paper's headline numbers from the shipped
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
import subprocess
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
STUDY_T = ACC / "study_t"
STUDY_U = ACC / "study_u"
STUDY_V = ACC / "study_v"
STUDY_W = ACC / "study_w"
STUDY_X = ACC / "study_x"
STUDY_Y = ACC / "study_y"
STUDY_Z = ACC / "study_z"
STUDY_ZB = ACC / "study_zb"
STUDY_RS = ACC / "study_rs"
EXPL_1105 = REPO / "exploratory_claude_2026-11-05"
EXPL_1106 = REPO / "exploratory_claude_2026-11-06"
STUDY_EXT = ACC / "study_ext"
STUDY_FT = ACC / "study_ft"
NOT_SHIPPED = {"data_panel_j/panel_FI.pkl", "data_panel_j/panel_NO2.pkl"}   # derived from eSett data (not redistributed)
FAILS: list[str] = []
REVISED: list[str] = []


def check_hashes(folder: Path) -> None:
    recorded: dict[str, set[str]] = defaultdict(set)
    for number, line in enumerate((folder / "PREREGISTRATION_HASHES.txt").read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        entry = re.fullmatch(r"\S+\s+([0-9a-f]{64})\s+\*(\S+)(?:\s+.*)?", line)
        if entry is None:
            name = f"{folder.name}/PREREGISTRATION_HASHES.txt:{number}"
            print(f"[FAIL] {name}: malformed hash entry")
            FAILS.append(name)
            continue
        recorded[entry[2]].add(entry[1])
    for rel, hashes in sorted(recorded.items()):
        # Study C's manifest predates its move into scaling; preserve the recorded digest.
        shipped_rel = "scaling/aggregate_c.py" if folder.name == LIAB.name and rel == "aggregate_c.py" else rel
        p = folder / shipped_rel
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
        location = f" (shipped as {shipped_rel})" if shipped_rel != rel else ""
        print(f"[{'ok' if ok else 'FAIL'}] {folder.name}/{rel}{location}")
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
        digests = {hashlib.sha256(data).hexdigest()}
        if log:
            cut = data.index(b"\n## After hashing")
            digests = {hashlib.sha256(data[:cut]).hexdigest()}                       # the registered file ended with one newline
            # Study W was hashed with the "After hashing" heading line already in the file; accept that prefix as well
            digests.add(hashlib.sha256(data[: data.index(b"\n", cut + 1) + 1]).hexdigest())
        ok = digest in digests
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


def study_t_headlines() -> None:
    """Study T (twin-identified reward, ORPO baseline): headline numbers recomputed from the registered run table."""
    df = pd.read_csv(STUDY_T / "runs_t.csv")
    check("T runs (Part A, of 90)", float((df.setting == "S_DK1").sum()), 90, 0)
    check("T runs (Part B, of 80)", float(df.setting.isin(["T_DK2", "T_DK1M"]).sum()), 80, 0)
    m = df.groupby(["setting", "arm"]).mean(numeric_only=True)
    for st, arm, x, e in (("S_DK1", "PLAIN", 0.600, 1.98), ("S_DK1", "ORACLE", 0.025, 20.46), ("S_DK1", "SPR", 0.0, 19.40),
                          ("S_DK1", "TIR", 0.025, 21.34), ("S_DK1", "OM1", 0.0, 7.70), ("S_DK1", "OM0.05", 0.0, 7.70),
                          ("S_DK1", "OM0.02", 0.287, 0.94), ("S_DK1", "OM0.01", 0.537, -0.07),
                          ("T_DK2", "PLAIN", 0.394, 21.85), ("T_DK2", "ORACLE", 0.0, 32.66), ("T_DK2", "SPR", 0.0, 33.34), ("T_DK2", "TIR", 0.013, 33.97),
                          ("T_DK1M", "PLAIN", 0.388, 6.72), ("T_DK1M", "ORACLE", 0.0, 11.63), ("T_DK1M", "SPR", 0.0, 13.45), ("T_DK1M", "TIR", 0.0, 13.25)):
        check(f"T {st} {arm}: X", m.loc[(st, arm), "X"], x, 0.0006)
        check(f"T {st} {arm}: booked edge (% K0)", m.loc[(st, arm), "E"], e, 0.006)
    supported = (STUDY_T / "VERDICTS_T.md").read_text(encoding="utf-8").count("| SUPPORTED |")
    check("T supported hypotheses (of 16)", float(supported), 16, 0)


def study_u_headlines() -> None:
    """Study U (twin-based rewards under settlement lag and terminal liability): headline numbers from the registered run table."""
    df = pd.read_csv(STUDY_U / "runs_u.csv")
    check("U runs (of 80)", float(len(df)), 80, 0)
    m = df.groupby("arm").X.mean()
    for arm, x in (("ZFG", 16.00), ("ZBK", 1.18), ("ZSPR", 1.45), ("ZTIR", 1.21), ("TFG", 10.84), ("TBK", 1.59), ("TSPR", 0.79), ("TTIR", 0.38)):
        check(f"U {arm}: mean X", m[arm], x, 0.006)
    supported = (STUDY_U / "VERDICTS_U.md").read_text(encoding="utf-8").count("| SUPPORTED |")
    check("U supported hypotheses (of 6)", float(supported), 6, 0)


def study_v_headlines() -> None:
    """Study V (ruin penalty, mean-variance reward, pooled-magnitude reward, ledger-free audit): headline numbers from the registered run table."""
    df = pd.read_csv(STUDY_V / "runs_v.csv")
    check("V runs (of 144)", float(len(df)), 144, 0)
    m = df.groupby(["setting", "arm"]).mean(numeric_only=True)
    for st, arm, x, e in (("S_DK1", "PLAIN", 0.781, -1.79), ("S_DK1", "ORACLE", 0.000, 20.89), ("S_DK1", "PMR", 0.062, 18.00),
                          ("S_DK1", "RUIN1", 0.336, 11.74), ("S_DK1", "RUIN5", 0.000, 18.97), ("S_DK1", "RUIN25", 0.000, 17.63),
                          ("S_DK1", "MV0.05", 0.000, 8.59), ("S_DK1", "MV0.2", 0.000, 4.48), ("S_DK1", "MV1", 0.031, 1.06),
                          ("S_DK1+Z", "PLAIN", 0.914, -17.59), ("S_DK1+Z", "ORACLE", 0.000, 19.84), ("S_DK1+Z", "TIR", 0.000, 20.79),
                          ("S_DK1+Z", "PMR", 0.000, 22.00), ("S_DK1+Z", "RUIN1", 0.641, -2.35), ("S_DK1+Z", "RUIN5", 0.000, 17.19),
                          ("S_DK1+Z", "RUIN25", 0.000, 4.18), ("T_DK2", "PMR", 0.000, 35.12), ("T_DK1M", "PMR", 0.000, 15.07)):
        check(f"V {st} {arm}: X", m.loc[(st, arm), "X"], x, 0.0006)
        check(f"V {st} {arm}: booked edge (% K0)", m.loc[(st, arm), "E"], e, 0.006)
    pmr = df[df.arm == "PMR"]
    check("V PMR agents (of 32)", float(len(pmr)), 32, 0)
    check("V PMR agents with X > 0", float((pmr.X > 0).sum()), 1, 0)
    a = df[df.setting.isin(["S_DK1", "S_DK1+Z"])]
    check("V audit agents", float(len(a)), 128, 0)
    check("V audit: max TIR error, edge market", a.err_tir_edge.max(), 0.385, 0.0006)
    check("V audit: max TIR error, placebo market", a.err_tir_plac.max(), 0.129, 0.0006)
    g = a[a.X >= 0.25]
    check("V audit: agents with X >= 0.25", float(len(g)), 28, 0)
    check("V audit: median reported-return error, gamblers", g.err_reported_plac.median(), 68.25, 0.006)
    supported = (STUDY_V / "VERDICTS_V.md").read_text(encoding="utf-8").count("| SUPPORTED |")
    check("V supported hypotheses (of 17; VA3, VA5, VC6 not supported)", float(supported), 14, 0)


def study_w_headlines() -> None:
    """Study W (ledger-free choice of the ruin-penalty strength): headline numbers from the registered run table."""
    df = pd.read_csv(STUDY_W / "runs_w.csv")
    check("W runs (of 48)", float(len(df)), 48, 0)
    m = df.groupby(["setting", "arm"]).mean(numeric_only=True)
    for st, arm, x, e, t in (("T_DK2", "RUIN1", 0.078, 30.85, 30.84), ("T_DK2", "RUIN5", 0.000, 33.27, 33.27), ("T_DK2", "RUIN25", 0.000, 30.58, 30.58),
                             ("T_DK1M", "RUIN1", 0.000, 11.22, 11.22), ("T_DK1M", "RUIN5", 0.000, 10.03, 10.03), ("T_DK1M", "RUIN25", 0.000, 7.32, 7.32)):
        check(f"W {st} {arm}: X", m.loc[(st, arm), "X"], x, 0.0006)
        check(f"W {st} {arm}: booked edge (% K0)", m.loc[(st, arm), "E"], e, 0.006)
        check(f"W {st} {arm}: TIR-audited edge (% K0)", m.loc[(st, arm), "TA"], t, 0.006)
    for st, best in (("T_DK2", "RUIN5"), ("T_DK1M", "RUIN1")):                       # audit-selected = booked-best = report-selected
        s = m.loc[st]
        check(f"W {st}: audit-selected strength is {best}", float(s.TA.idxmax() == best), 1, 0)
        check(f"W {st}: booked-best strength is {best}", float(s.E.idxmax() == best), 1, 0)
        check(f"W {st}: report-selected strength is {best} (W4 not supported)", float(s.R.idxmax() == best), 1, 0)
    check("W audit: max TIR error, edge market", df.err_tir_edge.max(), 0.077, 0.0006)
    check("W audit: max TIR error, placebo market", df.err_tir_plac.max(), 0.063, 0.0006)
    supported = (STUDY_W / "VERDICTS_W.md").read_text(encoding="utf-8").count("| SUPPORTED |")
    check("W supported decisions (of 6; W4 not supported)", float(supported), 5, 0)


def study_x_headlines() -> None:
    """Study X (twin rewards under a two-write floor): headline numbers recomputed from the registered run table."""
    df = pd.read_csv(STUDY_X / "runs_x.csv")
    check("X runs (of 32)", float(len(df)), 32, 0)
    m = df.groupby("arm").mean(numeric_only=True)
    for arm, x, e, ez in (("PLAIN", 0.836, -13.37, -25.83), ("ORACLE", 0.000, 18.46, 18.56), ("SPR", 0.000, 20.66, 21.37), ("TIR", 0.000, 18.19, 18.50)):
        check(f"X {arm}: X", m.loc[arm, "X"], x, 0.0006)
        check(f"X {arm}: booked edge (% K0)", m.loc[arm, "E"], e, 0.006)
        check(f"X {arm}: booked edge from zero equity (% K0)", m.loc[arm, "Ez"], ez, 0.006)
    plain = df[df.arm == "PLAIN"]
    check("X reported-reward agents that gambled (X >= 0.25, of 8)", float((plain.X >= 0.25).sum()), 8, 0)
    check("X SPR and TIR agents with any maximal-leverage probe (of 16)", float((df[df.arm.isin(["SPR", "TIR"])].X > 0).sum()), 0, 0)
    honest = df[df.X < 0.25]
    check("X non-gambling agents", float(len(honest)), 24, 0)
    check("X audit: max TIR error of non-gamblers, edge market", honest.err_tir_edge.max(), 0.001, 0.0006)
    check("X audit: max TIR error of non-gamblers, placebo market", honest.err_tir_plac.max(), 0.006, 0.0006)
    g = df[df.X >= 0.25]
    check("X audit: median TIR error of gamblers, edge market", g.err_tir_edge.median(), 1.357, 0.0006)
    check("X audit: median TIR error of gamblers, placebo market", g.err_tir_plac.median(), 0.054, 0.0006)
    check("X audit: median reported-return error of gamblers, placebo market", g.err_reported_plac.median(), 93.747, 0.006)
    supported = (STUDY_X / "VERDICTS_X.md").read_text(encoding="utf-8").count("| SUPPORTED |")
    check("X supported decisions (of 6)", float(supported), 6, 0)


def study_y_headlines() -> None:
    """Study Y (the twin rewards with A2C): headline numbers recomputed from the registered run table."""
    path = STUDY_Y / "runs_y.csv"
    if not path.exists():
        check("Y registered run table present", 0.0, 1, 0)
        return
    df = pd.read_csv(path)
    check("Y runs (of 80)", float(len(df)), 80, 0)
    for (setting, arm), (x, e) in Y_EXPECTED.items():
        sub = df[(df.setting == setting) & (df.arm == arm)]
        check(f"Y {setting} {arm}: X", sub.X.mean(), x, 0.0006)
        check(f"Y {setting} {arm}: booked edge (% K0)", sub.E.mean(), e, 0.006)
    supported = (STUDY_Y / "VERDICTS_Y.md").read_text(encoding="utf-8").count("| SUPPORTED |")
    check(f"Y supported decisions (of 13)", float(supported), Y_SUPPORTED, 0)


Y_EXPECTED: dict = {          # (setting, arm): (mean X, mean booked edge, % K0 per episode), from the registered analysis of 3 November 2026
    ("S_DK1+Z", "PLAIN"): (0.3125, 5.0907), ("S_DK1+Z", "ORACLE"): (0.0, 21.0735), ("S_DK1+Z", "SPR"): (0.0, 22.0062), ("S_DK1+Z", "TIR"): (0.0, 21.0735),
    ("X_DK1+Z", "PLAIN"): (0.3828, 2.6157), ("X_DK1+Z", "ORACLE"): (0.0, 21.0735), ("X_DK1+Z", "SPR"): (0.0, 21.0735), ("X_DK1+Z", "TIR"): (0.0, 22.0062),
    ("S_DK1", "PLAIN"): (0.0, 20.2182), ("S_DK1", "ORACLE"): (0.0, 20.2182)}
Y_SUPPORTED = 12               # of 13; YC1 (acquisition without zero starts) not supported


def study_z_headlines() -> None:
    """Study Z (losses floored per account, Part A, or per isolated position, Part B): headline numbers from the registered run tables,
    the exact checks of the split-account proposition, and the numbers the main text quotes."""
    for folder, csv, n_runs, expected, tag, n_sup in ((STUDY_Z, "runs_z.csv", 72, Z_EXPECTED, "Z", Z_SUPPORTED),
                                                      (STUDY_ZB, "runs_zb.csv", 64, ZB_EXPECTED, "ZB", ZB_SUPPORTED)):
        path = folder / csv
        if not path.exists():
            check(f"{tag} registered run table present", 0.0, 1, 0)
            continue
        df = pd.read_csv(path)
        check(f"{tag} runs (of {n_runs})", float(len(df)), n_runs, 0)
        for arm, (x, e) in expected.items():
            sub = df[df.arm == arm]
            check(f"{tag} {arm}: X", sub.X.mean(), x, 0.0006)
            check(f"{tag} {arm}: booked edge (% K0)", sub.E.mean(), e, 0.006)
        check(f"{tag} runs whose twin bracket held on every audited step and episode", float(df.bracket_ok.sum()), n_runs, 0)
        verdicts = (folder / f"VERDICTS_{tag}.md").read_text(encoding="utf-8")
        check(f"{tag} supported decisions (of 13)", float(verdicts.count("| SUPPORTED |")), n_sup, 0)
    if (STUDY_ZB / "runs_zb.csv").exists():
        b = pd.read_csv(STUDY_ZB / "runs_zb.csv")
        nf = b[b.arm != "FINDER"]
        check("ZB finder: bracket width per edge-market week (points of K0)", b[b.arm == "FINDER"].width_edge.mean(), 429.665, 0.006)
        check("ZB ledger: bracket width per edge-market week (points of K0)", b[b.arm == "LEDGER"].width_edge.mean(), 17.234, 0.006)
        check("ZB largest TIR audit error, all 64 runs (ZIB2 as run)", max(b.err_tir_edge.max(), b.err_tir_plac.max()), 1.446, 0.0006)
        check("ZB largest TIR audit error, the 56 non-finder runs (ERRATA_ZB.md)", max(nf.err_tir_edge.max(), nf.err_tir_plac.max()),
              0.289, 0.0006)
    for folder, script in ((STUDY_Z, "test_envs_z.py"), (STUDY_ZB, "test_envs_zb.py")):    # exact checks, run before any training
        r = subprocess.run([sys.executable, "-B", script], cwd=folder, capture_output=True, text=True)
        check(f"{folder.name}/{script}: exact checks pass", float("ALL PASS" in r.stdout), 1, 0)


Z_EXPECTED: dict = {           # arm: (mean X, mean booked edge, % K0 per episode), from the registered analysis of 3 November 2026
    "LEDGER": (0.0039, 33.8159), "P_REP": (0.9844, -21.5336), "P_RUIN5": (0.1484, 28.3868), "REP": (0.6719, -18.3049),
    "RUINA5": (0.0508, 26.6655), "RUINP25": (0.2969, 11.3474), "RUINP5": (0.6953, -13.5941), "SPR": (0.0273, 39.0274),
    "TIR": (0.0039, 36.0410)}
ZB_EXPECTED: dict = {          # from the registered analysis of 4 November 2026 (01:11)
    "FINDER": (1.0000, -292.9842), "LEDGER": (0.0000, 20.5533), "LIQ5": (0.0938, 7.3802), "REP": (0.6797, -18.9911),
    "RUINP25": (0.3906, 4.0973), "RUINP5": (0.5703, -10.6292), "SPR": (0.0000, 20.3152), "TIR": (0.0000, 21.0643)}
Z_SUPPORTED = 11               # of 13; ZA10 (pooled penalty within 0.10 of the ledger) and ZB3 (park share >= 0.50) not supported
ZB_SUPPORTED = 11              # of 13; ZIB2 (TIR audit within 0.5; text says 56 runs, code tests 64) and ZIB3 not supported


def check_hashes_rs() -> None:
    """Study RS was hashed in Paper1/study_rs_2026-11-04, next to github/, so its manifest lists the reused files as ../github/accounting-options-rl/<path>;
    here they are checked at <path>. Its hasher stopped at the marker quoted in line 3 of the run log, so the registered digest of RUN_LOG_RS.md covers
    only the first 147 bytes (disclosed in the study's README and the supplement). The digest of the intended prefix was recorded after the run."""
    prefix, marker = "../github/accounting-options-rl/", b"## After hashing (not part of the registered prefix)"
    snapshot = json.loads((STUDY_RS / "independent_verification" / "post_launch_integrity.json").read_text(encoding="utf-8"))
    intended_recorded = {r["file"]: r["actual_sha256"] for r in snapshot["integrity"]["records"]}["RUN_LOG_RS.md"]
    for line in (STUDY_RS / "PREREGISTRATION_HASHES_RS.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        if rel == "RUN_LOG_RS.md":
            data = (STUDY_RS / rel).read_bytes()
            registered = data[: data.index(b"\n", data.index(marker)) + 1]
            ok = hashlib.sha256(registered).hexdigest() == digest
            print(f"[{'ok' if ok else 'FAIL'}] study_rs/{rel} (registered digest, covers only the first {len(registered)} bytes: the disclosed defect)")
            if not ok:
                FAILS.append(f"study_rs/{rel}")
            lines = data.splitlines(keepends=True)
            at = [i for i, x in enumerate(lines) if x.rstrip(b"\r\n") == marker]
            intended = b"".join(lines[: at[0] + 1])
            check("RS run log: bytes covered by the registered digest", float(len(registered)), 147, 0)
            check("RS run log: bytes of the intended prefix", float(len(intended)), 3555, 0)
            check("RS run log: intended prefix matches the digest recorded after the run", float(hashlib.sha256(intended).hexdigest() == intended_recorded), 1, 0)
            continue
        p = REPO / rel[len(prefix):] if rel.startswith(prefix) else STUDY_RS / rel
        ok = hashlib.sha256(p.read_bytes()).hexdigest() == digest
        print(f"[{'ok' if ok else 'FAIL'}] study_rs/{rel}")
        if not ok:
            FAILS.append(f"study_rs/{rel}")


RS_EXPECTED: dict = {          # arm: (mean X, mean booked edge, % of the allocation per episode), from the registered analysis of 5 November 2026
    "PLAIN": (0.8255, -100.5667), "ORACLE": (0.0, 8.5164), "SPR": (0.0, 5.8548), "TIR": (0.0, 8.4939), "RUIN5": (0.0938, -9.4750)}


def study_rs_headlines() -> None:
    """Study RS (real signs): the run table, the registered decisions recomputed from it with the registered bootstrap stream, the audit from the raw
    per-run files, and the exploratory numbers the manuscript quotes (exploratory_claude_2026-11-05, re-run here with --print and compared)."""
    df = pd.read_csv(STUDY_RS / "runs_rs.csv")
    check("RS runs (of 80)", float(len(df)), 80, 0)
    check("RS distinct arm-seed pairs", float(len(df[["arm", "seed"]].drop_duplicates())), 80, 0)
    for arm, (x, e) in RS_EXPECTED.items():
        check(f"RS {arm}: X", df[df.arm == arm].X.mean(), x, 0.0006)
        check(f"RS {arm}: booked edge (% K0)", df[df.arm == arm].E.mean(), e, 0.006)
    verdicts = (STUDY_RS / "VERDICTS_RS.md").read_text(encoding="utf-8")
    check("RS supported decisions (of 8)", float(verdicts.count("| SUPPORTED |")), 6, 0)
    check("RS decisions not supported", float(verdicts.count("| NOT SUPPORTED |")), 2, 0)
    rng = np.random.default_rng(20261104)
    eo = df[df.arm == "ORACLE"].sort_values("seed").E.to_numpy()
    check("RS ORACLE edge, registered bootstrap lower bound", float(np.quantile([rng.choice(eo, len(eo)).mean() for _ in range(10_000)], 0.05)), 6.6039, 0.0006)
    for arm, ratio, lb in (("SPR", 0.68747, 0.369859), ("TIR", 0.99736, 0.754601)):
        ea = df[df.arm == arm].sort_values("seed").E.to_numpy()
        boot = [rng.choice(ea, len(ea)).mean() / rng.choice(eo, len(eo)).mean() for _ in range(10_000)]
        check(f"RS {arm}: edge ratio", float(ea.mean() / eo.mean()), ratio, 0.00001)
        check(f"RS {arm}: registered bootstrap lower bound", float(np.quantile(boot, 0.05)), lb, 0.00001)
    recount = json.loads((STUDY_RS / "independent_verification" / "recount.json").read_text(encoding="utf-8"))
    changed = sum(hashlib.sha256((STUDY_RS / "results_rs" / k).read_bytes()).hexdigest() != v for k, v in recount["raw_file_sha256"].items())
    check("RS result files changed since the separate recount (of 80)", float(changed), 0, 0)
    steps = inside = ident = inexact = episodes = outside = 0
    tir_err = {}
    for path in sorted((STUDY_RS / "results_rs").glob("*.json")):
        d = json.loads(path.read_text())
        for mk, tk in (("audit_edge", "test_edge"), ("audit_placebo", "test_placebo")):
            a = d[mk]
            steps += a["steps"]
            inside += a["steps_inside_bracket"]
            ident += a["steps_identified"]
            inexact += a["identified_steps_tir_inexact"]
            b, r, lo = (np.asarray(v, float) for v in (d[tk]["booked_episodes"], a["episodes_reported"], a["episodes_lower"]))
            episodes += len(b)
            outside += int(((b < lo - 1e-6) | (b > r + 1e-6)).sum())
            tir_err.setdefault(d["arm"], []).append(abs(np.mean(a["episodes_tir"]) - b.mean()))
    check("RS audited steps (runner's counters)", float(steps), 10_752_000, 0)
    check("RS steps outside the step bracket (runner's counters)", float(steps - inside), 0, 0)
    check("RS identified steps (runner's counters)", float(ident), 6_106_409, 0)
    check("RS identified steps where TIR is inexact (runner's counters)", float(inexact), 0, 0)
    check("RS audited episodes, recomputed from the stored arrays", float(episodes), 16_000, 0)
    check("RS episodes outside the bracket, recomputed", float(outside), 0, 0)
    audited = 60_800 + 200 * (len(pd.read_csv(STUDY_Z / "runs_z.csv")) + len(pd.read_csv(STUDY_ZB / "runs_zb.csv"))) + episodes
    check("Audited test episodes in the bracket statement (V, W, X, Y; Z; RS)", float(audited), 104_000, 0)
    check("RS largest TIR audit error, ORACLE, SPR and TIR runs (RS-D1)", max(max(tir_err[a]) for a in ("ORACLE", "SPR", "TIR")), 0.00237, 0.00001)
    check("RS largest TIR audit error, all 80 runs", max(max(v) for v in tir_err.values()), 0.92205, 0.00001)
    r = subprocess.run([sys.executable, "-B", str(STUDY_RS / "portable_rs.py"), "test_envs_rs.py"], capture_output=True, text=True, encoding="utf-8")
    check("study_rs/test_envs_rs.py: exact checks pass", float(r.returncode == 0 and "309221 checks pass" in r.stdout), 1, 0)
    for script, stored in (("rs_reanalysis.py", "rs_reanalysis.json"), ("bracket_limits.py", "bracket_limits.json")):
        r = subprocess.run([sys.executable, "-B", str(EXPL_1105 / script), "--print"], capture_output=True, text=True, encoding="utf-8")
        check(f"{script} --print reproduces {stored}", float(r.returncode == 0 and json.loads(r.stdout) == json.loads((EXPL_1105 / stored).read_text(encoding="utf-8"))), 1, 0)
    a = json.loads((EXPL_1105 / "rs_reanalysis.json").read_text(encoding="utf-8"))
    h = a["head_to_head"]
    check("RS exploratory: TIR minus SPR, booked edge", h["difference"], 2.639, 0.001)
    check("RS exploratory: 95% interval, lower end", h["ci95"][0], -0.87, 0.01)
    check("RS exploratory: 95% interval, upper end", h["ci95"][1], 6.29, 0.01)
    check("RS exploratory: Welch p", h["welch_p"], 0.1764, 0.0001)
    check("RS exploratory: paired t p", h["paired_t_p"], 0.1252, 0.0001)
    check("RS exploratory: seeds with TIR above SPR", float(h["seeds_tir_above_spr"]), 12, 0)
    check("RS exploratory: SPR ratio interval contains 0.80", float(a["ratio_ci95"]["SPR"][0] <= 0.80 <= a["ratio_ci95"]["SPR"][1]), 1, 0)
    check("RS exploratory: share of test quarters covered by the 100 windows", a["windows"]["share_of_test_quarters_covered"], 0.948, 0.001)
    check("RS exploratory: overlapping window pairs", float(a["windows"]["overlapping_pairs"]), 340, 0)
    check("RS exploratory: non-overlapping weeks in the test panel", float(a["windows"]["non_overlapping_weeks_in_panel"]), 31, 0)
    check("RS exploratory: test windows containing a gap", float(a["windows"]["windows_with_gap"]), 19, 0)
    check("RS exploratory: TIR lower bound, seeds and test windows", a["lower_bound_5pct"]["two_way"]["TIR"], 0.737, 0.001)
    check("RS exploratory: SPR lower bound, seeds and test windows", a["lower_bound_5pct"]["two_way"]["SPR"], 0.261, 0.001)
    check("RS exploratory: TIR lower bound, seeds and months", a["lower_bound_5pct"]["seeds_and_months"]["TIR"], 0.704, 0.001)
    check("RS exploratory: SPR lower bound, seeds and months", a["lower_bound_5pct"]["seeds_and_months"]["SPR"], -0.018, 0.001)
    ts = a["time_semantics_DK2"]
    check("Panel: missing quarters, training split", float(ts["train"]["missing_quarters"]), 784, 0)
    check("Panel: missing quarters, test split", float(ts["test"]["missing_quarters"]), 14, 0)
    check("Panel: 8-row lag is exactly 2 h, training split", ts["train"]["lag_exactly_2h"], 0.9952, 0.0001)
    check("Panel: 8-row lag is exactly 2 h, test split", ts["test"]["lag_exactly_2h"], 0.9977, 0.0001)
    check("Panel: 672-row windows longer than a week, training split", ts["train"]["windows_longer_than_week"], 0.338, 0.001)
    check("Panel: 672-row windows longer than a week, test split", ts["test"]["windows_longer_than_week"], 0.157, 0.001)
    check("Panel: longest 672-row window, training split (days)", ts["train"]["window_span_max_days"], 10.0, 0.001)
    check("Panel: longest 672-row window, test split (days)", ts["test"]["window_span_max_days"], 7.094, 0.001)
    b = json.loads((EXPL_1105 / "bracket_limits.json").read_text(encoding="utf-8"))
    z = b["carried_deficit"]["ZFG"]
    check("Bracket limits, settlement lag: steps", float(z["steps"]), 88359, 0)
    check("Bracket limits, settlement lag: steps above the upper end", float(z["step_upper_fails"]), 3000, 0)
    check("Bracket limits, settlement lag: steps below the lower end", float(z["step_lower_fails"]), 2973, 0)
    check("Bracket limits, settlement lag: episodes above the upper end (of 300)", float(z["episode_upper_fails"]), 80, 0)
    check("Bracket limits, settlement lag: episodes below the lower end (of 300)", float(z["episode_lower_fails"]), 73, 0)
    t = b["carried_deficit"]["TFG"]
    check("Bracket limits, terminal liability: step and episode failures", float(t["step_upper_fails"] + t["step_lower_fails"] + t["episode_upper_fails"] + t["episode_lower_fails"]), 0, 0)
    m = b["mixed_legs"]
    check("Mixed legs: unequal pairs", float(m["random_actions"]["unequal_pairs"]), 24630, 0)
    check("Mixed legs: TIR wrong on unequal pairs", float(m["random_actions"]["tir_wrong"]), 1595, 0)
    check("Mixed legs: largest TIR error (% K0)", m["random_actions"]["largest_error"], 12.28, 0.01)
    check("Mixed legs: bracket failures", float(m["random_actions"]["bracket_fails"]), 0, 0)
    w = b["liquidation_fee"]["whole_trade"]
    check("Liquidation fee: booked below the bracket's lower end", float(w["booked"] < w["lower"]), 1, 0)
    check("Liquidation fee: positive width", w["width"], 40.0, 1e-6)
    check("Liquidation fee, one tick: lower end with the fee undeclared", b["liquidation_fee"]["one_tick"]["lower_fee_undeclared"], -332.0, 1e-6)
    v = b["width_vs_realized"]
    check("Width against realized forgiveness: finder, width", v["ZB FINDER, edge"]["width"], 429.665, 0.006)
    check("Width against realized forgiveness: finder, realized", v["ZB FINDER, edge"]["realized"], 254.524, 0.006)
    check("Width against realized forgiveness: ledger, realized", v["ZB LEDGER, edge"]["realized"], 7.141, 0.006)
    check("Width against realized forgiveness: largest ratio", max(x["width_over_realized"] for x in v.values()), 2.42, 0.01)
    check("Width against realized forgiveness: smallest ratio", min(x["width_over_realized"] for x in v.values()), 1.35, 0.01)


def engine_twin_headlines() -> None:
    """The sign twin on the engine's own paths (exploratory): hashes, decisions re-derived from the run table, the run table against the
    audit's table, the shipped traces recomputed, the runtime settings, and the numbers the manuscript quotes. Reads stored results; writes nothing."""
    T = EXPL_1105 / "engine_twin"
    for line in (T / "PROTOCOL_HASHES.txt").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        digest, rel = line.split()
        if rel.startswith("engine/"):   # the audit engine built from the private Prototype5 folder
            continue
        src = T / rel if (T / rel).is_file() else REPO / "audit" / rel
        check(f"engine twin: registered digest of {rel}", float(hashlib.sha256(src.read_bytes()).hexdigest() == digest), 1, 0)
    stored = json.loads((T / "results" / "engine_twin_summary.json").read_text(encoding="utf-8"))
    gap = stored["decisions"]["G1_largest_json_minus_campaign_pp"]
    r = subprocess.run([sys.executable, "-B", str(T / "summarize_engine_twin.py"), "--results", str(T / "results"), "--jobs", str(T / "JOBS.txt"), "--audit-gap", repr(gap), "--print"],
                       capture_output=True, text=True, encoding="utf-8")
    check("engine twin: decisions re-derived from the run table", float(r.returncode == 0 and json.dumps(json.loads(r.stdout), sort_keys=True) == json.dumps(stored, sort_keys=True)), 1, 0)
    d = stored["decisions"]
    for key, want in (("runs", 93), ("bound", 54), ("comparison", 39), ("cap_on", 22), ("G2_max_parity_dkk", 0), ("G2_unknown_writes", 0), ("E2_steps_outside", 0), ("E2_runs_containing", 93),
                      ("reconcile_flags_bound", 53), ("reconcile_flags_comparison", 0), ("E3_flags_bound", 53), ("E3_flags_comparison", 0), ("E4_flags_bound", 53), ("E4_flags_comparison", 1),
                      ("E5_horizon_tir_better", 24), ("E5_horizon_runs", 27), ("E5_percent_spr_better", 27), ("E5_percent_runs", 27), ("E6_other_formula_over_5pp_bound", 53),
                      ("E7_runs_flagged_bound", 53), ("E7_large_tir_steps", 57173), ("E7_large_tir_flagged", 57120)):
        check(f"engine twin: {key}", float(d[key]), want, 0)
    for key, want in (("E1", True), ("E2", True), ("E3", True), ("E4", True), ("E5", False), ("E6", True), ("E7", False)):
        check(f"engine twin: decision {key}", float(d[key] == want), 1, 0)
    check("engine twin: G1, largest difference from the audit's table below 1e-12 points", float(gap < 1e-12), 1, 0)
    check("engine twin: E1, smallest credit above -1e-3 DKK", float(d["E1_min_real_credit_dkk"] > -1e-3), 1, 0)
    check("engine twin: E6, largest direct-calculation error below 1e-5 points", float(d["E6_max_direct_error_pp_no_liquidation"] <= 1e-5), 1, 0)
    check("engine twin: E7, share of large-error steps flagged", d["E7_share"], 0.999, 0.0005)
    runs = pd.read_csv(T / "results" / "engine_twin_runs.csv")
    audit = runs[~runs.variant.str.endswith("_capon")]
    camp = pd.read_csv(REPO / "results" / "campaign" / "results_long.csv")
    merged = audit.merge(camp, on=["variant", "agent", "seed", "region"], how="left", validate="one_to_one")
    check("engine twin: every audit run is in the audit's table", float(merged.sleeve_return_pct.notna().sum()), 93, 0)
    check("engine twin: run table against the audit's table, largest difference in reported return below 1e-3 points", float((merged.reported_pct - merged.sleeve_return_pct).abs().max() < 1e-3), 1, 0)
    check("engine twin: floor-bound jobs are the audit's floor-bound runs", float(int(merged.sleeve_cash_floor_hit.sum())), 54, 0)
    check("engine twin: steps over the 93 runs", float(audit.steps.sum()), 3655923, 0)
    check("engine twin: largest direct-calculation error over all 93 runs below 1e-4 points", float(audit.direct_error_pp.abs().max() < 1e-4), 1, 0)
    check("engine twin: TIR estimate outside the bracket, steps", float(audit.steps_tir_outside.sum()), 0, 0)
    check("engine twin: steps where the pair differs", float(audit.steps_identified.sum()), 1641606, 0)
    check("engine twin: of which TIR wrong", float(audit.steps_tir_wrong.sum()), 490437, 0)
    groups = {g["group"]: g for g in stored["groups"]}
    v, p = groups["floor bound, volume payoff"], groups["floor bound, percent payoff"]
    for label, got, want, tol in (("volume, median booked", v["booked"], -1935.8, 0.05), ("volume, median width", v["width"], 2877.8, 0.05), ("volume, TIR error", v["tir_error"], 43.1, 0.05), ("volume, TIR largest", v["tir_error_max"], 389.8, 0.05),
                                  ("volume, SPR error", v["spr_error"], 617.7, 0.05), ("volume, SPR largest", v["spr_error_max"], 1409.6, 0.05), ("volume, other formula", v["other_error"], 1924.6, 0.05),
                                  ("volume, width over credit, smallest", v["width_over_credit_min"], 1.11, 0.005), ("volume, width over credit, largest", v["width_over_credit_max"], 2.39, 0.005),
                                  ("percent, TIR error", p["tir_error"], 607.8, 0.05), ("percent, TIR largest", p["tir_error_max"], 1903.6, 0.05), ("percent, SPR error", p["spr_error"], 10.9, 0.05), ("percent, SPR largest", p["spr_error_max"], 254.8, 0.05),
                                  ("percent, other formula", p["other_error"], 838.4, 0.05), ("percent, width over credit, smallest", p["width_over_credit_min"], 1.91, 0.005), ("percent, width over credit, largest", p["width_over_credit_max"], 2.04, 0.005)):
        check(f"engine twin: {label}", got, want, tol)
    bound_rows = audit[audit.name.isin(merged[merged.sleeve_cash_floor_hit == True].name)]  # noqa: E712
    for label, mode, want in (("volume", "horizon", -805.1), ("percent", "percent", -1174.9)):
        g = bound_rows[bound_rows["mode"] == mode]
        check(f"engine twin: {label}, median lower end minus booked return", (g.lower_pct - g.booked_pct).median(), want, 0.05)
    capon = {(c["label"], c["zone"]): c for c in stored["cap"]}
    for zone, rep, med, booked, rule, floor in (("DK1", -38.5, -45.4, -41.3, -7.9, 1), ("DK2", -45.0, -53.1, -50.3, -13.7, 1)):
        c = capon[("cap on", zone)]
        check(f"cap on, {zone}: MAPPO reported mean", c["marl_reported_mean"], rep, 0.05)
        check(f"cap on, {zone}: MAPPO reported median", c["marl_reported_median"], med, 0.05)
        check(f"cap on, {zone}: MAPPO booked mean", c["marl_booked_mean"], booked, 0.05)
        check(f"cap on, {zone}: rule", c["anchor_reported"], rule, 0.05)
        check(f"cap on, {zone}: MAPPO runs with a floor binding", float(c["floor_runs"]), floor, 0)
    on = runs[runs.variant.str.endswith("_capon")]
    for zone, region in (("DK1", "original"), ("DK2", "v2")):
        g = on[on.region == region]
        check(f"cap on, {zone}: MAPPO seeds above the rule", float((g[g.agent == "marl"].reported_pct > g[g.agent == "anchor"].reported_pct.mean()).sum()), 3, 0)
    check("cap on: largest TIR audit error", float(on.tir_error_pp.abs().max()), 23.5, 0.05)
    check("cap on: largest SPR audit error", float(on.spr_error_pp.abs().max()), 0.10, 0.005)
    sh = json.loads((T / "results" / "shift_tolerance.json").read_text(encoding="utf-8"))
    for key, want in (("bound_runs_flagged_plain", 53), ("bound_runs_flagged_scaled", 53), ("comparison_runs_flagged_plain", 26), ("comparison_runs_flagged_scaled", 2), ("comparison_steps_flagged_scaled", 2)):
        check(f"engine twin, shift tolerance: {key}", float(sh[key]), want, 0)
    me = json.loads((T / "results" / "mechanisms.json").read_text(encoding="utf-8"))
    check("engine twin, mechanisms: empty-account steps", float(me["volume_empty_steps"]), 9549, 0)
    check("engine twin, mechanisms: SPR's magnitude over the gross, smallest run median", me["volume_min_spr_over_gross"], 0.4999, 0.0001)
    check("engine twin, mechanisms: SPR's magnitude over the gross, largest run median", me["volume_max_spr_over_gross"], 0.5000, 0.0001)
    check("engine twin, mechanisms: both branches credited on TIR's large errors, median run", me["percent_median_share_both_credited"], 1.0, 0.0005)
    check("engine twin, mechanisms: smallest share of TIR's error on closures that met the floor", float(me["percent_min_abs_error_share_on_closure"] >= 0.96), 1, 0)
    # the shipped traces: digests, and the run table recomputed from them
    want_digest = {x.split()[1].lstrip("*"): x.split()[0] for x in (T / "results" / "trace_sha256.txt").read_text(encoding="utf-8").splitlines() if x.strip()}
    check("engine twin: digests listed for all 115 traces", float(len(want_digest)), 115, 0)
    shipped = sorted((T / "traces").glob("*.npz"))
    check("engine twin: shipped traces", float(len(shipped)), 5, 0)
    for p in shipped:
        check(f"engine twin: digest of {p.name}", float(hashlib.sha256(p.read_bytes()).hexdigest() == want_digest[p.name]), 1, 0)
    r = subprocess.run([sys.executable, "-B", str(T / "analyze_engine_twin.py"), "--traces", str(T / "traces"), "--print"], capture_output=True, text=True, encoding="utf-8")
    ok = r.returncode == 0
    if ok:
        for row in json.loads(r.stdout):
            one = runs[runs.name == row["name"]].iloc[0]
            ok &= abs(row["reported_pct"] - one.reported_pct) < 1e-9 and abs(row["booked_pct"] - one.booked_pct) < 1e-9 and abs(row["lower_pct"] - one.lower_pct) < 1e-9 \
                and abs(row["tir_pct"] - one.tir_pct) < 1e-9 and abs(row["spr_pct"] - one.spr_pct) < 1e-9 and row["steps_outside"] == one.steps_outside and row["shift_flagged"] == one.shift_flagged
    check("engine twin: run table recomputed from the five shipped traces", float(ok), 1, 0)
    for name, low, high, label in (("L7_thesis_like__marl__seed7__original", 10.0, 11.0, "off: applied return reaches over 1,000% in a step"),
                                   ("L7_thesis_like_capon__marl__seed7__original", 0.00099, 0.00101, "on: applied return stops at 0.1% per step")):
        z = np.load(T / "traces" / f"{name}.npz", allow_pickle=False)
        cols = [str(c) for c in z["columns"]]
        rows = z["rows"]
        pos, gross = rows[:, cols.index("position_pre")], rows[:, cols.index("gross")]
        held = np.abs(pos) > 1e3
        check(f"engine twin, cap {label}", float(low <= np.abs(gross[held] / pos[held]).max() <= high), 1, 0)
    # the runtime settings: the cap is the one setting that differed from the description
    dump = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in (T / "config_dump").glob("*.json")}
    check("engine twin: protocols with recorded runtime settings", float(len(dump)), 16, 0)
    off = [k for k, v in dump.items() if v["config"]["mtm_apply_price_return_cap"] is False]
    check("engine twin: the percent payoff's cap is off in every audit protocol", float(len(off)), 15, 0)
    check("engine twin: the cap is on only in the cap-on runs", float(dump["L7_thesis_like_capon"]["config"]["mtm_apply_price_return_cap"] is True), 1, 0)
    for key, value in (("mtm_return_model", "percent_capped"), ("market_fee_model", "legacy_notional_fixed"), ("liquidity_participation_cap_fraction", 0.0), ("enable_trading_sleeve_margin", False), ("distribution_rate", 0.1)):
        check(f"engine twin: L7 runs with {key} = {value}", float(dump["L7_thesis_like"]["config"][key] == value), 1, 0)

def study_ext_headlines() -> None:
    """Study EXT (an unmodified public simulator): the vendored source against its manifest, the decisions re-derived from the run table and the stored scan, and the numbers the manuscript quotes."""
    from scipy import stats
    man = json.loads((STUDY_EXT / "vendor" / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    src = [f for f in man["files"] if f["path"].startswith("src/")]
    same = all(hashlib.sha256((STUDY_EXT / "vendor" / f["path"].replace("src/", "")).read_bytes()).hexdigest() == f["sha256"] for f in src)
    check("EXT: vendored source files byte-identical to the manifest of the pinned commit", float(same and len(src) > 0), 1, 0)
    d = pd.read_csv(STUDY_EXT / "runs_ext.csv")
    fresh = pd.DataFrame([json.loads(f.read_text()) for f in sorted((STUDY_EXT / "results_ext").glob("EXT__*.json"))])
    check("EXT: runs (70; the hashed text says 62, see ERRATA_EXT.md)", float(len(d)), 70, 0)
    check("EXT: stored run files", float(len(fresh)), 70, 0)
    g = {k: v for k, v in d.groupby("arm")}
    for arm, n, x, bust, rep, bkd in (("REP", 8, 0.888, 0.606, 50.35, -9.80), ("LEDGER", 8, 0.031, 0.016, 1.13, 0.13), ("SPR", 8, 0.003, 0.009, 2.73, 2.18), ("TIR", 8, 0.000, 0.005, 0.18, -0.14), ("TSR", 4, 0.044, 0.020, 1.42, 0.35),
                                      ("RUIN5", 8, 0.924, 0.560, 67.03, 5.02), ("RUIN25", 8, 0.789, 0.516, 49.21, 1.35), ("RUIN100", 8, 0.183, 0.121, 14.80, 2.64), ("FINDER", 8, 0.993, 0.698, 69.86, -27.12), ("FINDER_DEF", 2, 0.000, 0.000, 1.47, 1.47)):
        check(f"EXT {arm}: runs", float(len(g[arm])), n, 0)
        check(f"EXT {arm}: X", g[arm].X.mean(), x, 0.0006)
        check(f"EXT {arm}: bust share", g[arm].bust.mean(), bust, 0.0006)
        check(f"EXT {arm}: reported, % per episode", g[arm].reported.mean(), rep, 0.006)
        check(f"EXT {arm}: booked, % per episode", g[arm].booked.mean(), bkd, 0.006)

    def mw(a, b, alt):
        return float(stats.mannwhitneyu(a, b, alternative=alt).pvalue)

    L = g["LEDGER"]

    def present(arm):
        return mw(g[arm].X, L.X, "greater") < 0.05 and g[arm].X.mean() >= L.X.mean() + 0.10

    def removes(arm):
        return mw(g[arm].X, g["REP"].X, "less") < 0.05 and g[arm].X.mean() <= L.X.mean() + 0.10

    scan = json.loads((STUDY_EXT / "results_ext" / "scan_ext.json").read_text(encoding="utf-8"))
    d0, d1 = scan["default_0_1"], scan["leverage_10"]
    dec = dict(EA1=present("REP"), EA2=(g["REP"].reported - g["REP"].booked).mean() >= 10, EA3=mw(g["REP"].bust, L.bust, "greater") < 0.05 and g["REP"].bust.mean() >= L.bust.mean() + 0.10, EA4=removes("SPR"), EA5=removes("TIR"), EA6=removes("TSR"),
               EA7=present("RUIN5"), EA8=present("RUIN25"), EA9=removes("RUIN100"), EA10=(g["FINDER"].X >= 0.5).sum() >= 7, EA11=g["FINDER_DEF"].max_width.max() <= 1e-6, EA12=(d.inside == d.episodes).all(), EA13=d.max_err_line.max() <= 0.01,
               EA14=g["REP"].mae_spr.mean() >= 5 and g["REP"].mae_tir.mean() <= 0.01, ES1=d0["exact_within_1e_6"] >= 0.999 * d0["states"] and d1["exact_within_1e_6"] >= 0.999 * d1["states"],
               ES2=d1["jumps_at_true_location"] >= 0.99 * d1["jumps"] and d1["jump_equals_wealth"] >= 0.99 * d1["jumps"], ES3=d0["width_nonzero"] == 0)
    want = {k: k != "EA9" for k in dec}
    check("EXT: decisions re-derived, 16 of 17 supported, EA9 not", float({k: bool(v) for k, v in dec.items()} == want), 1, 0)
    check("EXT: supported decisions in the stored verdicts", float((STUDY_EXT / "VERDICTS_EXT.md").read_text(encoding="utf-8").count("| SUPPORTED |")), 16, 0)
    check("EXT: RUIN100, seeds that kept the gamble (X above 0.1)", float((g["RUIN100"].X > 0.1).sum()), 3, 0)
    check("EXT: RUIN100, largest X among the other seeds", float(g["RUIN100"][g["RUIN100"].X <= 0.1].X.max()), 0.07, 0.005)
    check("EXT: finder, seeds at maximal leverage at half the probes or more", float((g["FINDER"].X >= 0.5).sum()), 8, 0)
    check("EXT: reading's largest audit error, points", float(d.max_err_line.max() < 3e-11), 1, 0)
    check("EXT: SPR's audit error on the reported-reward agents, points", g["REP"].mae_spr.mean(), 45.33, 0.006)
    check("EXT: TIR's audit error on the reported-reward agents below 1e-12 points", float(g["REP"].mae_tir.mean() < 1e-12), 1, 0)
    check("EXT scan: states, both position sets", float(d0["states"] + d1["states"]), 8000, 0)
    check("EXT scan: tail reading exact on all of them", float(d0["exact_within_1e_6"] + d1["exact_within_1e_6"]), 8000, 0)
    check("EXT scan: jumps in the leverage set", float(d1["jumps"]), 3505, 0)
    check("EXT scan: jumps at the true location", float(d1["jumps_at_true_location"]), 3505, 0)
    check("EXT scan: jumps of a size equal to the current wealth", float(d1["jump_equals_wealth"]), 3505, 0)
    check("EXT scan: jumps further along the line in the default set", float(d0["jumps"]), 1881, 0)
    check("EXT scan: width non-zero at t = +-1 in the default set", float(d0["width_nonzero"]), 0, 0)
    check("EXT scan: width non-zero at t = +-1 in the leverage set", float(d1["width_nonzero"]), 27, 0)
    check("EXT scan: median distance to the jump at position 10", d1["median_abs_t_star_by_position"]["10"], 5.46, 0.01)
    check("EXT scan: median distance to the jump at position 1", d1["median_abs_t_star_by_position"]["1"], 49.1, 0.05)


def study_ft_headlines() -> None:
    """Study FT (accountings that a sign query cannot read): the run table against the stored runs, the decisions re-derived with the registered analysis's own functions, in its order and with its generator, and the numbers the manuscript quotes."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("aggregate_ft_verify", STUDY_FT / "aggregate_ft.py")
    ag = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ag)
    df = ag.load(STUDY_FT / "results_ft")
    stored = pd.read_csv(STUDY_FT / "runs_ft.csv")
    check("FT: runs", float(len(df)), 81, 0)
    check("FT: the run table equals the stored runs (booked edge from a deficit, summed)", float(stored.Ed.sum()), float(df.Ed.sum()), 1e-9)
    A, B, D = ({k: df[(df.part == part) & (df.arm == k)] for k in ag.ARMS} for part in "ABC")
    dec = {}
    dec["FA1"] = ag.present(A["REP"], A["LEDGER"], "REP")[0]
    for i, j, arm in (("FA2", "FA3", "SPR"), ("FA4", "FA5", "TIR"), ("FA6", "FA7", "TSR")):
        dec[i] = ag.removes(A[arm], A["REP"], A["LEDGER"], arm)[0]
        dec[j] = ag.keeps(A[arm], A["LEDGER"], arm)[0]
    dec["FB1"] = ag.present(B["REP"], B["LEDGER"], "REP")[0]
    for i, arm in (("FB2", "SPR"), ("FB3", "TIR"), ("FB4", "TSR")):
        dec[i] = ag.removes(B[arm], B["REP"], B["LEDGER"], arm)[0]
    for i, arm in (("FB5", "TSR"), ("FB6", "SPR"), ("FB7", "TIR")):
        dec[i] = ag.keeps(B[arm], B["LEDGER"], arm, col="Ed")[0]
    dec["FD1"] = ag.present(D["REP"], D["LEDGER"], "REP")[0]
    for i, arm in (("FD2", "SPR"), ("FD3", "TIR"), ("FD4", "TSR")):
        dec[i] = ag.removes(D[arm], D["REP"], D["LEDGER"], arm)[0]
    dec["FD5"] = ag.keeps(D["TSR"], D["LEDGER"], "TSR", col="Ed")[0]
    for i, arm in (("FD6", "SPR"), ("FD7", "TIR")):
        dec[i] = ag.loses(D[arm], D["LEDGER"], arm)[0]
    dec["FC1"] = float(max(df.max_tsr_edge.max(), df.max_tsr_deficit.max(), df.max_tsr_placebo.max())) <= 1e-6
    dec["FC2"] = bool(((A["REP"].inside_edge == 100) & (A["REP"].inside_placebo == 100)).all() and all(((A[k].inside_edge == 100) & (A[k].inside_placebo == 100)).all() for k in ag.ARMS))
    want = {k: k not in ("FB4", "FB6", "FB7") for k in dec}
    check("FT: 23 decisions re-derived, 20 supported, FB4, FB6 and FB7 not", float(len(dec) == 23 and {k: bool(v) for k, v in dec.items()} == want), 1, 0)
    check("FT: supported decisions in the stored verdicts", float((STUDY_FT / "VERDICTS_FT.md").read_text(encoding="utf-8").count("| SUPPORTED |")), 20, 0)
    g = stored.groupby(["part", "arm"]).mean(numeric_only=True)
    for part, arm, x, e, ed in (("A", "REP", 0.984, -16.29, -30.64), ("A", "LEDGER", 0.042, 20.76, 20.60), ("A", "SPR", 0.049, 16.64, 16.17), ("A", "TIR", 0.097, 17.49, 17.50), ("A", "TSR", 0.0, 19.53, 19.20),
                                ("B", "REP", 0.333, 7.43, -1.34), ("B", "LEDGER", 0.0, 20.23, 20.41), ("B", "SPR", 0.0, 14.39, 11.24), ("B", "TIR", 0.0, 14.82, 11.79), ("B", "TSR", 0.0, 18.47, 18.81),
                                ("C", "REP", 0.750, -15.42, -27.34), ("C", "LEDGER", 0.0, 16.32, 20.76), ("C", "SPR", 0.0, 0.32, 0.23), ("C", "TIR", 0.0, -0.03, 0.05), ("C", "TSR", 0.0, 16.13, 19.52)):
        check(f"FT {part} {arm}: X", g.loc[(part, arm), "X"], x, 0.0006)
        check(f"FT {part} {arm}: booked edge, funded", g.loc[(part, arm), "E"], e, 0.006)
        check(f"FT {part} {arm}: booked edge, from a deficit", g.loc[(part, arm), "Ed"], ed, 0.006)
    for part, arm, col, ratio in (("A", "SPR", "E", 0.80), ("A", "TIR", "E", 0.84), ("A", "TSR", "E", 0.94), ("B", "SPR", "Ed", 0.55), ("B", "TIR", "Ed", 0.58), ("B", "TSR", "Ed", 0.92), ("B", "SPR", "E", 0.71), ("B", "TIR", "E", 0.73), ("C", "SPR", "Ed", 0.011), ("C", "TIR", "Ed", 0.0024), ("C", "TSR", "Ed", 0.94)):
        check(f"FT {part} {arm}: {col} over the ledger arm's", g.loc[(part, arm), col] / g.loc[(part, "LEDGER"), col], ratio, 0.005 if ratio > 0.1 else 0.0005)
    tsr, led = stored[stored.arm == "TSR"].set_index(["part", "seed"]), stored[stored.arm == "LEDGER"].set_index(["part", "seed"])
    same = float(max((tsr[c] - led.loc[tsr.index, c]).abs().max() for c in ("E", "Ez", "Ed", "X", "placebo_booked")))
    check("FT: TSR's results equal the ledger arm's on each of its seeds, largest difference", same, 0.0, 0)
    check("FT: largest TSR audit error below 2e-12 points", float(max(stored.max_tsr_edge.max(), stored.max_tsr_deficit.max(), stored.max_tsr_placebo.max()) < 2e-12), 1, 0)
    for part, arm, tir, spr in (("B", "LEDGER", 30.77, 30.95), ("C", "LEDGER", 31.26, 31.43), ("B", "TSR", 28.31, 28.45), ("C", "TSR", 29.29, 29.43), ("B", "REP", 58.48, 58.89), ("C", "REP", 91.99, 93.15)):
        check(f"FT {part} {arm}: TIR audit error from a deficit", g.loc[(part, arm), "mae_tir_deficit"], tir, 0.006)
        check(f"FT {part} {arm}: SPR audit error from a deficit", g.loc[(part, arm), "mae_spr_deficit"], spr, 0.006)
    check("FT B REP: windows with every step inside the bracket, funded", g.loc[("B", "REP"), "inside_edge"], 78.83, 0.01)
    check("FT B REP: windows with every step inside the bracket, from a deficit", g.loc[("B", "REP"), "inside_deficit"], 0.0, 0)
    b = stored[stored.part == "B"]
    check("FT B REP: seeds at X = 0", float((b[b.arm == "REP"].X == 0).sum()), 2, 0)
    check("FT B LEDGER: seeds at +22.0 from a deficit", float((b[b.arm == "LEDGER"].Ed.round(1) == 22.0).sum()), 5, 0)
    check("FT B TIR: smallest and largest seed from a deficit", b[b.arm == "TIR"].Ed.min() + b[b.arm == "TIR"].Ed.max(), 17.8, 0.1)
    check("FT B SPR: smallest and largest seed from a deficit", b[b.arm == "SPR"].Ed.min() + b[b.arm == "SPR"].Ed.max(), 18.9, 0.1)
    r = subprocess.run([sys.executable, "-B", str(STUDY_FT / "test_envs_ft.py")], cwd=STUDY_FT, capture_output=True, text=True, encoding="utf-8")
    check("FT: exact checks of the environments pass", float(r.returncode == 0), 1, 0)
    log = sorted((STUDY_EXT / "results_log").glob("LOG__s*.json"))
    d = [json.loads(f.read_text(encoding="utf-8")) for f in log]
    check("EXT, exploratory default logarithmic reward: seeds", float(len(d)), 8, 0)
    check("EXT, exploratory default logarithmic reward: largest X", max(x["X"] for x in d), 0.0111, 0.0006)
    check("EXT, exploratory default logarithmic reward: largest bust share", max(x["bust"] for x in d), 0.0, 0.0006)
    check("EXT, exploratory default logarithmic reward: mean probed position", float(np.mean([x["mean_abs_probe"] for x in d])), 0.147, 0.006)


def reward_diagram_checks() -> None:
    """The reward diagram: the exact checks of the theory, re-run here in rational arithmetic, and the read-out in the study sleeves (stored; exploratory)."""
    r = subprocess.run([sys.executable, "-B", str(ACC / "twin_reward" / "test_reward_diagram.py")], capture_output=True, text=True, encoding="utf-8")
    total = re.search(r"total (\d+)", r.stdout)
    check("reward diagram: exact checks pass, in rational arithmetic", float(r.returncode == 0 and total is not None), 1, 0)
    if total:
        check("reward diagram: number of exact checks", float(total.group(1)), 88702, 0)
    r = subprocess.run([sys.executable, "-B", str(ACC / "twin_reward" / "test_line_wrapper.py")], capture_output=True, text=True, encoding="utf-8")
    check("reward diagram: the training wrapper's reward equals the booked P&L, transitions unchanged", float(r.returncode == 0 and "ALL PASS" in r.stdout), 1, 0)
    rows = json.loads((EXPL_1106 / "reward_diagram" / "sleeve_readout.json").read_text(encoding="utf-8"))
    check("sleeve read-out: accountings", float(len(rows)), 10, 0)
    for row in rows:
        name = row["name"]
        check(f"sleeve read-out, {name}: the tail reading is exact on every state", float(row["tail_exact"] == row["states"]), 1, 0)
        check(f"sleeve read-out, {name}: the shape is classified correctly on every state", float(row["shape_agree"] == row["states"]), 1, 0)
        if row["read_tried"]:
            check(f"sleeve read-out, {name}: every read-out exact", float(row["read_exact"] == row["read_tried"]), 1, 0)
    by = {row["name"]: row for row in rows}
    check("sleeve read-out: a funded clone misses under isolated margin", float(by["Isolated margin"]["funded_exact"] < by["Isolated margin"]["states"]), 1, 0)
    check("sleeve read-out: and nowhere else", float(all(row["funded_exact"] == row["states"] for row in rows if row["name"] != "Isolated margin")), 1, 0)
    check("sleeve read-out: TIR exact on a carried deficit, per cent of states", 100.0 * by["Carried deficit"]["tir_exact"] / by["Carried deficit"]["states"], 45.9, 0.05)
    check("sleeve read-out: SPR exact on a carried deficit, per cent of states", 100.0 * by["Carried deficit"]["spr_exact"] / by["Carried deficit"]["states"], 31.3, 0.05)
    a = json.loads((EXPL_1106 / "sign_cost" / "results" / "sign_cost_summary.json").read_text(encoding="utf-8"))
    for key, want in (("runs", 93), ("floor_bound", 54), ("comparison", 39), ("sign_cost_flags_bound", 53), ("sign_cost_flags_comparison", 0), ("positive_reported_negative_booked_runs", 19), ("false_profits_certified_negative_without_twin", 13), ("l3_negative_upper", 20), ("six_run_headline_negative_upper", 0)):
        check(f"sign-cost certificate: {key}", float(a[key]), want, 0)
    check("sign-cost certificate: the same flags as the ledger", float(a["same_flags_as_ledger"]), 1, 0)
    d = pd.read_csv(EXPL_1106 / "sign_cost" / "results" / "sign_cost_runs.csv")
    six = d[(d.variant == "L7_minus_percent_mtm") & (d.agent == "marl")]
    l3 = d[(d.variant == "L3_price_taker_no_solvency") & (d.agent == "marl")]
    check("engine: own cost over booked loss, six headline runs, median", float((six.net_cost_pct / -six.booked_pct).median()), 0.972, 0.0005)
    check("engine: own cost over booked loss, L3 MAPPO, median", float((l3.net_cost_pct / -l3.booked_pct).median()), 1.026, 0.0005)
    check("engine: gross positive, of the six headline runs", float((six.gross_pct > 0).sum()), 2, 0)
    check("engine: gross positive, of the 20 L3 MAPPO runs", float((l3.gross_pct > 0).sum()), 11, 0)


def engine_scan_headlines() -> None:
    """The reward diagram on real clones of the engine (exploratory): hashes, decisions re-derived from the run table, the shipped scans recomputed, the numbers the manuscript quotes."""
    T = EXPL_1106 / "engine_scan"
    for line in (T / "PROTOCOL_SCAN_HASHES.txt").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        digest, rel = line.split()
        if rel.endswith("twin_trace.py"):
            src = EXPL_1105 / "engine_twin" / "twin_trace.py"
        elif rel.endswith("run_scarcity_tail_campaign.py"):
            src = REPO / "audit" / "run_scarcity_tail_campaign.py"
        else:
            src = T / rel
        data = src.read_bytes()
        digests = {hashlib.sha256(data).hexdigest()}
        if rel == "RUN_LOG_SCAN.md":                        # append-only after the hash
            cut = data.index(b"\n## After hashing")
            digests.add(hashlib.sha256(data[: data.index(b"\n", cut + 1) + 1]).hexdigest())
        check(f"engine scan: registered digest of {rel}", float(digest in digests), 1, 0)
    R = T / "results"
    runs = pd.read_csv(R / "scan_runs.csv")
    check("engine scan: runs", float(len(runs)), 26, 0)
    want = dict(G1=True, G2=True, S1=False, S2=False, S3=True, S4=True, S5=True, S6=False, S7=True, S8=True, S9=False, S10=True)
    stored = json.loads((R / "scan_verdicts.json").read_text(encoding="utf-8"))
    check("engine scan: stored decisions are those the text reports", float(stored == want), 1, 0)
    V, K, N, P = (runs[runs.group == g] for g in "VKNP")
    K5, K7 = K[K.run.str.contains("seed5001")], K[K.run.str.contains("seed7__")]
    PF = P[~P.run.str.contains("seed42")]
    re_derived = dict(
        G1=bool((runs.g1_steps_equal & (runs.g1_dr <= 1e-3) & (runs.g1_dcash <= 1e-3)).all()), G2=bool((runs.g2_phi1 <= 1e-3).all()),
        S1=bool((pd.concat([V, N]).s1_twin_replay <= 1.0).all()), S2=bool((runs.s2_outside == 0).all()),
        S3=bool(((V.audit_err <= 0.01) & (V.audit_err < V.audit_err_tir) & (V.audit_err < V.audit_err_spr)).all()), S4=bool((V.step_exact_share >= 0.98).all()),
        S5=bool((N.audit_err <= 0.01).all() and (K7.sum_err <= 0.01).all()), S6=bool((K5.sum_err <= 1.0).all()),
        S7=bool(((PF.sum_err < PF.sum_err_tir) & (PF.sum_err <= 2.0 * PF.sum_err_spr)).all()), S8=bool((K.cap_share >= 0.90).all() and (P.cap_share <= 0.05).all()),
        S9=bool(V.flat_hit.sum() >= 0.99 * V.flat_try.sum() and V.flat_try.sum() > 0), S10=bool((V.mean_clones <= 8).all() and (N.mean_clones <= 2.5).all()))
    check("engine scan: decisions re-derived from the run table", float(re_derived == want), 1, 0)
    n = json.loads((R / "scan_numbers.json").read_text(encoding="utf-8"))
    for label, got, value, tol in (("read steps", n["read_steps"], 170326, 0), ("clones", n["clones"], 674523, 0), ("full diagrams", n["diagrams"], 6658, 0),
                                   ("volume, clones per step, smallest", n["volume"]["clones"][0], 2.54, 0.005), ("volume, clones per step, largest", n["volume"]["clones"][1], 5.12, 0.005),
                                   ("volume, stage 3 steps", n["volume"]["stage3"], 0, 0), ("volume, largest audit error", n["volume"]["audit_err"][1], 0.0001, 0.00001),
                                   ("volume, TIR audit error, smallest", n["volume"]["audit_err_tir"][0], 9.1, 0.05), ("volume, TIR audit error, largest", n["volume"]["audit_err_tir"][1], 389.8, 0.05),
                                   ("volume, SPR audit error, smallest", n["volume"]["audit_err_spr"][0], 31.1, 0.05), ("volume, SPR audit error, largest", n["volume"]["audit_err_spr"][1], 1409.6, 0.05),
                                   ("percent, floor bound, error, smallest", n["percent_bound"]["sum_err"][0], 12.9, 0.05), ("percent, floor bound, error, largest", n["percent_bound"]["sum_err"][1], 18.4, 0.05),
                                   ("percent, floor bound, TIR, smallest", n["percent_bound"]["sum_tir"][0], 825.8, 0.05), ("percent, floor bound, TIR, largest", n["percent_bound"]["sum_tir"][1], 1227.9, 0.05),
                                   ("percent, floor bound, SPR, largest", n["percent_bound"]["sum_spr"][1], 19.0, 0.05), ("cap and floor, error, smallest", n["cap_floor"]["sum_err"][0], 4.76, 0.005), ("cap and floor, error, largest", n["cap_floor"]["sum_err"][1], 7.35, 0.005),
                                   ("cap and floor, fallback, smallest", n["cap_floor"]["sum_fb"][0], 0.20, 0.005), ("cap and floor, fallback, largest", n["cap_floor"]["sum_fb"][1], 0.31, 0.005), ("fallback never worse than the better twin by more than", n["fallback_worst_over_best_twin"], 0.0021, 0.0001),
                                   ("cap signature on, smallest", n["cap_share_on"][0], 0.985, 0.0005), ("cap signature on, largest", n["cap_share_on"][1], 0.994, 0.0005), ("cap signature off, largest", n["cap_share_off"][1], 0.006, 0.0005),
                                   ("flat level tried", n["flat"]["tried"], 1198, 0), ("flat level within 1 DKK", n["flat"]["within_1_dkk"], 1040, 0), ("flat level within 1,000 DKK", n["flat"]["within_1000_dkk"], 1186, 0),
                                   ("twin against replay, worst in V and N", n["twin_vs_replay_group_vn_worst"], 82.1, 0.05), ("twin against replay, steps in V and N", n["twin_replay_gap_steps_vn"], 2, 0),
                                   ("bracket, steps outside", n["bracket"]["outside"], 5278, 0), ("bracket, largest miss, DKK", n["bracket"]["largest_dkk"], 484.9, 0.05), ("bracket, misses with a cost gap", n["bracket"]["outside_on_cost_gap"], 5278, 0), ("bracket, misses with each branch's cost", n["bracket"]["outside_with_each_branch_cost"], 0, 0),
                                   ("audit without the declared cost, smallest", n["no_declared_cost_volume"][0], 188.0, 0.05), ("audit without the declared cost, largest", n["no_declared_cost_volume"][1], 881.9, 0.05),
                                   ("funded clone exact", n["funded"]["exact"], 6420, 0), ("funded clone exact, anchor", n["funded"]["anchor_exact"], 128, 0), ("funded clone, anchor steps", n["funded"]["anchor_diagrams"], 366, 0), ("zero-cost clone exact", n["zerocost_exact"], 6133, 0),
                                   ("percent misses, sampled", n["percent_misses"], 458, 0), ("of which the position moves within 8", n["percent_misses_position_moves"], 0, 0)):
        check(f"engine scan: {label}", float(got), float(value), tol)
    # the shipped scans: digests, and the run table's step errors recomputed from them, with the frozen reader
    want_digest = {x.split()[1].lstrip("*"): x.split()[0] for x in (R / "scan_sha256.txt").read_text(encoding="utf-8").splitlines() if x.strip()}
    check("engine scan: digests listed for all 26 scans", float(len(want_digest)), 26, 0)
    shipped = sorted((T / "scans").glob("*.npz"))
    check("engine scan: shipped scans", float(len(shipped)), 5, 0)
    sys.path.insert(0, str(T))
    from reward_diagram import read_diagram
    E0 = 662068965.5172415
    for f in shipped:
        check(f"engine scan: digest of {f.name}", float(hashlib.sha256(f.read_bytes()).hexdigest() == want_digest[f.name]), 1, 0)
        z = dict(np.load(f, allow_pickle=False))
        grid, phi, C = z["read_grid"], z["phi"], z["own_1"]
        idx = {float(t): j for j, t in enumerate(grid)}
        r1, rm = phi[:, idx[1.0]], phi[:, idx[-1.0]]
        tol = 1e-3 + 1e-9 * np.abs(z["cash_pre"] + z["mtm_pre"])
        est = r1.copy()
        for k in np.flatnonzero(np.abs(r1 + rm + 2.0 * C) > tol):
            have = ~np.isnan(phi[k])
            order = np.argsort(grid[have])
            est[k] = read_diagram(grid[have][order], phi[k][have][order], rel_tol=1e-9, abs_tol=1e-3).gross - C[k]
        cols = [str(c) for c in z["real_columns"]]
        real = {c: z["real"][:, i] for i, c in enumerate(cols)}
        x = real["gross"] - (real["access"] + real["roll"] + real["transaction"] + real["funding"] - real["battery"])
        row = runs[runs.run == f.stem].iloc[0]
        check(f"engine scan: summed error of {f.stem} recomputed from its scan", float(np.abs(est - x).sum() * 100 / E0), float(row.sum_err), 1e-6)
    sys.path.remove(str(T))


def identification_statement_checks() -> None:
    """Brute-force, data-free check of the corrected identification statements of the paper (identified sets, SPR's non-sharpness,
    TIR's error dominance, the minimax regret Q*A*B/(A+B), the finite-pool counterexamples); a few seconds."""
    script = REPO / "exploratory_claude_2026-11-02" / "check_identification_statements.py"
    result = subprocess.run([sys.executable, "-B", str(script)], capture_output=True, text=True, encoding="utf-8")
    check("identification statements: script exit code (0 = all pass)", float(result.returncode), 0, 0)
    check("identification statements: 'all checks pass' reported", float("all checks pass" in result.stdout), 1, 0)


def twin_reward_tests() -> None:
    """The released twin-reward package: unit tests of TIR, SPR, the pooled-magnitude reward and the cost-shift test on toy environments."""
    script = ACC / "twin_reward" / "test_twin_reward.py"
    result = subprocess.run([sys.executable, "-B", str(script)], capture_output=True, text=True, encoding="utf-8")
    check("twin_reward tests: script exit code (0 = all pass)", float(result.returncode), 0, 0)
    check("twin_reward tests: 'all checks pass' reported", float("all checks pass" in result.stdout), 1, 0)


def _audit_rows(folder: Path, settings=None) -> list:
    """Per agent: exploit share X and the absolute errors of the TIR and reported estimates of its mean booked return (raw result files)."""
    rows = []
    for p in sorted(folder.glob("*.json")):
        d = json.loads(p.read_text())
        if settings is not None and d["setting"] not in settings:
            continue
        low = [abs(d["lev"][f"{e}|{z}|{r}"]) == 16.0 for e in ("0", "0.02", "0.05", "0.1") for z in ("-1", "+1") for r in ("0.05", "0.5")]
        a, b = d["audit_edge"], d["audit_placebo"]
        rows.append(dict(X=float(np.mean(low)), tir_edge=abs(a["tir"] - a["booked"]), tir_plac=abs(b["tir"] - b["booked"]),
                         rep_plac=abs(b["reported"] - b["booked"])))
    return rows


def revision_2026_11_03() -> None:
    """Numbers added after the fourth AI-assisted review (3 November 2026): the engine audit's legacy-protocol headline, the audit cohorts
    behind the abstract, Study W under the common criterion, the Holm sensitivity, the exact cost-shift test and the agent totals. Reads
    stored results only and writes nothing."""
    from scipy import stats
    # engine audit, legacy protocol L7 (results/campaign/results_long.csv): MAPPO in DK1 ("original"), ten seeds
    d = pd.read_csv(REPO / "results" / "campaign" / "results_long.csv")
    m = d[(d.variant == "L7_thesis_like") & (d.agent == "marl") & (d.region == "original")]
    check("Engine L7, MAPPO DK1: seeds", float(len(m)), 10, 0)
    check("Engine L7, MAPPO DK1: reported (sleeve) return, mean (%)", m.sleeve_return_pct.mean(), 269.86, 0.01)
    check("Engine L7, MAPPO DK1: booked (investor ledger) return, mean (%)", m.investor_return_pct.mean(), -534.76, 0.01)
    rule = d[(d.variant == "L7_thesis_like") & (d.agent == "anchor") & (d.region == "original")]
    check("Engine L7, rule DK1: reported (sleeve) return (%)", rule.sleeve_return_pct.mean(), -35.94, 0.01)
    # the headline after the cap correction: L7 without its percentage payoff (volume payoff, untouched by the uncapped-percent error)
    h = d[(d.variant == "L7_minus_percent_mtm") & (d.agent == "marl")]
    check("Engine L7 without percent payoff, MAPPO: runs", float(len(h)), 6, 0)
    check("Engine L7 without percent payoff, MAPPO: runs reporting above the rule's best", float((h.sleeve_return_pct > d[(d.variant == "L7_minus_percent_mtm") & (d.agent == "anchor")].sleeve_return_pct.max()).sum()), 6, 0)
    check("Engine L7 without percent payoff, MAPPO: smallest reported return (%)", h.sleeve_return_pct.min(), 161.8, 0.05)
    check("Engine L7 without percent payoff, MAPPO: largest reported return (%)", h.sleeve_return_pct.max(), 795.1, 0.05)
    check("Engine L7 without percent payoff, MAPPO: reported return, mean (%)", h.sleeve_return_pct.mean(), 437.5, 0.05)
    check("Engine L7 without percent payoff, MAPPO: largest booked return (%)", h.investor_return_pct.max(), -1821.8, 0.05)
    check("Engine L7 without percent payoff, MAPPO: smallest booked return (%)", h.investor_return_pct.min(), -3654.6, 0.05)
    check("Engine L7 without percent payoff, MAPPO: booked return, mean (%)", h.investor_return_pct.mean(), -2980.2, 0.05)
    check("Engine L7 without percent payoff, rule: reported return, mean (%)", d[(d.variant == "L7_minus_percent_mtm") & (d.agent == "anchor")].sleeve_return_pct.mean(), 100.9, 0.05)
    l3 = d[(d.variant == "L3_price_taker_no_solvency") & (d.agent == "marl")]
    check("Engine L3, MAPPO: runs", float(len(l3)), 20, 0)
    check("Engine L3, MAPPO: runs reporting a positive return", float((l3.sleeve_return_pct > 0).sum()), 13, 0)
    check("Engine L3, MAPPO: largest booked return (%)", l3.investor_return_pct.max(), -737.1, 0.05)
    check("Engine L3, MAPPO: smallest booked return (%)", l3.investor_return_pct.min(), -3338.1, 0.05)
    # audit cohorts from the raw result files
    v = _audit_rows(STUDY_V / "results_v", ("S_DK1", "S_DK1+Z"))
    w = _audit_rows(STUDY_W / "results_w")
    x = _audit_rows(STUDY_X / "results_x")
    mx = lambda rs, k: max(r[k] for r in rs)  # noqa: E731
    check("Audit cohort: Study V Parts A and C, agents", float(len(v)), 128, 0)
    check("Audit cohort: Study W, agents", float(len(w)), 48, 0)
    check("Audit cohort: single-floor agents of V and W", float(len(v) + len(w)), 176, 0)
    check("Audit cohort: V and W, largest TIR error, edge market (below 0.4)", mx(v + w, "tir_edge"), 0.3845, 0.0001)
    check("Audit cohort: V and W, largest TIR error, placebo market", mx(v + w, "tir_plac"), 0.1286, 0.0001)
    vg = [r for r in v if r["X"] >= 0.25]
    check("Audit cohort: Study V gamblers (X >= 0.25)", float(len(vg)), 28, 0)
    check("Audit cohort: Study V gamblers, median reported-return error, placebo", float(np.median([r["rep_plac"] for r in vg])), 68.25, 0.006)
    if (STUDY_Y / "runs_y.csv").exists():
        vb = _audit_rows(STUDY_V / "results_v", ("T_DK2", "T_DK1M"))
        ys = _audit_rows(STUDY_Y / "results_y", ("S_DK1+Z", "S_DK1"))
        yx = _audit_rows(STUDY_Y / "results_y", ("X_DK1+Z",))
        single = v + vb + w + ys
        check("Audit cohort: every single-floor agent of Studies V, W and Y (PPO and A2C)", float(len(single)), 240, 0)
        check("Audit cohort: those 240 agents, largest TIR error, edge market (below 0.4)", mx(single, "tir_edge"), 0.3845, 0.0001)
        check("Audit cohort: those 240 agents, largest TIR error, placebo market", mx(single, "tir_plac"), 0.1286, 0.0001)
        check("Audit cohort: Study Y two-write gamblers, largest TIR error, edge market", mx([r for r in yx if r["X"] >= 0.25], "tir_edge"), 1.3935, 0.0001)
        check("Audit cohort: Study Y two-write non-gamblers, largest TIR error, edge market", mx([r for r in yx if r["X"] < 0.25], "tir_edge"), 0.0013, 0.00005)
    xg, xh = [r for r in x if r["X"] >= 0.25], [r for r in x if r["X"] < 0.25]
    check("Audit cohort: Study X gamblers, largest TIR error, edge market", mx(xg, "tir_edge"), 1.5658, 0.0001)
    check("Audit cohort: Study X non-gamblers, largest TIR error, edge market (at most 0.0013)", mx(xh, "tir_edge"), 0.0013, 0.00005)
    check("Audit cohort: Study X non-gamblers, largest TIR error, placebo market (at most 0.0065)", mx(xh, "tir_plac"), 0.0065, 0.00005)
    # the twin bracket [2*SPR - reported, reported] on every stored audit episode, and on the engine's hand cases
    if (STUDY_Y / "runs_y.csv").exists():
        eps = out = lbc = n_am = 0
        for folder in (STUDY_V / "results_v", STUDY_W / "results_w", STUDY_X / "results_x", STUDY_Y / "results_y"):
            for p in sorted(folder.glob("*.json")):
                d = json.loads(p.read_text())
                for mk in ("audit_edge", "audit_placebo"):
                    ep = d[mk]["episodes"]
                    b, rep, s = (np.asarray(ep[k], float) for k in ("booked", "reported", "spr"))
                    lo = 2 * s - rep
                    eps += len(b)
                    out += int(((b < lo - 1e-6) | (b > rep + 1e-6)).sum())
                    lbc += int(lo.mean() - 1.645 * lo.std(ddof=1) / np.sqrt(len(lo)) <= b.mean() + 1e-6)
                    n_am += 1
        check("Twin bracket: audited test episodes (Studies V, W, X, Y)", float(eps), 60800, 0)
        check("Twin bracket: episodes whose booked return lies outside the bracket", float(out), 0, 0)
        check("Twin bracket: agent-markets whose one-sided 95% bound lies below the booked mean (of 608)", float(lbc), float(n_am), 0)
    hand = json.loads((REPO / "exploratory_codex_2026-11-01" / "native_reward_identification_001" / "runs" / "hand_v1" / "RESULTS.json").read_text())
    cov = tir_wrong = 0
    for row in hand["rows"]:
        e0 = row["known_receipts"]["initial_cash"]
        r, rp = row["reported_ending_cash"]["real"] - e0, row["reported_ending_cash"]["shadow"] - e0
        g = row["evaluator_only_gold"]["real"]
        c, booked = g["cost"] + g["funding"], float(Fraction(g["booked"]))
        cov += int(-rp - 2 * c - 1e-9 <= booked <= r + 1e-9)
        tir_wrong += int(abs(row["naive_TIR_on_cash_increment"] - booked) > 1e-9)
    check("Twin bracket: engine hand cases covered (of 6)", float(cov), 6, 0)
    check("Twin bracket: engine hand cases where TIR's point estimate is wrong", float(tir_wrong), 4, 0)
    # Study W under the common criterion (one-sided Mann-Whitney against Study T's reported-reward arm)
    W, T = pd.read_csv(STUDY_W / "runs_w.csv"), pd.read_csv(STUDY_T / "runs_t.csv")
    for setting, arm, p_exp in (("T_DK2", "RUIN1", 0.0758), ("T_DK2", "RUIN5", 0.0063), ("T_DK1M", "RUIN1", 0.0010)):
        p = stats.mannwhitneyu(W[(W.setting == setting) & (W.arm == arm)].X, T[(T.setting == setting) & (T.arm == "PLAIN")].X, alternative="less").pvalue
        check(f"Study W common criterion: {setting} {arm}, one-sided Mann-Whitney p", float(p), p_exp, 0.00006)
    # Holm sensitivity of every registered significance test of Studies S to Z, RS, EXT and FT (within each study and across all), written nowhere
    row_re = re.compile(r"^\|\s*([A-Z][^|]*?)\s*\|\s*(.*)\|\s*(SUPPORTED|NOT SUPPORTED|NOT TESTABLE)\s*\|\s*$")
    p_re = re.compile(r"p = (\d+\.\d+) \(need < 0\.05\)")
    tests = []
    holm_files = [(s, ACC / f"study_{s.lower()}" / f"VERDICTS_{s}.md") for s in "STUVWXY"] + [("Z", STUDY_Z / "VERDICTS_Z.md"), ("Z", STUDY_ZB / "VERDICTS_ZB.md"), ("RS", STUDY_RS / "VERDICTS_RS.md"), ("EXT", STUDY_EXT / "VERDICTS_EXT.md"), ("FT", STUDY_FT / "VERDICTS_FT.md")]
    for s, f in holm_files:
        for line in f.read_text(encoding="utf-8").splitlines():
            mm = row_re.match(line)
            pm = p_re.search(mm.group(2)) if mm else None
            if mm and pm and mm.group(3) != "NOT TESTABLE":
                tests.append((s, mm.group(3) == "SUPPORTED", max(float(pm.group(1)), 0.00005)))

    def holm(ps):
        order, adj, run = sorted(range(len(ps)), key=lambda i: ps[i]), [0.0] * len(ps), 0.0
        for rank, i in enumerate(order):
            run = max(run, (len(ps) - rank) * ps[i])
            adj[i] = min(1.0, run)
        return adj
    adj_all = holm([t[2] for t in tests])
    lost_all = sum(1 for t, a in zip(tests, adj_all) if t[1] and a >= 0.05)
    within = {}
    for s in sorted({t[0] for t in tests}):
        sub = [t for t in tests if t[0] == s]
        within[s] = sum(1 for t, a in zip(sub, holm([t[2] for t in sub])) if t[1] and a >= 0.05)
    check("Holm sensitivity: decisions with a significance test, all studies", float(len(tests)), 69, 0)
    check("Holm sensitivity: supported decisions with a test", float(sum(t[1] for t in tests)), 65, 0)
    check("Holm sensitivity: supported decisions that change across all 69", float(lost_all), 17, 0)
    check("Holm sensitivity: largest adjusted p among the supported, across all 69", max(a for t, a in zip(tests, adj_all) if t[1]), 0.0825, 0.00005)
    check("Holm sensitivity: supported decisions that change within Study FT", float(within["FT"]), 4, 0)
    check("Holm sensitivity: supported decisions that change within any other study", float(sum(v for k, v in within.items() if k != "FT")), 0, 0)
    old_tests = [t for t in tests if t[0] not in ("EXT", "FT")]
    adj_old = holm([t[2] for t in old_tests])
    check("Holm sensitivity: decisions with a test, Studies S to Z and RS", float(len(old_tests)), 49, 0)
    check("Holm sensitivity: supported decisions that change across Studies S to Z and RS", float(sum(1 for t, a in zip(old_tests, adj_old) if t[1] and a >= 0.05)), 0, 0)
    check("Holm sensitivity: largest adjusted p among the supported, Studies S to Z and RS", max(a for t, a in zip(old_tests, adj_old) if t[1]), 0.0495, 0.00005)
    # the exact cost-shift test (exploratory_claude_2026-11-03/exact_cost_shift/)
    ex = REPO / "exploratory_claude_2026-11-03" / "exact_cost_shift"
    c = json.loads((ex / "contract_exact_results.json").read_text(encoding="utf-8"))
    sf, mf = c["single_floor_sleeve"], c["multi_floor_sleeve"]
    check("Cost-shift (exact): single-floor sleeve, states tested", float(sf["tested_states"]), 5122, 0)
    check("Cost-shift (exact): single-floor sleeve, states flagged",
          float(sf["flagged_exact_among_informative"] + sf["flagged_exact_among_uninformative"] + sf["flagged_exact_among_free"]), 0, 0)
    check("Cost-shift (exact): two-write sleeve, informative binding states", float(mf["informative_among_them"]), 761, 0)
    check("Cost-shift (exact): two-write sleeve, informative binding states flagged", float(mf["flagged_exact_among_informative"]), 761, 0)
    check("Cost-shift (exact): two-write sleeve, states with no binding fee flagged (of 4,273)", float(mf["flagged_exact_among_free"]), 0, 0)
    eng = {row["starting_cash"]: row["violated_exact"] for row in c["engine_original_methods"]}
    check("Cost-shift (exact): engine flagged at starting cash 0, 5 and not 100", float(eng[0.0] and eng[5.0] and not eng[100.0]), 1, 0)
    val = json.loads((ex / "validate_exact_results.json").read_text(encoding="utf-8"))
    check("Cost-shift (exact): disagreements with the brute force", float(sum(g["mismatches"] for g in val["exactness"].values())), 0, 0)
    check("Cost-shift (exact): false alarms in 200,000 single-floor worlds",
          float(val["false_alarms_noise_free"]["flagged_exact"] + val["false_alarms_with_noise"]["flagged_exact"]), 0, 0)
    check("Cost-shift (exact): cost-dependent floors flagged (of 4,000)", float(val["power"]["cost-dependent floor"]["flagged_exact_test"]), 3912, 0)
    check("Cost-shift (earlier check): cost-dependent floors flagged (of 4,000)", float(val["power"]["cost-dependent floor"]["flagged_previous_test"]), 265, 0)
    # agent totals: B-P counted in counts_and_learnability (824); Q, R and S-Z from their run tables
    n = {"Q": len(pd.read_csv(STUDY_Q / "runs_q.csv")), "R": len(pd.read_csv(ACC / "study_r" / "runs_r.csv")),
         "S": len(pd.read_csv(STUDY_S / "runs_s_A.csv")) + len(pd.read_csv(STUDY_S / "runs_s_B.csv")),
         "T": len(pd.read_csv(STUDY_T / "runs_t.csv")), "U": len(pd.read_csv(STUDY_U / "runs_u.csv")), "V": len(pd.read_csv(STUDY_V / "runs_v.csv")),
         "W": len(pd.read_csv(STUDY_W / "runs_w.csv")), "X": len(pd.read_csv(STUDY_X / "runs_x.csv"))}
    if (STUDY_Y / "runs_y.csv").exists():
        n["Y"] = len(pd.read_csv(STUDY_Y / "runs_y.csv"))
    sy = sum(n[k] for k in "STUVWXY" if k in n)
    check("Agents in the registered remedy studies S to Y", float(sy), 734, 0)
    check("Distinct trained agents of the registered programme (B-P 824 + Q + R + S to Y)", float(824 + n["Q"] + n["R"] + sy), 1718, 0)
    nz = sum(len(pd.read_csv(f)) for f in (STUDY_Z / "runs_z.csv", STUDY_ZB / "runs_zb.csv") if f.exists())
    check("Agents in the registered remedy studies S to Z", float(sy + nz), 870, 0)
    check("Distinct trained agents of the registered programme (B-P 824 + Q + R + S to Z)", float(824 + n["Q"] + n["R"] + sy + nz), 1854, 0)
    nrs = len(pd.read_csv(STUDY_RS / "runs_rs.csv"))
    check("Agents in the registered remedy studies S to Z and RS", float(sy + nz + nrs), 950, 0)
    check("Distinct trained agents of the registered programme with RS", float(824 + n["Q"] + n["R"] + sy + nz + nrs), 1934, 0)
    next_, nft = len(pd.read_csv(STUDY_EXT / "runs_ext.csv")), len(pd.read_csv(STUDY_FT / "runs_ft.csv"))
    check("Agents in the registered remedy studies S to Z, RS, EXT and FT", float(sy + nz + nrs + next_ + nft), 1101, 0)
    check("Distinct trained agents of the registered programme with EXT and FT", float(824 + n["Q"] + n["R"] + sy + nz + nrs + next_ + nft), 2085, 0)


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


# ---------------------------------------------------------------------------------------------------------------------------------
# Added 18 November 2026: Study RE; Studies GC, DR and XE (registered 14 to 16 November); the same-engine backtesting.py control and
# the tests of the closed forms (both exploratory); and the gym-mtsim exploratory records that the paper quotes. Only stored
# results are read. Each registered aggregator is re-run in a scratch copy of its folder, so the repository is not written to.
STUDY_RE = ACC / "study_re"
STUDY_XE = REPO / "study_xe_2026-11-16"
STUDY_GC = REPO / "study_gc_2026-11-14"
STUDY_DR = REPO / "study_dr_2026-11-15"
CONTROL_XE = REPO / "xe_patch_control_2026-11-17"
CLOSED_FORMS = REPO / "monitoring_law_2026-11-17"
EXPL_LINE = REPO / "exploratory_line_2026-11-07"
EXPL_MTSIM = REPO / "exploratory_mtsim_reach_2026-11-08"
XE_EQ, XE_LEFT = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0), (0.25, 0.5, 0.75)


def check_hashes_manifest(folder: Path, tag: str) -> None:
    """'<sha256>  <path>' lines and '#' comments; a path may point into a sibling study ('../'). RUN_LOG_<tag>.md is append-only:
    its registered version is the whole file, or the prefix that ends at the line '## After hashing' (without it, with its leading
    newline, or with the heading line), as the registration took it."""
    for line in (folder / f"PREREGISTRATION_HASHES_{tag}.txt").read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 2 or line.startswith("#"):
            continue
        digest, rel = parts
        name = f"{folder.name}/{rel}"
        p = folder / rel
        if not p.exists():
            print(f"[FAIL] {name}: MISSING")
            FAILS.append(name)
            continue
        data = p.read_bytes()
        digests = {hashlib.sha256(data).hexdigest()}
        log = rel == f"RUN_LOG_{tag}.md" and b"\n## After hashing" in data
        if log:
            cut = data.index(b"\n## After hashing")
            digests |= {hashlib.sha256(data[:cut]).hexdigest(), hashlib.sha256(data[:cut + 1]).hexdigest(),
                        hashlib.sha256(data[: data.index(b"\n", cut + 1) + 1]).hexdigest()}
        ok = digest in digests
        print(f"[{'ok' if ok else 'FAIL'}] {name}{' (registered prefix)' if log else ''}")
        if not ok:
            FAILS.append(name)


def rerun_aggregator(folder: Path, script: str, outputs: list[str]) -> None:
    """Re-run a study's registered aggregator on its stored results in a scratch copy of the folder and require it to reproduce
    every stored output, line endings aside."""
    import shutil
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / folder.name
        shutil.copytree(folder, dst, ignore=shutil.ignore_patterns("__pycache__", "policies", "policies_smoke"))
        for rel in outputs:
            (dst / rel).unlink(missing_ok=True)
        r = subprocess.run([sys.executable, "-B", script], cwd=dst, capture_output=True, text=True, encoding="utf-8")
        for rel in outputs:
            stored = (folder / rel).read_text(encoding="utf-8").replace("\r\n", "\n")
            again = (dst / rel).read_text(encoding="utf-8").replace("\r\n", "\n") if (dst / rel).exists() else None
            ok = r.returncode == 0 and again == stored
            print(f"[{'ok' if ok else 'FAIL'}] {folder.name}/{script} re-run reproduces {rel}")
            if not ok:
                FAILS.append(f"{folder.name}/{rel} (re-aggregation)")


def _verdicts(path: Path, prefix: str) -> dict:
    """The registered decisions of a VERDICTS file: {'XE1': True, ...}, read from lines ending '-> SUPPORTED' or '-> NOT SUPPORTED'."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(rf"-\s+({prefix}\d+)\b.*->\s+(NOT SUPPORTED|SUPPORTED)\s*$", line)
        if m:
            out[m[1]] = m[2] == "SUPPORTED"
    return out


def _xe_share(r: dict) -> float:
    """Study XE's learner measure: the share of the nine low-equity probes (equity at most 0.25) at the largest level."""
    return float(np.mean([r["probe"][f"{e:g}|{l:g}"] >= max(r["levels"]) for e in XE_EQ if e <= 0.25 for l in XE_LEFT]))


def _gap(r: dict, cash: str) -> np.ndarray:
    t = r["tests"][cash]
    return np.array(t["draws_reported"]) - np.array(t["draws_booked"])


def study_re_headlines() -> None:
    rerun_aggregator(STUDY_RE, "aggregate_re.py", ["runs_re.csv", "cells_re.csv", "VERDICTS_RE.md"])
    check("RE: registered decisions RE1 to RE3 supported", float(_verdicts(STUDY_RE / "VERDICTS_RE.md", "RE") ==
                                                               {"RE1": True, "RE2": True, "RE3": True}), 1, 0)
    cells = pd.read_csv(STUDY_RE / "cells_re.csv").set_index("scale")
    for c, reach, d in ((1.0, 855.8, 13.15), (0.3, 442.8, 15.55), (0.1, 148.5, 6.15), (0.03, 5.73, -3.15), (0.01, 0.0, 0.0)):
        check(f"RE c={c:g}: reach predicted before training", float(cells.loc[c, "reach_R"]), reach, 0.05)
        check(f"RE c={c:g}: mean exploit gap D", float(cells.loc[c, "mean_D"]), d, 0.005)
    for c, lo, hi in ((1.0, 609, 737), (0.3, 280, 372), (0.1, 10, 102), (0.03, 0, 4), (0.01, 0, 0)):
        check(f"RE c={c:g}: fewest floor hits in floored training", float(cells.loc[c, "ZF_floor_episodes_min"]), lo, 0)
        check(f"RE c={c:g}: most floor hits in floored training", float(cells.loc[c, "ZF_floor_episodes_max"]), hi, 0)
    check("RE c=0.01: pairs with identical probes", float(cells.loc[0.01, "pairs_identical_probe"]), 5, 0)
    check("RE c=0.01: pairs with identical parameters", float(cells.loc[0.01, "pairs_identical_params"]), 5, 0)
    check("RE c=0.03: pairs with identical probes", float(cells.loc[0.03, "pairs_identical_probe"]), 4, 0)
    check("RE c=0.1: pairs with a positive gap", float(cells.loc[0.1, "pairs_D_positive"]), 3, 0)


def study_xe_headlines() -> None:
    from scipy.stats import fisher_exact
    rerun_aggregator(STUDY_XE, "aggregate_xe.py", ["VERDICTS_XE.md"])
    check("XE: decisions as reported (XE1, XE4, XE5, XE6 supported; XE2, XE3, XE7 not)", float(
        _verdicts(STUDY_XE / "VERDICTS_XE.md", "XE") == {"XE1": True, "XE2": False, "XE3": False, "XE4": True, "XE5": True,
                                                       "XE6": True, "XE7": False}), 1, 0)
    census = pd.read_csv(STUDY_XE / "xe_census.csv")
    low, top = census[census.L <= 5], census[census.L == 200]
    check("XE1: largest |backtesting.py - backtrader| at leverage <= 5 (x 1e16)", 1e16 * float((low.btpy_rep - low.bt_rep).abs().max()), 9.4, 0.05)
    check("XE2: backtesting.py reported minus its own trade records at leverage 200 (points)", 100 * float((top.btpy_rep - top.btpy_book).mean()), 37.82, 0.005)
    check("XE2: backtrader reported at leverage 200 (%)", 100 * float(top.bt_rep.mean()), 14.8, 0.05)
    check("XE2: the census sample's 1x ledger (%; a fair market loses the fee)", 100 * float(census[census.L == 1].ledger.mean()), 0.074, 0.0005)
    runs = {}
    for p in sorted((STUDY_XE / "results_xe").glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        runs.setdefault((r["engine"], r["tag"]), []).append(r)
    xb, xt = runs[("btpy", "_xe5")], runs[("bt", "_xe5")]
    kb, kt = sum(_xe_share(r) >= 0.5 for r in xb), sum(_xe_share(r) >= 0.5 for r in xt)
    check("XE5: learners through backtesting.py (of 16)", float(kb), 12, 0)
    check("XE5: learners through backtrader (of 16)", float(kt), 1, 0)
    check("XE5: Fisher one-sided p (x 1e5)", 1e5 * fisher_exact([[kb, 16 - kb], [kt, 16 - kt]], alternative="greater")[1], 8.54, 0.005)
    check("XE6: backtesting.py seed-mean reported - booked from 10,000 USD (points)", float(np.mean([_gap(r, "10000").mean() for r in xb])), 26.94, 0.005)
    check("XE6: backtesting.py seed-mean reported - booked from 1,000 USD (points)", float(np.mean([_gap(r, "1000").mean() for r in xb])), 31.36, 0.005)
    check("XE6: backtrader largest |reported - booked| (points)", max(float(np.abs(_gap(r, c)).max()) for r in xt for c in ("10000", "1000")), 0, 1e-6)
    learned = [r for r in xb if _xe_share(r) >= 0.5]
    check("XE learners from 10,000 USD: reported (%)", float(np.mean([r["tests"]["10000"]["reported_pct"] for r in learned])), 2.64, 0.005)
    check("XE learners from 10,000 USD: booked (%)", float(np.mean([r["tests"]["10000"]["booked_pct"] for r in learned])), -33.28, 0.005)
    everywhere = [r for r in xb if all(v == 200.0 for v in r["probe"].values())]
    check("XE agents at 200x at every probe", float(len(everywhere)), 9, 0)
    check("XE agents at 200x at every probe: mean reported - booked from 10,000 USD (points)", float(np.mean([_gap(r, "10000").mean() for r in everywhere])), 36.96, 0.005)
    for eng, short, long_ in (("btpy", 6, 2), ("bt", 4, 3)):
        rr = runs[(eng, "_xe7")]
        check(f"XE7 {eng}: seeds short at half or more of the probes", float(sum(np.mean([v < 0 for v in r["probe"].values()]) >= 0.5 for r in rr)), short, 0)
        check(f"XE7 {eng}: seeds long at half or more of the probes", float(sum(np.mean([v > 0 for v in r["probe"].values()]) >= 0.5 for r in rr)), long_, 0)


def study_gc_headlines() -> None:
    rerun_aggregator(STUDY_GC, "aggregate_gc.py", ["results_gc/runs_gc.csv", "results_gc/VERDICTS_GC.md"])
    check("GC: decisions as reported (GC1 to GC6 supported, GC7 not)", float(
        _verdicts(STUDY_GC / "results_gc" / "VERDICTS_GC.md", "GC") == {f"GC{i}": i != 7 for i in range(1, 8)}), 1, 0)
    runs = pd.read_csv(STUDY_GC / "results_gc" / "runs_gc.csv")
    check("GC: runs", float(len(runs)), 40, 0)
    crypto = runs[runs.market == "crypto"].groupby("condition")
    check("GC crypto: low-equity exposure, published reward", float(crypto.xlow.mean()["ORIG"]), 0.278, 0.0005)
    check("GC crypto: low-equity exposure, floor lines deleted", float(crypto.xlow.mean()["FLPATCH"]), 0.078, 0.0005)
    check("GC crypto: low-equity exposure, reading-trained", float(crypto.xlow.mean()["LINE"]), 0.054, 0.0005)
    check("GC crypto: reported - booked, published reward (points)", float(crypto.gap.mean()["ORIG"]), 3.36, 0.005)
    orig = runs[(runs.market == "crypto") & (runs.condition == "ORIG")]
    check("GC crypto: share of training episodes in which the floor wrote", float((orig.train_floored_episodes / orig.train_episodes).mean()), 0.76, 0.01)
    fx = runs[runs.market == "forex"]
    check("GC forex: test orders per episode, every agent", float(fx.orders_per_episode.abs().max()), 0, 0)
    fxo = fx[fx.condition == "ORIG"]
    check("GC forex: fewest training steps on which the floor wrote", float(fxo.train_floor_steps.min()), 97, 0)
    check("GC forex: most training steps on which the floor wrote", float(fxo.train_floor_steps.max()), 871, 0)
    check("GC7: the reading's largest error (USD)", float(runs.reading_max_abs_err_usd.max()), 4.40, 0.005)


def study_dr_headlines() -> None:
    rerun_aggregator(STUDY_DR, "aggregate_dr.py", ["results_dr/runs_dr.csv", "results_dr/VERDICTS_DR.md"])
    v = STUDY_DR / "results_dr" / "VERDICTS_DR.md"
    check("DR: decisions as reported (DR4 supported, DR1 to DR3 not)", float(_verdicts(v, "DR") ==
                                                                          {"DR1": False, "DR2": False, "DR3": False, "DR4": True}), 1, 0)
    m = re.search(r"one-sided permutation p = ([0-9.]+)", v.read_text(encoding="utf-8"))
    check("DR4: Jonckheere-Terpstra one-sided p", float(m[1]) if m else float("nan"), 0.035, 0.0005)
    runs = pd.read_csv(STUDY_DR / "results_dr" / "runs_dr.csv")
    check("DR: runs", float(len(runs)), 64, 0)
    learners = runs[runs.condition == "ORIG"].groupby("arm").learner.sum()
    for arm, k in (("V2", 3), ("V4", 4), ("V8", 8), ("V16", 8), ("L30", 0), ("L500", 2)):
        check(f"DR {arm}: learners", float(learners[arm]), k, 0)


def control_xe_headlines() -> None:
    from scipy.stats import fisher_exact
    rows = {}
    for p in sorted((CONTROL_XE / "results_ctl").glob("*_ctl.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        rows.setdefault(r["engine"], {})[r["seed"]] = r
    seeds = range(301, 309)
    check("Control: runs", float(sum(len(v) for v in rows.values())), 16, 0)
    kp = sum(_xe_share(rows["btpy"][s]) >= 0.5 for s in seeds)
    kc = sum(_xe_share(rows["btpyfl"][s]) >= 0.5 for s in seeds)
    check("Control: learners through the published engine (of 8)", float(kp), 6, 0)
    check("Control: learners with the floor lines replaced (of 8)", float(kc), 0, 0)
    check("Control: Fisher one-sided p (x 1e4)", 1e4 * fisher_exact([[kp, 8 - kp], [kc, 8 - kc]], alternative="greater")[1], 34.97, 0.01)
    check("Control: largest |reported - booked| with the floor lines replaced (points)",
          max(float(np.abs(_gap(rows["btpyfl"][s], c)).max()) for s in seeds for c in ("10000", "1000")), 0, 1e-9)
    check("Control: published engine, seed-mean reported - booked from 10,000 USD (points)", float(np.mean([_gap(rows["btpy"][s], "10000").mean() for s in seeds])), 22.80, 0.005)
    lrn = [rows["btpy"][s] for s in seeds if _xe_share(rows["btpy"][s]) >= 0.5]
    check("Control learners from 10,000 USD: reported (%)", float(np.mean([r["tests"]["10000"]["reported_pct"] for r in lrn])), 28.96, 0.005)
    check("Control learners from 10,000 USD: booked (%)", float(np.mean([r["tests"]["10000"]["booked_pct"] for r in lrn])), -1.44, 0.005)
    g = _gap(rows["btpy"][302], "10000")                       # seed 302 reproduces 'always 200x' on the whole bank (verify_ctl.py)
    check("Control bank, 200x everywhere: forgiven (points)", float(g.mean()), 36.48, 0.005)
    check("Control bank, 200x everywhere: standard error (points)", float(g.std(ddof=1) / np.sqrt(len(g))), 2.72, 0.005)


def closed_form_headlines() -> None:
    a2 = json.loads((CLOSED_FORMS / "test_a2_sim.json").read_text(encoding="utf-8"))
    check("Closed forms on additive P&L: values compared", float(3 * len(a2["A2"]["rows"])), 36, 0)
    check("Closed forms on additive P&L: largest relative deviation (%)", 100 * a2["A2"]["worst"], 1.8, 0.05)
    eng = json.loads((CLOSED_FORMS / "test_c_engine.json").read_text(encoding="utf-8"))
    ratios = [r["ratio"] for r in eng["rows"]]
    check("Closed forms in backtesting.py: settings meeting the pre-stated criterion (of 18)", float(sum(r["ok"] for r in eng["rows"])), 18, 0)
    check("Closed forms in backtesting.py: smallest ratio to the prediction", min(ratios), 0.94, 0.005)
    check("Closed forms in backtesting.py: largest ratio to the prediction", max(ratios), 1.02, 0.005)
    check("Closed forms in backtesting.py: largest deviation (%)", 100 * max(abs(x - 1) for x in ratios), 6.0, 0.05)
    check("Closed forms in backtesting.py: flattest fee-free slope", max(eng["slopes"].values()), -0.465, 0.0005)
    check("Closed forms in backtesting.py: steepest fee-free slope", min(eng["slopes"].values()), -0.484, 0.0005)
    d = json.loads((CLOSED_FORMS / "test_d_existing.json").read_text(encoding="utf-8"))
    d1 = {r["L"]: r for r in d["D1"]}
    check("Closed forms against the XE census at leverage 200: predicted (points)", 100 * d1[200]["pred"], 37.47, 0.005)
    check("Closed forms against the XE census at leverage 200: measured (points)", 100 * d1[200]["xe"], 37.82, 0.005)
    check("Closed forms against XE's agents at 200x everywhere: agents", float(d["D2"]["n_everywhere"]), 9, 0)
    check("Closed forms against XE's agents at 200x everywhere: mean over-report (points)", d["D2"]["gap_mean"], 36.96, 0.005)
    check("First-order Nordic formula: mean absolute error (above its pre-stated limit of 0.05)", d["D4"]["mae"], 0.1255, 0.0005)
    nordic = pd.read_csv(CLOSED_FORMS / "test_d4b_nordic.csv")
    check("Nordic ratios, post hoc form: zone-years within two standard errors (of 24)", float((nordic.err_b.abs() <= 2 * nordic.se).sum()), 24, 0)
    check("Nordic ratios, post hoc form: mean absolute error", float(nordic.err_b.abs().mean()), 0.017, 0.0005)
    check("Nordic overshoot constant: smallest (sd of the hourly spread)", float(nordic.rho_over_s.min()), 0.9, 0.01)
    check("Nordic overshoot constant: largest (sd of the hourly spread)", float(nordic.rho_over_s.max()), 12.9, 0.05)
    check("Nordic ratios of the paper's ladder table: smallest", float(nordic.table.min()), 1.62, 0.005)
    check("Nordic ratios of the paper's ladder table: largest", float(nordic.table.max()), 1.99, 0.005)


def mtsim_exploratory_headlines() -> None:
    s = json.loads((EXPL_LINE / "results.json").read_text(encoding="utf-8"))["summary"]
    check("gym-mtsim formula gap: positions meeting every rule (of 100)", float(s["public_pass"]), 100, 0)
    check("gym-mtsim formula gap: control positions meeting every rule (of 100)", float(s["control_pass"]), 100, 0)
    check("gym-mtsim formula gap: largest |reward - its own definition| (USD)", s["public_formula_gap_max"], 0, 1e-6)
    check("gym-mtsim formula gap: smallest forgiven loss at a 16% fall (USD)", s["public_shortfall_min"], 11587, 0.5)
    check("gym-mtsim formula gap: largest forgiven loss at a 16% fall (USD)", s["public_shortfall_max"], 11846, 0.5)
    check("gym-mtsim formula gap: slope's largest relative error (x 1e14)", 1e14 * s["public_slope_rel_err_max"], 1.1, 0.05)
    check("gym-mtsim formula gap: line's largest error (x 1e10 USD, below 1.6)", 1e10 * s["public_book_err_max"], 1.55, 0.05)
    r = json.loads((EXPL_LINE / "results_range.json").read_text(encoding="utf-8"))
    check("gym-mtsim logged witness: transitions", float(r["n"]), 5000, 0)
    check("gym-mtsim logged witness: largest reward difference with and without the floor (USD)", r["max_abs_gap_inside"], 0, 0)
    check("gym-mtsim logged witness: slope relative error (x 1e16)", 1e16 * r["slope_rel_err"], 5.2, 0.05)
    check("gym-mtsim logged witness: fitted line minus the public reward at a 16% fall (USD)", r["abs_pred_minus_public"], 11827, 0.5)
    learn = json.loads((EXPL_MTSIM / "learn_public.json").read_text(encoding="utf-8"))
    c = learn["summary"]["commits"]["0.012"]
    check("gym-mtsim commit rule at 1.2%: published commits to the maximum (of 20)", float(c["published_max"]), 16, 0)
    check("gym-mtsim commit rule at 1.2%: floor deleted stays flat (of 20)", float(c["nofloor_flat"]), 18, 0)
    check("gym-mtsim commit rule at 1.2%: line stays flat (of 20)", float(c["line_flat"]), 18, 0)
    check("gym-mtsim line against the floor-deleted step (x 1e9 USD)", 1e9 * learn["summary"]["line_max_abs_err_vs_nofloor"], 1.5, 0.05)
    excess = [x["mean_published"] - x["mean_line"] for x in learn["rows"] if x["scale"] == 0.012]
    check("gym-mtsim commit rule at 1.2%: seeds with the published mean above the line", float(sum(e > 0 for e in excess)), 20, 0)
    check("gym-mtsim commit rule at 1.2%: smallest excess (USD)", min(excess), 504, 0.5)
    check("gym-mtsim commit rule at 1.2%: largest excess (USD)", max(excess), 750, 0.5)
    check("gym-mtsim commit rule at 1.2%: mean excess (USD)", float(np.mean(excess)), 606, 0.5)


def main() -> int:
    print("== Part 1: pre-registration hashes")
    check_hashes(GEN)
    check_hashes(LIAB)
    check_hashes_n()
    check_hashes_n(STUDY_O, "O")
    check_hashes_n(STUDY_P, "P")
    check_hashes_n(STUDY_Q, "Q")
    check_hashes_n(STUDY_S, "S")
    check_hashes_n(STUDY_T, "T")
    check_hashes_n(STUDY_U, "U")
    check_hashes_n(STUDY_V, "V")
    check_hashes_n(STUDY_W, "W")
    check_hashes_n(STUDY_X, "X")
    check_hashes_n(STUDY_Y, "Y")
    check_hashes_n(STUDY_Z, "Z")
    check_hashes_n(STUDY_ZB, "ZB")
    check_hashes_rs()
    check_hashes_n(STUDY_EXT, "EXT")
    check_hashes_n(STUDY_FT, "FT")
    check_hashes_manifest(STUDY_RE, "RE")
    check_hashes_manifest(STUDY_GC, "GC")
    check_hashes_manifest(STUDY_DR, "DR")
    check_hashes_manifest(STUDY_XE, "XE")
    print("\n== Part 2: headline numbers")
    headlines()
    study_s_headlines()
    study_t_headlines()
    study_u_headlines()
    study_v_headlines()
    study_w_headlines()
    study_x_headlines()
    study_y_headlines()
    study_z_headlines()
    study_rs_headlines()
    engine_twin_headlines()
    reward_diagram_checks()
    engine_scan_headlines()
    study_ext_headlines()
    study_ft_headlines()
    identification_statement_checks()
    twin_reward_tests()
    revision_2026_11_03()
    headlines_new()
    counts_and_learnability()
    headlines_q()
    cost_baseline_headlines()
    adaptive_witness_headlines()
    study_r_integrity()
    study_re_headlines()
    study_gc_headlines()
    study_dr_headlines()
    study_xe_headlines()
    control_xe_headlines()
    closed_form_headlines()
    mtsim_exploratory_headlines()
    if REVISED:
        print(f"\nnote: {len(REVISED)} rebuilt panel(s) differ from the registered ones because eSett revised the source data "
              f"after the study (see THIRD_PARTY_LICENSES.md, 'eSett data'): {REVISED}")
    print("\nall checks pass" if not FAILS else f"\n{len(FAILS)} check(s) FAILED: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
