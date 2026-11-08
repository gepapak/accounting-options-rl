"""Study V analysis (PREREGISTRATION_V.md): writes runs_v.csv and VERDICTS_V.md.

    python aggregate_v.py                         # registered analysis of results_v/*.json
    python aggregate_v.py --results DIR --print   # descriptive summary of another folder (pilots); writes nothing

Part B compares the clone-free reward in two further settings with Study T's PLAIN and ORACLE runs in the same settings (runs_t.csv, hashed
before Study T ran); the evaluation test windows are identical.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
RUNS_T = HERE.parent / "study_t" / "runs_t.csv"
B_BOOT = 10_000
rng = np.random.default_rng(20261101)
LOW_E = ("0", "0.02", "0.05", "0.1")
SETTING_A = "S_DK1"
SETTINGS_B = ("T_DK2", "T_DK1M")
SETTING_C = "S_DK1+Z"
RUIN_ARMS = ("RUIN1", "RUIN5", "RUIN25")
MV_ARMS = ("MV0.05", "MV0.2", "MV1")
GAMBLER_X = 0.25


def load(folder: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(folder.glob("*.json")):
        d = json.loads(p.read_text())
        low = [abs(d["lev"][f"{e}|{z}|{r}"]) == 16.0 for e in LOW_E for z in ("-1", "+1") for r in ("0.05", "0.5")]
        row = dict(setting=d["setting"], arm=d["arm"], seed=d["seed"], X=float(np.mean(low)), E=d["test_edge"]["booked_pct_K0"],
                   Ez=d["test_zero"]["booked_pct_K0"] if "test_zero" in d else float("nan"),
                   placebo_reported=d["test_placebo"]["reported_pct_K0"], placebo_forgiven=d["test_placebo"]["forgiven_pct_K0"],
                   pool_mean=d.get("pool_mean_eur_per_mwh", float("nan")))
        for mk, key in (("edge", "audit_edge"), ("plac", "audit_placebo")):
            a = d.get(key)
            for est in ("reported", "spr", "tir", "pmr"):
                row[f"err_{est}_{mk}"] = abs(a[est] - a["booked"]) if a else float("nan")    # |estimated - booked| of the 100-episode mean
        rows.append(row)
    return pd.DataFrame(rows)


def mw_less(a, b) -> float:
    return float(stats.mannwhitneyu(a, b, alternative="less").pvalue)


def verdict(ok) -> str:
    if ok is None:
        return "NOT TESTABLE"
    return "SUPPORTED" if ok else "NOT SUPPORTED"


def edge_ratio(arm_df, oracle_df, min_ratio, min_lb):
    eo, ea = oracle_df.E.to_numpy(), arm_df.E.to_numpy()
    ratio = ea.mean() / eo.mean() if eo.mean() > 0 else float("nan")
    boot = np.array([rng.choice(ea, len(ea)).mean() / rng.choice(eo, len(eo)).mean() for _ in range(B_BOOT)])
    lb = float(np.quantile(boot, 0.05))
    ok = eo.mean() > 0 and ratio >= min_ratio and lb >= min_lb
    return ok, (f"E {ea.mean():+.2f} / ORACLE {eo.mean():+.2f} = {ratio:.3f} (need >= {min_ratio:.2f}); "
                f"one-sided 95% bootstrap LB {lb:.3f} (need >= {min_lb:.2f})")


def removes(arm_df, plain_df, oracle_df, name):
    p = mw_less(arm_df.X, plain_df.X)
    ok = p < 0.05 and arm_df.X.mean() <= oracle_df.X.mean() + 0.10
    return ok, (f"X {name} {arm_df.X.mean():.3f} < PLAIN {plain_df.X.mean():.3f}: p = {p:.4f} (need < 0.05); "
                f"{name} <= ORACLE + 0.10 = {oracle_df.X.mean() + 0.10:.3f}")


def no_arm_does_both(d, arms, O):
    both = []
    for arm in arms:
        k = d[d.arm == arm]
        if len(k):
            both.append((arm, k.X.mean() <= O.X.mean() + 0.10 and k.E.mean() >= 0.80 * O.E.mean(), k.X.mean(), k.E.mean()))
    ok = len(both) == len(arms) and not any(b[1] for b in both)
    if O.E.mean() <= 0:
        ok = None
    text = "; ".join(f"{a_}: X {x:.3f}, E {e:+.2f} -> {'both' if b else 'not both'}" for a_, b, x, e in both)
    return ok, text + f" (both = X <= {O.X.mean() + 0.10:.3f} and E >= {0.80 * O.E.mean():+.2f})"


def table(df: pd.DataFrame, setting: str, arms) -> list[str]:
    d = df[df.setting == setting]
    zc = d.Ez.notna().any()
    lines = [f"| {setting} arm | runs | mean X (share of low-equity probes at leverage 16) | runs with X > 0 | mean E (booked % K0 / episode) |"
             + (" mean E from zero equity |" if zc else "") + " placebo reported | placebo forgiven |",
             "|---|---|---|---|---|" + ("---|" if zc else "") + "---|---|"]
    for arm in arms:
        x = d[d.arm == arm]
        if len(x):
            lines.append(f"| {arm} | {len(x)} | {x.X.mean():.3f} | {int((x.X > 0).sum())} | {x.E.mean():+.2f} |"
                         + (f" {x.Ez.mean():+.2f} |" if zc else "") + f" {x.placebo_reported.mean():+.2f} | {x.placebo_forgiven.mean():.2f} |")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results_v"))
    ap.add_argument("--print", dest="print_only", action="store_true")
    a = ap.parse_args()
    df = load(Path(a.results))
    t = pd.read_csv(RUNS_T)
    n_a = int((df.setting == SETTING_A).sum()) if len(df) else 0
    n_b = int(df.setting.isin(SETTINGS_B).sum()) if len(df) else 0
    n_c = int((df.setting == SETTING_C).sum()) if len(df) else 0
    lines = ["# Study V verdicts", "", f"Runs: Part A {n_a} of 72, Part B {n_b} of 16, Part C {n_c} of 56.", ""]
    verdicts = []
    if n_a:
        lines += ["## Part A: the Study T edge-market test (DK1, PPO) with rewards that need no clone", ""]
        lines += table(df, SETTING_A, ("PLAIN", "ORACLE", "PMR") + RUIN_ARMS + MV_ARMS) + [""]
        d = df[df.setting == SETTING_A]
        P, O, M = d[d.arm == "PLAIN"], d[d.arm == "ORACLE"], d[d.arm == "PMR"]
        p1 = mw_less(O.X, P.X)
        present = p1 < 0.05
        verdicts.append(("VA1", f"X PLAIN {P.X.mean():.3f} > ORACLE {O.X.mean():.3f}: one-sided Mann-Whitney p = {p1:.4f} (need < 0.05)", present))
        ok2, r2 = removes(M, P, O, "PMR")
        ok3, r3 = edge_ratio(M, O, 0.80, 0.60)
        ok4, r4 = no_arm_does_both(d, MV_ARMS, O)
        ok5, r5 = no_arm_does_both(d, RUIN_ARMS, O)
        rest = [("VA2", r2, ok2), ("VA3", "PMR " + r3, ok3), ("VA4", r4, ok4), ("VA5", r5, ok5)]
        if not present:                                            # the other hypotheses are not testable if the exploit was not acquired
            rest = [(i, r + "; the exploit was not acquired (VA1 failed)", None) for i, r, _ in rest]
        verdicts += rest
    for st in SETTINGS_B:
        if int((df.setting == st).sum()):
            m = df[(df.setting == st) & (df.arm == "PMR")]
            ref = t[t.setting == st]
            P, O = ref[ref.arm == "PLAIN"], ref[ref.arm == "ORACLE"]
            lines += [f"## Part B: {st} (PMR against Study T's PLAIN and ORACLE runs in the same setting)", "",
                      f"| {st} arm | runs | mean X | runs with X > 0 | mean E |", "|---|---|---|---|---|"]
            for name, x in (("PLAIN (Study T)", P), ("ORACLE (Study T)", O), ("PMR", m)):
                lines.append(f"| {name} | {len(x)} | {x.X.mean():.3f} | {int((x.X > 0).sum())} | {x.E.mean():+.2f} |")
            lines.append("")
            ok1, r1 = removes(m, P, O, "PMR")
            ok2, r2 = edge_ratio(m, O, 0.80, 0.60)
            verdicts += [(f"VB-{st}-1", r1, ok1), (f"VB-{st}-2", "PMR " + r2, ok2)]
    if n_c:
        lines += ["## Part C: the same test when half of the training episodes start at zero equity", ""]
        lines += table(df, SETTING_C, ("PLAIN", "ORACLE", "TIR", "PMR") + RUIN_ARMS) + [""]
        d = df[df.setting == SETTING_C]
        P, O, M, T = (d[d.arm == k] for k in ("PLAIN", "ORACLE", "PMR", "TIR"))
        p1 = mw_less(O.X, P.X)
        present = p1 < 0.05
        verdicts.append(("VC1", f"X PLAIN {P.X.mean():.3f} > ORACLE {O.X.mean():.3f}: one-sided Mann-Whitney p = {p1:.4f} (need < 0.05)", present))
        ok2, r2 = removes(M, P, O, "PMR")
        ok3, r3 = edge_ratio(M, O, 0.80, 0.60)
        ok4, r4 = removes(T, P, O, "TIR")
        ok5, r5 = edge_ratio(T, O, 0.80, 0.60)
        ok6, r6 = no_arm_does_both(d, RUIN_ARMS, O)
        rest = [("VC2", r2, ok2), ("VC3", "PMR " + r3, ok3), ("VC4", r4, ok4), ("VC5", "TIR " + r5, ok5), ("VC6", r6, ok6)]
        if not present:
            rest = [(i, r + "; the exploit was not acquired (VC1 failed)", None) for i, r, _ in rest]
        verdicts += rest
    if n_a or n_c:
        d = df[df.setting.isin([SETTING_A, SETTING_C])]
        lines += ["## Part D: ledger-free audit of the trained agents (twin queries at evaluation time)", "",
                  "| estimate | max error, edge market | max error, placebo market | median error, placebo market, agents with X >= 0.25 |", "|---|---|---|---|"]
        g = d[d.X >= GAMBLER_X]
        for est in ("reported", "spr", "tir", "pmr"):
            lines.append(f"| {est} | {d[f'err_{est}_edge'].max():.3f} | {d[f'err_{est}_plac'].max():.3f} | "
                         + (f"{g[f'err_{est}_plac'].median():.3f} |" if len(g) else "n/a |"))
        lines.append(f"\nAgents audited: {len(d)}; agents with X >= {GAMBLER_X}: {len(g)}. Errors are |estimated - booked| of the 100-episode mean return, percent of K0 per episode.\n")
        ok1 = float(max(d.err_tir_edge.max(), d.err_tir_plac.max())) <= 1.0
        verdicts.append(("VD1", f"max |TIR - booked| = {d.err_tir_edge.max():.3f} (edge market), {d.err_tir_plac.max():.3f} (placebo market) over {len(d)} agents (need <= 1.0)", ok1))
        if len(g) >= 5:
            ratio = float((g.err_tir_plac / g.err_reported_plac).median())
            medrep = float(g.err_reported_plac.median())
            ok2 = ratio <= 0.05 and medrep >= 5.0
            verdicts.append(("VD2", f"{len(g)} agents with X >= {GAMBLER_X}: median reported-return error {medrep:.2f} (need >= 5), median TIR/reported error ratio {ratio:.4f} (need <= 0.05), placebo market", ok2))
        else:
            verdicts.append(("VD2", f"only {len(g)} agents with X >= {GAMBLER_X} (need 5)", None))
    lines += ["| ID | Result | Verdict |", "|---|---|---|"] + [f"| {i} | {r} | {verdict(ok)} |" for i, r, ok in verdicts]
    text = "\n".join(lines) + "\n"
    if a.print_only:
        print(text)
        return
    df.to_csv(HERE / "runs_v.csv", index=False)
    (HERE / "VERDICTS_V.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
