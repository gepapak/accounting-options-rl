"""Study W analysis (PREREGISTRATION_W.md): writes runs_w.csv and VERDICTS_W.md.

    python aggregate_w.py                                   # registered analysis of results_w/*.json
    python aggregate_w.py --results ../study_v/results_v --demo   # software check on Study V's ruin-penalty runs; writes nothing

The references (ORACLE and PLAIN of the same setting) are Study T's runs in ../study_t/runs_t.csv (hashed before Study T ran); the test windows
are identical. In --demo mode the references are the ORACLE and PLAIN arms of the results folder itself.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RUNS_T = HERE.parent / "study_t" / "runs_t.csv"
SETTINGS = ("T_DK2", "T_DK1M")
GRID = ("RUIN1", "RUIN5", "RUIN25")
LOW_E = ("0", "0.02", "0.05", "0.1")
TOL = 1.0                      # points of the allocation per episode: the audit-accuracy bound of Study V (VD1)
N_EXPECTED = 8


def load(folder: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(folder.glob("*.json")):
        d = json.loads(p.read_text())
        low = [abs(d["lev"][f"{e}|{z}|{r}"]) == 16.0 for e in LOW_E for z in ("-1", "+1") for r in ("0.05", "0.5")]
        ae, ap = d["audit_edge"], d["audit_placebo"]
        rows.append(dict(setting=d["setting"], arm=d["arm"], seed=d["seed"], X=float(np.mean(low)), E=d["test_edge"]["booked_pct_K0"],
                         R=ae["reported"], TA=ae["tir"], err_tir_edge=abs(ae["tir"] - ae["booked"]), err_tir_plac=abs(ap["tir"] - ap["booked"]),
                         err_spr_edge=abs(ae["spr"] - ae["booked"]), err_rep_edge=abs(ae["reported"] - ae["booked"])))
    return pd.DataFrame(rows)


def verdict(ok) -> str:
    return "NOT TESTABLE" if ok is None else ("SUPPORTED" if ok else "NOT SUPPORTED")


def both(x: pd.DataFrame, oracle: pd.DataFrame) -> bool:
    return bool(x.X.mean() <= oracle.X.mean() + 0.10 and x.E.mean() >= 0.80 * oracle.E.mean())


def analyse(df: pd.DataFrame, refs: pd.DataFrame, settings, demo: bool):
    lines, verdicts, picks = [], [], {}
    for st in settings:
        d = df[df.setting == st]
        n = {a: int((d.arm == a).sum()) for a in GRID}
        ref = refs[refs.setting == st]
        O, P = ref[ref.arm == "ORACLE"], ref[ref.arm == "PLAIN"]
        lines += [f"## {st}", "", f"Runs per strength: {n}. References ({'this folder' if demo else 'Study T'}): ORACLE {len(O)} runs "
                  f"(X {O.X.mean():.3f}, E {O.E.mean():+.2f}), PLAIN {len(P)} runs (X {P.X.mean():.3f}, E {P.E.mean():+.2f}).", "",
                  "| strength | runs | mean X | mean E (booked) | mean T (TIR audit) | mean R (reported) | both? |", "|---|---|---|---|---|---|---|"]
        testable = all(n[a] >= 6 for a in GRID) and len(O) > 0
        means = {}
        for a in GRID:
            x = d[d.arm == a]
            if len(x):
                means[a] = dict(E=x.E.mean(), T=x.TA.mean(), R=x.R.mean(), X=x.X.mean(), both=both(x, O) if len(O) else None)
                lines.append(f"| {a} | {len(x)} | {x.X.mean():.3f} | {x.E.mean():+.2f} | {x.TA.mean():+.2f} | {x.R.mean():+.2f} | {means[a]['both']} |")
        lines.append("")
        if not testable:
            verdicts += [(f"W1-{st}", f"fewer than 6 runs in an arm: {n}", None), (f"W2-{st}", f"fewer than 6 runs in an arm: {n}", None)]
            continue
        sel_t = max(GRID, key=lambda a: means[a]["T"])
        sel_b = max(GRID, key=lambda a: means[a]["E"])
        sel_r = max(GRID, key=lambda a: means[a]["R"])
        picks[st] = dict(audit=sel_t, booked=sel_b, reported=sel_r, gap_audit=means[sel_b]["E"] - means[sel_t]["E"],
                         gap_reported=means[sel_b]["E"] - means[sel_r]["E"])
        lines += [f"Audit-selected {sel_t} (E {means[sel_t]['E']:+.2f}); booked-best {sel_b} (E {means[sel_b]['E']:+.2f}); "
                  f"report-selected {sel_r} (E {means[sel_r]['E']:+.2f}).", ""]
        verdicts.append((f"W1-{st}", f"audit-selected {sel_t} E {means[sel_t]['E']:+.2f} vs booked-best {sel_b} E {means[sel_b]['E']:+.2f}: "
                                     f"shortfall {picks[st]['gap_audit']:.2f} (need <= {TOL})", picks[st]["gap_audit"] <= TOL))
        verdicts.append((f"W2-{st}", f"audit-selected {sel_t}: X {means[sel_t]['X']:.3f} (need <= ORACLE {O.X.mean():.3f} + 0.10), "
                                     f"E {means[sel_t]['E']:+.2f} (need >= {0.80 * O.E.mean():+.2f} = 0.80 x ORACLE {O.E.mean():+.2f})", means[sel_t]["both"]))
    if len(df):
        e1, e2 = float(df.err_tir_edge.max()), float(df.err_tir_plac.max())
        verdicts.append(("W3", f"max |TIR - booked| = {e1:.3f} (edge market), {e2:.3f} (placebo market) over {len(df)} agents (need <= {TOL})", max(e1, e2) <= TOL))
    if picks:
        gaps = {st: p["gap_reported"] for st, p in picks.items()}
        verdicts.append(("W4", "booked-best minus report-selected mean E: " + ", ".join(f"{s} {g:.2f} ({picks[s]['reported']})" for s, g in gaps.items())
                         + f" (need >= {TOL} in at least one setting)", any(g >= TOL for g in gaps.values())))
    return lines, verdicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results_w"))
    ap.add_argument("--demo", action="store_true", help="software check on Study V's ruin-penalty runs; writes nothing")
    a = ap.parse_args()
    df = load(Path(a.results))
    if a.demo:
        settings = ("S_DK1", "S_DK1+Z")
        refs = df
    else:
        settings = SETTINGS
        refs = pd.read_csv(RUNS_T)
    lines = ["# Study W verdicts" if not a.demo else "# Study W software check on Study V's ruin-penalty runs (not a result of Study W)", "",
             f"Runs: {len(df)} of {len(settings) * len(GRID) * N_EXPECTED if not a.demo else 'n/a'}.", ""]
    body, verdicts = analyse(df, refs, settings, a.demo)
    lines += body + ["| ID | Result | Verdict |", "|---|---|---|"] + [f"| {i} | {r} | {verdict(ok)} |" for i, r, ok in verdicts]
    text = "\n".join(lines) + "\n"
    if a.demo:
        print(text)
        return
    df.to_csv(HERE / "runs_w.csv", index=False)
    (HERE / "VERDICTS_W.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
