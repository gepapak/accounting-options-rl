"""Study U analysis (PREREGISTRATION_U.md): writes runs_u.csv and VERDICTS_U.md.

    python aggregate_u.py                         # registered analysis of results_u/*.json
    python aggregate_u.py --results DIR --print   # descriptive summary of another folder (pilots); writes nothing
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
B_BOOT = 10_000
rng = np.random.default_rng(20261031)
Z_PROBE_N = (-0.5, -0.25, -0.1, -0.02)
Z_PROBE_K = (96, 48, 12)
T_PROBE_L = (0.02, 0.1, 0.25)
T_PROBE_TAU = (1, 4, 16, 48, 96)
SETTINGS = {"Z": ("ZFG", "ZBK", "ZSPR", "ZTIR"), "T": ("TFG", "TBK", "TSPR", "TTIR")}
NAME = {"Z": "settlement lag (zombie window)", "T": "terminal limited liability"}


def outcome(d: dict) -> float:
    """Mean absolute leverage over the registered probes: insolvent probes (zombie) or ledger 0.02/0.1/0.25 with at most 96 quarter-hours to go."""
    if d["arm"][0] == "Z":
        vals = [d["probes"][f"zombie_n{n:g}_k{k}"] for n in Z_PROBE_N for k in Z_PROBE_K]
    else:
        vals = [d["probes"][f"L{L:g}_tau{tau}"] for L in T_PROBE_L for tau in T_PROBE_TAU]
    return float(np.mean(vals))


def load(folder: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(folder.glob("*.json")):
        d = json.loads(p.read_text())
        rows.append(dict(arm=d["arm"], setting=d["arm"][0], seed=d["seed"], region=d["region"], X=outcome(d),
                         booked=d["drawmean_booked_pct"], reported=d["drawmean_reported_pct"]))
    return pd.DataFrame(rows)


def boot_diff(a: np.ndarray, b: np.ndarray):
    """Run-level bootstrap of mean(a) - mean(b): (point, 5th percentile, 95th percentile)."""
    d = np.array([rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(B_BOOT)])
    return float(a.mean() - b.mean()), float(np.quantile(d, 0.05)), float(np.quantile(d, 0.95))


def verdict(ok) -> str:
    if ok is None:
        return "NOT TESTABLE"
    return "SUPPORTED" if ok else "NOT SUPPORTED"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results_u"))
    ap.add_argument("--print", dest="print_only", action="store_true")
    a = ap.parse_args()
    df = load(Path(a.results))
    lines = ["# Study U verdicts", "", f"Runs: {len(df)} of 80.", ""]
    verdicts = []
    for s, (fg, bk, spr, tir) in SETTINGS.items():
        d = df[df.setting == s]
        if not len(d):
            continue
        X = {arm: d[d.arm == arm].X.to_numpy() for arm in (fg, bk, spr, tir)}
        lines += [f"## {NAME[s]}", "", "| arm | runs | mean X | mean booked (% of allocation) | mean reported |", "|---|---|---|---|---|"]
        for arm in (fg, bk, spr, tir):
            x = d[d.arm == arm]
            if len(x):
                lines.append(f"| {arm} | {len(x)} | {x.X.mean():.2f} | {x.booked.mean():+.1f} | {x.reported.mean():+.1f} |")
        lines.append("")
        if min(len(v) for v in X.values()) < 2:
            continue
        pt, lo, _ = boot_diff(X[fg], X[bk])
        ok1 = pt >= 8 and lo > 4
        verdicts.append((f"U1 {s}", f"X({fg}) - X({bk}) = {pt:+.2f} (need >= 8), one-sided 95% bootstrap lower bound {lo:+.2f} (need > 4)", ok1))
        for tag, arm in (("U2", spr), ("U3", tir)):
            pt, lo, _ = boot_diff(X[fg], X[arm])
            near = X[arm].mean() - X[bk].mean()
            ok = pt >= 4 and lo > 0 and near < 4
            res = f"X({fg}) - X({arm}) = {pt:+.2f} (need >= 4), lower bound {lo:+.2f} (need > 0); X({arm}) - X({bk}) = {near:+.2f} (need < 4)"
            if not ok1:
                verdicts.append((f"{tag} {s}", res + "; the exploit was not acquired (U1 failed)", None))
            else:
                verdicts.append((f"{tag} {s}", res, ok))
    lines += ["| ID | Result | Verdict |", "|---|---|---|"] + [f"| {i} | {r} | {verdict(ok)} |" for i, r, ok in verdicts]
    text = "\n".join(lines) + "\n"
    if a.print_only:
        print(text)
        return
    df.to_csv(HERE / "runs_u.csv", index=False)
    (HERE / "VERDICTS_U.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
