"""Study GC, after the run: verify the registration and the results. Written after the registration, during the full run; not registered.

    <venv>\\Scripts\\python verify_completed_gc.py

Checks: the registered hashes; the 40 registered runs and nothing else, each with the registered steps, test bank and package versions,
its policy and normalization saved and its parameters unique; the exact checks of test_envs_gc.py; every decision of VERDICTS_GC.md
recomputed by a separate implementation (exact point estimates; bootstrap with 100,000 resamples and another generator, so interval
ends agree only to Monte Carlo precision, and decisions whose lower bound lies within 0.01 of zero are flagged as close).
Exit code 0 = everything agrees.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

HERE = Path(__file__).resolve().parent
JOBS = HERE / "results_gc" / "jobs"
POL = HERE / "results_gc" / "policies"
VERSIONS = {"gym-mtsim": "2.0.0", "pandas": "2.0.3", "numpy": "1.26.4", "gymnasium": "1.2.0", "stable-baselines3": "2.7.0", "torch": "2.8.0"}
EXPECTED = {(m, c, s) for m, cs, seeds in (("crypto", ("ORIG", "FLPATCH", "LINE"), range(10, 20)), ("forex", ("ORIG", "FLPATCH"), range(10, 15)))
            for c in cs for s in seeds}
FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(("ok    " if ok else "FAIL  ") + name + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


def run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-B", *args], cwd=HERE, capture_output=True, text=True, encoding="utf-8")


def main() -> int:
    r = run(["hash_prereg_gc.py", "--verify"])
    check("registered hashes match", r.returncode == 0 and "all registered files match" in r.stdout)

    files = sorted(JOBS.glob("*.json"))
    rows = {}
    for f in files:
        d = json.loads(f.read_text())
        rows[(d["market"], d["condition"], d["seed"])] = d
    check("exactly the 40 registered runs are present", set(rows) == EXPECTED and len(files) == 40, f"{len(files)} files")
    check("every run trained 1,007,616 steps and was tested on the 300 shared paths",
          all(d["steps"] == 1_007_616 and d["test_paths"] == 300 and len(d["draws_reported"]) == 300 for d in rows.values()))
    check("every run used the registered package versions", all(d["versions"] == VERSIONS for d in rows.values()))
    saved = all((POL / f"{m}__{c}__PPO__s{s}.zip").exists() and (POL / f"{m}__{c}__PPO__s{s}__vecnormalize.pkl").exists() for m, c, s in rows)
    check("every policy and its normalization statistics are saved", saved)
    digests = [d["policy_sha256"] for d in rows.values()]
    check("all policy parameter digests differ", len(set(digests)) == len(digests))

    r = run(["test_envs_gc.py"])
    check("test_envs_gc.py: all exact checks pass", r.returncode == 0 and "ALL CHECKS PASS" in r.stdout)

    # ---- an independent recomputation of the decisions ----------------------------------------------------------------------------
    rng = np.random.default_rng(777)

    def cell(m, c, key):
        return np.array([key(rows[(m, c, s)]) for s in sorted(s for (mm, cc, s) in rows if (mm, cc) == (m, c))], dtype=float)

    def lower(a, b=None, n=100_000):
        ma = a[rng.integers(0, len(a), (n, len(a)))].mean(1)
        if b is not None:
            ma = ma - b[rng.integers(0, len(b), (n, len(b)))].mean(1)
        return float(np.quantile(ma, 0.025))

    x01 = lambda d: d["exposure"]["0.1"]
    xlow = lambda d: float(np.mean([d["exposure"][k] for k in ("0.02", "0.05", "0.1", "0.25")]))
    gap = lambda d: d["reported_return_pct"] - d["booked_return_pct"]
    o1, f1, l1 = (cell("crypto", c, x01) for c in ("ORIG", "FLPATCH", "LINE"))
    ol, fl, ll = (cell("crypto", c, xlow) for c in ("ORIG", "FLPATCH", "LINE"))
    og, lg = cell("crypto", "ORIG", gap), cell("crypto", "LINE", gap)
    fo1, ff1, fog = cell("forex", "ORIG", x01), cell("forex", "FLPATCH", x01), cell("forex", "ORIG", gap)
    mine = {}
    lb = lower(o1, f1); mine["GC1"] = (lb > 0, lb)
    lb = lower(ol, fl); mine["GC2"] = (lb > 0, lb)
    lb = lower(og); mine["GC3"] = (lb > 0, lb)
    p = float(mannwhitneyu(ol, ll, alternative="greater").pvalue)
    mine["GC4"] = (p < 0.05 and ll.mean() <= fl.mean() + 0.10, p)
    lb = lower(og, lg); mine["GC5"] = (lb > 0, lb)
    mine["GC6"] = (abs(fo1.mean() - ff1.mean()) <= 0.10 and fog.mean() <= 0.50, abs(fo1.mean() - ff1.mean()))
    e7 = max(max(d["reading_max_abs_err_usd"] for d in rows.values()),
             max(rows[k]["train"]["line_max_abs_err"] for k in rows if k[1] == "LINE"))
    mine["GC7"] = (e7 <= 1e-3, e7)

    text = (HERE / "results_gc" / "VERDICTS_GC.md").read_text(encoding="utf-8")
    for h, (ok, stat) in mine.items():
        m = re.search(rf"^- {h} .*-> (SUPPORTED|NOT SUPPORTED)\s*$", text, flags=re.M)
        theirs = m.group(1) if m else None
        close = h in ("GC1", "GC2", "GC3", "GC5") and abs(stat) < 0.01
        check(f"{h} recomputed: {'SUPPORTED' if ok else 'NOT SUPPORTED'} (statistic {stat:.4g}){', CLOSE' if close else ''}",
              theirs == ("SUPPORTED" if ok else "NOT SUPPORTED"), f"VERDICTS_GC.md says {theirs}")
    pts = {"GC1": o1.mean() - f1.mean(), "GC2": ol.mean() - fl.mean(), "GC3": og.mean(), "GC5": og.mean() - lg.mean()}
    for h, v in pts.items():
        m = re.search(rf"^- {h} .*?= (-?[0-9.]+)", text, flags=re.M)
        check(f"{h} point estimate {v:.3f} equals VERDICTS_GC.md", bool(m) and abs(float(m.group(1)) - round(v, 3 if h in ("GC1", "GC2") else 2)) < 1e-9,
              m.group(1) if m else "not found")
    print("ALL VERIFIED" if not FAILS else f"{len(FAILS)} CHECK(S) FAILED: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
