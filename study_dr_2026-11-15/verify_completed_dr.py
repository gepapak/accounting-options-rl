"""Study DR, after the run: verify the registration and the results. Written after the registration files, before the full run finished;
not registered.

    <venv>\\Scripts\\python verify_completed_dr.py

Checks:
  - the registered hashes;
  - the 64 registered runs and nothing else, each with the registered steps, both test banks and the package versions, its policy and
    normalization saved, and its parameters unique;
  - the exact checks of test_envs_dr.py;
  - every decision of VERDICTS_DR.md recomputed by a separate implementation: bootstrap with 100,000 resamples and another generator,
    so interval ends agree only to Monte Carlo precision, and lower bounds within 0.01 of zero are flagged as close; the
    Jonckheere-Terpstra p-value from 200,000 permutations with another generator and, as a cross-check, its normal approximation with
    the tie correction.
Exit code 0 = everything agrees.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
JOBS = HERE / "results_dr" / "jobs"
POL = HERE / "results_dr" / "policies"
VERSIONS = {"gym-mtsim": "2.0.0", "pandas": "2.0.3", "numpy": "1.26.4", "gymnasium": "1.2.0", "stable-baselines3": "2.7.0", "torch": "2.8.0"}
SEEDS = {("V2", "ORIG"): range(600, 606), ("V4", "ORIG"): range(600, 610), ("V8", "ORIG"): range(600, 610),
         ("V16", "ORIG"): range(600, 610), ("L30", "ORIG"): range(600, 606), ("L500", "ORIG"): range(600, 610),
         ("V16", "FLPATCH"): range(600, 606), ("L500", "FLPATCH"): range(600, 606)}
EXPECTED = {(arm, c, s) for (arm, c), seeds in SEEDS.items() for s in seeds}
FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(("ok    " if ok else "FAIL  ") + name + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


def run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-B", *args], cwd=HERE, capture_output=True, text=True, encoding="utf-8")


def jt_pairs(groups):
    s = 0.0
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            for x in groups[i]:
                s += float(np.sum(groups[j] > x)) + 0.5 * float(np.sum(groups[j] == x))
    return s


def main() -> int:
    r = run(["hash_prereg_dr.py", "--verify"])
    check("registered hashes match", r.returncode == 0 and "all registered files match" in r.stdout)

    files = sorted(JOBS.glob("*.json"))
    rows = {}
    for f in files:
        d = json.loads(f.read_text())
        rows[(d["arm"], d["condition"], d["seed"])] = d
    check("exactly the 64 registered runs are present", set(rows) == EXPECTED and len(files) == 64, f"{len(files)} files")
    check("every run trained 1,007,616 steps and was tested on both banks of 300 paths",
          all(d["steps"] == 1_007_616 and d["test_nominal"]["test_paths"] == 300 and d["test_stressed_k4"]["test_paths"] == 300
              and len(d["test_nominal"]["draws_reported"]) == 300 for d in rows.values()))
    check("every run used the registered package versions", all(d["versions"] == VERSIONS for d in rows.values()),
          str(next(iter(rows.values()))["versions"]) if rows else "")
    saved = all((POL / f"{a}__{c}__PPO__s{s}.zip").exists() and (POL / f"{a}__{c}__PPO__s{s}__vecnormalize.pkl").exists() for a, c, s in rows)
    check("every policy and its normalization statistics are saved", saved)
    digests = [d["policy_sha256"] for d in rows.values()]
    check("all policy parameter digests differ", len(set(digests)) == len(digests))

    r = run(["test_envs_dr.py"])
    check("test_envs_dr.py: all exact checks pass", r.returncode == 0 and "ALL CHECKS PASS" in r.stdout)

    # ---- an independent recomputation of the decisions ----------------------------------------------------------------------------
    rng = np.random.default_rng(777)
    xlow = lambda d: float(np.mean([d["exposure"][k] for k in ("0.02", "0.05", "0.1", "0.25")]))  # noqa: E731

    def cell(arm, cond):
        return np.array([xlow(rows[(arm, cond, s)]) for s in sorted(s for (a, c, s) in rows if (a, c) == (arm, cond))], dtype=float)

    def lower(a, b, n=100_000):
        ma = a[rng.integers(0, len(a), (n, len(a)))].mean(1) - b[rng.integers(0, len(b), (n, len(b)))].mean(1)
        return float(np.quantile(ma, 0.025))

    fv, fl = cell("V16", "FLPATCH"), cell("L500", "FLPATCH")
    mine, close = {}, {}
    lb8, lb16 = lower(cell("V8", "ORIG"), fv), lower(cell("V16", "ORIG"), fv)
    mine["DR1"], close["DR1"] = lb8 > 0 and lb16 > 0, min(abs(lb8), abs(lb16)) < 0.01
    lb5 = lower(cell("L500", "ORIG"), fl)
    mine["DR2"], close["DR2"] = lb5 > 0, abs(lb5) < 0.01
    mine["DR3"] = all(int(np.sum(cell(arm, "ORIG") >= 0.10)) <= 1 for arm in ("V2", "L30"))
    groups = [cell(arm, "ORIG") for arm in ("V2", "V4", "V8", "V16")]
    jt = jt_pairs(groups)
    pooled, sizes = np.concatenate(groups), np.cumsum([len(g) for g in groups])[:-1]
    hits = sum(jt_pairs(np.split(rng.permutation(pooled), sizes)) >= jt - 1e-9 for _ in range(200_000))
    p_perm = (hits + 1) / 200_001
    n = np.array([len(g) for g in groups])
    N = n.sum()
    _, t = np.unique(pooled, return_counts=True)
    mean = (N ** 2 - np.sum(n ** 2)) / 4
    var = ((N ** 2 * (2 * N + 3) - np.sum(n ** 2 * (2 * n + 3))) / 72
           - np.sum(t * (t - 1) * (2 * t + 5)) / 72)                    # tie-corrected (first-order) variance
    p_norm = float(norm.sf((jt - mean) / np.sqrt(var))) if var > 0 else float("nan")
    mine["DR4"], close["DR4"] = p_perm < 0.05, 0.02 < p_perm < 0.10
    print(f"recomputed: DR1 lower bounds V8 {lb8:.4f}, V16 {lb16:.4f}; DR2 lower bound {lb5:.4f}; DR4 J = {jt:.1f}, permutation p = "
          f"{p_perm:.5f}, normal approximation p = {p_norm:.5f}")

    text = (HERE / "results_dr" / "VERDICTS_DR.md").read_text(encoding="utf-8")
    for h, ok in mine.items():
        m = re.search(rf"^- {h} .*-> (SUPPORTED|NOT SUPPORTED)\s*$", text, flags=re.M)
        theirs = m.group(1) if m else None
        check(f"{h} recomputed: {'SUPPORTED' if ok else 'NOT SUPPORTED'}{', CLOSE' if close.get(h) else ''}",
              theirs == ("SUPPORTED" if ok else "NOT SUPPORTED"), f"VERDICTS_DR.md says {theirs}")
    m = re.search(r"J = ([0-9.]+) of", text)
    check(f"DR4 statistic J = {jt:.1f} equals VERDICTS_DR.md", bool(m) and abs(float(m.group(1)) - round(jt, 1)) < 1e-9,
          m.group(1) if m else "not found")
    print("ALL VERIFIED" if not FAILS else f"{len(FAILS)} CHECK(S) FAILED: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
