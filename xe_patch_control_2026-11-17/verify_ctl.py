"""Independent checks of the same-engine control, run after training (nothing here was used to score the predictions).
1. The patched broker differs from backtesting.py's own _Broker.next only in the floor lines.
2. The control's two wrapper files differ from Study XE's registered files only where intended, and those files are the registered ones.
3. The recorded test draws are reproduced by fresh engine runs on the regenerated test bank.
4. The monitoring law's prediction for 'always 200x' (written at 01:50, before this run) against this bank."""
import difflib
import hashlib
import inspect
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["TQDM_DISABLE"] = "1"
import numpy as np
import backtesting
from backtesting.backtesting import _Broker

import rl_engine_env as R

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
XE = HERE.parent / "study_xe_2026-11-16"

print(f"1. backtesting {backtesting.__version__}: lines that differ between the package's _Broker.next and BookedBroker.next")
a = inspect.getsource(_Broker.next).splitlines()
b = inspect.getsource(R.BookedBroker.next).splitlines()
for line in difflib.unified_diff(a, b, lineterm="", n=0):
    if not line.startswith(("@@", "---", "+++")):
        print("   " + line)

prereg = (XE / "PREREGISTRATION_HASHES_XE.txt").read_text(encoding="utf-8")       # the registration's hash manifest
for name in ("rl_engine_env.py", "rl_engine_pilot.py"):
    h = hashlib.sha256((XE / name).read_bytes()).hexdigest()
    a = (XE / name).read_text(encoding="utf-8").splitlines()
    b = (HERE / name).read_text(encoding="utf-8").splitlines()
    diff = [x for x in difflib.unified_diff(a, b, lineterm="", n=0) if not x.startswith(("@@", "---", "+++"))]
    print(f"2. {name}: Study XE's copy hash {h[:12]}... {'is' if h in prereg else 'is NOT'} in PREREGISTRATION_HASHES_XE.txt; "
          f"{sum(x.startswith('+') for x in diff)} lines added, {sum(x.startswith('-') for x in diff)} removed or changed:")
    for x in diff:
        if x.startswith("-"):
            print("   " + x)

mags = np.load(HERE / "eurusd_mags.npy")
rng = np.random.default_rng(20261211)
bank = [R.placebo(rng, mags, 1.0) for _ in range(200)]
kw = dict(levels=(0.0, 200.0), l_max=200.0, mag_scale=1.0)


def run(cls, act, cash):
    env = cls(mags, seed=0, train=False, **kw)
    rep, book = [], []
    for close in bank:
        o, _ = env.reset(options={"path": close, "cash": cash})
        done = False
        while not done:
            o, r, te, tr, info = env.step(act)
            done = te or tr
        s = info["stats"]
        rep.append(100 * (s["reported_equity"] / cash - 1))
        book.append(100 * (s["booked_equity"] / cash - 1))
    env.close()
    return np.array(rep), np.array(book)


fresh = {(e, act, c): run(cls, act, float(c)) for e, cls in (("btpy", R.BtpyEnv), ("btpyfl", R.BtpyBookedEnv))
         for act in (0, 1) for c in ("10000", "1000")}
print("3. recorded test draws against fresh runs on the regenerated bank (largest absolute difference, points)")
for p in sorted((HERE / "results_ctl").glob("*_ctl.json")):
    r = json.loads(p.read_text())
    out = []
    for c in ("10000", "1000"):
        rec = (np.array(r["tests"][c]["draws_reported"]), np.array(r["tests"][c]["draws_booked"]))
        m = {lab: max(float(np.abs(rec[0] - fresh[(r["engine"], act, c)][0]).max()), float(np.abs(rec[1] - fresh[(r["engine"], act, c)][1]).max()))
             for act, lab in ((1, "always 200x"), (0, "always flat"))}
        best = min(m, key=m.get)
        out.append(f"from {int(c):>6,}: {best if m[best] < 1e-9 else 'neither (mixed policy)'}")
    print(f"   {r['engine']:<6} seed {r['seed']}: " + "; ".join(out))
pub, ctl = fresh[("btpy", 1, "10000")], fresh[("btpyfl", 1, "10000")]
print(f"   'always 200x' from 10,000 USD: published engine reports {pub[0].mean():+.2f}% and books {pub[1].mean():+.2f}%; "
      f"floor removed reports {ctl[0].mean():+.2f}% and books {ctl[1].mean():+.2f}% (booked identical: {float(np.abs(pub[1] - ctl[1]).max()):.1e})")

law = next(x for x in json.loads((HERE.parent / "monitoring_law_2026-11-17" / "predictions_v2.json").read_text())["D1"] if x["L"] == 200)
g = pub[0] - pub[1]
print(f"4. the monitoring law (written 01:50, before this run) for 200x with a 2 bp fee: forgiven {100 * law['forgiven']:.2f} points, "
      f"P(ruin) {law['p_ruin']:.3f}; this bank: {g.mean():.2f} (standard error {g.std(ddof=1) / np.sqrt(len(g)):.2f}), "
      f"ruin share {float(np.mean(pub[0] <= -100.0 + 1e-9)):.3f}")
