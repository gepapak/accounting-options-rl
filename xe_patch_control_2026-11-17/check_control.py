"""Exact checks of the same-engine control before any training: the published engine and the floor-removed engine on the same 300
paths, under 'always 200x' and 'always flat'."""
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np

import rl_engine_env as R

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
mags = np.load("eurusd_mags.npy")
rng = np.random.default_rng(11)
bank = [R.placebo(rng, mags) for _ in range(300)]
res = {}
for name, cls in (("floor (as published)", R.BtpyEnv), ("floor removed", R.BtpyBookedEnv)):
    for act, lab in ((1, "always 200x"), (0, "always flat")):
        env = cls(mags, seed=0, train=False, levels=(0.0, 200.0))
        rep, book, zero, rsum = [], [], [], []
        for close in bank:
            o, _ = env.reset(options={"path": close, "cash": 10_000.0})
            done, tot = False, 0.0
            while not done:
                o, r, te, tr, info = env.step(act)
                tot += r
                done = te or tr
            s = info["stats"]
            rep.append(s["reported_equity"]); book.append(s["booked_equity"]); zero.append(s["hit_zero"]); rsum.append(tot)
        env.close()
        rep, book, zero, rsum = np.array(rep), np.array(book), np.array(zero), np.array(rsum)
        res[(name, lab)] = (rep, book, zero)
        print(f"{name:<22} {lab:<12}: reported {100 * (rep.mean() / 1e4 - 1):+8.2f}%  booked {100 * (book.mean() / 1e4 - 1):+8.2f}%  "
              f"max |reported - booked| {np.abs(rep - book).max():12.4f} USD  ruin {zero.mean():.3f}  lowest reported equity {rep.min():12.1f}  "
              f"|sum of rewards - reported change| {np.abs(rsum * 1e4 - (rep - 1e4)).max():.1e}")
a, b = res[("floor (as published)", "always 200x")], res[("floor removed", "always 200x")]
print("booked equity identical in the two engines on every path (max abs difference, USD):", float(np.abs(a[1] - b[1]).max()))
print("ruin flags identical:", bool((a[2] == b[2]).all()), "| reported identical on the paths without ruin:", float(np.abs(a[0][~a[2]] - b[0][~a[2]]).max()))
print("floor removed: reported equals its own trade records on every path:", bool(np.abs(b[0] - b[1]).max() < 1e-6))
