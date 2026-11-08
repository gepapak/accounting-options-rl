"""Study RS checks (prints only; writes nothing): the real-sign market arrays, the inherited step arithmetic, the twin, the step bracket,
TIR's exactness on identified steps, the training-reward wrappers, the placebo signal and the zero-equity training starts.

    python test_envs_rs.py
"""
from __future__ import annotations

import copy

import numpy as np

from envs_rs import EPISODE, FEE, K0, LAG, LEV, PLACEBO_MIN_SHIFT, Q1, SETTINGS, EdgeEnvRS, load_market, market_arrays
from rewards_v import make_ruin
from spr_t import make_spr, make_tir

TOL = 1e-9
n_checks = 0


def check(cond, msg):
    global n_checks
    assert cond, msg
    n_checks += 1


# 1. market arrays are the real spreads; the signal is the sign of the spread LAG quarters earlier
for st, cfg in SETTINGS.items():
    for sp in ("train", "test"):
        m = load_market(cfg["region"], sp, "RM")
        s = np.asarray(m.spread, dtype=float)
        a, e, z, c = market_arrays(st, sp)
        check(np.all(np.isfinite(s)) and np.all(np.isfinite(c)), f"{st} {sp}: non-finite data")
        check(np.allclose(a * e, s, atol=1e-12, rtol=0), f"{st} {sp}: |s| * sign(s) != s")
        check(set(np.unique(e)) <= {-1.0, 0.0, 1.0} and set(np.unique(z)) <= {-1.0, 0.0, 1.0}, f"{st} {sp}: sign values")
        check(np.array_equal(z[LAG:], np.sign(s[:-LAG])) and np.all(z[:LAG] == 0), f"{st} {sp}: signal is not sign(s[t-{LAG}])")
        check(not a.flags.writeable, f"{st} {sp}: arrays must be read-only")

# 2. the inherited step reproduces the documented arithmetic (both reward modes), on random policies and low start equities
rng = np.random.default_rng(1)
for st, cfg in SETTINGS.items():
    for mode in ("reported", "booked"):
        env = EdgeEnvRS(st, "train", "edge", mode, seed=3)
        a_, e_, z_, c_ = market_arrays(st, "train")
        for ep in range(6):
            o, _ = env.reset(options={"e0": float(rng.choice([0.0, 0.01, 0.05, 0.3, 1.5]))})
            check(env.t_end - env.t0 == EPISODE and 200 <= env.t0 <= env.n - EPISODE, "episode window")
            ledger, rep, forg = env.e0, env.e0, 0.0
            done = False
            while not done:
                t = env.t
                check(o[1] == z_[t] and abs(o[2] - (env.t_end - t) / EPISODE) < 1e-6, "observation: signal and remaining share")
                check(abs(o[0] - np.clip(rep / K0, -5, 5)) < 1e-5, "observation: reported equity")
                act = int(rng.integers(len(LEV)))
                q = float(np.clip(LEV[act] * Q1, -c_[t], c_[t]))
                cost = FEE * abs(q) + cfg["kappa"] * q * q
                pnl = q * a_[t] * e_[t] - cost
                ledger += pnl
                f = max(0.0, -(rep + pnl))
                rep = max(rep + pnl, 0.0)
                forg += f
                o, r, done, _, info = env.step(act)
                want = (pnl + f) if mode == "reported" else pnl
                check(abs(r - want / K0 * 100) < 1e-9 and abs(info["pnl"] - pnl) < 1e-9 and abs(info["cost"] - cost) < 1e-12, "step arithmetic")
                check(abs(env.ledger - ledger) < 1e-6 and abs(env.reported - rep) < 1e-6 and abs(env.forgiven - forg) < 1e-6, "state")

# 3. the twin: a shallow copy with flip_next reverses the real sign, leaves the real simulator untouched, and the flag is one-shot
# 4. the step bracket -r' - 2c <= x <= r on every step; TIR exact on identified steps (r != r'); otherwise |TIR - x| <= g <= c
n_steps = n_ident = n_unid = 0
for st in SETTINGS:
    env = EdgeEnvRS(st, "train", "edge", "reported", seed=7)
    for ep in range(30):
        env.reset(options={"e0": float(rng.choice([0.0, 0.0, 0.005, 0.02, 0.05, 0.2, 1.0]))})
        done = False
        while not done:
            act = int(rng.choice([0, 8, 1, 7, 4], p=[0.3, 0.3, 0.15, 0.15, 0.1]))
            state = (env.t, env.ledger, env.reported, env.forgiven, env.flip_next)
            eps_t = float(env.eps_ep[env.t - env.t0])
            twin = copy.copy(env)
            twin.flip_next = True
            _, r_t, _, _, i_t = twin.step(act)
            check(state == (env.t, env.ledger, env.reported, env.forgiven, env.flip_next), "twin step mutated the real simulator")
            check(twin.flip_next is False, "flip_next must be one-shot")
            _, r, done, _, info = env.step(act)
            c = info["cost"] / K0 * 100
            x, x_t = info["pnl"] / K0 * 100, i_t["pnl"] / K0 * 100
            check(abs(x + x_t + 2 * c) < 1e-9, "twin must reverse the gross P&L and keep the cost")
            check(-r_t - 2 * c <= x + TOL and x <= r + TOL, "step bracket violated")
            g = abs(x + c)
            sigma = float(np.sign(LEV[act])) * eps_t
            ident = abs(r - r_t) > 1e-12
            g_hat = max(r, r_t) + c if ident else 0.0
            tir = sigma * g_hat - c
            if ident:
                n_ident += 1
                check(abs(tir - x) < 1e-9, "TIR must be exact on identified steps")
            else:
                n_unid += 1
                check(abs(tir - x) <= g + 1e-9 and g <= c + 1e-9, "unidentified step: error bounded by g <= c")
            n_steps += 1
check(n_ident > 1000 and n_unid > 100, "both kinds of steps must occur")

# 5. the wrappers return the documented rewards on the real-sign environment
for st in SETTINGS:
    for name in ("SPR", "TIR", "RUIN5"):
        base = EdgeEnvRS(st, "train", "edge", "reported", seed=11)
        ref = EdgeEnvRS(st, "train", "edge", "reported", seed=11)
        w = {"SPR": make_spr, "TIR": make_tir, "RUIN5": lambda e_: make_ruin(e_, 5.0)}[name](base)
        w.reset(options={"e0": 0.03})
        ref.reset(options={"e0": 0.03})
        done = False
        while not done:
            act = int(rng.choice([0, 8, 4, 2, 6]))
            eps_t = float(ref.eps_ep[ref.t - ref.t0])
            before = ref.reported
            tw = copy.copy(ref)
            tw.flip_next = True
            r_t = tw.step(act)[1]
            _, r, _, _, info = ref.step(act)
            c = info["cost"] / K0 * 100
            if name == "SPR":
                want = 0.5 * (r - r_t) - c
            elif name == "TIR":
                want = float(np.sign(LEV[act])) * eps_t * (max(r, r_t) + c if abs(r - r_t) > 1e-12 else 0.0) - c
            else:
                want = r - (5.0 if (before > 0 and ref.reported <= 0) else 0.0)
            _, got, done, _, _ = w.step(act)
            check(abs(got - want) < 1e-9, f"{name} wrapper reward")

# 6. placebo: real moves of the window, signal from a window at least PLACEBO_MIN_SHIFT away
for st in SETTINGS:
    pp = EdgeEnvRS(st, "test", "placebo", "reported", seed=9500)
    a_, e_, z_, _ = market_arrays(st, "test")
    agree = []
    for ep in range(50):
        pp.reset(options={"e0": 0.5})
        check(np.array_equal(pp.eps_ep, e_[pp.t0:pp.t0 + EPISODE]), "placebo moves must be the real moves of the window")
        nz = (pp.z_ep != 0) & (pp.eps_ep != 0)
        agree.append(float(np.mean(pp.z_ep[nz] == pp.eps_ep[nz])))
    check(abs(np.mean(agree) - 0.5) < 0.12, f"{st}: placebo signal should be roughly uninformative ({np.mean(agree):.3f})")

# 7. determinism: two environments with the same seed give the same windows and start equities
for st in SETTINGS:
    u, v = EdgeEnvRS(st, "test", "edge", "reported", seed=9000), EdgeEnvRS(st, "test", "edge", "reported", seed=9000)
    for _ in range(20):
        u.reset()
        v.reset()
        check(u.t0 == v.t0 and u.e0 == v.e0, "determinism")

# 8. zero-equity training starts: half of the unspecified starts at zero, the draw order of Study V's EdgeEnvZ, explicit starts untouched
from rewards_v import EdgeEnvZ  # noqa: E402
from envs_rs import E0_TRAIN  # noqa: E402

for st, tst in (("RS_DK1", "S_DK1"), ("RS_DK2", "T_DK2")):
    zs, ref = EdgeEnvRS(st, "train", "edge", "reported", seed=21, zero_starts=True), EdgeEnvZ(tst, "train", "edge", "reported", seed=21)
    plain_ = EdgeEnvRS(st, "train", "edge", "reported", seed=21)
    check(zs.n == ref.n, "same training window range as Study V")
    n0 = 0
    for _ in range(2000):
        zs.reset()
        ref.reset()
        check(zs.t0 == ref.t0 and zs.e0 == ref.e0, "zero starts must follow EdgeEnvZ's draw order")
        n0 += int(zs.e0 == 0.0)
        check(zs.e0 == 0.0 or E0_TRAIN[0] * K0 - 1e-9 <= zs.e0 <= E0_TRAIN[1] * K0 + 1e-9, "non-zero start in the training range")
        plain_.reset()
        check(plain_.e0 > 0.0, "no zero starts without zero_starts")
    check(0.45 < n0 / 2000 < 0.55, f"{st}: share of zero starts {n0 / 2000:.3f}")
    for x in (0.0, 0.05, 1.0):
        zs.reset(options={"e0": x})
        check(zs.e0 == x * K0 and zs.reported == x * K0 and zs.ledger == x * K0, "explicit start equity untouched")
    ru = make_ruin(EdgeEnvRS(st, "train", "edge", "reported", seed=22, zero_starts=True), 5.0)
    twin_free = EdgeEnvRS(st, "train", "edge", "reported", seed=22, zero_starts=True)
    rng_ = np.random.default_rng(5)
    for ep in range(10):
        ru.reset(options={"e0": 0.0})
        twin_free.reset(options={"e0": 0.0})
        done = False
        while not done:
            a = int(rng_.integers(len(LEV)))
            before = ru.unwrapped.reported
            _, r1, done, _, _ = ru.step(a)
            _, r0, _, _, _ = twin_free.step(a)
            hit = before > 0.0 and ru.unwrapped.reported <= 0.0
            check(abs(r1 - (r0 - (5.0 if hit else 0.0))) < 1e-9, "RUIN5 from a zero start")
            check(not (before == 0.0 and r1 != r0), "no ruin event can fire at zero equity")

print(f"OK: {n_checks} checks pass ({n_steps} twin steps, {n_ident} identified, {n_unid} not identified; "
      f"placebo minimum shift {PLACEBO_MIN_SHIFT} quarters)")
