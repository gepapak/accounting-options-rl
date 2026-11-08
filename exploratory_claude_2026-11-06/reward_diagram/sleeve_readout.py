"""What the reward diagram reads in the study sleeves, where the truth is known. Exploratory; no learning; the registered environments are imported unchanged.

Each state: a random equity, a random action, one clone per queried point, the market input moved along the line through the realized move. The reading sees only the rewards; the own
cost is declared. The ledger and the environment's internal balances are used only to score it. Output: sleeve_readout.json and the LaTeX table in sleeve_readout.tex.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STUDIES = REPO / "ACCOUNTING_OPTIONS_2026-10-14"
for folder in ("twin_reward", "study_t", "study_x", "study_z", "study_zb", "study_ft"):
    sys.path.insert(0, str(STUDIES / folder))

import envs_ft  # noqa: E402
import envs_t  # noqa: E402
import envs_x  # noqa: E402
import envs_z  # noqa: E402
import envs_zb  # noqa: E402
from reward_diagram import kink_position, read_diagram  # noqa: E402

K0 = float(envs_t.K0)
LEV = np.asarray(envs_t.LEV, dtype=float)
NLEV = len(LEV)
SEED = 20261107
N_STATES = 2000
EQUITIES = (0.0, 0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0)
POS = 2.0 ** np.arange(0, 41)
GRID = np.concatenate((-POS[::-1], [0.0], POS))
ZERO = 1e-9


def eps_query(env, action, t):
    c = copy.copy(env)
    c.eps_ep = env.eps_ep.copy()
    c.eps_ep[env.t - env.t0] *= t
    return float(c.step(action)[1])


def scale_query(env, action, t):
    c = copy.copy(env)
    c.move_scale = t
    return float(c.step(action)[1])


def far_slopes(ts, phi):
    return (phi[1] - phi[0]) / (ts[1] - ts[0]), (phi[-1] - phi[-2]) / (ts[-1] - ts[-2])


def shape_label(ts, phi, d):
    """Far-tail shape: the reading's classification."""
    s_left, s_right = far_slopes(ts, phi)
    big = max(abs(s_left), abs(s_right))
    if big <= ZERO:
        return "no exposure"
    if d.concave_kinks:
        return "cap"
    rel = np.abs(s_left - s_right) <= 1e-9 * big
    if rel:
        return "affine"
    if s_left * s_right < 0.0:
        return "opposite"
    if min(abs(s_left), abs(s_right)) <= 1e-9 * big:
        return "floor"
    return "partial"


def tail_estimate(s_left, s_right):
    if s_left * s_right < 0.0:
        return s_left + s_right
    return s_left if abs(s_left) >= abs(s_right) else s_right


def single_kink(ts, phi):
    """Position of a single kink: where the two affine pieces meet, each taken at the grid point nearest the kink (far points would lose the intercept to rounding)."""
    s = np.diff(phi) / np.diff(ts)
    s_left, s_right = s[0], s[-1]
    if abs(s_left - s_right) <= 1e-12:
        return None
    same_left = np.flatnonzero(np.abs(s - s_left) <= 1e-9 * max(abs(s_left), abs(s_right)))
    same_right = np.flatnonzero(np.abs(s - s_right) <= 1e-9 * max(abs(s_left), abs(s_right)))
    i, j = same_left[same_left < len(s)].max(), same_right.min()
    return kink_position(ts[i], phi[i], s_left, ts[j + 1], phi[j + 1], s_right)


FUND = 1e7       # EUR on the clone: far above any move at these positions


def funded_reward(env, action):
    """One clone with the balance raised, no other change. It needs write access to each balance; it cannot reach a margin rule."""
    c = copy.copy(env)
    if hasattr(c, "ea"):
        c.ea, c.eb = c.ea + FUND, c.eb + FUND
        before = c.ea + c.eb
    elif hasattr(c, "cash"):
        c.cash = c.cash + FUND
    elif isinstance(env, envs_ft.DeficitEnv):
        c.ledger = c.ledger + FUND
    else:
        c.reported = c.reported + FUND
    return float(c.step(action)[1])


def requested_gross(env, action):
    """The direct calculation from the requested position: leverage times Q1 times the move, without the liquidity cap, which the trainer does not see. Percent of K0."""
    a = np.asarray(action).ravel()
    t = env.t
    move = float(env.abs_s[t] * env.eps_ep[t - env.t0])
    if hasattr(env, "ea"):
        q = float(LEV[int(a[0])] + LEV[int(a[1])]) * float(envs_t.Q1)
    else:
        q = float(LEV[int(a[0])]) * float(envs_t.Q1)
    return q * move / K0 * 100.0


def run(name, make_env, draw, query, truth_label, readout, rng):
    env = make_env()
    rows = []
    labels = []
    read_ok = []
    read_n = 0
    kinks = []
    for _ in range(N_STATES):
        options, action, meta = draw(env, rng)
        env.reset(options=options)
        meta = meta(env, action) if callable(meta) else meta
        q = lambda t: query(env, action, float(t))  # noqa: E731, B023
        phi = np.array([q(t) for t in GRID])
        real = copy.copy(env)
        _, r, _, _, info = real.step(action)
        x = info["pnl"] / K0 * 100.0
        c = info["cost"] / K0 * 100.0
        assert abs(q(1.0) - r) < 1e-9
        rm = q(-1.0)
        g_true = x + c
        if abs(g_true) <= 1e-9 and max(np.abs(far_slopes(GRID, phi))) <= ZERO:
            continue                                                  # no exposure: nothing to read
        d = read_diagram(GRID, phi, rel_tol=1e-12, abs_tol=1e-9)
        s_left, s_right = far_slopes(GRID, phi)
        est = tail_estimate(s_left, s_right) - c
        tir = np.sign(r - rm) * (max(r, rm) + c) - c if abs(r - rm) > 1e-12 else -c
        spr = 0.5 * (r - rm) - c
        fund = funded_reward(env, action)
        direct = requested_gross(env, action) - c
        rows.append((abs(est - x), abs(tir - x), abs(spr - x), abs(r - x), abs(fund - x), abs(direct - x)))
        label = shape_label(GRID, phi, d)
        labels.append((label, truth_label(env, action, meta)))
        kinks.append(len(d.convex_kinks))
        if readout is not None:
            res = readout(env, action, meta, GRID, phi, d, c, label)
            if res is not None:
                read_n += 1
                read_ok.append(res)
    R = np.array(rows)
    agree = sum(a == b for a, b in labels)
    return dict(name=name, states=len(R), tail_exact=int((R[:, 0] < 1e-6).sum()), tail_max=float(R[:, 0].max()), tir_exact=int((R[:, 1] < 1e-6).sum()), tir_mae=float(R[:, 1].mean()), spr_exact=int((R[:, 2] < 1e-6).sum()), spr_mae=float(R[:, 2].mean()),
                reported_exact=int((R[:, 3] < 1e-6).sum()), reported_mae=float(R[:, 3].mean()), funded_exact=int((R[:, 4] < 1e-6).sum()), funded_mae=float(R[:, 4].mean()), requested_exact=int((R[:, 5] < 1e-6).sum()), requested_mae=float(R[:, 5].mean()), shape_agree=int(agree), shapes={k: int(sum(1 for a, b in labels if b == k)) for k in sorted({b for _, b in labels})},
                confusions=sorted({f"{b}->{a}" for a, b in labels if a != b}), read_tried=read_n, read_exact=int(sum(read_ok)), convex_kinks_median=float(np.median(kinks)))


def main():
    rng = np.random.default_rng(SEED)

    def eq_draw(env, rng):
        return {"t0": int(rng.integers(200, env.n - 700)), "e0": float(rng.choice(EQUITIES))}

    def d_one(env, rng):
        return eq_draw(env, rng), int(rng.integers(NLEV)), None

    def level_read(env, action, meta, ts, phi, d, c, label):
        if label != "floor":
            return None
        return abs(d.hidden_level - env.reported / K0 * 100.0) < 1e-6

    res = []
    res.append(run("Single floor", lambda: envs_t.EdgeEnvT("S_DK1", "test", "edge", "reported", seed=SEED), d_one, eps_query, lambda e, a, m: "floor", level_read, rng))
    res.append(run("Two writes, fee floored first", lambda: envs_x.EdgeEnvMF("S_DK1", "test", "edge", "reported", seed=SEED), d_one, eps_query, lambda e, a, m: "floor", level_read, rng))

    def d_pair(env, rng):
        return eq_draw(env, rng), rng.integers(0, NLEV, size=2), None

    res.append(run("Pooled legs, one floor", lambda: envs_z.PooledEnv("test", "edge", "reported", seed=SEED), d_pair, eps_query, lambda e, a, m: "floor", level_read, rng))

    def d_split(env, rng):
        o = eq_draw(env, rng)
        share = float(rng.choice((0.0, 0.01, 0.25, 0.5, 0.99, 1.0)))
        o.update(ea=o["e0"] * share, eb=o["e0"] * (1.0 - share))
        return o, rng.integers(0, NLEV, size=2), None

    def split_truth(env, action, m):
        qa, qb, _, _ = env.legs(action)
        return "opposite" if qa * qb < 0 else "floor"

    def split_read(env, action, meta, ts, phi, d, c, label):
        """Opposite legs: each tail slope is one account's exposure, read without knowing that there are two accounts."""
        qa, qb, _, _ = env.legs(action)
        m = env.abs_s[env.t] * env.eps_ep[env.t - env.t0] / K0 * 100.0
        if qa * qb >= 0 or m == 0.0:
            return None
        s_left, s_right = far_slopes(ts, phi)
        return abs(max(s_left, s_right) - max(qa * m, qb * m)) < 1e-6 and abs(min(s_left, s_right) - min(qa * m, qb * m)) < 1e-6

    res.append(run("Split accounts", lambda: envs_z.SplitEnv("test", "edge", "reported", seed=SEED), d_split, eps_query, split_truth, split_read, rng))

    def d_iso(env, rng):
        return eq_draw(env, rng), np.array([rng.integers(NLEV), rng.integers(2)]), None

    def iso_read(env, action, meta, ts, phi, d, c, label):
        """The hidden margin: in isolated mode the flat level is the margin, not the balance."""
        q, cost, iso = env.position(action)
        if label != "floor" or not iso or q == 0.0 or env.reported <= 0.0:
            return None
        return abs(d.hidden_level - min(cost + envs_zb.M_ISO * abs(q), env.reported) / K0 * 100.0) < 1e-6

    res.append(run("Isolated margin", lambda: envs_zb.IsolatedEnv("test", "edge", "reported", seed=SEED), d_iso, eps_query, lambda e, a, m: "floor", iso_read, rng))

    def d_bucket(env, rng):
        return {"t0": int(rng.integers(200, env.n - 700)), "cash0": float(rng.choice((0.0, 0.0, 0.02, 0.1, 1.0))), "bucket0": float(rng.choice((0.0, -0.05, -0.5, -2.0, 0.3)))}, np.array([rng.integers(NLEV), rng.integers(3)]), None

    def bucket_truth(env, action, m):
        return {0: "affine", 1: "partial", 2: "floor"}[int(action[1])]

    def ratio_read(share):
        def f(env, action, meta, ts, phi, d, c, label):
            if label != "partial":
                return None
            s_left, s_right = far_slopes(ts, phi)
            return abs((1.0 - min(abs(s_left), abs(s_right)) / max(abs(s_left), abs(s_right))) - share(env, action)) < 1e-6
        return f

    res.append(run("Unrealized bucket", lambda: envs_ft.BucketEnv("test", "edge", "reported", seed=SEED), d_bucket, scale_query, bucket_truth, ratio_read(lambda e, a: envs_ft.REALIZE[int(a[1])]), rng))
    for lam in (0.1, 0.5):
        res.append(run(f"Partial forgiveness {lam:g}", lambda lam=lam: envs_ft.PartialEnv("test", "edge", "reported", seed=SEED, lam=lam), d_one, scale_query, lambda e, a, m: "partial", ratio_read(lambda e, a, lam=lam: lam), rng))

    def d_deficit(env, rng):
        return {"t0": int(rng.integers(200, env.n - 700)), "e0": float(rng.choice((0.0, 0.05, 1.0, -0.02, -0.5, -2.0)))}, int(rng.integers(NLEV)), None

    def deficit_read(env, action, meta, ts, phi, d, c, label):
        """The carried deficit, which the reported equity (0) hides: from the position of the kink."""
        if label != "floor" or env.ledger >= 0.0:
            return None
        s_left, s_right = far_slopes(ts, phi)
        s = s_left if abs(s_left) > abs(s_right) else s_right
        tk = single_kink(ts, phi)
        return abs((abs(tk * s) - c) - (-env.ledger / K0 * 100.0)) < 1e-6 * (1.0 + abs(env.ledger / K0 * 100.0))

    res.append(run("Carried deficit", lambda: envs_ft.DeficitEnv("test", "edge", "reported", seed=SEED), d_deficit, scale_query, lambda e, a, m: "floor", deficit_read, rng))
    res.append(run("No floor (negative control)", lambda: envs_ft.PartialEnv("test", "edge", "reported", seed=SEED, lam=0.0), d_one, scale_query, lambda e, a, m: "affine", None, rng))

    (HERE / "sleeve_readout.json").write_text(json.dumps(res, indent=1))
    what = {"Single floor": "balance", "Two writes, fee floored first": "balance", "Pooled legs, one floor": "balance", "Split accounts": "each exposure", "Isolated margin": "margin", "Unrealized bucket": "share realized",
            "Partial forgiveness 0.1": "share forgiven", "Partial forgiveness 0.5": "share forgiven", "Carried deficit": "deficit", "No floor (negative control)": "--"}
    lines = [r"\begin{tabular}{@{}lrrrrrrrlr@{}}", r"\toprule", r"Accounting & States & Reading & TIR & SPR & Funded clone & Requested position & Shape & Read from the reward & Exact \\", r"\midrule"]
    for r in res:
        n = r["states"]
        pct = lambda v: f"{100.0 * v / n:.1f}"  # noqa: E731, B023
        th = lambda v: f"{v:,}".replace(",", "{,}")  # noqa: E731
        exact = f"{th(r['read_exact'])}/{th(r['read_tried'])}" if r["read_tried"] else "--"
        lines.append(f"{r['name']} & {th(n)} & {pct(r['tail_exact'])} & {pct(r['tir_exact'])} & {pct(r['spr_exact'])} & {pct(r['funded_exact'])} & {pct(r['requested_exact'])} & {pct(r['shape_agree'])} & {what[r['name']]} & {exact} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (HERE / "sleeve_readout.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for r in res:
        print(f"{r['name']:32s} n={r['states']:5d} tail {r['tail_exact']:5d} (max err {r['tail_max']:.1e}) | TIR {r['tir_exact']:5d} mae {r['tir_mae']:.3f} | SPR {r['spr_exact']:5d} mae {r['spr_mae']:.3f} | rep {r['reported_exact']:5d} | funded {r['funded_exact']:5d} mae {r['funded_mae']:.3f} | requested {r['requested_exact']:5d} mae {r['requested_mae']:.3f} | shape ok {r['shape_agree']:5d} {r['confusions']} | read {r['read_exact']}/{r['read_tried']} | kinks med {r['convex_kinks_median']}")


if __name__ == "__main__":
    main()
