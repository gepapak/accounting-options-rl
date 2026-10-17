"""Exact checks of the 'accounting rules are options' results (enumeration over all sign paths).

Setting: no-edge market, x_t = q_t*eps_t*m_t - fee*|q_t|, eps_t fair +-1 independent of the past and of m_t.
Ledger L_t = e + sum_{u<t} x_u. Forgiven-loss (subsidy) of each rule, for the SAME position process:
  FL 0;  LL (-L_{s0})^+ at first passage s0 of L <= 0;  LAG_W (-L_{min(s0+W,T)})^+;  TERM (-L_T)^+;  ZF max_t (-L_t)^+.
Claims checked:
  C1  0 <= LL <= LAG_W (nondecreasing in W) <= TERM <= ZF in expectation, for every non-anticipating policy, fees >= 0.
  C2  fixed positions, no fee: ZF <= 2 * TERM for every e >= 0 (Levy's inequality); Brownian limit ratio -> 2.
  C3  is C2 true for adaptive policies? (search for counterexamples)
  C4  non-identifiability: two markets/policies with the same law of reported (zero-floor) paths, booked means of opposite sign.
"""
import itertools, numpy as np
rng = np.random.default_rng(11)

def paths(T):
    return np.array(list(itertools.product([-1.0, 1.0], repeat=T)))      # all 2^T sign paths, equally likely

def run(eps, mags, e, fee, policy):
    """policy(t, L_t, hist_eps) -> q_t >= 0 (non-anticipating). Returns ledger matrix (paths x T+1)."""
    P, T = eps.shape
    L = np.zeros((P, T + 1)); L[:, 0] = e
    for t in range(T):
        q = np.array([policy(t, L[p, t], eps[p, :t]) for p in range(P)])
        L[:, t + 1] = L[:, t] + q * eps[:, t] * mags[t] - fee * q
    return L

def subsidies(L, Ws):
    P, T1 = L.shape; T = T1 - 1
    neg = L[:, 1:] <= 0
    hit = neg.any(axis=1); s0 = np.where(hit, neg.argmax(axis=1) + 1, T)
    D = np.maximum(-L, 0.0)
    out = {"LL": np.where(hit, D[np.arange(P), s0], 0.0)}
    for W in Ws:
        out[f"LAG{W}"] = np.where(hit, D[np.arange(P), np.minimum(s0 + W, T)], 0.0)
    out["TERM"] = D[:, T]; out["ZF"] = D.max(axis=1)
    return {k: v.mean() for k, v in out.items()}

T = 12; eps = paths(T); Ws = [1, 3, 6]
viol = 0
for trial in range(300):
    mags = rng.lognormal(0, 1.0, T) * rng.uniform(0.2, 2)
    e = rng.uniform(0, 3); fee = rng.choice([0.0, rng.uniform(0, 0.3)])
    kind = trial % 3
    if kind == 0:   # fixed schedule
        qs = rng.uniform(0, 2, T); pol = lambda t, L, h, qs=qs: qs[t]
    elif kind == 1: # equity-dependent (gamble near zero)
        b = rng.uniform(0, 3); pol = lambda t, L, h, b=b: 2.0 if L < b else 0.2
    else:           # history-dependent (double after a loss)
        pol = lambda t, L, h: 0.5 * (2.0 ** min(int((h == -1).sum()), 3))
    S = subsidies(run(eps, mags, e, fee, pol), Ws)
    chain = [0.0, S["LL"], S["LAG1"], S["LAG3"], S["LAG6"], S["TERM"], S["ZF"]]
    if any(chain[i] > chain[i + 1] + 1e-12 for i in range(len(chain) - 1)): viol += 1
print(f"C1 ordering violations in 300 random (policy, market) cases: {viol}")

ratios = []
for trial in range(300):
    mags = rng.lognormal(0, rng.uniform(0, 2), T); qs = rng.uniform(0, 2, T); e = rng.uniform(0, 2) * np.sqrt((mags ** 2 * qs ** 2).sum())
    S = subsidies(run(eps, mags, e, 0.0, lambda t, L, h, qs=qs: qs[t]), [])
    if S["TERM"] > 1e-12: ratios.append(S["ZF"] / S["TERM"])
print(f"C2 fixed positions, no fee: max ZF/TERM = {max(ratios):.4f} (bound 2), min = {min(ratios):.4f}")
for n in (100, 1000, 10000):
    Z = rng.standard_normal((20000, n)) / np.sqrt(n); S = np.cumsum(Z, axis=1)
    for e in (0.0, 0.5):
        zf = np.maximum(np.max(-S, axis=1) - e, 0).mean(); term = np.maximum(-S[:, -1] - e, 0).mean()
        print(f"   Gaussian n={n:5d} e={e}: ZF/TERM = {zf / term:.3f}")

best = (0, None)
for trial in range(400):
    mags = rng.lognormal(0, 1.0, T); e = rng.uniform(0, 1.5)
    a, b = rng.uniform(0, 3, 2)
    pol = lambda t, L, h, a=a, b=b: (3.0 if L > a else 0.1) if t % 2 == 0 else (0.1 if L < b else 3.0)
    S = subsidies(run(eps, mags, e, 0.0, pol), [])
    r = S["ZF"] / S["TERM"] if S["TERM"] > 1e-12 else 0
    if r > best[0]: best = (r, (round(a, 2), round(b, 2), round(e, 2)))
print(f"C3 largest ZF/TERM found for adaptive policies (no fee): {best[0]:.3f}  params {best[1]}")

for name, (g, p, loss) in {"A": (10.0, 0.5, 4.0), "B": (10.0, 0.5, 30.0)}.items():
    print(f"C4 market {name}: reported law {{{g}: {p}, 0: {1-p}}}, booked mean {p * g - (1 - p) * loss:+.1f}")

# C3b: a doubling ('martingale') policy -- bet 2^k after k consecutive losses, stop after the first win
def doubling(t, L, h):
    if (h == 1).any(): return 0.0
    return 2.0 ** len(h)
for T2 in (6, 8, 10, 12):
    eps2 = paths(T2)
    S = subsidies(run(eps2, np.ones(T2), 0.5, 0.0, doubling), [])
    print(f"C3b doubling policy T={T2}: ZF={S['ZF']:.3f} TERM={S['TERM']:.3f} ratio={S['ZF'] / S['TERM']:.2f}")
