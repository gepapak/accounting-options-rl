"""The monitoring law: closed forms for what three accounting rules forgive, from leverage, bar volatility, horizon and time step.

Setting (the paper's Section 3): a position of fixed size with notional L x equity, a market without edge, bars with mean-zero
returns of standard deviation s, n bars. In units of the initial equity, the booked loss walk is Y_k = -L (P_k / P_0 - 1).
    LL    capped loss, the account closes at the first bar with equity <= 0 and the overshoot is forgiven (backtesting.py's rule)
    TERM  terminal limited liability: the ledger may go negative, the owner receives max(L_T, 0)
    ZF    zero floor, trading continues: equity is reset to zero whenever it would go negative
Subsidies (expected forgiven loss), with rho the mean overshoot of the bar-return walk over a far barrier:
    S_LL   = P(ruin) * (L rho + fee term)         P(ruin) = 2 Phi(-(b + rho) / (s sqrt(n))),  b = barrier distance in return units
    S_TERM = v phi(1/v) - Phi(-1/v)               v = L s sqrt(n)                        (Bachelier call; no time-step term)
    S_ZF   = 2 [ v phi(d) - (1 + L rho) Phi(-d) ] d = (1 + L rho) / v                    (discrete lookback)
rho = BETA * s for Gaussian bars, BETA = -zeta(1/2) / sqrt(2 pi) = 0.5826 (Siegmund 1985; Broadie, Glasserman and Kou 1997), and in
general Siegmund's integral over the characteristic function of the bar return (rho_siegmund).
Refining the time step by m (n -> n m, s -> s / sqrt(m)) leaves v unchanged and divides rho by sqrt(m):
    S_LL falls like 1/sqrt(m) to zero, S_TERM does not move, S_ZF rises to 2 S_TERM (fee-free, Brownian limit).
"""
from math import erf, exp, log, pi, sqrt

import numpy as np

BETA = 0.5825971579390107      # -zeta(1/2) / sqrt(2 pi)


def Phi(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def phi(x):
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def p_ruin(b, rho, s, n):
    """P(the loss walk crosses the barrier b (return units) within n bars), with the continuity correction."""
    return 2.0 * Phi(-(b + rho) / (s * sqrt(n)))


def s_ll(L, s, n, rho=None, fee=0.0):
    """Capped loss with closure. fee = commission per side as a share of notional; it lowers the equity at entry (barrier nearer) and
    the exit commission at ruin is forgiven too. Returns (P(ruin), subsidy in units of the initial equity)."""
    rho = BETA * s if rho is None else rho
    b = 1.0 / L - fee
    p = p_ruin(b, rho, s, n)
    return p, p * (L * rho + fee * (L - 1.0))


def s_term(L, s, n):
    v = L * s * sqrt(n)
    return v * phi(1.0 / v) - Phi(-1.0 / v)


def s_zf(L, s, n, rho=None):
    rho = BETA * s if rho is None else rho
    v = L * s * sqrt(n)
    d = (1.0 + L * rho) / v
    return 2.0 * (v * phi(d) - (1.0 + L * rho) * Phi(-d))


# ---------------------------------------------------------------------------------------------------------------------------------
# Compounded prices (added 17 Nov after Test A: the additive forms above were 12% high at L = 20 on compounded prices, in all three
# rules, because a long position's loss is L (1 - P/P_0) with P a product of bar returns). The same law with the log price as the walk:
# barrier at K = 1 - e1 / L (e1 = equity after the entry fee), b = -ln K, log drift -s^2/2 per bar, overshoot rho of the log walk.
def p_min_below(B, mu, s, n):
    """P(min over n bars of a walk with drift mu and standard deviation s per bar is <= -B) (Brownian formula)."""
    sd = s * sqrt(n)
    return Phi((-B - mu * n) / sd) + exp(-2.0 * mu * B / (s * s)) * Phi((-B + mu * n) / sd)


def ln_forms(L, s, n, rho=None, fee=0.0):
    """Returns dict(p_ruin, LL, TERM, ZF) in units of the initial equity for a long position of notional L x equity in a price that
    compounds bar returns of standard deviation s (no edge). fee = commission per side on notional."""
    rho = BETA * s if rho is None else rho
    e1 = 1.0 - fee * L
    K = 1.0 - e1 / L
    b = -log(K)
    mu = -0.5 * s * s
    sd = s * sqrt(n)
    p = p_min_below(b + rho, mu, s, n)
    ll = p * L * K * (rho + fee)                        # overshoot of the log price, and the exit commission, both forgiven at ruin
    d1 = (b + 0.5 * sd * sd) / sd
    term = L * (K * Phi(-(d1 - sd)) - Phi(-d1))         # a put on the price struck at K
    z = np.linspace(b + rho, b + rho + 14.0 * sd, 20001)    # zero floor: L E[(K - discrete min)^+], discrete min = continuous min x e^rho
    f = np.exp(-z) * np.array([Phi((-x + 0.5 * sd * sd) / sd) for x in z]) + np.array([Phi((-x - 0.5 * sd * sd) / sd) for x in z])
    zf = L * exp(rho) * float(np.sum((f[1:] + f[:-1]) * 0.5 * np.diff(z)))
    return dict(p_ruin=p, LL=ll, TERM=term, ZF=zf)


def overshoot_log_samples(mags, barrier_sd=25.0, paths=40000, max_steps=6000, seed=4):
    """Samples of the overshoot of the LOG price below a far barrier (its stationary law), for bar returns (fair sign) x M."""
    rng = np.random.default_rng(seed)
    m = np.asarray(mags, float)
    B = barrier_sd * sqrt(float(np.mean(m ** 2)))
    pos = np.zeros(paths)
    alive = np.ones(paths, bool)
    over = []
    for _ in range(max_steps):
        k = int(alive.sum())
        if k == 0:
            break
        pos[alive] += np.log1p(rng.choice([-1.0, 1.0], size=k) * m[rng.integers(0, len(m), size=k)])
        crossed = alive & (pos <= -B)
        over.append(-B - pos[crossed])
        alive &= ~crossed
        alive &= pos < 8 * B
    return np.concatenate(over)


def rho_log_mc(mags, barrier_sd=25.0, paths=20000, max_steps=6000, seed=2):
    """Mean overshoot of the LOG price below a far barrier, for bar returns (fair sign) x M: increments ln(1 +/- M)."""
    rng = np.random.default_rng(seed)
    m = np.asarray(mags, float)
    s = sqrt(float(np.mean(m ** 2)))
    B = barrier_sd * s
    pos = np.zeros(paths)
    alive = np.ones(paths, bool)
    over = []
    for _ in range(max_steps):
        k = int(alive.sum())
        if k == 0:
            break
        pos[alive] += np.log1p(rng.choice([-1.0, 1.0], size=k) * m[rng.integers(0, len(m), size=k)])
        crossed = alive & (pos <= -B)
        over.append(-B - pos[crossed])
        alive &= ~crossed
        alive &= pos < 8 * B
    over = np.concatenate(over)
    return float(over.mean()), len(over)


def rho_siegmund(mags, U=3000.0, du=0.004):
    """Mean overshoot over a far barrier, in the units of mags, for the symmetric bar return X = (fair sign) x M with M drawn from
    mags: rho = -(1/pi) int_0^inf t^-2 log{ 2 (1 - phi_X(t)) / (s^2 t^2) } dt   (Siegmund 1985; the third-moment term is zero by
    symmetry). Computed in u = t s on [0, U] with the analytic tail for phi_X ~ 0."""
    m = np.asarray(mags, float)
    s = sqrt(float(np.mean(m ** 2)))
    x = m / s
    u = np.arange(du / 2, U, du)
    cf = np.zeros_like(u)
    for i in range(0, len(x), 64):
        cf += np.cos(np.outer(u, x[i:i + 64])).sum(axis=1)
    cf /= len(x)
    g = np.log(np.maximum(2.0 * (1.0 - cf), 1e-300) / u ** 2) / u ** 2
    tail = (log(2.0) - 2.0 * (log(U) + 1.0)) / U
    return -(g.sum() * du + tail) / pi * s


def rho_far_barrier_mc(mags, barrier_sd=25.0, paths=20000, max_steps=6000, seed=1):
    """Monte Carlo cross-check of rho: mean overshoot of the fair-sign walk over a barrier barrier_sd standard deviations away."""
    rng = np.random.default_rng(seed)
    m = np.asarray(mags, float)
    s = sqrt(float(np.mean(m ** 2)))
    B = barrier_sd * s
    pos = np.zeros(paths)
    alive = np.ones(paths, bool)
    over = []
    for _ in range(max_steps):
        k = int(alive.sum())
        if k == 0:
            break
        step = rng.choice([-1.0, 1.0], size=k) * m[rng.integers(0, len(m), size=k)]
        pos[alive] += step
        crossed = alive & (pos >= B)
        over.append(pos[crossed] - B)
        alive &= ~crossed
        alive &= pos > -8 * B           # paths far on the wrong side are dropped (they would cross much later)
    over = np.concatenate(over)
    return float(over.mean()), len(over)
