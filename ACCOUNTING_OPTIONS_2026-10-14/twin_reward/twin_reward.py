"""Twin-based rewards for simulators whose reward channel is censored by an accounting credit (manuscript Section 8).

Try the direct calculation first. If you hold the market input, the position and the payoff formula, the booked P&L is position x move - own
cost: exact, with no clone, and its gap to the reward exposes every credit. It is only as good as that formula. This module is for a simulator
whose payoff is internal, changes between versions or is not trusted as a formula, and which can still be cloned and fed a reversed input.

Under the zero floor, and under the other rules of the accounting ladder at their crediting steps, the reported step reward is r = max(x, tau):
the booked P&L x raised to a threshold tau that the trainer cannot see (Assumption C). Two cases are outside this form and are not covered:
the reported increment of an owner's claim while insolvent (the zombie window of a settlement lag, which censors x minus the carried deficit),
and simulators that floor several cash writes in sequence (for example a fee written first with its own floor). Check the form first with
`cost_shift_test`. Repeat the step from the same state with the same action and the exogenous sign reversed (the *sign twin*), with reward r'.
Then r != r' exactly when the favourable branch is uncensored, and the gross magnitude is g = max(r, r') + c (c = the agent's own cost, known).

    spr(r, r', c)        sign-paired reward  (r - r')/2 - c: uses |r - r'|/2, a valid but generally non-sharp lower bound on g
                         (g = 10, c = 1, tau = 0 gives the pair (9, 0): g = 10 is identified, SPR uses 4.5)
    tir(r, r', c)        twin-identified reward: exact booked P&L where r != r', and the conservative value -c where r == r'
                         (i.e. the sharp lower end of the identified set for g; its error is never above SPR's)
    tmr(r, r', c, sign)  twin-maximum reward: the sharp upper end (needs the realized sign of the gross P&L)
    twin_bracket(r, r', c)  the interval [2*SPR - r, r] that contains the booked P&L for ANY accounting that only forgives (credits >= 0 at
                         every step on both branches), without the single-floor assumption; its width is the credit paid on the real and
                         twin branches together, an upper bound on the credit realized on the path

`TwinRewardWrapper` applies one of them to any Gymnasium environment that can reverse the sign of its next exogenous move. It never reads
a ledger or the accounting rule. Requirements on the simulator: it can be deep-copied before a step, it has a way to reverse the
exogenous sign of the clone's next step (default: `env.reverse_next_sign()`; or pass `reverse=`), and the agent's own cost is known (default:
`info["own_cost"]` of the real step; or pass `own_cost=`).

`cost_shift_pairs(env, action, knob_values, set_knob)` and `cost_shift_test(pairs)` test the assumption behind all of the above (manuscript
proposition "Testing Assumption C"): that the reward is ONE monotone censor of the booked P&L with a threshold that does not move with the own
cost. That cannot be seen in one twin pair, but it can be falsified by shifting the own cost: `single_floor_fit` decides exactly whether any
single-floor world reproduces the pairs. They never read a ledger either. Passing is not a certificate: a violation that produces pairs a single
floor could have produced (a hidden fixed fee on a censored loss branch, an inflated gross credit) is invisible to every reward-only test.
"""
from __future__ import annotations

import copy
from typing import Callable, Optional

import gymnasium as gym
import numpy as np

TOL = 1e-12


def spr(r, r_twin, own_cost):
    """Sign-paired reward (r - r')/2 - c."""
    return 0.5 * (np.asarray(r, dtype=float) - np.asarray(r_twin, dtype=float)) - np.asarray(own_cost, dtype=float)


def identified(r, r_twin):
    """True where the pair identifies the gross magnitude (the favourable branch is uncensored)."""
    return np.abs(np.asarray(r, dtype=float) - np.asarray(r_twin, dtype=float)) > TOL


def tir(r, r_twin, own_cost):
    """Twin-identified reward: sign(r - r') * (max(r, r') + c) - c where r != r', and -c where r == r'.

    The realized sign of the gross P&L is sign(r - r') on an identified step (the favourable branch has the larger reward), so the
    trainer needs only the pair and the own cost."""
    r, rt, c = (np.asarray(v, dtype=float) for v in (r, r_twin, own_cost))
    sigma = np.sign(r - rt)
    return np.where(np.abs(r - rt) > TOL, sigma * (np.maximum(r, rt) + c) - c, -c)


def tmr(r, r_twin, own_cost, sign):
    """Twin-maximum reward: sign * (max(r, r') + c) - c, with the realized sign of the gross P&L. Exact where TIR is, but where both
    branches are censored it inherits the whole threshold as noise; included for comparison."""
    r, rt, c, s = (np.asarray(v, dtype=float) for v in (r, r_twin, own_cost, sign))
    return s * (np.maximum(r, rt) + c) - c


def twin_bracket(r, r_twin, own_cost):
    """Interval [2*SPR - r, r] = [-r' - 2c, r] that contains the booked P&L for ANY accounting that only forgives (credits >= 0).

    Needs no single-floor assumption: only that the twin reverses the gross P&L and that the own cost is the same on both branches and known.
    Reporting that withholds a profit while a deficit is carried (the zombie window of a settlement lag) and a charge on one branch only are
    outside this class, and the interval can then fail.
    The width r + r' + 2c is the total credit paid on the pair (zero where nothing is forgiven, so the interval is then the booked P&L
    itself); a negative width is impossible under such an accounting and exposes an undeclared charge, while a positive width does not exclude one. Over all accountings with non-negative
    credits the interval is the sharp identified set, and SPR is its midpoint. Returns (lower, upper); sums over steps bracket an episode."""
    r, rt, c = (np.asarray(v, dtype=float) for v in (r, r_twin, own_cost))
    return -rt - 2.0 * c, r


ESTIMATORS = {"spr": spr, "tir": tir}


class PooledMagnitudeWrapper(gym.Wrapper):
    """Clone-free pooled-magnitude reward (PMR; manuscript proposition "Pooled magnitude", Study V).

    On a favourable step (gross sign +1, position q != 0) the reported reward r = max(|q| m - c, tau) gives the pseudo-magnitude
    U = (r + c) / |q|, which equals the per-unit move m whenever the favourable branch is uncensored and exceeds it otherwise. The wrapper keeps
    the running mean m_bar of these pseudo-magnitudes, pooled over all steps, and rewards  gross_sign * |q| * m_bar - c.

    Its conditional bias is E[PMR - x | history] = |q| d (m_bar - mu), where d is the conditional mean of the gross sign and mu the true mean
    move under Assumption M (the move does not depend on the equity, the action, the history or the sign). Hence in a conditionally fair
    market (d = 0) the expectation is -c in every state, whatever the pool, including states where a credit is certain; with an edge the reward
    is unbiased only if the pooled mean equals mu. Censored favourable draws enter the pool above their move, so a pool fed by censored draws
    overstates mu and PMR can pay for exposure: with q = 1, m = 1, c = 1, a carried deficit that sets tau = 10 and a favourable-sign probability
    of 0.75, a pool of such draws has m_bar = 11 and the expected PMR reward is +4.5 while the booked expectation is -0.5, although Assumptions
    C and M hold. A finite pool can also lie below mu. Use PMR where favourable draws are rarely censored, or calibrate the mean on uncensored
    exposures; it is not exact step by step, and it needs no clone.

    Requirements: info["gross_sign"] (the realized sign of the gross P&L, sign(q * eps)), info["position"] (q) and info["own_cost"] of the
    real step, or the callables `gross_sign`, `position` and `own_cost` of (env, action, info). The mean used for a step excludes that
    step's own reward. `info` gets "pooled_magnitude" (the mean used) and "pool_size".
    """

    def __init__(self, env: gym.Env, gross_sign: Optional[Callable] = None, position: Optional[Callable] = None, own_cost: Optional[Callable] = None):
        super().__init__(env)
        self._gross_sign = gross_sign or (lambda env, action, info: info["gross_sign"])
        self._position = position or (lambda env, action, info: info["position"])
        self._own_cost = own_cost or (lambda env, action, info: info["own_cost"])
        self.total, self.n = 0.0, 0

    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)
        sigma, q, c = float(self._gross_sign(self.env, action, info)), abs(float(self._position(self.env, action, info))), float(self._own_cost(self.env, action, info))
        m_bar = self.total / self.n if self.n else 0.0
        if q > 0.0 and sigma > 0.0:
            self.total += (float(r) + c) / q
            self.n += 1
        info = dict(info, reported_reward=float(r), pooled_magnitude=m_bar, pool_size=self.n)
        return obs, float(sigma * q * m_bar - c), terminated, truncated, info


def single_floor_fit(pairs, tol: float = 1e-9):
    """Exact test of Assumption C on twin pairs at several own costs. Returns (consistent, witness, detail).

    pairs: iterable of (cost, r, r_twin) for ONE state, action and exogenous draw, with the own cost shifted over several values. The pairs are
    generated by a single-floor world iff there are sigma in {+1, -1}, g >= 0 and tau in [-inf, inf) with r_k = max(sigma*g - c_k, tau) and
    r'_k = max(-sigma*g - c_k, tau) for every cost c_k. Writing u_k and l_k for the favourable and unfavourable rewards, it suffices to try
    g in {0} + {u_k + c_k} + {-l_k - c_k} and tau in {-inf} + {u_k} + {l_k}, for both assignments of the branches: if a favourable branch is
    uncensored it fixes g, if none is then g = 0 reproduces the pairs whenever any g does, and tau is either -inf or a censored, hence observed,
    value. consistent: True iff some single-floor world reproduces every pair to within tol * (1 + largest |value|); witness: (sigma, g, tau)
    of one such world (sigma = +1 if the real step is the favourable branch), or None; detail: a short reason. The tolerance scales with the pairs'
    own values: where rewards are differences of large balances, pass a tol that also covers the rounding of the balance, or rounding is flagged.
    """
    data = sorted((float(c), float(r), float(p)) for c, r, p in pairs)
    if not data:
        return True, None, "no pairs"
    c = np.array([d[0] for d in data])
    r = np.array([d[1] for d in data])
    p = np.array([d[2] for d in data])
    eps = tol * (1.0 + float(np.max(np.abs(np.concatenate([c, r, p])))))
    d = r - p
    if np.any(d > eps) and np.any(d < -eps):
        return False, None, "the branch with the larger reward changes with the cost"
    best = (np.inf, None)
    for sigma in (+1.0, -1.0):
        u, l = (r, p) if sigma > 0 else (p, r)
        G = np.unique(np.concatenate(([0.0], u + c, -l - c)))
        G = np.maximum(G[G >= -eps], 0.0)
        T = np.concatenate(([-np.inf], u, l))
        pu = np.maximum(G[:, None, None] - c[None, None, :], T[None, :, None])
        pl = np.maximum(-G[:, None, None] - c[None, None, :], T[None, :, None])
        res = np.maximum(np.abs(pu - u).max(axis=2), np.abs(pl - l).max(axis=2))
        i, j = np.unravel_index(int(np.argmin(res)), res.shape)
        if res[i, j] < best[0]:
            best = (float(res[i, j]), (sigma, float(G[i]), float(T[j])))
        if res[i, j] <= eps:
            return True, (sigma, float(G[i]), float(T[j])), "consistent with a single monotone censor"
    return False, None, f"no single (g, tau) reproduces the pairs (best misfit {best[0]:.3g}, tolerance {eps:.3g})"


def cost_shift_test(pairs, tol: float = 1e-9, exact: bool = True):
    """Falsification test of Assumption C (reward = max(sigma * g - c, tau), with g and tau independent of the own cost c).

    pairs: iterable of (cost, r, r_twin) for ONE state, action and exogenous draw, with the own cost shifted over several values. Returns
    (violated, reason). exact=True (default) uses `single_floor_fit`: a violation means that no single-floor world reproduces the pairs.
    exact=False is the earlier check of two implied conditions only (the inferred magnitude max(r, r') + c is constant where the pair is unequal;
    an equal pair above -c keeps its value at every higher cost), which misses, for example, a threshold that moves with the cost on the
    unfavourable branch. "No violation" is not a certificate: the test has power only where some tested cost changes which branch is censored,
    and violations that a single floor could have produced are invisible to it.
    """
    if exact:
        ok, _, why = single_floor_fit(pairs, tol)
        return (not ok), why
    ordered = sorted(pairs)
    censored_value = None
    ghat = []
    for c, r, rp in ordered:
        if abs(r - rp) > tol:                                         # identified: g_hat = max(r, r') + c
            if censored_value is not None:
                return True, "identified again after a censored pair at a lower cost"
            ghat.append(max(r, rp) + c)
        else:
            if r + c < -tol:
                return True, "an equal pair books less than the own cost"
            if abs(r + c) <= tol:                                     # r = -c: zero magnitude, no information
                continue
            if censored_value is not None and abs(r - censored_value) > tol:
                return True, "censored value changes with the cost"
            censored_value = r
    if ghat and max(ghat) - min(ghat) > max(tol, 1e-7 * (1 + max(abs(x) for x in ghat))):
        return True, "inferred magnitude varies with the cost"
    return False, "consistent with a single monotone censor"


def cost_shift_pairs(env, action, knob_values, set_knob: Callable, reverse: Optional[Callable] = None, own_cost: Optional[Callable] = None):
    """Twin pairs of one step at several values of a cost knob, for `cost_shift_test`.

    For each value, the environment is deep-copied twice (the two copies share the random-number state, hence the exogenous draw),
    `set_knob(copy, value)` is applied to both, the sign of the second copy's next step is reversed (`reverse`, default
    `copy.reverse_next_sign()`) and both are stepped with `action`. Returns [(own cost, r, r'), ...] with the own cost read from `info`
    of the real copy (`own_cost(copy, action, info)`, default `info["own_cost"]`)."""
    reverse = reverse or (lambda clone: clone.reverse_next_sign())
    own_cost = own_cost or (lambda clone, action, info: info["own_cost"])
    out = []
    for value in knob_values:
        real, twin = copy.deepcopy(env), copy.deepcopy(env)
        set_knob(real, value)
        set_knob(twin, value)
        reverse(twin)
        _, r, _, _, info = real.step(action)
        _, r_twin, _, _, _ = twin.step(action)
        out.append((float(own_cost(real, action, info)), float(r), float(r_twin)))
    return out


class TwinRewardWrapper(gym.Wrapper):
    """Replace the reward by a twin-based estimate of the booked P&L.

    estimator   "tir" (default) or "spr".
    reverse     callable(clone) that makes the clone's NEXT step use the opposite exogenous sign (default: clone.reverse_next_sign()).
    own_cost    callable(env, action, info) -> the agent's own cost of the step (default: info["own_cost"] of the real step).
    `info` of every step gets "reported_reward", "twin_reward" and "identified" (the pair identified the gross magnitude).
    """

    def __init__(self, env: gym.Env, estimator: str = "tir", reverse: Optional[Callable] = None, own_cost: Optional[Callable] = None):
        super().__init__(env)
        if estimator not in ESTIMATORS:
            raise ValueError(f"estimator must be one of {sorted(ESTIMATORS)}")
        self.estimator = ESTIMATORS[estimator]
        self._reverse = reverse or (lambda clone: clone.reverse_next_sign())
        self._own_cost = own_cost or (lambda env, action, info: info["own_cost"])

    def step(self, action):
        clone = copy.deepcopy(self.env)                 # the twin starts from the same state, with the same random-number state
        self._reverse(clone)
        _, r_twin, _, _, _ = clone.step(action)
        obs, r, terminated, truncated, info = self.env.step(action)
        c = float(self._own_cost(self.env, action, info))
        info = dict(info, reported_reward=float(r), twin_reward=float(r_twin), identified=bool(identified(r, r_twin)))
        return obs, float(self.estimator(r, r_twin, c)), terminated, truncated, info
