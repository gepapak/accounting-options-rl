"""Cost-shift contract test for Assumption C (a single monotone censor) -- exploratory, data-free apart from the shipped market panels.

Assumption C: the reward of a step is r = max(sigma*g - c, tau), with a threshold tau that does not depend on the sign sigma or on the own cost c.
It cannot be tested from one twin pair (Proposition on non-testability), but it has testable implications when the own cost can be shifted:
for one state, action and exogenous draw, and an increasing grid of own costs c_1 < ... < c_K,
  (i)   a pair that is unequal at c_k is identified, and its inferred magnitude  g_hat(c) = max(r, r') + c  is the same at every such cost;
  (ii)  once a pair is equal (both branches censored) it stays equal at every higher cost, at the same value tau.
A violation of (i) or (ii) falsifies C at that state. The test needs only the twin pair and a cost knob; it never reads the ledger.

Three demonstrations:
  1. the Study T sleeve (single floor)                 -> no violation at any tested state;
  2. a multi-floor variant (the fee is written with its own floor before the gross P&L) -> violation exactly where the account cannot pay the fee;
  3. the original methods of the public engine (hand states of the engine's cash writes, compiled unchanged by the independent review's harness).

    python contract_test.py            # writes contract_test_results.json next to this file
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
sys.path[:0] = [str(CODE / "LIABILITY_STUDY_2026-09-29"), str(CODE / "ACCOUNTING_OPTIONS_2026-10-14" / "study_t")]
from envs_t import FEE, K0, LEV, Q1, EdgeEnvT  # noqa: E402

TOL = 1e-9


# --------------------------------------------------------------------------------------------------- the test itself
def contract_test(pairs, tol: float = TOL):
    """pairs: iterable of (cost, r, r_twin) at one state, action and exogenous draw. Returns (violated, reason)."""
    pairs = sorted(pairs)
    censored_value = None
    ghat = []
    for c, r, rp in pairs:
        if abs(r - rp) > tol:                                     # identified: g_hat = max(r, r') + c
            if censored_value is not None:
                return True, "identified again after a censored pair at a lower cost"
            ghat.append(max(r, rp) + c)
        else:
            if r + c < -tol:
                return True, "an equal pair books less than the own cost"
            if abs(r + c) <= tol:                                 # r = -c: an uncensored draw with zero magnitude, no information
                continue
            if censored_value is not None and abs(r - censored_value) > tol:
                return True, "censored value changes with the cost"
            censored_value = r
    if ghat and max(ghat) - min(ghat) > max(tol, 1e-7 * (1 + max(abs(x) for x in ghat))):
        return True, "inferred magnitude varies with the cost"
    return False, "consistent with a single monotone censor"


# --------------------------------------------------------------------------------------------------- multi-floor variant of the sleeve
class EdgeEnvMF(EdgeEnvT):
    """The Study T sleeve with a second floor: the fee is written first and floored (an account that cannot pay the fee is not charged for
    the unpaid part), then the gross P&L is written and floored. The ledger books the whole fee."""

    def step(self, action):
        t = self.t
        q = float(np.clip(float(LEV[int(action)]) * Q1, -self.cap[t], self.cap[t]))
        cost = FEE * abs(q) + self.kappa * q * q
        sign = self.eps_ep[t - self.t0] * (-1.0 if self.flip_next else 1.0)
        self.flip_next = False
        gross = q * self.abs_s[t] * sign
        e0 = self.reported
        e1 = max(e0 - cost, 0.0)                 # fee write, floored
        e2 = max(e1 + gross, 0.0)                # gross write, floored
        pnl = gross - cost
        self.ledger += pnl
        self.forgiven += (e2 - e0) - pnl
        self.reported = e2
        r = (e2 - e0) if self.mode == "reported" else pnl
        self.t += 1
        done = self.t >= self.t_end
        obs = self._obs() if not done else np.zeros(4, np.float32)
        return obs, float(r / K0 * 100.0), done, False, {"pnl": pnl, "cost": cost, "forgiven": (e2 - e0) - pnl}


def sleeve_pair(env, action, kappa):
    a = copy.copy(env)
    a.kappa = kappa
    b = copy.copy(env)
    b.kappa = kappa
    b.flip_next = True
    _, r, _, _, info = a.step(action)
    _, rp, _, _, _ = b.step(action)
    return r, rp, info["cost"] / K0 * 100.0


def sweep_sleeve(env_cls, n_episodes=60, stride=7, seed=11):
    rng = np.random.default_rng(seed)
    kappa0 = 4.0
    kappas = [0.0] + [kappa0 * m for m in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)]
    starts = [0.0, 1e-4, 1e-3, 5e-3, 0.02, 0.1, 0.5, 1.0]
    rows = []
    env = env_cls("S_DK1", "train", "edge", "reported", seed=int(rng.integers(1 << 30)))
    for ep in range(n_episodes):
        e0 = starts[ep % len(starts)]
        env.reset(options={"e0": e0})
        done = False
        k = 0
        while not done:
            a = int(rng.integers(len(LEV)))
            if k % stride == 0 and abs(LEV[a]) > 0:
                pairs = []
                for kp in kappas:
                    r, rp, c = sleeve_pair(env, a, kp)
                    pairs.append((c, r, rp))
                violated, why = contract_test(pairs)
                cost_budget = max(p[0] for p in pairs) * K0 / 100.0
                rows.append(dict(equity=env.reported / K0, violated=bool(violated), why=why,
                                 informative=any(abs(p[1] - p[2]) > TOL for p in pairs),   # the draw identified the magnitude at some tested cost
                                 binds=bool(env.reported < cost_budget)))     # the largest tested fee exceeds the account
            _, _, done, _, _ = env.step(a)
            k += 1
    return rows


def summarize(rows):
    n = len(rows)
    binds = [r for r in rows if r["binds"]]
    free = [r for r in rows if not r["binds"]]
    inf_binds = [r for r in binds if r["informative"]]
    return dict(tested_states=n, flagged=sum(r["violated"] for r in rows),
                states_where_a_tested_fee_exceeds_the_account=len(binds), flagged_among_them=sum(r["violated"] for r in binds),
                informative_draws_among_them=len(inf_binds), flagged_among_informative=sum(r["violated"] for r in inf_binds),
                states_where_no_tested_fee_exceeds_the_account=len(free), flagged_among_free=sum(r["violated"] for r in free))


# --------------------------------------------------------------------------------------------------- the public engine's own source methods
def engine_sweep():
    sys.path.insert(0, str(CODE / "exploratory_codex_2026-11-01" / "native_reward_identification_001"))
    import floor_inverse as I  # noqa: N812
    import run_source_hand_gate as G
    counter = dict(top_attempted=0, top_returned=0, helper_attempted=0, helper_returned=0)
    cls, np_ = G.make_source_class(counter)

    def pair(cash, fee):
        counter.update(top_attempted=0, top_returned=0, helper_attempted=0, helper_returned=0)
        case = dict(name="SWEEP", cash=float(cash), fee=float(fee), funding=0.0, entry=500.0, settlement=501.0, mixed=False)
        original = G.make_proxy(cls, np_, case)
        due = {int(c["delivery_step"]): float(original._horizon_settlement_price_raw[int(c["delivery_step"])])
               for c in original._horizon_contracts if c["settle_step"] <= 12}
        shadows = I.due_reflection_map(original._horizon_contracts, 12, due)
        real, shadow = copy.deepcopy(original), copy.deepcopy(original)
        for d, v in shadows.items():
            shadow._horizon_settlement_price_raw[d] = v
        end = {}
        for name, env in (("real", real), ("shadow", shadow)):
            env._open_horizon_settlement_contract(12, {"wind_instrument_value": 500.0})
            cost = float(env._last_horizon_roll_cost)
            env._settle_horizon_contracts_for_step(12)
            end[name] = (float(env.budget), cost)
        return end["real"][0] - cash, end["shadow"][0] - cash, end["real"][1]

    fees = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    out = []
    for cash in (0.0, 5.0, 100.0):
        pairs = []
        for f in fees:
            r, rp, c = pair(cash, f)
            pairs.append((c, r, rp))
        violated, why = contract_test(pairs)
        out.append(dict(starting_cash=cash, fees=fees, real_increment=[p[1] for p in pairs], shadow_increment=[p[2] for p in pairs],
                        inferred_magnitude=[(max(p[1], p[2]) + p[0]) if abs(p[1] - p[2]) > TOL else None for p in pairs],
                        violated=bool(violated), why=why))
    return out


def estimator_errors(env_cls, n_episodes=120, seed=5):
    """Mean absolute error (percent of K0 per quarter-hour) of the reported reward, SPR and TIR against the booked P&L on credited steps
    of random-action rollouts from low starting equity. Credited = the account forgave part of the step's loss or fee."""
    rng = np.random.default_rng(seed)
    env = env_cls("S_DK1", "train", "edge", "reported", seed=int(rng.integers(1 << 30)))
    errs = {"reported": [], "SPR": [], "TIR": []}
    cost_levels = []
    for ep in range(n_episodes):
        env.reset(options={"e0": [0.0, 1e-4, 1e-3, 0.01][ep % 4]})
        done = False
        while not done:
            a = int(rng.integers(len(LEV)))
            twin = copy.copy(env)
            twin.flip_next = True
            sign = float(np.sign(LEV[a])) * float(env.eps_ep[env.t - env.t0])
            _, rp, _, _, _ = twin.step(a)
            _, r, done, _, info = env.step(a)
            if abs(LEV[a]) > 0 and info["forgiven"] > 1e-12:
                c = info["cost"] / K0 * 100.0
                x = info["pnl"] / K0 * 100.0
                tir = sign * (max(r, rp) + c) - c if abs(r - rp) > TOL else -c
                spr = 0.5 * (r - rp) - c
                errs["reported"].append(abs(r - x))
                errs["SPR"].append(abs(spr - x))
                errs["TIR"].append(abs(tir - x))
                cost_levels.append(c)
    out = {k: (float(np.mean(v)) if v else None) for k, v in errs.items()}
    out["credited_steps"] = len(errs["TIR"])
    out["mean_cost"] = float(np.mean(cost_levels)) if cost_levels else None
    return out


def main():
    res = {}
    for name, cls in (("single_floor_sleeve", EdgeEnvT), ("multi_floor_sleeve", EdgeEnvMF)):
        res[name + "_estimator_errors"] = estimator_errors(cls)
        print(name, "estimator errors on credited steps:", json.dumps(res[name + "_estimator_errors"]))
    for name, cls in (("single_floor_sleeve", EdgeEnvT), ("multi_floor_sleeve", EdgeEnvMF)):
        rows = sweep_sleeve(cls)
        res[name] = summarize(rows)
        by_bucket = {}
        for lo, hi, lab in ((-1, 1e-3, "equity <= 0.001 K0"), (1e-3, 0.02, "0.001-0.02 K0"), (0.02, 0.2, "0.02-0.2 K0"), (0.2, 99, "> 0.2 K0")):
            sel = [r for r in rows if lo < r["equity"] <= hi]
            by_bucket[lab] = (len(sel), sum(r["violated"] for r in sel))
        res[name]["by_equity_bucket(tested, flagged)"] = by_bucket
        print(name, json.dumps(res[name]))
    res["engine_original_methods"] = engine_sweep()
    for row in res["engine_original_methods"]:
        print("engine cash", row["starting_cash"], "violated:", row["violated"], "|", row["why"], "| g_hat:", row["inferred_magnitude"])
    (HERE / "contract_test_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
