r"""Read-only step tracer for the audit engine: the sign twin of every step, from the engine's own cash writes.

Nothing in the engine's source changes. `install()` puts a property on the class attribute `budget`, so every write to
trading cash passes through a setter that records where it came from and the nominal amount the writing method held in
its local variables, before any floor. After each `step` the writes are replayed, in the engine's own order and with its
own floors:

  real branch   the amounts as written; it must reproduce the engine's cash at every step (parity);
  twin branch   the same step from the same pre-step state, same action and same own cash flows, gross P&L reversed.

The twin is one step off the real path: the state is not carried over, and the executed action, costs and battery flow
are held at the real branch's values. It is a replay of the cash writes, not a second run of the engine's step.

Horizon settlement: at maturity `budget = max(0, budget + total_pnl)`; the twin uses -total_pnl. Percent mark-to-market
(the L7 family): the step's payoff m_k goes to an unfloored bucket per asset, and a position closed by a fraction f moves
f of the bucket to cash; the twin's bucket is lower by 2 m_k and its realized part is (bucket - 2 m_k) f.

Distributions are the step's last write. The reported return adds them back, so they do not enter the reported change.

Cost shift: on steps with a non-zero gross and a positive own cost the same replay is repeated with the access fee, roll
cost, transaction cost and funding multiplied by each of COST_MULTIPLIERS, which gives the twin pairs of the exact
cost-shift test (twin_reward.single_floor_fit); the analysis runs the test offline.

Per step the tracer stores: r (reported change of sleeve value plus distributions), r_twin, the components of the booked
P&L (gross, own costs, battery), the replay's parity error, the pairs at the shifted costs, and the public quantities a
direct calculation would use (own position and the three price series).
"""
from __future__ import annotations

import atexit
import linecache
import os
import sys

import numpy as np

ASSETS = ("wind_instrument_value", "solar_instrument_value", "hydro_instrument_value")
COST_MULTIPLIERS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
NOMINAL_LOCAL = {
    "_apply_market_access_fee": ("access", "fee"),
    "_charge_horizon_roll_cost": ("roll", "roll_cost"),
    "_settle_horizon_contracts_for_step": ("settle", "total_pnl"),
    "_apply_collateral_cash_drag": ("funding", "cost"),
    "_update_finance": ("battery", "trading_cash_flow"),
    "_distribute_excess_cash": ("distribution", "distribution_amount"),
    "_liquidate_trading_sleeve_for_margin": ("liquidation", "realized_cash"),
    "_check_emergency_reallocation": ("reallocation", "new_budget"),
}
COLUMNS = (
    "t", "cash_pre", "cash_post", "mtm_pre", "mtm_post", "dist_pre", "dist_post",
    "r", "r_twin", "gross", "access", "roll", "transaction", "funding", "battery",
    "parity", "n_floor_real", "n_floor_twin", "unknown_writes", "liquidated",
    "position_pre", "position_post", "spot", "entry", "settlement", "open_notional",
) + tuple(f"r_k{i}" for i in range(len(COST_MULTIPLIERS))) + tuple(f"rt_k{i}" for i in range(len(COST_MULTIPLIERS)))


def _mtm_sum(env) -> float:
    return float(sum(float(v) for v in (getattr(env, "financial_mtm_positions", {}) or {}).values()))


def _position_sum(env) -> float:
    return float(sum(float(v) for v in (getattr(env, "financial_positions", {}) or {}).values()))


def _safe_float(fn, *a) -> float:
    try:
        return float(fn(*a))
    except Exception:
        return float("nan")


def install(cls, out_path: str) -> None:
    instances = []

    def get_budget(self):
        try:
            return self.__dict__["_tw_budget"]
        except KeyError:
            raise AttributeError("budget") from None

    def set_budget(self, value):
        d = self.__dict__
        old = d.get("_tw_budget", 0.0)
        d["_tw_budget"] = value
        writes = d.get("_tw_writes")
        if writes is None:
            return
        frame = sys._getframe(1)
        site = frame.f_code.co_name
        rec = {"site": site, "before": float(old), "after": float(value)}
        if site in NOMINAL_LOCAL:
            kind, name = NOMINAL_LOCAL[site]
            rec["kind"] = kind
            rec["amount"] = float(frame.f_locals.get(name, float("nan")))
        elif site == "_settle":
            loc = frame.f_locals
            rec["kind"] = "realize"
            rec["key"] = str(loc["key"])
            rec["old_mtm"] = float(loc["old_mtm"])
            rec["fraction"] = float(np.clip(float(loc["closed_fraction"]), 0.0, 1.0))
            rec["amount"] = float(loc["realized"])
        elif site == "_execute_investor_trades":
            text = linecache.getline(frame.f_code.co_filename, frame.f_lineno)
            if "max(0.0" in text:
                rec["kind"] = "floor"
            elif "-=" in text:
                rec["kind"] = "transaction"
                rec["amount"] = float(frame.f_locals["transaction_costs"])
            else:
                rec["kind"] = "unknown"
        else:
            rec["kind"] = "unknown"
        writes.append(rec)

    cls.budget = property(get_budget, set_budget)

    orig_mtm = cls._compute_mtm_for_step

    def compute_mtm(self, i, current_price):
        out = orig_mtm(self, i, current_price)
        self.__dict__["_tw_gross"] = float(out[0])
        self.__dict__["_tw_per_asset"] = {k: float(v) for k, v in (out[1] or {}).items()}
        return out

    cls._compute_mtm_for_step = compute_mtm

    orig_step = cls.step

    def step(self, actions):
        d = self.__dict__
        t = int(getattr(self, "t", 0))
        if t >= int(getattr(self, "max_steps", 10**12)):
            return orig_step(self, actions)
        if "_tw_rows" not in d:
            d["_tw_rows"] = []
            instances.append(self)
        d["_tw_gross"] = 0.0
        d["_tw_per_asset"] = {}
        cash_pre = float(self.budget)
        mtm_pre = _mtm_sum(self)
        dist_pre = float(getattr(self, "total_distributions", 0.0) or 0.0)
        position_pre = _position_sum(self)
        d["_tw_writes"] = []
        try:
            out = orig_step(self, actions)
        finally:
            writes = d.pop("_tw_writes", [])
        d["_tw_rows"].append(_close_step(self, t, writes, cash_pre, mtm_pre, dist_pre, position_pre))
        return out

    cls.step = step

    def save() -> None:
        best = max(instances, key=lambda e: len(e.__dict__.get("_tw_rows", [])), default=None)
        if best is None or not best.__dict__.get("_tw_rows"):
            return
        arr = np.asarray(best.__dict__["_tw_rows"], dtype=np.float64)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        tmp = out_path + ".tmp.npz"
        np.savez_compressed(tmp, columns=np.array(COLUMNS), rows=arr, e0=float(best.__dict__.get("_tw_e0", arr[0, 1])),
                            cost_multipliers=np.array(COST_MULTIPLIERS))
        os.replace(tmp, out_path)

    atexit.register(save)


def _replay(writes, per_asset, cash_pre, k):
    """Both branches through the step's cash writes, with the own costs multiplied by k. Returns the cash before the
    distribution (which the reported change adds back) and the number of floors that bound on each branch."""
    c_real, c_twin = cash_pre, cash_pre
    floors_real = floors_twin = 0
    for w in writes:
        kind = w["kind"]
        if kind in ("access", "roll", "funding"):
            amount = k * w["amount"]
            floors_real += int(c_real - amount < 0.0)
            floors_twin += int(c_twin - amount < 0.0)
            c_real, c_twin = max(0.0, c_real - amount), max(0.0, c_twin - amount)
        elif kind == "settle":
            floors_real += int(c_real + w["amount"] < 0.0)
            floors_twin += int(c_twin - w["amount"] < 0.0)
            c_real, c_twin = max(0.0, c_real + w["amount"]), max(0.0, c_twin - w["amount"])
        elif kind == "battery":
            floors_real += int(c_real + w["amount"] < 0.0)
            floors_twin += int(c_twin + w["amount"] < 0.0)
            c_real, c_twin = max(0.0, c_real + w["amount"]), max(0.0, c_twin + w["amount"])
        elif kind == "realize":
            c_real += w["amount"]
            c_twin += (w["old_mtm"] - 2.0 * per_asset.get(w["key"], 0.0)) * w["fraction"]
        elif kind == "transaction":
            c_real -= k * w["amount"]
            c_twin -= k * w["amount"]
        elif kind == "floor":
            floors_real += int(c_real < 0.0)
            floors_twin += int(c_twin < 0.0)
            c_real, c_twin = max(0.0, c_real), max(0.0, c_twin)
    return c_real, c_twin, floors_real, floors_twin


def _close_step(env, t, writes, cash_pre, mtm_pre, dist_pre, position_pre):
    d = env.__dict__
    d.setdefault("_tw_e0", cash_pre + mtm_pre)
    horizon = bool(env._is_horizon_settlement_mtm())
    per_asset = {} if horizon else d.get("_tw_per_asset", {})  # in horizon mode the payoff never enters the bucket
    amounts = dict(access=0.0, roll=0.0, transaction=0.0, funding=0.0, battery=0.0)
    fractions = {}
    distributed = 0.0
    unknown = liquidated = 0
    for w in writes:
        kind = w["kind"]
        if kind in amounts:
            amounts[kind] += w["amount"]
        elif kind == "realize":
            fractions[w["key"]] = w["fraction"]
        elif kind == "distribution":
            distributed += w["before"] - w["after"]
        elif kind == "liquidation":
            liquidated += 1
        elif kind not in ("settle", "floor"):
            unknown += 1

    c_real, c_twin, n_floor_real, n_floor_twin = _replay(writes, per_asset, cash_pre, 1.0)
    cash_post = float(env.budget)
    mtm_post = _mtm_sum(env)
    dist_post = float(getattr(env, "total_distributions", 0.0) or 0.0)
    parity = (c_real - distributed) - cash_post

    # The twin's bucket: the reversed payoff of this step, on the part that stayed open.
    twin_mtm_post = mtm_post
    for key, m in per_asset.items():
        if key in ASSETS:
            twin_mtm_post += -2.0 * float(m) * (1.0 - fractions.get(key, 0.0))

    gross = float(d.get("_tw_gross", 0.0))
    base = max(0.0, cash_pre) + mtm_pre
    r = (max(0.0, cash_post) + mtm_post + dist_post) - (base + dist_pre)
    r_twin = (c_twin + twin_mtm_post) - base

    settlement = _safe_float(env._horizon_settlement_price, t)
    entry = _safe_float(env._horizon_entry_price, t)
    spot = float(np.clip(env._price_raw[t], -1000.0, 1e9)) if t < len(env._price_raw) else float("nan")
    open_notional = float(sum(sum(abs(float(v)) for v in (c.get("exposures", {}) or {}).values())
                              for c in (getattr(env, "_horizon_contracts", []) or [])))

    own_positive = amounts["access"] + amounts["roll"] + amounts["transaction"] + amounts["funding"]
    if gross != 0.0 and own_positive > 0.0:
        shifted_r, shifted_t = [], []
        for k in COST_MULTIPLIERS:
            a, b, _, _ = _replay(writes, per_asset, cash_pre, k)
            shifted_r.append(a + mtm_post - base)
            shifted_t.append(b + twin_mtm_post - base)
    else:
        shifted_r = shifted_t = [float("nan")] * len(COST_MULTIPLIERS)

    return (
        float(t), cash_pre, cash_post, mtm_pre, mtm_post, dist_pre, dist_post,
        r, r_twin, gross, amounts["access"], amounts["roll"], amounts["transaction"], amounts["funding"], amounts["battery"],
        parity, float(n_floor_real), float(n_floor_twin), float(unknown), float(liquidated),
        position_pre, _position_sum(env), spot, entry, settlement, open_notional,
        *shifted_r, *shifted_t,
    )
