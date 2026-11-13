r"""Read-only step tracer for the audit engine: the sign twin of every step, from the engine's own cash writes.

Nothing in the engine's source changes. `install()` puts a property on the class attribute `budget`, so every write to
trading cash passes through a setter that records where it came from and the nominal amount the writing method held in
its local variables, before any floor. After each `step` the writes are replayed twice, in the engine's own order and
with its own floors:

  real branch   the amounts as written; it must reproduce the engine's cash at every step (parity);
  twin branch   the same step from the same pre-step state, same action and same own cash flows, gross P&L reversed.

The twin is one step off the real path: the state is not carried over, and the executed action, costs and battery flow
are held at the real branch's values. The floor rule and the order of writes are the engine's.

Horizon settlement: at maturity `budget = max(0, budget + total_pnl)`; the twin uses -total_pnl. Capped percent MTM
(the L7 family): the step's payoff m_k goes to an unfloored bucket per asset, and a position closed by a fraction f moves
f of the bucket to cash; the twin's bucket is lower by 2 m_k and its realized part is (bucket - 2 m_k) f.

Distributions are the step's last write. The reported return adds them back, so they do not enter the reported change.

Per step the tracer stores: r (reported change of sleeve value plus distributions), r_twin, the components of the
booked P&L (gross, own costs, battery), the replay's parity error, and the public quantities a direct calculation would
use (own position, and the three price series).
"""
from __future__ import annotations

import atexit
import linecache
import os
import sys

import numpy as np

ASSETS = ("wind_instrument_value", "solar_instrument_value", "hydro_instrument_value")
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
)


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
        row = _close_step(self, t, writes, cash_pre, mtm_pre, dist_pre, position_pre)
        d["_tw_rows"].append(row)
        return out

    cls.step = step

    def save() -> None:
        best = max(instances, key=lambda e: len(e.__dict__.get("_tw_rows", [])), default=None)
        if best is None or not best.__dict__.get("_tw_rows"):
            return
        arr = np.asarray(best.__dict__["_tw_rows"], dtype=np.float64)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        tmp = out_path + ".tmp.npz"
        np.savez_compressed(tmp, columns=np.array(COLUMNS), rows=arr, e0=float(best.__dict__.get("_tw_e0", arr[0, 1])))
        os.replace(tmp, out_path)

    atexit.register(save)


def _close_step(env, t, writes, cash_pre, mtm_pre, dist_pre, position_pre):
    d = env.__dict__
    d.setdefault("_tw_e0", cash_pre + mtm_pre)
    per_asset = d.get("_tw_per_asset", {})
    amounts = dict(access=0.0, roll=0.0, transaction=0.0, funding=0.0, battery=0.0)
    settle_gross = 0.0
    c_real, c_twin = cash_pre, cash_pre
    n_floor_real = n_floor_twin = 0
    unknown = liquidated = 0
    fractions = {}
    distributed = 0.0

    def floored(value):
        return max(0.0, value), value < 0.0

    for w in writes:
        kind = w["kind"]
        if kind in ("access", "roll", "funding"):
            amounts[kind] += w["amount"]
            c_real, a = floored(c_real - w["amount"])
            c_twin, b = floored(c_twin - w["amount"])
        elif kind == "settle":
            settle_gross += w["amount"]
            c_real, a = floored(c_real + w["amount"])
            c_twin, b = floored(c_twin - w["amount"])
        elif kind == "battery":
            amounts["battery"] += w["amount"]
            c_real, a = floored(c_real + w["amount"])
            c_twin, b = floored(c_twin + w["amount"])
        elif kind == "realize":
            key = w["key"]
            fractions[key] = w["fraction"]
            c_real += w["amount"]
            c_twin += (w["old_mtm"] - 2.0 * per_asset.get(key, 0.0)) * w["fraction"]
            a = b = False
        elif kind == "transaction":
            amounts["transaction"] += w["amount"]
            c_real -= w["amount"]
            c_twin -= w["amount"]
            a = b = False
        elif kind == "floor":
            c_real, a = floored(c_real)
            c_twin, b = floored(c_twin)
        elif kind == "distribution":
            distributed += w["before"] - w["after"]
            a = b = False
        elif kind == "liquidation":
            liquidated += 1
            a = b = False
        else:
            unknown += 1
            a = b = False
        n_floor_real += int(a)
        n_floor_twin += int(b)

    cash_post = float(env.budget)
    mtm_post = _mtm_sum(env)
    dist_post = float(getattr(env, "total_distributions", 0.0) or 0.0)
    parity = (c_real - distributed) - cash_post

    # The twin's bucket: the reversed payoff of this step, on the part that stayed open.
    twin_mtm_post = mtm_post
    for key, m in per_asset.items():
        if key in ASSETS:
            twin_mtm_post += -2.0 * float(m) * (1.0 - fractions.get(key, 0.0))

    gross = float(d.get("_tw_gross", 0.0)) if not writes or settle_gross == 0.0 else settle_gross
    r = (max(0.0, cash_post) + mtm_post + dist_post) - (max(0.0, cash_pre) + mtm_pre + dist_pre)
    r_twin = (c_twin + twin_mtm_post) - (max(0.0, cash_pre) + mtm_pre)

    settlement = _safe_float(env._horizon_settlement_price, t)
    entry = _safe_float(env._horizon_entry_price, t)
    spot = float(np.clip(env._price_raw[t], -1000.0, 1e9)) if t < len(env._price_raw) else float("nan")
    open_notional = float(sum(sum(abs(float(v)) for v in (c.get("exposures", {}) or {}).values())
                              for c in (getattr(env, "_horizon_contracts", []) or [])))

    return (
        float(t), cash_pre, cash_post, mtm_pre, mtm_post, dist_pre, dist_post,
        r, r_twin, gross, amounts["access"], amounts["roll"], amounts["transaction"], amounts["funding"], amounts["battery"],
        parity, float(n_floor_real), float(n_floor_twin), float(unknown), float(liquidated),
        position_pre, _position_sum(env), spot, entry, settlement, open_notional,
    )
