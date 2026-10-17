"""Unfrozen source-derived v2 support, including alpha0 new-claim cash funding.

No environment/actor/random query occurs on import. Functions require a fresh
clone after audit-only cap removal, before audit transitions/sign disclosure.
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent
ORIGINAL_SUPPORT_SHA256 = "b6134d8bc41151d417c61197ecc07117a4ccdc5204427f69d432e9953aa3411a"


def _original_support():
    name = "_joint_fault_v2_original_support"
    if name not in sys.modules:
        path = ORIGINAL / "native_support.py"
        if hashlib.sha256(path.read_bytes()).hexdigest() != ORIGINAL_SUPPORT_SHA256:
            raise RuntimeError("Preserve exact original calendar support source")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


base = _original_support()
GAMMA = base.GAMMA


def _require(value, message):
    if not bool(value):
        raise AssertionError(message)


def cash_inflow_support(episode, gamma=GAMMA):
    env, cfg, spec, np = episode.env, episode.env.config, episode.scenario.spec, episode.runtime.np
    H, first = int(spec.horizon), int(spec.burn_in)
    _require(not episode.step_records and int(env.t) == first, "Pre-draw initial clone only")
    _require(not bool(cfg.enable_trading_sleeve_margin), "No margin asset-to-cash conversion")
    _require(not bool(cfg.allow_operational_reallocation), "No emergency operational recapitalization")
    _require(env._is_horizon_settlement_mtm(), "No legacy realized-MTM cash conversion")
    _require(float(cfg.liquidity_participation_cap_fraction) == 0., "Audit-only missing participation cap")
    _require(0 < float(gamma) <= 1, "Nonincreasing nonnegative financial event weights")
    for key in ("distribution_rate", "target_cash_ratio", "min_distribution_threshold_ratio"):
        value = float(getattr(cfg, key))
        _require(math.isfinite(value) and 0 <= value <= 1, "Native distribution range: " + key)
    override = getattr(cfg, "eval_distribution_rate", None)
    _require(override is None or (math.isfinite(float(override)) and 0 <= float(override) <= 1),
             "Native eval distribution override cannot overdraw")
    # FinancialEngine.calculate_fund_nav returns max(fund_equity,0) in BOTH
    # branches. Thus target_cash>=0 and any actual excess withdrawal<=budget.
    initial = float(env.budget)
    capacity = float(env.physical_assets["battery_capacity_mwh"])
    c_rate = float(env.batt_power_c_rate)
    eta_c, eta_d = float(env.batt_eta_charge), float(env.batt_eta_discharge)
    degradation = float(env.batt_degradation_cost)
    _require(all(math.isfinite(v) and v >= 0 for v in (initial, capacity, c_rate, eta_c, eta_d, degradation)),
             "Finite physical battery constants and nonnegative degradation")
    _require(0 <= float(env.batt_soc_min) <= float(env.batt_soc_max) <= 1, "Native bounded battery energy state")
    _require(0 <= float(env.operational_state["battery_energy"]) <= capacity,
             "Actual initial energy in source-invariant capacity interval")
    lower, upper = float(cfg.minimum_price_filter), float(cfg.maximum_price_cap)
    _require(math.isfinite(lower) and math.isfinite(upper) and lower <= upper, "Native finite battery price clip")
    prices = np.clip(np.asarray(env._price_raw[first:first+H], dtype=float), lower, upper)
    _require(len(prices) == H and np.all(np.isfinite(prices)), "Complete fixed calendar battery price path")
    # TradingEngine uses literal ten-minute step_h in execute_battery_operations.
    # Every action decoder caps requested energy by this power limit; efficiency
    # and nonnegative degradation can only reduce the positive cash bound below.
    moved = capacity*c_rate*(10./60.)
    battery = moved*np.maximum(eta_d*np.maximum(prices, 0.),
                               np.maximum(-prices, 0.)/max(eta_c, 1e-6))
    old = np.zeros(H)
    old_batches = {}
    old_absolute_terms = {}
    lag = int(cfg.mtm_settlement_horizon_steps)
    _require(lag == 6, "Frozen direct-old-maturity mask has fixed lag6")
    for contract in env._horizon_contracts:
        opened, mature = int(contract["open_step"]), int(contract["settle_step"])
        _require(opened < first and mature == opened+lag and first <= mature < first+lag,
                 "All inherited claims belong exactly to relative steps0..5")
        delivery = int(contract.get("delivery_step", opened))
        unit = float(env._horizon_unit_payoff(float(contract["entry_price"]),
                                             float(env._horizon_settlement_price(delivery))))
        volumes = contract["volumes_mwh"]
        keys = ("wind_instrument_value", "solar_instrument_value", "hydro_instrument_value")
        _require(set(volumes) <= set(keys), "Native three-asset inherited volume contract")
        terms = [float(volumes.get(k, 0.))*unit for k in keys]
        gross = sum(terms)
        _require(math.isfinite(gross), "Finite predetermined inherited unsigned amplitude")
        relative = mature-first
        old_batches[relative] = old_batches.get(relative, 0.)+gross
        old_absolute_terms.setdefault(relative, []).extend(abs(v) for v in terms)
    for relative, gross in old_batches.items():
        # All contracts due at one step share one post-action fresh sign.
        # Triangle envelope avoids cancellation/accumulation-order sensitivity
        # between sum(volume)*unit and the native per-asset maturity loop.
        old[relative] = math.fsum(old_absolute_terms[relative])
    battery = np.asarray([base._up(v) for v in battery])
    old = np.asarray([base._up(v) for v in old])
    weights = float(gamma)**np.arange(H)
    K = base._up(initial+float(weights@(battery+old)))
    return dict(cash_inflow_bound=K, cash_inflow_initial_cash=initial,
        cash_inflow_battery_credit_bound=base._up(float(weights@battery)),
        cash_inflow_old_maturity_credit_bound=base._up(float(weights@old)),
        calendar_battery_positive_credit_bound=battery.tolist(),
        calendar_old_positive_credit_bound=old.tolist(),
        old_maturity_unsigned_batch_gross={str(k):v for k,v in sorted(old_batches.items())},
        cash_target="alpha0: ALL forward financial costs/cost injections; direct inherited maturity G/I excluded",
        cash_bound_source="initialcash + deterministic physical battery price/power bound + known inherited absolute per-asset amplitudes",
        cash_bound_requires_financial_pre_and_post_nonnegative=True,
        cash_bound_requires_full_cash_write_classification=True,
        cash_bound_distribution_repair=False, n_cash_bound_moment_queries=0,
        n_cash_bound_payoff_point_evaluations=len(env._horizon_contracts))


def tightened_calendar_support(episode, gamma=GAMMA):
    result = base.tightened_calendar_support(episode, gamma)
    result.update(cash_inflow_support(episode, gamma))
    result["joint_fault_support_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def union_context_support(supports):
    rows = list(supports)
    result = base.union_context_support(rows)
    result["cash_inflow_bound"] = max(s["cash_inflow_bound"] for s in rows)
    result["cash_inflow_bound_contexts"] = len(rows)
    result["cash_target"] = rows[0]["cash_target"]
    result["joint_fault_support_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def check_trace_support(trace, support):
    base.check_trace_support(trace, support)
    import numpy as np
    _require(np.all(np.asarray(trace["event_cash_before"]) >= 0) and
             np.all(np.asarray(trace["event_cash_after"]) >= 0), "Actual financial pre/post cash nonnegative")
    _require(np.all(np.asarray(trace["path_native_cash_before"]) >= 0) and
             np.all(np.asarray(trace["path_native_cash_after"]) >= 0), "Complete native step-boundary cash nonnegative")
    H = int(support["horizon"])
    steps, kinds = np.asarray(trace["transfer_step"]), np.asarray(trace["transfer_kind"])
    returned = np.asarray(trace["transfer_returned_cashflow"], dtype=float)
    inside = np.asarray(trace["transfer_cash_change_inside_method"], dtype=float)
    _require(not np.any(kinds == "margin_conversion"), "Disabled margin branch has no asset conversion")
    battery_bound = np.asarray(support["calendar_battery_positive_credit_bound"])
    for t in range(H):
        battery = (steps == t)&(kinds == "battery")
        distribution = (steps == t)&(kinds == "distribution")
        _require(int(battery.sum()) == int(distribution.sum()) == 1, "Exactly one battery and distribution call per step")
        _require(abs(float(inside[battery][0])) <= 1e-7, "Battery method returns cashflow without cash mutation")
        _require(float(inside[distribution][0]) <= 1e-7, "Distribution does not supply positive trading cash")
        _require(max(float(returned[battery][0]), 0.) <= battery_bound[t]+1e-5,
                 "All battery positive inflows covered by source bound")
        excluded = float(np.asarray(trace["path_excluded_native_cash_change"])[t])
        _require(excluded <= max(float(returned[battery][0]), 0.)+1e-4,
                 "No unexplained positive excluded credit beyond physical battery bound")
    events = np.asarray(trace["event_step"])
    old = (np.asarray(trace["event_sign_epsilon"]) != 0)&(events < 6)
    weights = float(support["gamma"])**events
    credit = np.maximum(np.asarray(trace["event_gross"])+np.asarray(trace["event_injection"]), 0.)
    _require(float(np.sum(weights[old]*credit[old])) <= support["cash_inflow_old_maturity_credit_bound"]+1e-4,
             "Excluded inherited cash credits obey predetermined batch bound")
    financial = ((np.asarray(trace["event_gross"])-np.asarray(trace["event_cost"])+
                  np.asarray(trace["event_injection"]))[~old])
    raw_new = float(np.sum(weights[~old]*financial))
    _require(raw_new >= -support["cash_inflow_bound"]-1e-3, "Cash-funded alpha0 new-claim raw lower endpoint")
    return True
