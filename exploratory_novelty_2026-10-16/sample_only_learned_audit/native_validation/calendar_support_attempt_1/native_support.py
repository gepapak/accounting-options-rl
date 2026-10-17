"""Pre-draw calendar support for the authentic fixed-cadence native engine.

No policy inference, environment step, random draw or endpoint is run on import.
Only configured hard capacities, the full declared physical context and frozen
initial pending contracts are used. No conditional payoff expectation is queried.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import numpy as np

GAMMA = .995
SCHEMA = "native_financial_calendar_support_v1"


def _require(value, message):
    if not bool(value):
        raise AssertionError(message)


def _up(value):
    value = float(value)
    _require(math.isfinite(value) and value >= 0, "Finite nonnegative support constant")
    if value == 0:
        return 0.0
    return float(np.nextafter(value+max(1e-7,1e-12*value), math.inf))


def tightened_calendar_support(episode, gamma=GAMMA):
    """Bound a fresh warmed clone before its audit steps, without reading coins.

    The 96-row arrays use RELATIVE audit time. Carried contracts were paid before
    the audit, so only their future gross and any outstanding funding enter here.
    New roll fees occur at admissible opening clocks only. Funding is bounded
    after current maturity processing, matching native _update_finance ordering.
    """
    env, spec = episode.env, episode.scenario.spec
    cfg = env.config
    H, start = int(spec.horizon), int(spec.burn_in)
    end = start+H
    _require(not episode.step_records and int(env.t)==start, "Support must precede audit transitions")
    _require(0<float(gamma)<=1, "Finite-horizon discount")
    _require(str(cfg.mtm_return_model)=="horizon_settlement", "Native fixed maturity model")
    _require(str(cfg.mtm_horizon_payoff_denominator_mode)=="mwh_volume", "Native MWh payoff units")
    _require(env._get_investor_sizing_base_mode()=="initial_trading_sleeve", "Original fixed initial sleeve sizing")
    _require(str(cfg.mtm_entry_price_mode)=="current_price", "Current-price entry bound")
    lag, frequency = int(cfg.mtm_settlement_horizon_steps), int(env.investment_freq)
    _require(lag>0 and frequency>0, "Positive fixed clocks")
    _require(int(cfg.meta_freq_min)==frequency==int(cfg.meta_freq_max), "Meta updates cannot change native opening cadence")
    _require(len(env._price_raw)==end, "Complete native burn-in/audit window, with no uncensored tail")
    original_F = float(env._initial_trading_sleeve_dkk())
    conversion = float(env._mwh_volume_conversion_price(0.0))
    _require(original_F>=0 and math.isfinite(original_F) and conversion>0, "Original sleeve and fixed conversion")
    # Native tradeable allocation is clipped to [0,1]. This conservative bound
    # does not confuse the class .75 with the saved config's .8 meta maximum.
    q_size = _up(original_F*float(cfg.max_position_size)*
                 max(1.,float(cfg.risk_exposure_cap_max))/conversion*(1.+2e-6))
    _require(float(cfg.max_position_size)>0 and float(cfg.risk_exposure_cap_max)>=0, "Monotone fixed sizing")
    cap_fraction = max(float(cfg.liquidity_participation_cap_fraction),0.)
    raw_friction = getattr(cfg,"friction_cost_multiplier",1.)
    raw_spread = getattr(cfg,"half_spread_bp",0.)
    friction = 1. if raw_friction is None else float(raw_friction)
    spread = 0. if raw_spread is None else float(raw_spread)
    impact_raw = getattr(cfg,"impact_coef_bp",0.)
    impact = 0. if impact_raw is None else float(impact_raw)
    exponent = float(getattr(cfg,"impact_exponent",.5) or .5)
    _require(min(friction,spread,impact,exponent)>=0, "Native roll-cost monotonicity")
    settlement_low = float(cfg.mtm_external_settlement_min_price_dkk_per_mwh)
    settlement_high = float(cfg.mtm_external_settlement_max_price_dkk_per_mwh)
    _require(math.isfinite(settlement_low) and math.isfinite(settlement_high) and settlement_low<=settlement_high,
             "Configured settlement support")
    roll = np.zeros(H)
    access = np.zeros(H)
    funding = np.zeros(H)
    gross = np.zeros(H)
    openings = []
    effective_caps = np.zeros(H)
    opening_possible = np.zeros(H,dtype=bool)
    zero_cap_fallback = np.zeros(H,dtype=bool)
    source_helper = episode.runtime.environment
    fee = float(source_helper.market_access_fee_for_step(
        annual_fee_dkk=float(cfg.annual_market_access_fee_dkk),
        allocation_fraction=float(cfg.market_access_fee_allocation_fraction),
        time_step_hours=float(cfg.time_step_hours)))
    access[:] = _up(fee)
    initial_claims = []
    for contract in env._horizon_contracts:
        maturity = int(contract["settle_step"])
        _require(start<=maturity<end, "Carried claim matures within the complete audit window")
        volumes = contract.get("volumes_mwh",{})
        volume = _up(sum(abs(float(v)) for v in volumes.values()))
        entry = float(contract["entry_price"])
        claim = {"origin":"prepaid_burn_in", "open_native_step":int(contract["open_step"]),
                 "settle_native_step":maturity,"volume_bound_mwh":volume,"entry_price":entry}
        initial_claims.append(claim)
        openings.append(claim)
    for relative,t in enumerate(range(start,end)):
        # The native opener refuses t=0, off-cadence times and maturity>=end.
        allowed = t>0 and t%frequency==0 and t+lag<end
        opening_possible[relative] = allowed
        if not allowed:
            continue
        if cap_fraction>0:
            capacity = cap_fraction*min(float(env._market_liquidity_volume_mwh(t)),
                                       float(env._causal_market_liquidity_volume_mwh(t)))
        else:
            capacity = 0.
        _require(math.isfinite(capacity) and capacity>=0, "Finite native effective capacity")
        # Native _cap_exposures_by_liquidity keeps scale=1 on zero capacity.
        zero_cap_fallback[relative] = capacity<=0
        q = min(q_size,capacity) if capacity>0 else q_size
        q = _up(q)
        effective_caps[relative] = q
        components = env._estimate_horizon_roll_cost(conversion*q,t,q)
        roll[relative] = _up(float(components["roll_cost_dkk"]))
        openings.append({"origin":"possible_audit_opening","open_native_step":t,
                         "settle_native_step":t+lag,"volume_bound_mwh":q,
                         "entry_price":float(env._horizon_entry_price(t))})
    margin_coefficient = max(conversion*max(float(cfg.collateral_notional_margin_fraction),0.),
        max(float(cfg.collateral_stress_price_dkk_per_mwh),0.)*
        max(float(cfg.collateral_stress_loss_fraction),0.))
    funding_factor = (max(float(cfg.collateral_funding_rate_annual),0.)*
                      max(float(cfg.time_step_hours),1e-9)/(365.25*24))
    for relative,t in enumerate(range(start,end)):
        due = [claim for claim in openings if claim["settle_native_step"]==t]
        gross[relative] = _up(sum(claim["volume_bound_mwh"]*
            max(abs(settlement_low-claim["entry_price"]),abs(settlement_high-claim["entry_price"]))
            for claim in due))
        # Native funding is paid after current maturity and margin liquidation.
        # Retaining all possible outstanding claims is an upper bound because
        # margin, no-trade rules, learned gates and cash can only remove claims.
        active_volume = sum(claim["volume_bound_mwh"] for claim in openings
                            if claim["open_native_step"]<=t<claim["settle_native_step"])
        if bool(cfg.enable_collateral_cash_drag):
            funding[relative] = _up(active_volume*margin_coefficient*funding_factor)
    costs = np.asarray([_up(a+r+f) for a,r,f in zip(access,roll,funding)])
    weights = float(gamma)**np.arange(H)
    exact_access = float(weights.sum())*fee
    access_lower = max(0.,float(np.nextafter(
        exact_access-max(1e-7,abs(exact_access)*1e-12),-math.inf)))
    result = {"schema":SCHEMA,"gamma":float(gamma),"horizon":H,"burn_in":start,
        "native_first_step":start,"native_last_step":end-1,
        "original_initial_trading_sleeve":original_F,
        "fixed_initial_sizing_volume_guard_mwh":q_size,"allocation_upper_bound":1.,
        "configured_meta_cap_max_descriptive":float(cfg.meta_cap_max),
        "investment_frequency":frequency,"settlement_lag":lag,
        "discounted_cost_bound":_up(float(weights@costs)),
        "discounted_gross_bound":_up(float(weights@gross)),
        "unavoidable_discounted_access_cost":access_lower,
        "calendar_cost_bound":costs.tolist(),"calendar_gross_bound":gross.tolist(),
        "calendar_roll_cost_bound":roll.tolist(),"calendar_access_cost_bound":access.tolist(),
        "calendar_collateral_cost_bound":funding.tolist(),
        "calendar_new_opening_possible":opening_possible.tolist(),
        "calendar_effective_opening_volume_bound_mwh":effective_caps.tolist(),
        "calendar_zero_capacity_sizing_fallback":zero_cap_fallback.tolist(),
        "initial_prepaid_claims":initial_claims,"possible_new_openings":int(opening_possible.sum()),
        "n_payoff_moment_queries":0,
        "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "native_source_sha256":episode.runtime.source_sha256,
        "scope":"deterministic configured/context support before audit; native chronological event order; not sample maxima or conditional moments",
        "numerics":"positive scalar/row envelopes rounded upward with abs1e-7/rel1e-12 cushion and nextafter"}
    _require(result["discounted_cost_bound"]<=episode.support["discounted_cost_bound"]+1e-3,
             "Calendar cost envelope contained by broad adapter support")
    _require(result["discounted_gross_bound"]<=episode.support["discounted_gross_bound"]+1.,
             "Calendar gross envelope contained by broad adapter support")
    return result


def union_context_support(context_bounds):
    """Scalar support for an unknown/declared mixture of the fixed contexts."""
    rows = list(context_bounds)
    _require(len(rows)>0,"Nonempty context law")
    _require(len({(r["gamma"],r["horizon"]) for r in rows})==1,"Same audit discount/horizon")
    return {"schema":SCHEMA,"n_contexts":len(rows),"gamma":rows[0]["gamma"],"horizon":rows[0]["horizon"],
        "discounted_cost_bound":max(row["discounted_cost_bound"] for row in rows),
        "discounted_gross_bound":max(row["discounted_gross_bound"] for row in rows),
        "unavoidable_discounted_access_cost":max(0.,min(row["unavoidable_discounted_access_cost"] for row in rows)),
        "scope":"max of full predetermined context supports; not maxima of realized episodes",
        "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def check_trace_support(trace, bounds, *, tolerance=2e-5):
    """Arithmetic-only checks on a completed trace; no estimator/certificate."""
    H = int(bounds["horizon"])
    steps = np.asarray(trace["event_step"],dtype=int)
    _require(np.all((steps>=0)&(steps<H)),"Relative native audit clock")
    costs = np.asarray(trace["event_cost"],float)
    gross = np.asarray(trace["event_gross"],float)
    injections = np.asarray(trace["event_injection"],float)
    kinds = np.asarray(trace["event_kind"],str)
    coins = np.asarray(trace["event_sign_epsilon"],int)
    _require(np.all(np.isfinite(np.r_[costs,gross,injections])) and np.all(costs>=0),"Finite monetary trace")
    _require(set(kinds)<=set(("roll_cost","access_cost","collateral_cost","maturity")),"Complete declared financial hooks")
    aggregate_cost = np.bincount(steps,weights=costs,minlength=H)
    aggregate_gross = np.bincount(steps,weights=np.abs(gross),minlength=H)
    _require(np.all(aggregate_cost<=np.asarray(bounds["calendar_cost_bound"])+tolerance),"Calendar financial cost support")
    _require(np.all(aggregate_gross<=np.asarray(bounds["calendar_gross_bound"])+tolerance),"Calendar absolute gross support")
    for kind,key in (("roll_cost","calendar_roll_cost_bound"),("access_cost","calendar_access_cost_bound"),
                     ("collateral_cost","calendar_collateral_cost_bound")):
        active = kinds==kind
        by_row = np.bincount(steps[active],weights=costs[active],minlength=H)
        counts = np.bincount(steps[active],minlength=H)
        _require(np.all(counts<=1),"At most one native "+kind+" call per audit step")
        _require(np.all(by_row<=np.asarray(bounds[key])+tolerance),kind+" calendar support")
    maturity_counts = np.bincount(steps[coins!=0],minlength=H)
    _require(np.all(maturity_counts<=1),"One shared-sign aggregate maturity batch per row")
    _require(np.all(injections<=costs+np.abs(gross)+tolerance),"Local financial floor injection envelope")
    weights = float(bounds["gamma"])**np.arange(H)
    total_cost,total_gross=float(weights@aggregate_cost),float(weights@aggregate_gross)
    _require(total_cost<=float(bounds["discounted_cost_bound"])+tolerance,"Whole-path cost support")
    _require(total_gross<=float(bounds["discounted_gross_bound"])+tolerance,"Whole-path gross support")
    _require(np.array_equal(np.asarray(trace["path_step"],int),np.arange(H)),"Complete fixed audit trajectory")
    _require(int(np.asarray(trace["path_open_contracts"])[-1])==0,"No unsettled terminal contract")
    return {"pass":True,"calendar_rows":H,"financial_events":len(steps),
            "all_cost_kinds_checked":True,"terminal_queue_empty":True}
