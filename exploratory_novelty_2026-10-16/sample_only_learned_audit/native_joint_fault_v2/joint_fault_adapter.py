"""Unfrozen, source-only joint-fault adapter; native execution remains held.

Reuses the byte-exact run001 adapter. The original vulnerable .25-cap warm
state is cloned, then the participation cap is removed for the audit only.
No actor, environment, warm-up or random draw occurs on import.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent
ORIGINAL_ADAPTER_SHA256 = "da93030da39c1f2a4d0eaef3666177fc6628b7da97161ed892d1b9abfa38c879"


def _load_original():
    path = ORIGINAL / "native_engine_adapter.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != ORIGINAL_ADAPTER_SHA256:
        raise RuntimeError("Preserve exact original numerical adapter")
    name = "_joint_fault_v2_original_adapter"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


base = _load_original()
# This uniquely named module is private to v2. Only engineering output roots
# change in memory; frozen run001 source bytes and its live modules do not.
base.HERE = HERE
NativeRuntime = base.NativeRuntime
EpisodeSpec = base.EpisodeSpec
fixed_context_grid = base.fixed_context_grid
file_sha = base.file_sha
state_sha = base.state_sha
native_state_sha = base.native_state_sha
jsonable = base.jsonable
ENGINE = base.ENGINE


class NativeScenario(base.NativeScenario):
    def __init__(self, runtime, spec):
        if spec.profile != "joint_fault" or spec.burn_in != 384 or spec.horizon != 96:
            raise ValueError("Held v2 contract: joint_fault,384 warm steps,96 audit steps")
        # The exact original constructor performs the original vulnerable warm
        # protocol, under its original fixed-cache key and config snapshot.
        super().__init__(runtime, replace(spec, profile="vulnerable"))
        self.warm_spec = self.spec
        self.spec = spec
        self.metadata["warm_spec"] = self.warm_spec.__dict__.copy()
        self.metadata["spec"] = self.spec.__dict__.copy()
        self.metadata["warm_profile_intervention"] = self.metadata["profile_intervention"].copy()
        self.metadata["profile_intervention"] = {
            "profile": "joint_fault", "warm_profile": "vulnerable",
            "enable_trading_sleeve_margin": False,
            "mtm_loss_exit_threshold_pct": "disabled_inf",
            "warm_liquidity_participation_cap_fraction": .25,
            "audit_liquidity_participation_cap_fraction": 0.,
            "cash_fraction_after_burn_in": spec.cash_fraction,
            "preserve_original_sizing_base": True, "preserve_native_halt_flags": True}
        self.metadata["joint_fault_intervention"] = {
            "version": "native_joint_fault_v2_source_draft",
            "warm_profile": "vulnerable", "warm_participation_fraction": .25,
            "audit_participation_fraction": 0., "audit_only": True,
            "preserve_original_warm_parent_and_config": True,
            "preserve_fund_drawdown_halt": True,
            "observation_rebuilt_after_cap_change": True,
            "additional_distribution_or_recapitalization_repair": False,
            "scope": "L3-style audit fault after capped native warm-up; not exact historical L3 redeployment"}

    def _prepare_native_warm_snapshot(self):
        cfg = self.config_snapshot
        if float(cfg.liquidity_participation_cap_fraction) != .25:
            raise AssertionError("Original capped vulnerable warm-up is required")
        if bool(cfg.enable_trading_sleeve_margin) or not math.isinf(float(cfg.mtm_loss_exit_threshold_pct)):
            raise AssertionError("Preserve original vulnerable margin/loss settings")
        super()._prepare_native_warm_snapshot()

    def new_episode(self, sign_sequence):
        assert state_sha(self.config_snapshot.__dict__) == self.config_snapshot_sha256
        return NativeEpisode(self, sign_sequence)


class NativeEpisode(base.NativeEpisode):
    def __init__(self, scenario, sign_sequence):
        super().__init__(scenario, sign_sequence)
        cfg = self.env.config
        assert float(cfg.liquidity_participation_cap_fraction) == .25
        assert not bool(cfg.enable_trading_sleeve_margin)
        assert math.isinf(float(cfg.mtm_loss_exit_threshold_pct))
        self.warm_clone_config_sha256 = state_sha(cfg.__dict__)
        cfg.liquidity_participation_cap_fraction = 0.
        # Rebuild authentic native cap-aware20D features before the first actor.
        # Every cached warm-parent field and original config snapshot is kept.
        self.env._fill_obs()
        self.observations = {k: self.runtime.np.asarray(v).copy() for k, v in self.env._obs_buf.items()}
        self.audit_clone_config_sha256 = state_sha(cfg.__dict__)
        assert native_state_sha(scenario.warm_env_snapshot) == scenario.warm_snapshot_sha256
        assert state_sha(scenario.config_snapshot.__dict__) == scenario.config_snapshot_sha256
        self.support = self._support_bounds()

    def _support_bounds(self):
        """Broad finite envelope from fixed sizing, without participation cap."""
        env, cfg, np = self.env, self.env.config, self.runtime.np
        assert env._get_investor_sizing_base_mode() == "initial_trading_sleeve"
        assert str(cfg.mtm_return_model) == "horizon_settlement"
        assert str(cfg.mtm_horizon_payoff_denominator_mode) == "mwh_volume"
        assert 0 <= float(cfg.liquidity_participation_cap_fraction) <= 1
        conversion = float(env._mwh_volume_conversion_price(0.))
        q = (float(env._initial_trading_sleeve_dkk()) * float(cfg.max_position_size) *
             max(1., float(cfg.risk_exposure_cap_max)) / conversion * (1.+2e-6))
        assert math.isfinite(q) and q >= 0 and conversion > 0
        for claim in env._horizon_contracts:
            assert sum(abs(float(v)) for v in claim["volumes_mwh"].values()) <= q + 1e-5
        notional = q * conversion
        min_ref = max(float(np.min(env._impact_ref_notional_by_step)), 1.)
        fee = self.runtime.environment.base_execution_fee_components(
            model=cfg.market_fee_model, abs_notional_dkk=notional, abs_volume_mwh=q,
            friction_multiplier=float(cfg.friction_cost_multiplier),
            transaction_cost_bps=float(cfg.transaction_cost_bps),
            transaction_fixed_cost_dkk=float(cfg.transaction_fixed_cost),
            transaction_fee_dkk_per_mwh=float(cfg.transaction_fee_dkk_per_mwh))
        roll = (float(fee["base_execution_fee_dkk"]) + notional*float(cfg.half_spread_bp)/10000. +
                notional*float(cfg.impact_coef_bp)*float(cfg.liquidity_tail_impact_max_multiplier)*
                (notional/min_ref)**float(cfg.impact_exponent)/10000.)
        max_open = math.ceil(int(cfg.mtm_settlement_horizon_steps)/int(env.investment_freq)) + 1
        collateral = max_open*q*max(conversion*float(cfg.collateral_notional_margin_fraction),
            float(cfg.collateral_stress_price_dkk_per_mwh)*float(cfg.collateral_stress_loss_fraction))
        funding = collateral*float(cfg.collateral_funding_rate_annual)*float(cfg.time_step_hours)/(365.25*24.)
        access = float(self.runtime.environment.market_access_fee_for_step(
            annual_fee_dkk=float(cfg.annual_market_access_fee_dkk),
            allocation_fraction=float(cfg.market_access_fee_allocation_fraction),
            time_step_hours=float(cfg.time_step_hours)))
        entry_bound = max(float(np.max(np.abs(env._price_raw))), 1000.)
        settle_bound = max(abs(float(cfg.mtm_external_settlement_min_price_dkk_per_mwh)),
                           abs(float(cfg.mtm_external_settlement_max_price_dkk_per_mwh)))
        gross = max_open*q*(settle_bound+entry_bound)
        costs = roll+funding+access
        assert all(math.isfinite(v) and v >= 0 for v in (roll, funding, access, gross, costs))
        discount_sum = sum(.995**t for t in range(self.scenario.spec.horizon))
        return dict(origin="fixed initial sizing/action guard and configured/full-context support; no participation requirement",
            fixed_initial_sizing_volume_guard_mwh=q, max_single_contract_volume_mwh=q,
            max_simultaneously_open_contracts=max_open, roll_cost_bound=roll,
            access_cost_bound=access, collateral_cost_bound=funding,
            per_step_cost_bound=costs, per_step_gross_bound=gross, audit_gamma=.995,
            discounted_cost_bound=discount_sum*costs, discounted_gross_bound=discount_sum*gross,
            undiscounted_episode_cost_bound=self.scenario.spec.horizon*costs)

    def _record_event(self, kind, before, after, gross, cost, epsilon=0, base_gross=0.):
        if before < 0 or after < 0:
            raise AssertionError("V2 financial attribution requires actual pre/post cash nonnegative")
        return super()._record_event(kind, before, after, gross, cost, epsilon, base_gross)

    def summary(self):
        result = super().summary()
        result["joint_fault_intervention"] = self.scenario.metadata["joint_fault_intervention"]
        result["warm_clone_config_sha256"] = self.warm_clone_config_sha256
        result["audit_clone_config_sha256"] = self.audit_clone_config_sha256
        return result


def prepare_scenario(runtime, checkpoint_dir, *, zone="original", start_row=0,
                     profile="joint_fault", cash_fraction=1e-4, burn_in=384, H=96, native_seed=7):
    return NativeScenario(runtime, EpisodeSpec(str(checkpoint_dir), zone=zone, start_row=start_row,
        horizon=H, native_seed=native_seed, profile=profile, cash_fraction=cash_fraction, burn_in=burn_in))
