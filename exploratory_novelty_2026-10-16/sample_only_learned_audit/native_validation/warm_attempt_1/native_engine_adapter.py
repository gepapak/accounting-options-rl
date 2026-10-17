"""Authentic Prototype5 frozen-policy financial-event audit adapter.

No experiment is launched on import. Native engine/checkpoint sources are read-only.
Every episode receives a fresh config and environment; only actual maturity cash
is fair-sign shocked. Forecast unit-payoff queries retain native semantics.

The estimand is the financial component G-C+(1-alpha)I_fin, NOT fund NAV or the
native shaped RL reward. Battery/physical/margin-conversion cashflows remain in
the native state but are excluded from this component's accounting return.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
from types import MethodType
import types
from collections import deque

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
PAPER1 = CODE.parents[1]
ENGINE = PAPER1 / "Prototype5"


@contextmanager
def local_working_directory():
    previous = Path.cwd()
    os.chdir(HERE)
    try:
        yield
    finally:
        os.chdir(previous)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonable(value):
    if isinstance(value, float) and not math.isfinite(value):
        return "disabled_inf" if value > 0 else str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def state_sha(value):
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True).encode()).hexdigest()


def native_state_sha(env):
    """Hash every native state buffer, excluding the approved memory resource.

    Object sharing/cycles are represented by traversal paths. The memory-manager
    lock and housekeeping counters are computational resources, not economic
    state. Every other attribute, including RNG/controller/normalizer buffers,
    is recursively covered rather than converted to an address-bearing repr.
    """
    import numpy as np
    import pandas as pd
    import logging
    seen = {}
    def encode(value, path):
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, (float, np.floating)):
            return jsonable(float(value))
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.ndarray):
            return {"array": hashlib.sha256(value.tobytes()).hexdigest(),
                    "shape": value.shape, "dtype": str(value.dtype)}
        if isinstance(value, np.random.Generator):
            return {"rng": encode(value.bit_generator.state, path+".rng")}
        if isinstance(value, pd.DataFrame):
            return {"dataframe": hashlib.sha256(pd.util.hash_pandas_object(value, index=True).values.tobytes()).hexdigest(),
                    "columns": list(value.columns), "dtypes": [str(t) for t in value.dtypes]}
        if isinstance(value, (pd.Series, pd.Index)):
            return {"series": hashlib.sha256(pd.util.hash_pandas_object(value, index=True).values.tobytes()).hexdigest()}
        if isinstance(value, (np.dtype, Path)):
            return str(value)
        if isinstance(value, logging.Logger):
            return {"logger": value.name}
        if isinstance(value, (type, types.FunctionType, types.BuiltinFunctionType)):
            return {"callable": value.__module__+"."+value.__qualname__}
        if isinstance(value, types.MethodType):
            return {"method": value.__func__.__module__+"."+value.__func__.__qualname__,
                    "self": encode(value.__self__, path+".self")}
        identity = id(value)
        if identity in seen:
            return {"reference": seen[identity]}
        seen[identity] = path
        if isinstance(value, dict):
            return {str(k): encode(v, path+"."+str(k)) for k,v in sorted(value.items(), key=lambda pair: str(pair[0]))}
        if isinstance(value, deque):
            return {"deque_maxlen": value.maxlen, "values": [encode(v, path+"."+str(i)) for i,v in enumerate(value)]}
        if isinstance(value, (tuple, list)):
            return [encode(v, path+"."+str(i)) for i,v in enumerate(value)]
        if isinstance(value, (set, frozenset)):
            return sorted([encode(v, path+".set") for v in value], key=str)
        if hasattr(value, "detach"):
            return encode(value.detach().cpu().numpy(), path+".tensor")
        if hasattr(value, "__dict__"):
            fields = vars(value)
            if value is env:
                fields = {k:v for k,v in fields.items() if k != "memory_manager"}
            return {"class": type(value).__module__+"."+type(value).__qualname__,
                    "fields": encode(fields, path+".fields")}
        representation = repr(value)
        if " at 0x" in representation:
            raise TypeError("Unsupported address-bearing native buffer: "+path+" "+type(value).__name__)
        return {"class": type(value).__module__+"."+type(value).__qualname__, "value": representation}
    return hashlib.sha256(json.dumps(encode(env,"env"), sort_keys=True).encode()).hexdigest()


def clone_native_environment(env):
    """Copy all economic state; replace only the native memory manager/lock."""
    manager = env.memory_manager
    replacement = type(manager)(max_memory_mb=manager.max_memory_mb)
    clone = copy.deepcopy(env, memo={id(manager): replacement})
    assert clone is not env and clone.config is not env.config
    return clone


class NativeRuntime:
    """One process-local native runtime; parallelize with processes, not threads."""
    def __init__(self):
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
            os.environ[name] = "1"
        for name in ("MPLCONFIGDIR", "NUMBA_CACHE_DIR", "TORCH_HOME"):
            target = HERE / "runtime_cache" / name.lower()
            target.mkdir(parents=True, exist_ok=True)
            os.environ[name] = str(target)
        sys.path.insert(0, str(ENGINE))
        with local_working_directory():
            self.evaluation = importlib.import_module("evaluation")
            self.environment = importlib.import_module("environment")
            self.config = importlib.import_module("config")
            self.generator = importlib.import_module("generator")
            self.forecast = importlib.import_module("forecast_prior")
            self.policy = importlib.import_module("policy")
            self.np = importlib.import_module("numpy")
            self.pd = importlib.import_module("pandas")
            torch = importlib.import_module("torch")
            torch.set_num_threads(1)
            logging = importlib.import_module("logging")
            importlib.import_module("logger").configure_logging(
                level=logging.WARNING, log_file=str(HERE / "runtime_cache/native_engine.log"),
                force_reconfigure=True)
        for module in (self.evaluation, self.environment, self.config, self.policy):
            if Path(module.__file__).resolve().parent != ENGINE.resolve():
                raise RuntimeError("Native engine module-name collision: " + str(module.__file__))
        self.source_sha256 = {name: file_sha(ENGINE / name) for name in
            ("environment.py", "observation_builder.py", "evaluation.py", "policy.py", "config.py")}
        self.data_cache = {}
        self.forecast_cache = {}
        self.model_cache = {}
        self.warm_cache = {}

    def load_models(self, checkpoint_dir):
        checkpoint_dir = Path(checkpoint_dir).resolve()
        key = str(checkpoint_dir)
        if key not in self.model_cache:
            self.model_cache[key] = {
                "investor_0": self.policy.CentralizedCriticPPO.load(
                    checkpoint_dir / "investor_0_policy.zip", device="cpu"),
                "battery_operator_0": importlib.import_module("stable_baselines3").DQN.load(
                    checkpoint_dir / "battery_operator_0_policy.zip", device="cpu"),
            }
            for model in self.model_cache[key].values():
                model.policy.set_training_mode(False)
        return self.model_cache[key]

    def market(self, zone):
        if zone not in ("original", "v2"):
            raise ValueError("zone must identify native original/v2 evaluation context")
        if zone not in self.data_cache:
            stem = "unseendata" + ("_v2" if zone == "v2" else "")
            source = ENGINE / "evaluation_dataset_ffill" / (stem + ".csv")
            with local_working_directory():
                frame = self.generator.load_energy_data(str(source))
            self.data_cache[zone] = (frame, source)
        return self.data_cache[zone]

    def forecasts(self, zone):
        if zone not in self.forecast_cache:
            root = ENGINE / "forecast_cache_settlement_hourly_v2/forecast_cache_eval_episode20_2025"
            suffix = "unseendata_v2" if zone == "v2" else "full"
            directory = root / ("forecast_cache_eval_episode20_2025-" + suffix)
            features = self.forecast.load_forecast_prior_features(str(directory), 0)
            if features is None:
                raise RuntimeError("Missing native causal forecast cache: " + str(directory))
            files = [p for p in directory.glob("precomputed_forecasts_*.csv") if "metadata" not in p.name]
            if len(files) != 1:
                raise RuntimeError("Forecast source is not uniquely determined")
            self.forecast_cache[zone] = (features, files[0])
        return self.forecast_cache[zone]

    def assert_sources_unchanged(self):
        assert self.source_sha256 == {name: file_sha(ENGINE / name) for name in self.source_sha256}


@dataclass(frozen=True)
class EpisodeSpec:
    checkpoint_dir: str
    zone: str = "original"
    start_row: int = 0
    horizon: int = 96
    native_seed: int = 7
    profile: str = "strict"
    cash_fraction: float = 1.0
    burn_in: int = 0


class NativeScenario:
    """Prepare a pre-draw immutable context/config; never selects audit outcomes."""
    def __init__(self, runtime: NativeRuntime, spec: EpisodeSpec):
        self.runtime, self.spec = runtime, spec
        np, pd = runtime.np, runtime.pd
        self.checkpoint_dir = Path(spec.checkpoint_dir).resolve()
        if not self.checkpoint_dir.is_relative_to(ENGINE):
            raise ValueError("Expected authentic local Prototype5 checkpoint")
        frame, source = runtime.market(spec.zone)
        total_horizon = spec.burn_in + spec.horizon
        if spec.horizon < 13 or spec.burn_in < 0 or spec.start_row < 0 or spec.start_row + total_horizon > len(frame):
            raise ValueError("Invalid complete native episode window")
        if spec.profile not in ("strict", "vulnerable") or not 0 < spec.cash_fraction <= 1:
            raise ValueError("Unknown solvency profile or initial cash fraction")
        self.data = frame.iloc[spec.start_row:spec.start_row + total_horizon].copy().reset_index(drop=True)
        self.models = runtime.load_models(self.checkpoint_dir)
        self.config_path = self.checkpoint_dir / "training_config.json"
        frozen = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.training_contract = frozen["runtime_contract"]
        cfg = runtime.evaluation._hydrate_eval_config_from_training_config(
            runtime.config.EnhancedConfig(), str(self.checkpoint_dir))
        cfg.seed = int(spec.native_seed)
        if spec.profile == "vulnerable":
            cfg.enable_trading_sleeve_margin = False
            # Zero is the most aggressive exit. Infinity explicitly disables it.
            cfg.mtm_loss_exit_threshold_pct = float("inf")
        cfg.enable_episode_csv_logs = False
        cfg.rolling_past_history_dir = str(ENGINE / "rolling_past_history_dataset_ffill")
        cfg.forecast_cache_dir = str(HERE / "runtime_cache/unused_forecasts")
        stem = "unseendata" + ("_v2" if spec.zone == "v2" else "")
        cfg.mtm_external_settlement_price_data_path = str(
            ENGINE / "evaluation_dataset_ffill" / (stem + "_settlement_real_v2.csv"))
        cfg.impact_volume_data_path = str(
            ENGINE / "evaluation_dataset_ffill" / (stem + "_liquidity_volume_real_v1.csv"))
        with local_working_directory():
            runtime.evaluation._prime_eval_rolling_past_from_history(cfg)
        # Advance the native historical price normalizer through the causal prefix.
        # No prefix cashflows or actor updates are run; this is a fresh episode.
        if spec.start_row:
            _, _, _, advanced = runtime.environment._online_normalize_with_state(
                frame.iloc[:spec.start_row]["price"].to_numpy(dtype=float),
                copy.deepcopy(cfg.rolling_past_price_state),
                prior_mean=float(frame["price"].iloc[0]),
                prior_std=float(cfg.rolling_past_price_std_floor), prior_count=1.0,
                std_floor=float(cfg.rolling_past_price_std_floor))
            cfg.rolling_past_price_state = advanced
        resource_sha = {str(source.relative_to(ENGINE)): file_sha(source),
                        "training_config": file_sha(self.config_path)}
        if bool(cfg.enable_forecast_utilization):
            features, forecast_source = runtime.forecasts(spec.zone)
            view = features[spec.start_row:spec.start_row + total_horizon]
            if len(view) != total_horizon:
                raise RuntimeError("Forecast/context row misalignment")
            cache_key = state_sha({"source": str(forecast_source), "start": spec.start_row,
                                   "H": total_horizon})[:20]
            cache_dir = HERE / "runtime_cache/forecast_views" / cache_key
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / "precomputed_forecasts_native_view.csv"
            if not cache_file.exists():
                pd.DataFrame(view, columns=runtime.forecast.FORECAST_PRIOR_ALL_COLS).to_csv(cache_file, index=False)
            cfg.forecast_cache_dir = str(cache_dir)
            resource_sha["native_forecast_source"] = file_sha(forecast_source)
            resource_sha["forecast_view"] = file_sha(cache_file)
        if str(cfg.mtm_return_model) != "horizon_settlement" or str(cfg.mtm_horizon_payoff_denominator_mode) != "mwh_volume":
            raise RuntimeError("Adapter is restricted to native fixed-horizon MWh contracts")
        if bool(cfg.allow_operational_reallocation):
            raise RuntimeError("Operational reallocation needs separate attribution")
        self.config_snapshot = copy.deepcopy(cfg)
        self.config_snapshot_sha256 = state_sha(cfg.__dict__)
        self.metadata = {"spec": spec.__dict__, "training_runtime_contract_hash": frozen["runtime_contract_hash"],
                         "training_contract": self.training_contract, "resource_sha256": resource_sha,
                         "config_snapshot_sha256": self.config_snapshot_sha256,
                         "source_sha256": runtime.source_sha256,
                         "estimand": "financial event contribution only; excludes battery/physical/margin-conversion transfers and native shaped rewards",
                         "episode_boundary": "native window ends at H; native refuses contracts maturing outside the window; no new contract queue carries into next episode",
                         "native_initial_state": "fresh native deployment/cash; causal normalizer advanced through observed prefix; all other buffers reset natively"}
        self.metadata["profile_intervention"] = {"profile": spec.profile,
            "enable_trading_sleeve_margin": bool(cfg.enable_trading_sleeve_margin),
            "mtm_loss_exit_threshold_pct": jsonable(float(cfg.mtm_loss_exit_threshold_pct)),
            "cash_fraction_after_burn_in": spec.cash_fraction,
            "preserve_original_sizing_base": True, "preserve_native_halt_flags": True}
        self.warm_env_snapshot = None
        self.warm_observations = None
        self.warm_snapshot_sha256 = None
        if spec.burn_in:
            self._prepare_native_warm_snapshot()

    def _native_actions(self, env, observations):
        actions = {}
        for agent in env.possible_agents:
            if agent in self.models:
                action, _ = self.models[agent].predict(observations[agent], deterministic=True)
            else:
                action = env.get_rule_based_agent_action(agent, observations[agent])
            action = self.runtime.evaluation._coerce_action_for_space(action, env.action_space(agent))
            if not env.action_space(agent).contains(action):
                raise AssertionError((agent, "native action outside trained support", action))
            actions[agent] = action
        return actions

    def _prepare_native_warm_snapshot(self):
        key = state_sha({"checkpoint": str(self.checkpoint_dir), "spec": self.spec.__dict__,
                         "config": self.config_snapshot_sha256})
        if key not in self.runtime.warm_cache:
            started, cpu = time.perf_counter(), time.process_time()
            with local_working_directory():
                env = self.runtime.environment.RenewableMultiAgentEnv(
                    self.data.copy(), investment_freq=int(self.config_snapshot.investment_freq),
                    config=copy.deepcopy(self.config_snapshot), log_dir=str(HERE / "runtime_cache/native_debug"))
                env.evaluation_mode = True
                observations, _ = env.reset(seed=int(self.spec.native_seed))
                for _ in range(self.spec.burn_in):
                    actions = self._native_actions(env, observations)
                    observations, _, _, _, _ = env.step(actions)
            digest = native_state_sha(env)
            self.runtime.warm_cache[key] = (env,
                {k:self.runtime.np.asarray(v).copy() for k,v in observations.items()}, digest,
                {"native_steps": self.spec.burn_in, "wall_seconds": time.perf_counter()-started,
                 "cpu_seconds": time.process_time()-cpu,
                 "financial_costs_paid_before_audit": float(getattr(env,"cumulative_transaction_costs",0.0)) +
                       float(getattr(env,"cumulative_collateral_funding_costs",0.0)),
                 "pending_prepaid_contracts": len(env._horizon_contracts),
                 "regime": "native original financial signs, profile-specific guards, original full cash; no audit hooks"})
        self.warm_env_snapshot, self.warm_observations, self.warm_snapshot_sha256, warm_metadata = self.runtime.warm_cache[key]
        self.metadata["burn_in"] = warm_metadata
        self.metadata["warm_snapshot_sha256"] = self.warm_snapshot_sha256

    def new_episode(self, sign_sequence):
        assert state_sha(self.config_snapshot.__dict__) == self.config_snapshot_sha256
        return NativeEpisode(self, sign_sequence)


class NativeEpisode:
    def __init__(self, scenario: NativeScenario, sign_sequence):
        self.scenario, self.runtime = scenario, scenario.runtime
        np = self.runtime.np
        self.signs = np.asarray(sign_sequence, dtype=np.int8).copy()
        if self.signs.shape != (scenario.spec.horizon,) or not np.all(np.isin(self.signs, [-1, 1])):
            raise ValueError("Supply one hidden fair-sign slot per engine step; validation may supply deterministic signs")
        self.signs.setflags(write=False)
        clone_start = time.perf_counter()
        if scenario.warm_env_snapshot is not None:
            before = native_state_sha(scenario.warm_env_snapshot)
            assert before == scenario.warm_snapshot_sha256
            self.env = clone_native_environment(scenario.warm_env_snapshot)
            self.clone_field_hash_equal = native_state_sha(self.env) == before
            assert self.clone_field_hash_equal
            assert native_state_sha(scenario.warm_env_snapshot) == before
            observations = {k:self.runtime.np.asarray(v).copy() for k,v in scenario.warm_observations.items()}
        else:
            with local_working_directory():
                self.env = self.runtime.environment.RenewableMultiAgentEnv(
                    scenario.data.copy(), investment_freq=int(scenario.config_snapshot.investment_freq),
                    config=copy.deepcopy(scenario.config_snapshot),
                    log_dir=str(HERE / "runtime_cache/native_debug"))
                self.env.evaluation_mode = True
                observations, _ = self.env.reset(seed=int(scenario.spec.native_seed))
            self.clone_field_hash_equal = True
        self.clone_seconds = time.perf_counter()-clone_start
        self.cash_before_initial_intervention = float(self.env.budget)
        if scenario.spec.cash_fraction != 1.0:
            self.env.budget = scenario.spec.cash_fraction * float(self.env._initial_trading_sleeve_dkk())
            self.env._fill_obs()
            observations = {k:self.runtime.np.asarray(v).copy() for k,v in self.env._obs_buf.items()}
        self.events, self.step_records, self.transfers = [], [], []
        self.current_step, self.current_native_step, self.action_mapped_step = -1, -1, -1
        self.current_observation = np.zeros(self.env.observation_space("investor_0").shape, dtype=float)
        self.current_action = 0.0
        self._install_hooks()
        self.observations = {key: np.asarray(value).copy() for key, value in observations.items()}
        self.initial_cash = float(self.env.budget)
        self.initial_nav = float(self.env._calculate_fund_nav())
        self.native_model_contract_checks = {}
        for agent, model in scenario.models.items():
            assert model.observation_space.shape == self.env.observation_space(agent).shape
            assert model.action_space == self.env.action_space(agent)
            self.native_model_contract_checks[agent] = {"observation_shape": list(model.observation_space.shape),
                                                       "action_shape": list(model.action_space.shape or ())}
        self.support = self._support_bounds()

    def _record_event(self, kind, before, after, gross, cost, epsilon=0, base_gross=0.0):
        np = self.runtime.np
        values = [before, after, gross, cost, base_gross]
        if not np.all(np.isfinite(values)) or cost < 0:
            raise RuntimeError("Nonfinite or negative-cost native financial event")
        injection = max(-float(before) - float(gross) + float(cost), 0.0)
        expected_after = max(float(before) + float(gross) - float(cost), 0.0)
        tolerance = 1e-7 + 2e-12 * max(abs(before), abs(after), abs(gross), cost)
        if abs(after - expected_after) > tolerance:
            raise AssertionError((kind, "native event cash reconciliation", after, expected_after))
        if abs(injection - (after - before - gross + cost)) > tolerance:
            raise AssertionError("Native injection attribution failed")
        if epsilon and abs(gross - epsilon * base_gross) > tolerance:
            raise AssertionError("Maturity is not the common fair-sign gross")
        self.events.append({"step": self.current_step, "native_step": self.current_native_step, "kind": kind,
            "cash_before": float(before), "gross": float(gross), "cost": float(cost),
            "injection": injection, "cash_after": float(after), "sign_epsilon": int(epsilon),
            "unsigned_base_gross": float(base_gross),
            "actor_observation": self.current_observation.copy(), "actor_action": self.current_action})

    def _install_hooks(self):
        env, audit = self.env, self
        execute = env._execute_investor_trades
        def mapped(_env, *args, **kwargs):
            result = execute(*args, **kwargs)
            audit.action_mapped_step = audit.current_native_step
            return result
        env._execute_investor_trades = MethodType(mapped, env)
        for method_name, kind in (("_charge_horizon_roll_cost", "roll_cost"),
                                  ("_apply_market_access_fee", "access_cost"),
                                  ("_apply_collateral_cash_drag", "collateral_cost")):
            original = getattr(env, method_name)
            def cost_event(_env, *args, _original=original, _kind=kind, **kwargs):
                before = float(_env.budget)
                cost = float(_original(*args, **kwargs))
                audit._record_event(_kind, before, float(_env.budget), 0.0, cost)
                return cost
            setattr(env, method_name, MethodType(cost_event, env))
        settle = env._settle_horizon_contracts_for_step
        def maturity(_env, step):
            mature = [contract for contract in _env._horizon_contracts
                      if int(contract.get("settle_step", 10**18)) <= int(step)]
            if not mature:
                return settle(step)
            if audit.action_mapped_step != int(step) or audit.current_native_step != int(step):
                raise AssertionError("Financial sign sampled outside post-action settlement")
            epsilon = int(audit.signs[audit.current_step])
            unit_payoff = _env._horizon_unit_payoff
            base_gross = 0.0
            for contract in mature:
                delivery = int(contract.get("delivery_step", contract["open_step"]))
                original_unit = float(unit_payoff(float(contract["entry_price"]),
                                                 float(_env._horizon_settlement_price(delivery))))
                base_gross += sum(float(v) for v in contract["volumes_mwh"].values()) * original_unit
            before = float(_env.budget)
            def signed_unit(_inner, entry_price, settlement_price):
                return epsilon * float(unit_payoff(entry_price, settlement_price))
            _env._horizon_unit_payoff = MethodType(signed_unit, _env)
            try:
                gross, per_asset = settle(step)
            finally:
                _env._horizon_unit_payoff = unit_payoff
            audit._record_event("maturity", before, float(_env.budget), float(gross), 0.0,
                                epsilon=epsilon, base_gross=base_gross)
            return gross, per_asset
        env._settle_horizon_contracts_for_step = MethodType(maturity, env)
        # These native flows affect cash/state but are outside the financial estimand.
        for name, kind in (("_liquidate_trading_sleeve_for_margin", "margin_conversion"),
                           ("_distribute_excess_cash", "distribution"),
                           ("_execute_battery_ops", "battery")):
            original = getattr(env, name)
            def transfer(_env, *args, _original=original, _kind=kind, **kwargs):
                before = float(_env.budget)
                result = _original(*args, **kwargs)
                after = float(_env.budget)
                audit.transfers.append({"step": audit.current_step, "native_step": audit.current_native_step, "kind": _kind,
                                        "cash_change_inside_method": after-before,
                                        "returned_cashflow": float(result) if isinstance(result, (float, int)) else 0.0})
                return result
            setattr(env, name, MethodType(transfer, env))

    def _support_bounds(self):
        """Deterministic declared-context support, never an observed rollout max."""
        env, np, cfg = self.env, self.runtime.np, self.env.config
        if not 0 < float(cfg.liquidity_participation_cap_fraction) <= 1:
            raise RuntimeError("A positive native hard volume cap is required")
        if str(cfg.liquidity_volume_source) != "impact_volume":
            raise RuntimeError("Bound implemented for the authentic activated-volume contract")
        horizon, frequency = int(cfg.mtm_settlement_horizon_steps), int(env.investment_freq)
        conversion = float(env._mwh_volume_conversion_price(0.0))
        if env._get_investor_sizing_base_mode() != "initial_trading_sleeve":
            raise RuntimeError("A fixed initial sleeve sizing guard is required")
        # Allocation is clipped to [0,1] in the native tradeable-capital helper;
        # use one rather than either the class .75 or configured .8 meta cap.
        # The small deterministic cushion covers float32 risk-weight summation.
        fixed_sizing_volume_guard = (float(env._get_investor_sizing_base_dkk()) *
            float(cfg.max_position_size) * max(1.0, float(cfg.risk_exposure_cap_max)) /
            conversion * (1.0 + 2e-6))
        native_calendar_caps = np.asarray([
            float(cfg.liquidity_participation_cap_fraction) * min(
                env._market_liquidity_volume_mwh(t), env._causal_market_liquidity_volume_mwh(t))
            for t in range(self.scenario.spec.burn_in+self.scenario.spec.horizon)], dtype=float)
        if not np.all(np.isfinite(native_calendar_caps)):
            raise RuntimeError("Nonfinite declared-context capacity bound")
        # Native code retains scale=1 if cap_volume==0. Keep the fixed-sizing
        # guard on that branch rather than assuming a zero executed volume.
        guarded_caps = np.where(native_calendar_caps > 0,
                                np.minimum(native_calendar_caps, fixed_sizing_volume_guard),
                                fixed_sizing_volume_guard)
        volume_cap = float(np.max(guarded_caps))
        notional_cap = volume_cap * conversion
        min_ref = max(float(np.min(env._impact_ref_notional_by_step)), 1.0)
        base = self.runtime.environment.base_execution_fee_components(
            model=cfg.market_fee_model, abs_notional_dkk=notional_cap, abs_volume_mwh=volume_cap,
            friction_multiplier=float(cfg.friction_cost_multiplier),
            transaction_cost_bps=float(cfg.transaction_cost_bps),
            transaction_fixed_cost_dkk=float(cfg.transaction_fixed_cost),
            transaction_fee_dkk_per_mwh=float(cfg.transaction_fee_dkk_per_mwh))
        roll_bound = (float(base["base_execution_fee_dkk"]) + notional_cap*float(cfg.half_spread_bp)/10000 +
                      notional_cap*float(cfg.impact_coef_bp)*float(cfg.liquidity_tail_impact_max_multiplier)*
                      (notional_cap/min_ref)**float(cfg.impact_exponent)/10000)
        max_open = math.ceil(horizon/frequency) + 1
        collateral = max_open*volume_cap*max(conversion*float(cfg.collateral_notional_margin_fraction),
                                            float(cfg.collateral_stress_price_dkk_per_mwh)*float(cfg.collateral_stress_loss_fraction))
        funding_bound = collateral*float(cfg.collateral_funding_rate_annual)*float(cfg.time_step_hours)/(365.25*24)
        access = float(self.runtime.environment.market_access_fee_for_step(
            annual_fee_dkk=float(cfg.annual_market_access_fee_dkk),
            allocation_fraction=float(cfg.market_access_fee_allocation_fraction),
            time_step_hours=float(cfg.time_step_hours)))
        entry_bound = max(abs(float(np.min(env._price_raw))), abs(float(np.max(env._price_raw))), 1000.0)
        settle_bound = max(abs(float(cfg.mtm_external_settlement_min_price_dkk_per_mwh)),
                           abs(float(cfg.mtm_external_settlement_max_price_dkk_per_mwh)))
        gross_bound = max_open*volume_cap*(settle_bound+entry_bound)
        discount_sum = sum(.995**t for t in range(self.scenario.spec.horizon))
        return {"origin": "configured hard participation cap and complete declared context arrays, not confirmation sample extrema",
                "fixed_initial_sizing_volume_guard_mwh": fixed_sizing_volume_guard,
                "all_native_calendar_caps_strictly_positive": bool(np.all(native_calendar_caps > 0)),
                "max_single_contract_volume_mwh": volume_cap, "max_simultaneously_open_contracts": max_open,
                "roll_cost_bound": roll_bound, "access_cost_bound": access,
                "collateral_cost_bound": funding_bound,
                "per_step_cost_bound": roll_bound+access+funding_bound,
                "per_step_gross_bound": gross_bound,
                "audit_gamma": .995,
                "discounted_cost_bound": discount_sum*(roll_bound+access+funding_bound),
                "discounted_gross_bound": discount_sum*gross_bound,
                "undiscounted_episode_cost_bound": self.scenario.spec.horizon*(roll_bound+access+funding_bound)}

    def step(self):
        np, env = self.runtime.np, self.env
        if len(self.step_records) >= self.scenario.spec.horizon:
            raise RuntimeError("Episode complete; create a fresh environment rather than reset reuse")
        self.current_native_step = int(env.t)
        self.current_step = self.current_native_step-self.scenario.spec.burn_in
        assert self.current_step == len(self.step_records)
        self.current_observation = np.asarray(self.observations["investor_0"], dtype=float).copy()
        actions = {}
        for agent in env.possible_agents:
            obs = self.observations[agent]
            if agent in self.scenario.models:
                action, _ = self.scenario.models[agent].predict(obs, deterministic=True)
            else:
                action = env.get_rule_based_agent_action(agent, obs)
            action = self.runtime.evaluation._coerce_action_for_space(action, env.action_space(agent))
            if not env.action_space(agent).contains(action):
                # Gym discrete spaces accept scalar rather than length-one arrays.
                raise AssertionError((agent, "native action outside trained support", action))
            actions[agent] = action
        self.current_action = float(np.asarray(actions["investor_0"]).reshape(-1)[0])
        event_start = len(self.events)
        before_cash = float(env.budget)
        started = time.perf_counter()
        with local_working_directory():
            observations, rewards, done, truncated, info = env.step(actions)
        duration = time.perf_counter()-started
        self.observations = {key: np.asarray(value).copy() for key, value in observations.items()}
        new_events = self.events[event_start:]
        costs = sum(event["cost"] for event in new_events)
        gross = sum(event["gross"] for event in new_events)
        injection = sum(event["injection"] for event in new_events)
        tolerance = 1e-6 + 1e-10*self.support["per_step_cost_bound"]
        assert costs <= self.support["per_step_cost_bound"] + tolerance
        assert abs(gross) <= self.support["per_step_gross_bound"] + tolerance
        assert len(env._horizon_contracts) <= self.support["max_simultaneously_open_contracts"]
        self.step_records.append({"step": self.current_step, "native_step": self.current_native_step, "native_cash_before": before_cash,
            "native_cash_after": float(env.budget), "native_fund_nav": float(env._calculate_fund_nav()),
            "financial_gross": gross, "financial_cost": costs, "financial_injection": injection,
            "excluded_native_cash_change": float(env.budget)-before_cash-gross+costs-injection,
            "native_investor_shaped_reward": float(rewards["investor_0"]),
            "open_contracts": len(env._horizon_contracts), "step_seconds_excluding_actor_inference": duration,
            "actor_observation": self.current_observation.copy(), "actor_action": self.current_action})
        if len(self.step_records) == self.scenario.spec.horizon and env._horizon_contracts:
            raise AssertionError("Complete native episode left unsettled financial contracts")
        return new_events

    def trace_arrays(self):
        np = self.runtime.np
        output = {}
        for key in ("step", "native_step", "kind", "cash_before", "gross", "cost", "injection", "cash_after",
                    "sign_epsilon", "unsigned_base_gross", "actor_observation", "actor_action"):
            output["event_"+key] = np.asarray([row[key] for row in self.events])
        for key in ("step", "native_step", "native_cash_before", "native_cash_after", "native_fund_nav", "financial_gross",
                    "financial_cost", "financial_injection", "excluded_native_cash_change",
                    "native_investor_shaped_reward", "open_contracts", "step_seconds_excluding_actor_inference",
                    "actor_observation", "actor_action"):
            output["path_"+key] = np.asarray([row[key] for row in self.step_records])
        for key in ("step", "native_step", "kind", "cash_change_inside_method", "returned_cashflow"):
            output["transfer_"+key] = np.asarray([row[key] for row in self.transfers])
        return output

    def summary(self):
        records = self.step_records
        return {"scope": self.scenario.metadata["estimand"], "steps": len(records),
            "event_count": len(self.events), "maturity_events": sum(row["kind"]=="maturity" for row in self.events),
            "financial_gross": sum(row["gross"] for row in self.events),
            "financial_cost": sum(row["cost"] for row in self.events),
            "financial_injection": sum(row["injection"] for row in self.events),
            "financial_booked_cashflow": sum(row["gross"]-row["cost"] for row in self.events),
            "native_shaped_reward_descriptive": sum(row["native_investor_shaped_reward"] for row in records),
            "native_fund_nav_change_descriptive": (records[-1]["native_fund_nav"]-self.initial_nav) if records else 0,
            "excluded_native_cash_change": sum(row["excluded_native_cash_change"] for row in records),
            "excluded_transfer_call_counts": {kind: sum(row["kind"]==kind for row in self.transfers)
                for kind in ("battery", "margin_conversion", "distribution")},
            "terminal_open_contracts": len(self.env._horizon_contracts), "support": self.support,
            "clone_seconds": self.clone_seconds, "clone_all_field_hash_equal_before_intervention": self.clone_field_hash_equal,
            "initial_cash_intervention_from": self.cash_before_initial_intervention,
            "initial_cash_intervention_to": self.initial_cash,
            "model_contract_checks": self.native_model_contract_checks,
            "immutable_warm_snapshot_unchanged": (self.scenario.warm_env_snapshot is None or
                native_state_sha(self.scenario.warm_env_snapshot)==self.scenario.warm_snapshot_sha256),
            "immutable_config_snapshot_unchanged": state_sha(self.scenario.config_snapshot.__dict__)==self.scenario.config_snapshot_sha256}


def prepare_scenario(runtime, checkpoint_dir, *, zone="original", start_row=0,
                     profile="strict", cash_fraction=.1, burn_in=384, H=96, native_seed=7):
    """Public pre-draw scenario API for the frozen runner."""
    return NativeScenario(runtime, EpisodeSpec(str(checkpoint_dir), zone=zone,
        start_row=start_row, horizon=H, native_seed=native_seed, profile=profile,
        cash_fraction=cash_fraction, burn_in=burn_in))


def fixed_context_grid(runtime, *, burn_in=384, H=96):
    """Eight predeclared chronological windows, no outcomes or random draws."""
    contexts = []
    for zone in ("original", "v2"):
        frame, source = runtime.market(zone)
        last_start = len(frame)-int(burn_in)-int(H)
        if last_start < 0:
            raise ValueError("Native context too short")
        for quantile_id, start in enumerate(runtime.np.linspace(0,last_start,4,dtype=int)):
            contexts.append({"context_id": len(contexts), "zone": zone,
                "quantile_id": quantile_id, "start_row": int(start),
                "burn_in": int(burn_in), "horizon": int(H), "source_rows": len(frame),
                "source_from_engine": source.relative_to(ENGINE).as_posix()})
    return contexts
