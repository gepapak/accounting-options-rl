r"""The reward diagram of the running engine: adaptive reading of every settlement step, and full diagrams on a sample, from real clones of the environment.

Every query is the full, unchanged step on a deep copy of the environment with the market input moved along one line, x(t) = reference + t * (input - reference):
  horizon settlement: for the deliveries due now, settlement price = entry price + t * (settlement price - entry price);
  percent payoff:     spot price now = previous price + t * (spot price - previous price).
t = 1 is the real step, t = -1 the sign twin, t = 0 no move, large |t| the far tails.

Reading, on every step with i % 6 == 0 (in the volume payoff every non-zero gross lies on these steps):
  stage 1  t = +1, -1. If the bracket's width r + r' + 2C vanishes, stop: nothing is forgiven on either branch.
  stage 2  t = 0, +-2, +-4, +-8.
  stage 3  t = +-16, +-64, +-256, +-1024, only if stage 2 found no stable slope on either side although the pair differs.
On a sample of those steps also the full grid of 23 points, the zero-cost clones (own costs set to zero through the engine's configuration, at t = 1, -1, 0) and the
funded clones (cash raised by FUND DKK, at t = 1, -1).

The estimator is computed afterwards by analyze_scan.py, from these rewards and the declared own cost. Nothing in the engine's source changes; random-number
states are saved and restored; the clone's reward logger is a null object; the run's own trace must still equal the audit's trace, step by step.
Usage: from the engine folder, TWIN_TRACE_OUT=<trace.npz> SCAN_OUT=<scan.npz> python scan_launcher.py <evaluation.py arguments>
"""
import atexit
import copy
import gc
import os
import random
import runpy
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = HERE.parent / "engine_twin_2026-11-05"
ENG = Path.cwd()
assert (ENG / "evaluation.py").is_file(), "run from the engine folder"
sys.path.insert(0, str(WORK))
import twin_trace  # noqa: E402

sys.path[0] = str(ENG)
sys.path.append(str(WORK))
sys.path.append(str(HERE))   # the reader frozen with this protocol (a copy of twin_reward/reward_diagram.py)
import environment  # noqa: E402
from reward_diagram import read_diagram  # noqa: E402

COST_SETTINGS = dict(friction_cost_multiplier=0.0, half_spread_bp=0.0, impact_coef_bp=0.0, annual_market_access_fee_dkk=0.0,
                     collateral_funding_rate_annual=0.0, transaction_cost_bps=0.0, transaction_fixed_cost=0.0, transaction_fee_dkk_per_mwh=0.0)
FUND = 1e13
STAGE2 = (0.0, 2.0, -2.0, 4.0, -4.0, 8.0, -8.0)
STAGE3 = (16.0, -16.0, 64.0, -64.0, 256.0, -256.0, 1024.0, -1024.0)
READ_GRID = (1.0, -1.0) + STAGE2 + STAGE3
POS = (1.0, 2.0, 4.0, 8.0, 16.0, 64.0, 256.0, 1024.0, 2.0**14, 2.0**18, 2.0**22)
GRID = (0.0,) + POS + tuple(-p for p in POS)
FIELDS = ("r", "gross", "access", "roll", "transaction", "funding", "battery", "floors", "position", "cash_post")
SAMPLE_P = float(os.environ.get("SCAN_P", "0.03"))
SAMPLE_P_EMPTY = float(os.environ.get("SCAN_P_EMPTY", "0.10"))
RNG = np.random.default_rng(int(os.environ.get("SCAN_SEED", "20261106")))


class Null:
    csv_writer = object()

    def __getattr__(self, name):
        return lambda *a, **k: None

    def __deepcopy__(self, memo):
        return self


def clone_env(env):
    new = object.__new__(type(env))
    memo = {id(env): new}
    for k, v in env.__dict__.items():
        if k == "debug_tracker":
            new.__dict__[k] = Null()
            continue
        if k in ("_tw_rows", "_tw_writes"):
            continue
        try:
            new.__dict__[k] = copy.deepcopy(v, memo)
        except Exception:  # noqa: BLE001
            new.__dict__[k] = v
    return new


def sleeve(env):
    return max(0.0, float(env.budget)) + twin_trace._mtm_sum(env) + float(getattr(env, "total_distributions", 0.0) or 0.0)


def set_line(c, i, t):
    if c._is_horizon_settlement_mtm():
        prices = c._horizon_settlement_price_raw
        for con in (getattr(c, "_horizon_contracts", []) or []):
            if int(con.get("settle_step", 10**18)) <= i:
                d = int(con.get("delivery_step", con.get("open_step", i)))
                entry = float(con["entry_price"])
                prices[d] = entry + t * (float(prices[d]) - entry)
    else:
        p = c._price_raw
        prev = float(np.clip(p[i - 1], -1000.0, 1e9))
        p[i] = prev + t * (float(np.clip(p[i], -1000.0, 1e9)) - prev)


def query(env, actions, i, t, pre, zero_cost=False, funded=False):
    c = clone_env(env)
    if zero_cost:
        for k, v in COST_SETTINGS.items():
            if hasattr(c.config, k):
                setattr(c.config, k, v)
    extra = 0.0
    if funded:
        c.budget = float(env.budget) + FUND
        extra = FUND
    set_line(c, i, t)
    c.__dict__["_tw_writes"] = []
    c.__dict__["_tw_gross"] = 0.0
    c.__dict__["_tw_per_asset"] = {}
    orig_step(c, copy.deepcopy(actions))
    writes = c.__dict__.pop("_tw_writes", [])
    am = dict(access=0.0, roll=0.0, transaction=0.0, funding=0.0, battery=0.0)
    for w in writes:
        if w["kind"] in am:
            am[w["kind"]] += w["amount"]
    _, _, floors, _ = twin_trace._replay(writes, c.__dict__.get("_tw_per_asset", {}), float(env.budget) + extra, 1.0)
    return (sleeve(c) - pre - extra, float(c.__dict__.get("_tw_gross", 0.0)), am["access"], am["roll"], am["transaction"], am["funding"], am["battery"], float(floors), twin_trace._position_sum(c), float(c.budget))


cls = environment.RenewableMultiAgentEnv
orig_step = cls.step
twin_trace.install(cls, os.environ["TWIN_TRACE_OUT"])
traced_step = cls.step
READS, DIAGRAMS = [], []
IDX = {t: j for j, t in enumerate(READ_GRID)}


def read_step(self, actions, i, pre):
    tol = 1e-3 + 1e-9 * abs(float(self.budget) + twin_trace._mtm_sum(self))
    phi = np.full(len(READ_GRID), np.nan)
    q1 = query(self, actions, i, 1.0, pre)
    qm = query(self, actions, i, -1.0, pre)
    phi[IDX[1.0]], phi[IDX[-1.0]] = q1[0], qm[0]
    own_1 = q1[2] + q1[3] + q1[4] + q1[5] - q1[6]
    own_m = qm[2] + qm[3] + qm[4] + qm[5] - qm[6]
    stage, n = 1, 2
    if abs(q1[0] + qm[0] + 2.0 * own_1) > tol:
        stage = 2
        for t in STAGE2:
            phi[IDX[t]] = query(self, actions, i, t, pre)[0]
            n += 1
        have = sorted(t for t in READ_GRID if not np.isnan(phi[IDX[t]]))
        d = read_diagram(np.array(have), np.array([phi[IDX[t]] for t in have]), rel_tol=1e-9, abs_tol=1e-3)
        if d.s_plus == 0.0 and d.s_minus == 0.0 and abs(q1[0] - qm[0]) > tol:
            stage = 3
            for t in STAGE3:
                phi[IDX[t]] = query(self, actions, i, t, pre)[0]
                n += 1
    return dict(t=i, stage=stage, n=n, phi=phi, own_1=own_1, own_m=own_m, gross_1=q1[1], gross_m=qm[1], pos_1=q1[8], pos_m=qm[8], cash_pre=float(self.budget), mtm_pre=twin_trace._mtm_sum(self))


STOP_AT = int(os.environ.get("SCAN_STOP_AT", "0"))   # development only: stop the run at this step (never set in the campaign)
GC_EVERY = int(os.environ.get("SCAN_GC_EVERY", "25"))  # clones form reference cycles; collect them regularly or the run outgrows the machine
DEBUG_MEM = bool(os.environ.get("SCAN_DEBUG_MEM"))
COUNT = [0]


def outer(self, actions):
    i = int(getattr(self, "t", 0))
    rec, dia = None, None
    if STOP_AT and i >= STOP_AT:
        sys.exit(0)
    if 6 <= i < int(getattr(self, "max_steps", 10**12)) - 1 and i % 6 == 0:
        py, npy = random.getstate(), np.random.get_state()
        try:
            import torch
            th = torch.get_rng_state()
        except Exception:  # noqa: BLE001
            torch, th = None, None
        t0 = time.perf_counter()
        pre = sleeve(self)
        rec = read_step(self, actions, i, pre)
        rec["wall"] = time.perf_counter() - t0
        empty = float(self.budget) <= 1e-6
        if RNG.random() < (SAMPLE_P_EMPTY if empty else SAMPLE_P):
            t0 = time.perf_counter()
            dia = dict(t=i, cash_pre=float(self.budget), mtm_pre=twin_trace._mtm_sum(self))
            dia["line"] = [query(self, actions, i, t, pre) for t in GRID]
            dia["zero"] = [query(self, actions, i, t, pre, zero_cost=True) for t in (1.0, -1.0, 0.0)]
            dia["funded"] = [query(self, actions, i, t, pre, funded=True) for t in (1.0, -1.0)]
            dia["wall"] = time.perf_counter() - t0
        random.setstate(py)
        np.random.set_state(npy)
        if torch is not None:
            torch.set_rng_state(th)
        COUNT[0] += 1
        if GC_EVERY and COUNT[0] % GC_EVERY == 0:
            gc.collect()
            if DEBUG_MEM:
                import psutil
                print(f"SCANMEM i={i} rss={psutil.Process().memory_info().rss / 1e6:.0f}MB objs={len(gc.get_objects())}", flush=True)
    out = traced_step(self, actions)
    if rec is not None:
        row = tuple(self.__dict__["_tw_rows"][-1])
        rec["real"] = row
        READS.append(rec)
        if dia is not None:
            dia["real"] = row
            DIAGRAMS.append(dia)
    return out


cls.step = outer


def save():
    if not READS:
        return
    out = os.environ["SCAN_OUT"]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    arrays = dict(
        t=np.array([r["t"] for r in READS]), stage=np.array([r["stage"] for r in READS]), n=np.array([r["n"] for r in READS]), wall=np.array([r["wall"] for r in READS]),
        read_grid=np.array(READ_GRID), phi=np.array([r["phi"] for r in READS]), own_1=np.array([r["own_1"] for r in READS]), own_m=np.array([r["own_m"] for r in READS]),
        gross_1=np.array([r["gross_1"] for r in READS]), gross_m=np.array([r["gross_m"] for r in READS]), pos_1=np.array([r["pos_1"] for r in READS]), pos_m=np.array([r["pos_m"] for r in READS]),
        cash_pre=np.array([r["cash_pre"] for r in READS]), mtm_pre=np.array([r["mtm_pre"] for r in READS]), real=np.array([r["real"] for r in READS]), real_columns=np.array(twin_trace.COLUMNS),
        fields=np.array(FIELDS), grid=np.array(GRID))
    if DIAGRAMS:
        arrays.update(d_t=np.array([d["t"] for d in DIAGRAMS]), d_cash_pre=np.array([d["cash_pre"] for d in DIAGRAMS]), d_mtm_pre=np.array([d["mtm_pre"] for d in DIAGRAMS]), d_wall=np.array([d["wall"] for d in DIAGRAMS]),
                      d_line=np.array([d["line"] for d in DIAGRAMS]), d_zero=np.array([d["zero"] for d in DIAGRAMS]), d_funded=np.array([d["funded"] for d in DIAGRAMS]), d_real=np.array([d["real"] for d in DIAGRAMS]))
    np.savez_compressed(out, **arrays)


atexit.register(save)
sys.argv = ["evaluation.py"] + sys.argv[1:]
runpy.run_path(str(ENG / "evaluation.py"), run_name="__main__")
