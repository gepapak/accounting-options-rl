r"""What did each audit protocol really run with? Starts each protocol's evaluation exactly as the campaign does, records the
engine's configuration at its first step, after the checkpoint's saved configuration has been restored, and stops.
No outcome is computed. Writes config_dump/<variant>.json and config_dump/DIFFERENCES.md (every setting that differs from L0).

Usage: python dump_runtime_config.py [--workers 2]            (parent)
       TWIN_CONFIG_OUT=<file> python dump_runtime_config.py --child <evaluation.py arguments>   (internal, cwd = engine)
"""
from __future__ import annotations

import json
import os
import runpy
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "config_dump"
VARIANTS = ["L0_strict", "L1_price_taker", "L2_no_solvency", "L3_price_taker_no_solvency", "L4_load_proxy_liquidity", "L5_percent_payoff",
            "L6_interpolated_prices", "L7_thesis_like", "L7_minus_interp", "L7_minus_percent_mtm", "L7_minus_price_taker", "L7_minus_no_solvency",
            "L7_minus_legacy_fee", "L7_minus_sweeper", "L8_pctmtm_nosolv", "L7_thesis_like_capon"]


def child() -> int:
    eng = Path.cwd()
    sys.path[0] = str(eng)
    import environment
    cls = environment.RenewableMultiAgentEnv
    orig = cls.step

    def step(self, actions):
        scalars = (bool, int, float, str, type(None))
        cfg = {k: v for k, v in sorted(vars(self.config).items()) if isinstance(v, scalars)}
        rec = dict(config=cfg, is_horizon_settlement=bool(self._is_horizon_settlement_mtm()), rows=int(len(self._price_raw)),
                   first_prices=[float(x) for x in self._price_raw[:8]], budget=float(self.budget))
        Path(os.environ["TWIN_CONFIG_OUT"]).write_text(json.dumps(rec, indent=0, sort_keys=True), encoding="utf-8")
        os._exit(0)

    cls.step = step
    sys.argv = ["evaluation.py"] + sys.argv[2:]
    runpy.run_path(str(eng / "evaluation.py"), run_name="__main__")
    return 1


def run_variant(variant: str) -> tuple[str, int]:
    sys.path.insert(0, str(HERE))
    import run_scarcity_tail_campaign as camp
    target = OUT / f"{variant}.json"
    if target.is_file():
        return variant, 0
    seed, region = 7, "original"
    cmd = camp.base_eval_cmd("marl", seed, region, camp.model_dir_for("marl", seed))
    capon = variant.endswith("_capon")
    cmd, cfg = camp.apply_variant(cmd, variant.removesuffix("_capon"), region)
    if capon:
        cfg = dict(cfg, mtm_apply_price_return_cap=True)
    tmp = OUT / "tmp" / variant
    cmd = camp.replace_arg(cmd, "--output_dir", str(tmp))
    assert cmd[1] == "evaluation.py"
    cmd = [cmd[0], "-B", str(Path(__file__).resolve()), "--child"] + cmd[2:]
    env = dict(os.environ, **camp.THREAD_ENV, SIM_AUDIT_VARIANT=variant, TWIN_CONFIG_OUT=str(target))
    if variant != "L0_strict":
        env["SIM_AUDIT_ALLOW_PROTOCOL_MISMATCH"] = "1"
    if cfg:
        env["SIM_AUDIT_CFG_OVERRIDES"] = json.dumps(cfg)
    tmp.mkdir(parents=True, exist_ok=True)
    with (OUT / f"{variant}.log").open("w", encoding="utf-8", errors="replace") as f:
        rc = subprocess.run(cmd, cwd=str(camp.ENG), env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    shutil.rmtree(tmp, ignore_errors=True)
    return variant, (0 if target.is_file() else (rc or 1))


def differences() -> None:
    base = json.loads((OUT / "L0_strict.json").read_text(encoding="utf-8"))
    lines = ["# Runtime settings that differ from L0 (MAPPO seed 7, DK1), read from the engine at its first step", ""]
    for v in VARIANTS[1:]:
        p = OUT / f"{v}.json"
        if not p.is_file():
            lines += [f"## {v}", "", "NOT RECORDED", ""]
            continue
        cur = json.loads(p.read_text(encoding="utf-8"))
        keys = sorted(k for k in set(base["config"]) | set(cur["config"]) if base["config"].get(k) != cur["config"].get(k))
        lines += [f"## {v}", "", f"horizon settlement: {cur['is_horizon_settlement']}; prices differ from L0: {cur['first_prices'] != base['first_prices']}", ""]
        lines += [f"- `{k}`: {base['config'].get(k)!r} -> {cur['config'].get(k)!r}" for k in keys] or ["- (no setting differs)"]
        lines.append("")
    (OUT / "DIFFERENCES.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        return child()
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 2
    OUT.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(run_variant, VARIANTS))
    print(results)
    differences()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
