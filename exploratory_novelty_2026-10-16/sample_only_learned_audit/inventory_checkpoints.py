"""Read-only policy/environment reconnaissance; no inference, draws, or training.

Writes only this new directory's inventory JSON. Zip metadata are decoded as JSON;
neither serialized Python objects nor normalization pickles are executed.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import zipfile

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
PAPER1 = CODE.parents[1]
ENGINE = PAPER1 / "Prototype5"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict:
    value = {"path_from_paper1": path.relative_to(PAPER1).as_posix(),
             "exists": path.is_file()}
    if path.is_file():
        value.update(bytes=path.stat().st_size, sha256=sha(path))
    return value


def policy(path: Path, normalizer: Path | None = None) -> dict:
    value = artifact(path)
    if path.is_file():
        with zipfile.ZipFile(path) as archive:
            metadata = json.loads(archive.read("data"))
            version = archive.read("_stable_baselines3_version").decode().strip()
        obs, action = metadata.get("observation_space", {}), metadata.get("action_space", {})
        klass = metadata.get("policy_class", {})
        value["checkpoint_metadata"] = {
            "observation_shape": obs.get("_shape"),
            "observation_low": obs.get("low_repr", obs.get("low")),
            "observation_high": obs.get("high_repr", obs.get("high")),
            "action_shape": action.get("_shape"),
            "action_n": action.get("n"),
            "action_low": action.get("low_repr", action.get("low")),
            "action_high": action.get("high_repr", action.get("high")),
            "policy_module": klass.get("__module__"),
            "policy_description": (klass.get("__doc__") or "").split("\n")[0],
            "num_timesteps": metadata.get("num_timesteps"),
            "gamma": metadata.get("gamma"), "stable_baselines3_version": version,
        }
    if normalizer is not None:
        value["normalization_artifact"] = artifact(normalizer)
    return value


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    study_l = CODE / "revision_experiments/2026-10-10/continuation_experiment/jobs"
    l_policies = [policy(p, p.with_name("normalization.pkl"))
                  for p in sorted(study_l.glob("*/policy.zip"))]
    study_r = CODE / "ACCOUNTING_OPTIONS_2026-10-14/study_r/results_r/models"
    r_policies = [policy(p, p.with_name(p.stem + "__normalization.pkl"))
                  for p in sorted(study_r.glob("*.zip"))]
    families = {}
    for name, base, pattern in [
        ("plain_mappo", ENGINE / "batch_tier_phase_runs/prototype5_mappo_marl_final_v1",
         "seed*/tier1_seed*/final_models/investor_0_policy.zip"),
        ("feasible_action", ENGINE / "Ablations/batch_tier_phase_runs/prototype5_mechanism_ablations_final_v1/feasible_action_full",
         "seed*/tier1_forecast_utilization_seed*/final_models/investor_0_policy.zip"),
        ("forecast_anchor", ENGINE / "Ablations/batch_tier_phase_runs/prototype5_mechanism_ablations_final_v1/focal_anchor",
         "seed*/tier1_forecast_utilization_seed*/final_models/investor_0_policy.zip"),
    ]:
        rows = []
        for path in sorted(base.glob(pattern)):
            record = policy(path)
            record["battery_policy"] = artifact(path.with_name("battery_operator_0_policy.zip"))
            config_path = path.with_name("training_config.json")
            record["training_config"] = artifact(config_path)
            if config_path.is_file():
                config = json.loads(config_path.read_text(encoding="utf-8"))
                record["frozen_contract"] = {
                    "seed": config.get("seed"), "flags": config.get("flags"),
                    "runtime_contract_hash": config.get("runtime_contract_hash"),
                    "runtime_contract": config.get("runtime_contract"),
                }
            rows.append(record)
        families[name] = rows
    key_paths = [
        "environment.py", "observation_builder.py", "policy.py", "evaluation.py", "marl.py",
        "evaluation_dataset_ffill/unseendata.csv",
        "evaluation_dataset_ffill/unseendata_v2.csv",
        "evaluation_dataset_ffill/unseendata_settlement_real_v2.csv",
        "evaluation_dataset_ffill/unseendata_v2_settlement_real_v2.csv",
        "evaluation_dataset_ffill/unseendata_liquidity_volume_real_v1.csv",
        "evaluation_dataset_ffill/unseendata_v2_liquidity_volume_real_v1.csv",
        "rolling_past_history_dataset_ffill/history_000.csv",
        "forecast_cache_settlement_hourly_v2/hourly_forecast_cache_v2_manifest.json",
    ]
    output = {
        "status": "RECONNAISSANCE ONLY: no policy inference, environment episode, random draw, or training",
        "created_date": "2026-10-16",
        "python_version": platform.python_version(),
        "installed_package_versions": {name: package_version(name) for name in
            ("numpy", "pandas", "torch", "stable-baselines3", "gymnasium", "pettingzoo", "tensorflow", "scipy", "scikit-learn")},
        "study_l": {"count": len(l_policies), "expected": 20, "policies": l_policies,
                    "observation_contract_from_source": "1D clip(reported_equity/20000,0,5), norm_obs=False, reward normalization only, Discrete9 signed leverage"},
        "study_r": {"count": len(r_policies), "expected": 60, "policies": r_policies,
                    "observation_contract_from_source": "1D clip(reported_equity/20000,0,5), norm_obs=False, reward normalization only, Discrete9 signed leverage"},
        "prototype5": {"root_from_paper1": "Prototype5", "families": families,
            "sources_and_data": [artifact(ENGINE / path) for path in key_paths],
            "scope": "Actual locally available engine/checkpoints, not shipped in code repository. Plain investor is 12D, feasible arm20D, scalar continuous exposure; one-factor sleeve, not independent multiasset control."},
        "proposal_status": "Await root agreement on audited cashflow/account definition before adapter implementation. No endpoint experiment authorized by this inventory.",
    }
    assert len(l_policies) == 20 and len(r_policies) == 60
    assert all(row["normalization_artifact"]["exists"] for row in l_policies + r_policies)
    destination = HERE / "checkpoint_inventory.json"
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"L": len(l_policies), "R": len(r_policies),
                      "engine_families": {name: len(rows) for name, rows in families.items()},
                      "output": str(destination)}))


if __name__ == "__main__":
    main()
