"""Independent post-run recount; never trains or invokes the registered aggregator.

--snapshot records post-launch integrity evidence without opening results.
The default command requires exactly 80 results AND VERDICTS_RS.md.
All new outputs go to independent_verification; registered files are read-only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.stats import mannwhitneyu

HERE = Path(__file__).resolve().parent
ARMS = ("PLAIN", "ORACLE", "SPR", "TIR", "RUIN5")
SEEDS = range(300, 316)
MARKER = b"## After hashing (not part of the registered prefix)"
EXPECTED = {f"RS_DK2__{arm}__s{seed}.json": (arm, seed) for arm in ARMS for seed in SEEDS}
IDS = ("RS1", "RS2", "RS3", "RS4", "RS5", "RS6", "RS-D1", "RS-D2")
KNOWN_LOG_DIGEST = "e0c5561e5f97be17a0f28fbec5a5189f4865c3145c223369a2291ba2e2d7aa2e"
KNOWN_ACTUAL_PREFIX = "c265e796477c5209b82e3c55d4f917d0abd6ac153ca7e14438856997fd44a72e"


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def log_prefix(data: bytes) -> bytes:
    lines = data.splitlines(keepends=True)
    positions = [i for i, line in enumerate(lines) if line.rstrip(b"\r\n") == MARKER]
    require(len(positions) == 1, "Expected one standalone run-log marker line")
    return b"".join(lines[:positions[0] + 1])


def registration(root: Path) -> dict:
    manifest = (root / "PREREGISTRATION_HASHES_RS.txt").read_bytes()
    records = []
    for line in manifest.decode("utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        recorded, name = line.split("  ", 1)
        data = (root / name).read_bytes()
        actual = sha(log_prefix(data) if name == "RUN_LOG_RS.md" else data)
        status = "MATCH"
        if name == "RUN_LOG_RS.md":
            require(recorded == KNOWN_LOG_DIGEST and actual == KNOWN_ACTUAL_PREFIX,
                    "Run-log prefix differs from the independently inspected post-launch bytes")
            first = data.index(MARKER)
            buggy_prefix = data[:data.index(b"\n", first) + 1]
            require(sha(buggy_prefix) == recorded, "Run-log recorded digest no longer matches")
            status = "KNOWN PREFIX-HASH DEFECT"
        else:
            require(actual == recorded, f"Registered file changed: {name}")
        records.append(dict(file=name, recorded_sha256=recorded, actual_sha256=actual, status=status))
    require(len(records) == 17 and len({r["file"] for r in records}) == 17, "Expected 17 distinct registered files")
    return {
        "manifest_sha256": sha(manifest),
        "records": records,
        "warning": (
            "The registered log hash covers only the first 147 bytes, ending at a quoted marker. "
            "The intended prefix is 3555 bytes. Its chronology was NOT fully covered by the launch-time hash. "
            "The other 16 registered files match. This snapshot is post-launch evidence, not retroactive preregistration."
        ),
    }


def ready(root: Path) -> list[Path]:
    names = {p.name for p in (root / "results_rs").glob("*.json")}
    require(names == set(EXPECTED),
            f"Results incomplete or unexpected: missing={sorted(set(EXPECTED) - names)}, extra={sorted(names - set(EXPECTED))}")
    require((root / "VERDICTS_RS.md").is_file(), "Registered verdicts do not exist; outcomes remain unopened")
    return [root / "results_rs" / name for name in sorted(names)]


def close(actual: float, expected: float, label: str) -> None:
    require(math.isfinite(actual) and math.isfinite(expected)
            and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-8),
            f"{label}: {actual!r} != {expected!r}")


def episode_values(value: object, label: str) -> list[float]:
    require(isinstance(value, list), f"{label}: not a list")
    if not isinstance(value, list):
        raise TypeError(label)
    require(len(value) == 100, f"{label}: expected 100 episodes")
    require(all(type(v) in (int, float) and math.isfinite(v) for v in value), f"{label}: non-finite or nonnumeric value")
    return [float(v) for v in value]


def read_run(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    arm, seed = EXPECTED[path.name]
    require((data["setting"], data["arm"], data["seed"]) == ("RS_DK2", arm, seed), f"Wrong job identity: {path.name}")
    require(data["steps"] == 1_000_000 and data["zero_starts"] is True, f"Wrong training configuration: {path.name}")
    probes = data["lev"]
    expected_keys = {f"{e}|{z}|{r}" for e in ("0", "0.02", "0.05", "0.1", "0.25", "0.5", "1", "2")
                     for z in ("-1", "+0", "+1") for r in ("0.05", "0.5")}
    require(set(probes) == expected_keys, f"Unexpected probe grid: {path.name}")
    require(all(v in (-16, -4, -1, -0.25, 0, 0.25, 1, 4, 16) for v in probes.values()), "Invalid probe leverage")
    values = [probes[f"{e}|{z}|{r}"] for e in ("0", "0.02", "0.05", "0.1")
              for z in ("-1", "+0", "+1") for r in ("0.05", "0.5")]
    row = dict(arm=arm, seed=seed, X=mean(abs(v) == 16 for v in values),
               X_long=mean(v == 16 for v in values), X_short=mean(v == -16 for v in values))
    for kind in ("edge", "placebo", "zero"):
        test = data[f"test_{kind}"]
        booked = episode_values(test["booked_episodes"], f"{path.name} {kind} booked")
        booked_mean = mean(booked)
        close(booked_mean, test["booked_pct_K0"], f"{kind} booked mean")
        close(float(np.std(booked, ddof=1) / 10), test["booked_se"], f"{kind} episode SE")
        close(test["reported_pct_K0"] - booked_mean, test["forgiven_pct_K0"], f"{kind} forgiveness identity")
        require(0 <= test["share_lev16"] <= 1 and test["forgiven_pct_K0"] >= -1e-8, "Invalid descriptive measure")
        row[f"E_{kind}"] = booked_mean
        row[f"share16_{kind}"] = test["share_lev16"]
        row[f"forgiven_{kind}"] = test["forgiven_pct_K0"]
        if kind == "zero":
            continue
        audit = data[f"audit_{kind}"]
        close(audit["booked"], booked_mean, f"{kind} audit replay")
        estimates = {}
        for name in ("reported", "spr", "tir", "lower"):
            estimates[name] = episode_values(audit[f"episodes_{name}"], f"{kind} {name}")
            estimate = mean(estimates[name])
            error = abs(estimate - booked_mean)
            close(estimate, audit[name], f"{kind} {name} mean")
            close(error, audit[f"err_{name}"], f"{kind} {name} error")
            close(mean(abs(a - b) for a, b in zip(estimates[name], booked)), audit[f"mae_{name}"], f"{kind} {name} MAE")
            row[f"err_{name}_{kind}"] = error
        close(mean(estimates["reported"]), test["reported_pct_K0"], f"{kind} reported replay")
        row[f"bias_spr_{kind}"] = mean(estimates["spr"]) - booked_mean
        row[f"bias_tir_{kind}"] = mean(estimates["tir"]) - booked_mean
        lower_count = sum(a <= b + 1e-6 for a, b in zip(estimates["lower"], booked))
        upper_count = sum(b <= a + 1e-6 for a, b in zip(estimates["reported"], booked))
        require(lower_count == audit["episodes_lower_le_booked"] == 100, "Episode lower bracket failure")
        require(upper_count == audit["episodes_booked_le_reported"] == 100, "Episode upper bracket failure")
        for key in ("steps", "steps_inside_bracket", "steps_identified", "identified_steps_tir_inexact"):
            require(type(audit[key]) is int and audit[key] >= 0, f"Invalid count: {key}")
            row[f"{key}_{kind}"] = audit[key]
        require(audit["steps"] == 67_200, "Wrong number of audited steps")
        require(audit["steps_inside_bracket"] <= audit["steps"], "Impossible bracket count")
        require(audit["identified_steps_tir_inexact"] <= audit["steps_identified"] <= audit["steps"], "Impossible identified count")
    return row


def decisions(rows: list[dict], repetitions: int = 10_000) -> dict:
    groups = {a: sorted((r for r in rows if r["arm"] == a), key=lambda r: r["seed"]) for a in ARMS}
    require(all([r["seed"] for r in g] == list(SEEDS) for g in groups.values()), "Each arm must have all 16 unique seeds")
    x = {a: np.array([r["X"] for r in g]) for a, g in groups.items()}
    e = {a: np.array([r["E_edge"] for r in g]) for a, g in groups.items()}
    rng = np.random.default_rng(20261104)
    oracle_lb = float(np.quantile([np.mean(rng.choice(e["ORACLE"], 16)) for _ in range(repetitions)], 0.05))
    positive_edge = e["ORACLE"].mean() > 0 and oracle_lb > 0
    pvalues = {tag: float(mannwhitneyu(x[arm], x["PLAIN"], alternative="less").pvalue)
               for tag, arm in (("RS1", "ORACLE"), ("RS2", "SPR"), ("RS4", "TIR"))}
    supports = {"RS1": pvalues["RS1"] < 0.05}
    ratios = {}
    for arm, remove_id, edge_id in (("SPR", "RS2", "RS3"), ("TIR", "RS4", "RS5")):
        supports[remove_id] = bool(pvalues[remove_id] < 0.05 and x[arm].mean() <= x["ORACLE"].mean() + 0.10)
        supports[edge_id] = None
        if positive_edge:
            boot = []
            nonpositive_denominators = 0
            for _ in range(repetitions):
                numerator = float(rng.choice(e[arm], 16).mean())
                denominator = float(rng.choice(e["ORACLE"], 16).mean())
                require(denominator != 0, "Undefined registered bootstrap ratio: denominator is zero")
                nonpositive_denominators += denominator < 0
                boot.append(numerator / denominator)
            ratio = float(e[arm].mean() / e["ORACLE"].mean())
            lb = float(np.quantile(boot, 0.05))
            ratios[arm] = dict(ratio=ratio, lower_bound=lb, nonpositive_bootstrap_denominators=nonpositive_denominators)
            supports[edge_id] = ratio >= 0.80 and lb >= 0.60
    supports["RS6"] = bool(x["RUIN5"].mean() <= x["ORACLE"].mean() + 0.10
                           and e["RUIN5"].mean() >= 0.80 * e["ORACLE"].mean()) if positive_edge else None
    if not supports["RS1"]:
        for tag in ("RS2", "RS3", "RS4", "RS5", "RS6"):
            supports[tag] = None
    audited = [r for r in rows if r["arm"] in ("ORACLE", "SPR", "TIR")]
    max_error = max(r[f"err_tir_{k}"] for r in audited for k in ("edge", "placebo"))
    counts = {key: sum(r[f"{key}_{k}"] for r in rows for k in ("edge", "placebo"))
              for key in ("steps", "steps_inside_bracket", "steps_identified", "identified_steps_tir_inexact")}
    supports["RS-D1"] = max_error <= 1.0
    supports["RS-D2"] = counts["steps"] == counts["steps_inside_bracket"] and counts["identified_steps_tir_inexact"] == 0
    adjusted, running = {}, 0.0
    for i, tag in enumerate(sorted(pvalues, key=lambda key: pvalues[key])):
        running = max(running, min(1.0, (3 - i) * pvalues[tag]))
        adjusted[tag] = running
    metrics = ("X", "X_long", "X_short", "E_edge", "E_placebo", "E_zero", "share16_edge", "forgiven_placebo")
    return {
        "verdicts": {tag: "NOT TESTABLE" if supports[tag] is None else "SUPPORTED" if supports[tag] else "NOT SUPPORTED" for tag in IDS},
        "means": {a: {key: mean(r[key] for r in g) for key in metrics} for a, g in groups.items()},
        "oracle_edge_lower_bound": oracle_lb, "edge_ratios": ratios, "pvalues": pvalues, "holm": adjusted,
        "D1_max_error": max_error, "audit_counts": counts,
    }


def parse_verdicts(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        fields = [part.strip() for part in line.split("|")]
        if len(fields) == 5 and fields[1] in IDS:
            require(fields[1] not in out, f"Duplicate verdict: {fields[1]}")
            require(fields[3] in ("SUPPORTED", "NOT SUPPORTED", "NOT TESTABLE"), "Unknown verdict")
            out[fields[1]] = fields[3]
    require(set(out) == set(IDS), "Missing registered verdicts")
    return out


def write_new(path: Path, data: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", action="store_true")
    args = parser.parse_args()
    output = HERE / "independent_verification"
    integrity = registration(HERE)
    print("WARNING:", integrity["warning"], flush=True)
    if args.snapshot:
        output.mkdir(exist_ok=True)
        write_new(output / "post_launch_integrity.json", {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "completed_result_files": len(list((HERE / "results_rs").glob("*.json"))),
            "registered_verdicts_present": (HERE / "VERDICTS_RS.md").is_file(),
            "purpose": "Integrity checkpoint, not preregistration or a blinded analysis plan.",
            "integrity": integrity,
            "verifier_sha256": sha(Path(__file__).read_bytes()),
        })
        print("Post-launch snapshot saved; no result file was opened.")
        return
    paths = ready(HERE)
    before = json.loads((output / "post_launch_integrity.json").read_text(encoding="utf-8"))
    require(before["integrity"] == integrity, "Integrity changed since the post-launch snapshot")
    require(before["verifier_sha256"] == sha(Path(__file__).read_bytes()), "Verifier changed since the snapshot")
    digests = {p.name: sha(p.read_bytes()) for p in paths}
    rows = [read_run(path) for path in paths]
    summary = decisions(rows)
    registered = parse_verdicts((HERE / "VERDICTS_RS.md").read_text(encoding="utf-8"))
    disagreements = {tag: [registered[tag], summary["verdicts"][tag]] for tag in IDS if registered[tag] != summary["verdicts"][tag]}
    summary.update(
        registered_verdicts=registered, disagreements=disagreements,
        verified_utc=datetime.now(timezone.utc).isoformat(), raw_file_sha256=digests, integrity=integrity,
        limitations=[
            "Seed bootstrap conditions on one reused test panel; it is not uncertainty over future markets.",
            "Step counts were independently summed, not replayed: per-step traces and learned weights were not saved.",
            "Episode means, error measures, episode brackets, and decisions were recomputed from saved arrays.",
            "The log-prefix hash defect is disclosed; no historical manifest was rewritten.",
        ],
    )
    with (HERE / "runs_rs.csv").open(newline="", encoding="utf-8") as stream:
        original_rows = list(csv.DictReader(stream))
    require(len(original_rows) == 80, "Registered CSV must contain 80 rows")
    originals = {(r["arm"], int(r["seed"])): r for r in original_rows}
    require(len(originals) == 80, "Duplicate registered CSV run")
    for row in rows:
        original = originals[(row["arm"], row["seed"])]
        for new_key, old_key in (("X", "X"), ("X_long", "X_long"), ("X_short", "X_short"),
                                 ("E_edge", "E"), ("E_zero", "E_zero"), ("E_placebo", "placebo_booked"),
                                 ("forgiven_placebo", "placebo_forgiven"), ("share16_edge", "share16")):
            close(row[new_key], float(original[old_key]), f"CSV {row['arm']} s{row['seed']} {old_key}")
    require(digests == {p.name: sha(p.read_bytes()) for p in paths}, "Results changed during recount")
    output.mkdir(exist_ok=True)
    require(not (output / "recount.json").exists() and not (output / "runs_recomputed.csv").exists(), "Recount output already exists")
    csv_text = io.StringIO(newline="")
    writer = csv.DictWriter(csv_text, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    with (output / "runs_recomputed.csv").open("x", encoding="utf-8", newline="") as stream:
        stream.write(csv_text.getvalue())
    write_new(output / "recount.json", summary)
    for tag in IDS:
        print(tag, "registered:", registered[tag], "independent:", summary["verdicts"][tag])
    print(json.dumps({key: summary[key] for key in ("means", "edge_ratios", "oracle_edge_lower_bound", "audit_counts")}, indent=2))
    require(not disagreements, f"Registered verdict disagreements: {disagreements}")
    print("PASS: 80 runs, all 8 verdicts agree; registration defect remains explicitly disclosed.")


if __name__ == "__main__":
    main()
