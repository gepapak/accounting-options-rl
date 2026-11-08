"""Synthetic checks for the independent recount; never reads training outcomes."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from verify_completed_rs import ARMS, EXPECTED, IDS, MARKER, SEEDS, decisions, episode_values, log_prefix, parse_verdicts, read_run, ready, write_new


def synthetic_rows() -> list[dict]:
    rows = []
    for arm in ARMS:
        for seed in SEEDS:
            row = dict(arm=arm, seed=seed, X=0.8 if arm == "PLAIN" else 0.0, X_long=0.0, X_short=0.0,
                       E_edge=10.0, E_zero=0.0, E_placebo=0.0, share16_edge=0.0, forgiven_placebo=0.0)
            for kind in ("edge", "placebo"):
                row.update({f"err_tir_{kind}": 0.0, f"steps_{kind}": 67_200,
                            f"steps_inside_bracket_{kind}": 67_200,
                            f"steps_identified_{kind}": 50_000, f"identified_steps_tir_inexact_{kind}": 0})
            rows.append(row)
    return rows


def synthetic_run() -> dict:
    result = dict(setting="RS_DK2", arm="PLAIN", seed=300, steps=1_000_000, zero_starts=True)
    result["lev"] = {f"{e}|{z}|{r}": 16.0
                     for e in ("0", "0.02", "0.05", "0.1", "0.25", "0.5", "1", "2")
                     for z in ("-1", "+0", "+1") for r in ("0.05", "0.5")}
    for kind in ("edge", "placebo", "zero"):
        result[f"test_{kind}"] = dict(booked_episodes=[0.0] * 100, booked_pct_K0=0.0,
                                     booked_se=0.0, reported_pct_K0=0.0,
                                     forgiven_pct_K0=0.0, share_lev16=1.0)
        if kind != "zero":
            audit = dict(booked=0.0, steps=67_200, steps_inside_bracket=67_200,
                         steps_identified=60_000, identified_steps_tir_inexact=0,
                         episodes_lower_le_booked=100, episodes_booked_le_reported=100)
            for name in ("reported", "spr", "tir", "lower"):
                audit.update({f"episodes_{name}": [0.0] * 100, name: 0.0,
                              f"err_{name}": 0.0, f"mae_{name}": 0.0})
            result[f"audit_{kind}"] = audit
    return result


class RecountTests(unittest.TestCase):
    def test_raw_run_recount_and_tampering(self):
        original = synthetic_run()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "RS_DK2__PLAIN__s300.json"
            path.write_text(json.dumps(original), encoding="utf-8")
            self.assertEqual(read_run(path)["X"], 1.0)
            for key, field, value in (("test_edge", "booked_pct_K0", 100.0),
                                       ("audit_edge", "err_tir", 1.0),
                                       ("audit_edge", "steps_identified", 1_000_000)):
                data = copy.deepcopy(original)
                data[key][field] = value
                path.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_run(path)

    def test_log_marker_is_a_complete_line(self):
        text = b'Header\nQuoted "' + MARKER + b'".\nFull chronology\n' + MARKER + b'\nAppended\n'
        self.assertEqual(log_prefix(text), text[:-len(b"Appended\n")])

    def test_log_preserves_crlf(self):
        text = b"Header\r\n" + MARKER + b"\r\nAfter\r\n"
        self.assertEqual(log_prefix(text), text[:-len(b"After\r\n")])

    def test_invalid_markers_rejected(self):
        for text in (b"Missing", MARKER + b"\n" + MARKER + b"\n"):
            with self.assertRaises(ValueError):
                log_prefix(text)

    def test_blind_gate_rejects_incomplete_and_missing_verdict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "results_rs").mkdir()
            with self.assertRaises(ValueError):
                ready(root)
            for name in EXPECTED:
                (root / "results_rs" / name).touch()
            with self.assertRaisesRegex(ValueError, "outcomes remain unopened"):
                ready(root)
            (root / "VERDICTS_RS.md").touch()
            self.assertEqual(len(ready(root)), 80)
            (root / "results_rs" / "extra.json").touch()
            with self.assertRaises(ValueError):
                ready(root)

    def test_normal_decisions(self):
        result = decisions(synthetic_rows(), repetitions=100)
        self.assertEqual(set(result["verdicts"].values()), {"SUPPORTED"})
        self.assertEqual(result["edge_ratios"]["TIR"]["lower_bound"], 1.0)
        self.assertEqual(result["audit_counts"]["steps"], 10_752_000)

    def test_no_exploit_masks_five_tests(self):
        rows = synthetic_rows()
        for row in rows:
            row["X"] = 0.0
        result = decisions(rows, repetitions=100)["verdicts"]
        self.assertEqual(result["RS1"], "NOT SUPPORTED")
        for tag in ("RS2", "RS3", "RS4", "RS5", "RS6"):
            self.assertEqual(result[tag], "NOT TESTABLE")

    def test_unreliable_oracle_masks_edge_only(self):
        for edge in (-1.0, 0.0):
            rows = synthetic_rows()
            for row in rows:
                if row["arm"] == "ORACLE":
                    row["E_edge"] = edge
            result = decisions(rows, repetitions=100)["verdicts"]
            for tag in ("RS3", "RS5", "RS6"):
                self.assertEqual(result[tag], "NOT TESTABLE")
            self.assertEqual(result["RS4"], "SUPPORTED")

    def test_positive_oracle_mean_but_negative_lower_bound(self):
        rows = synthetic_rows()
        for row in rows:
            if row["arm"] == "ORACLE":
                row["E_edge"] = 20.0 if row["seed"] == 300 else -1.0
        result = decisions(rows, repetitions=100)
        self.assertLess(result["oracle_edge_lower_bound"], 0)
        self.assertEqual(result["verdicts"]["RS5"], "NOT TESTABLE")

    def test_ratio_threshold(self):
        for value, expected in ((5.0, "NOT SUPPORTED"), (8.0, "SUPPORTED")):
            rows = synthetic_rows()
            for row in rows:
                if row["arm"] == "TIR":
                    row["E_edge"] = value
            self.assertEqual(decisions(rows, repetitions=100)["verdicts"]["RS5"], expected)

    def test_audit_failures(self):
        rows = synthetic_rows()
        row = next(r for r in rows if r["arm"] == "SPR")
        row["err_tir_edge"] = 1.01
        row["identified_steps_tir_inexact_placebo"] = 1
        result = decisions(rows, repetitions=100)["verdicts"]
        self.assertEqual(result["RS-D1"], "NOT SUPPORTED")
        self.assertEqual(result["RS-D2"], "NOT SUPPORTED")

    def test_missing_seed_rejected(self):
        with self.assertRaises(ValueError):
            decisions(synthetic_rows()[:-1], repetitions=100)

    def test_episode_shape_and_nonfinite(self):
        self.assertEqual(episode_values([1] * 100, "test"), [1.0] * 100)
        for value in ([], [float("nan")] * 100, [True] * 100):
            with self.assertRaises(ValueError):
                episode_values(value, "test")

    def test_verdict_shape(self):
        text = "\n".join(f"| {tag} | arithmetic | SUPPORTED |" for tag in IDS)
        self.assertEqual(len(parse_verdicts(text)), 8)
        with self.assertRaises(ValueError):
            parse_verdicts(text + "\n| RS1 | duplicate | SUPPORTED |")
        with self.assertRaises(ValueError):
            parse_verdicts("")

    def test_output_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.json"
            write_new(path, {"first": True})
            with self.assertRaises(FileExistsError):
                write_new(path, {"second": True})


if __name__ == "__main__":
    unittest.main()
