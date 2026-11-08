"""Regression tests for the historical registration manifest reader."""
import contextlib
import hashlib
import io
from pathlib import Path
import tempfile
import unittest

import verify_studies


class RegistrationHashesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name) / "LIABILITY_STUDY_2026-09-29"
        self.folder.mkdir()
        verify_studies.FAILS.clear()
        verify_studies.REVISED.clear()

    def check_manifest(self, text):
        (self.folder / "PREREGISTRATION_HASHES.txt").write_text(text, encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()) as output:
            verify_studies.check_hashes(self.folder)
        return output.getvalue()

    def test_annotated_entry_is_checked(self):
        data = b"registered script\n"
        (self.folder / "aggregate_b.py").write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        output = self.check_manifest(
            f"2026-09-29T18:49:09Z {digest} *aggregate_b.py (written during the run)\n"
        )
        self.assertIn("[ok] LIABILITY_STUDY_2026-09-29/aggregate_b.py", output)
        self.assertEqual(verify_studies.FAILS, [])

    def test_annotated_digest_mismatch_fails(self):
        (self.folder / "aggregate_b.py").write_bytes(b"changed script\n")
        digest = hashlib.sha256(b"registered script\n").hexdigest()
        self.check_manifest(
            f"2026-09-29T18:49:09Z {digest} *aggregate_b.py (written during the run)\n"
        )
        self.assertEqual(verify_studies.FAILS, ["LIABILITY_STUDY_2026-09-29/aggregate_b.py"])

    def test_relocated_script_retains_registered_digest(self):
        data = b"registered study C script\n"
        (self.folder / "scaling").mkdir()
        (self.folder / "scaling" / "aggregate_c.py").write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        output = self.check_manifest(
            f"2026-09-30T16:55:55Z {digest} *aggregate_c.py (written after the runs)\n"
        )
        self.assertIn("[ok]", output)
        self.assertIn("scaling/aggregate_c.py", output)
        self.assertEqual(verify_studies.FAILS, [])

    def test_pre_run_amendment_accepts_either_recorded_digest(self):
        data = b"amended registration\n"
        (self.folder / "PREREGISTRATION_B.md").write_bytes(data)
        previous = hashlib.sha256(b"original registration\n").hexdigest()
        current = hashlib.sha256(data).hexdigest()
        output = self.check_manifest(
            f"2026-09-29T18:00:00Z {previous} *PREREGISTRATION_B.md\n"
            f"2026-09-29T18:01:00Z {current} *PREREGISTRATION_B.md\n"
        )
        self.assertEqual(output.count("[ok]"), 1)
        self.assertEqual(verify_studies.FAILS, [])

    def test_missing_annotated_file_fails(self):
        digest = hashlib.sha256(b"missing\n").hexdigest()
        self.check_manifest(
            f"2026-09-29T18:49:09Z {digest} *missing.py (historical note)\n"
        )
        self.assertEqual(verify_studies.FAILS, ["LIABILITY_STUDY_2026-09-29/missing.py"])

    def test_malformed_entry_fails(self):
        output = self.check_manifest("2026-09-29T18:49:09Z not-a-digest *run.py\n")
        self.assertIn("malformed hash entry", output)
        self.assertEqual(
            verify_studies.FAILS, ["LIABILITY_STUDY_2026-09-29/PREREGISTRATION_HASHES.txt:1"]
        )


if __name__ == "__main__":
    unittest.main()
