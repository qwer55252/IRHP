import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class ReportTests(unittest.TestCase):
    def test_wer_units_and_missing_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps({"wer_unit": "fraction", "splits": {"test-clean": {"wer": .09}}}))
            result = subprocess.run([sys.executable, "cmd/compare_results.py", "--results", str(path),
                                     "--experiment", "wav2vec2-ls100"], capture_output=True, text=True, check=True)
            self.assertIn("| test-clean | 8.21 | 9.00 | +0.79 |", result.stdout)
            self.assertIn("Not evaluated", result.stdout)
