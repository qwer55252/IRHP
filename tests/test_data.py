import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import soundfile as sf
from irhp.data import ManifestDataset, clean_transcript


class DataTests(unittest.TestCase):
    def test_historical_normalization(self):
        self.assertEqual(clean_transcript("{BREATH} It's (2) a test! <sil>"), "IT'S A TEST")

    def test_bad_manifest_reports_line_number(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text(json.dumps({"audio_filepath": "relative.wav", "duration": 1, "text": "A"}) + "\n")
            with self.assertRaisesRegex(ValueError, r'bad.jsonl:1:'):
                ManifestDataset(path)

    def test_librispeech_conversion_and_audio_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chapter = root / "LibriSpeech" / "test-clean" / "1" / "2"
            chapter.mkdir(parents=True)
            sf.write(chapter / "1-2-0001.flac", np.zeros(1600), 16000)
            (chapter / "1-2.trans.txt").write_text("1-2-0001 HELLO WORLD\n")
            result = subprocess.run([sys.executable, "cmd/prepare_librispeech.py", "--root", str(root / "LibriSpeech"),
                                     "--splits", "test-clean", "--output-dir", str(root / "manifests")],
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            dataset = ManifestDataset(root / "manifests" / "test-clean.jsonl")
            self.assertEqual(len(dataset), 1)
            self.assertEqual(len(dataset[0]["audio"]["array"]), 1600)
            self.assertEqual(dataset[0]["text"], "HELLO WORLD")

    def test_tedlium_segmentation_skips_ignored_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ("train", "dev", "test"):
                (root / split / "sph").mkdir(parents=True)
                (root / split / "stm").mkdir(parents=True)
                sf.write(root / split / "sph" / "talk.sph", np.zeros(16000), 16000, format="NIST")
                (root / split / "stm" / "talk.stm").write_text(
                    "talk 1 speaker 0.0 0.1 <o,f0,male> hello world\n"
                    "talk 1 speaker 0.1 0.2 <o,f0,male> ignore_time_segment_in_scoring\n")
            result = subprocess.run([sys.executable, "cmd/prepare_tedlium.py", "--root", str(root),
                                     "--output-dir", str(root / "manifests")], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            dataset = ManifestDataset(root / "manifests" / "dev.jsonl")
            self.assertEqual(len(dataset), 1)
            self.assertEqual(len(dataset[0]["audio"]["array"]), 1600)
            self.assertEqual(dataset[0]["text"], "HELLO WORLD")

    def test_stereo_audio_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "stereo.wav"
            sf.write(audio, np.zeros((1600, 2)), 16000)
            path = root / "manifest.jsonl"
            path.write_text(json.dumps({"audio_filepath": str(audio), "duration": .1, "text": "A"}))
            dataset = ManifestDataset(path)
            with self.assertRaisesRegex(ValueError, "mono 16 kHz"):
                dataset[0]
