"""Exercise two pruning iterations, final training, export, and evaluation offline."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor
from test_pruning import tiny_model


class WorkflowTests(unittest.TestCase):
    def test_cpu_smoke_workflow(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pretrained = root / "pretrained"
            pretrained.mkdir()
            (pretrained / "vocab.json").write_text(json.dumps(
                {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4, "A": 5, "B": 6, "C": 7}))
            processor = Wav2Vec2Processor(
                Wav2Vec2FeatureExtractor(sampling_rate=16000, return_attention_mask=True),
                Wav2Vec2CTCTokenizer(str(pretrained / "vocab.json"), word_delimiter_token="|"))
            processor.save_pretrained(pretrained)
            cfg = tiny_model().config
            cfg.num_hidden_layers = 4
            Wav2Vec2ForCTC(cfg).save_pretrained(pretrained)
            rows = []
            for i in range(12):
                audio = root / f"{i}.wav"
                sf.write(audio, np.random.default_rng(i).normal(0, 0.1, 800), 16000)
                rows.append({"audio_filepath": str(audio), "duration": .05, "text": "A B"})
            train = root / "train.jsonl"
            dev = root / "dev.jsonl"
            train.write_text("".join(json.dumps(row) + "\n" for row in rows[:10]))
            dev.write_text("".join(json.dumps(row) + "\n" for row in rows[10:]))
            recipe = root / "recipe.json"
            recipe.write_text(json.dumps({"pretrained_model": str(pretrained), "keep_ratio": .8,
                                           "iterations": 2, "iteration_epochs": 1, "final_epochs": 1,
                                           "batch_size": 2, "seed": 42}))
            env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "1"}
            run = subprocess.run([sys.executable, "-m", "irhp.train", "--config", str(recipe),
                                  "--train-manifest", str(train), "--dev-manifest", str(dev),
                                  "--output-dir", str(root / "run"), "--device", "cpu", "--smoke-test"],
                                 env=env, text=True, capture_output=True, timeout=180)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            checkpoint = root / "run" / "final" / "export"
            metadata = json.loads((checkpoint / "pruning.json").read_text())
            self.assertTrue(metadata["smoke_test"])
            self.assertGreater(sum(map(len, metadata["pruned_heads"].values())), 0)
            output = root / "evaluation.json"
            evaluation = subprocess.run([sys.executable, "-m", "irhp.evaluate", "--checkpoint", str(checkpoint),
                                         "--manifest", f"dev={dev}", "--output", str(output), "--device", "cpu"],
                                        env=env, text=True, capture_output=True, timeout=60)
            self.assertEqual(evaluation.returncode, 0, evaluation.stdout + evaluation.stderr)
            result = json.loads(output.read_text())
            self.assertEqual(result["wer_unit"], "fraction")
            self.assertEqual(result["splits"]["dev"]["utterances"], 2)
            self.assertGreaterEqual(result["splits"]["dev"]["wer"], 0)
