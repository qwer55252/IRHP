"""Portable audio manifests and the historical transcript normalization."""
import json
import math
from pathlib import Path
import re


def clean_transcript(text):
    text = re.sub(r"\{.*?\}", "", text)
    text = re.sub(r"\(.*?\)", "", text)
    text = text.replace("<sil>", "").upper()
    text = re.sub(r"[^A-Z' ]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


class ManifestDataset:
    """Read JSONL records; load audio lazily so datasets need not fit in RAM."""
    def __init__(self, path):
        self.rows = []
        self.path = str(Path(path).resolve(strict=True))
        with open(self.path, encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                    audio = Path(row["audio_filepath"])
                    duration = float(row["duration"])
                    if not audio.is_absolute() or not audio.is_file():
                        raise ValueError("audio_filepath must be an existing absolute file")
                    if not math.isfinite(duration) or duration <= 0:
                        raise ValueError("duration must be finite and positive")
                    row["text"] = clean_transcript(row["text"])
                    if not row["text"]:
                        raise ValueError("transcript is empty after normalization")
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"{self.path}:{number}: {error}") from error
                self.rows.append(row)
        if not self.rows:
            raise ValueError(f"Empty manifest: {path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import soundfile as sf
        row = self.rows[index]
        audio, rate = sf.read(row["audio_filepath"], dtype="float32")
        if rate != 16000 or audio.ndim != 1:
            raise ValueError(f"Expected mono 16 kHz audio: {row['audio_filepath']}")
        return {"audio": {"array": audio, "sampling_rate": rate}, "text": row["text"]}


class CTCDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        # Keep historical full-utterance padding; do not silently truncate labels/audio.
        inputs = self.processor([f["audio"]["array"] for f in features],
                                sampling_rate=16000, return_attention_mask=True,
                                padding=True, return_tensors="pt")
        labels = self.processor.tokenizer([f["text"] for f in features],
                                          padding=True, return_tensors="pt")["input_ids"]
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"input_values": inputs["input_values"], "attention_mask": inputs["attention_mask"],
                "labels": labels}
