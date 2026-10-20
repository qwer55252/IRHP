#!/usr/bin/env python3
"""Extract STM segments from an official TED-LIUM release into mono WAV files."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Release directory containing train/dev/test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=10.0,
                        help="Historical Wav2Vec2 TED-LIUM duration limit")
    args = parser.parse_args()
    import soundfile as sf
    for split in ("train", "dev", "test"):
        files = sorted((args.root / split / "stm").glob("*.stm"))
        if not files:
            raise FileNotFoundError(f"No STM files for {split}")
        audio_dir = args.output_dir / "audio" / split
        audio_dir.mkdir(parents=True, exist_ok=True)
        manifest = args.output_dir / f"{split}.jsonl"
        temporary = manifest.with_suffix(".tmp")
        with temporary.open("w") as stream:
            for stm in files:
                sph = args.root / split / "sph" / f"{stm.stem}.sph"
                with sf.SoundFile(sph) as audio:
                    if audio.samplerate != 16000 or audio.channels != 1:
                        raise ValueError(f"Expected mono 16 kHz audio: {sph}")
                    for index, line in enumerate(stm.read_text().splitlines()):
                        fields = line.split(maxsplit=6)
                        if len(fields) < 7 or fields[6] == "ignore_time_segment_in_scoring":
                            continue
                        start, end = float(fields[3]), float(fields[4])
                        if end <= start or (args.max_seconds > 0 and end - start > args.max_seconds):
                            continue
                        audio.seek(int(start * 16000))
                        samples = audio.read(int(end * 16000) - int(start * 16000), dtype="float32")
                        output = audio_dir / f"{stm.stem}-{index:06d}.wav"
                        sf.write(output, samples, 16000, subtype="FLOAT")
                        stream.write(json.dumps({"audio_filepath": str(output.resolve()),
                                                 "duration": len(samples) / 16000,
                                                 "text": fields[6]}) + "\n")
        temporary.replace(manifest)
        print(manifest)


if __name__ == "__main__":
    main()
