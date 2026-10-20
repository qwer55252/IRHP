#!/usr/bin/env python3
"""Build IRHP manifests from extracted official LibriSpeech archives."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    import soundfile as sf
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        transcripts = sorted((args.root / split).rglob("*.trans.txt"))
        if not transcripts:
            raise FileNotFoundError(f"No transcripts for {split}")
        output = args.output_dir / f"{split}.jsonl"
        temporary = output.with_suffix(".tmp")
        with temporary.open("w") as stream:
            for transcript in transcripts:
                for line in transcript.read_text().splitlines():
                    utterance, text = line.split(maxsplit=1)
                    audio = transcript.parent / f"{utterance}.flac"
                    info = sf.info(audio)
                    if info.samplerate != 16000 or info.channels != 1:
                        raise ValueError(f"Expected mono 16 kHz: {audio}")
                    stream.write(json.dumps({"audio_filepath": str(audio.resolve()),
                                             "duration": info.duration, "text": text}) + "\n")
        temporary.replace(output)
        print(output)


if __name__ == "__main__":
    main()
