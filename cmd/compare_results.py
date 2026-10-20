#!/usr/bin/env python3
"""Compare local WER measurements to the published IRHP rows."""
import argparse
import json
from pathlib import Path

PAPER = {
    "wav2vec2-ls100": {"dev-clean": 7.80, "dev-other": 23.05, "test-clean": 8.21, "test-other": 23.67},
    "wav2vec2-ls460": {"dev-clean": 4.86, "dev-other": 15.97, "test-clean": 5.07, "test-other": 16.41},
    "wav2vec2-tedlium": {"dev": 19.96, "test": 22.69},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--experiment", choices=PAPER, required=True)
    args = parser.parse_args()
    result = json.loads(args.results.read_text())
    if result.get("wer_unit") != "fraction":
        parser.error("Expected a report with wer_unit='fraction'")
    print("| Split | Paper WER (%) | Measured WER (%) | Difference (pp) |")
    print("| --- | ---: | ---: | ---: |")
    for split, reference in PAPER[args.experiment].items():
        value = result["splits"].get(split)
        if value is None:
            print(f"| {split} | {reference:.2f} | Not evaluated | — |")
        else:
            measured = value["wer"] * 100
            print(f"| {split} | {reference:.2f} | {measured:.2f} | {measured-reference:+.2f} |")


if __name__ == "__main__":
    main()
