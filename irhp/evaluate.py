"""Evaluate a locally trained IRHP checkpoint with greedy CTC decoding."""
import argparse
import hashlib
import json
from pathlib import Path
from irhp.data import ManifestDataset, CTCDataCollator


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Directory containing model.safetensors and pruning.json")
    parser.add_argument("--manifest", action="append", required=True, metavar="SPLIT=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    datasets = {}
    for item in args.manifest:
        if "=" not in item:
            parser.error("Use --manifest SPLIT=PATH")
        split, path = item.split("=", 1)
        if not split or split in datasets:
            parser.error("Split names must be nonempty and unique")
        datasets[split] = ManifestDataset(path)
    import torch
    from jiwer import process_words
    from irhp.checkpoint import load_checkpoint
    model, processor, metadata = load_checkpoint(args.checkpoint)
    model.to(args.device).eval()
    collator = CTCDataCollator(processor)
    report = {"checkpoint": str(args.checkpoint.resolve()), "metadata": metadata,
              "model_sha256": sha256(args.checkpoint / "model.safetensors"),
              "decoding": "greedy_ctc", "wer_unit": "fraction", "splits": {}}
    for split, dataset in datasets.items():
        counts = {"substitutions": 0, "deletions": 0, "insertions": 0, "hits": 0}
        for start in range(0, len(dataset), args.batch_size):
            samples = [dataset[i] for i in range(start, min(start + args.batch_size, len(dataset)))]
            batch = collator(samples)
            with torch.inference_mode():
                logits = model(input_values=batch["input_values"].to(args.device),
                               attention_mask=batch["attention_mask"].to(args.device)).logits
            lengths = model._get_feat_extract_output_lengths(batch["attention_mask"].sum(-1))
            predictions = [processor.decode(ids[:int(length)].cpu().tolist())
                           for ids, length in zip(logits.argmax(-1), lengths)]
            result = process_words([sample["text"] for sample in samples], predictions)
            for key in counts:
                counts[key] += getattr(result, key)
        reference_words = counts["hits"] + counts["substitutions"] + counts["deletions"]
        score = (counts["substitutions"] + counts["deletions"] + counts["insertions"]) / reference_words
        report["splits"][split] = {"wer": score, "utterances": len(dataset), **counts,
                                    "manifest_sha256": sha256(dataset.path)}
        print(f"{split}: WER {score:.2%}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
