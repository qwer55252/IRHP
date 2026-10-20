"""Train Wav2Vec2 IRHP from pretrained weights and explicit audio manifests."""
import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
from types import SimpleNamespace

from irhp.data import ManifestDataset, CTCDataCollator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke-test", action="store_true", help="Use one optimizer step per phase; not a paper run")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                        help="CPU is supported only for smoke tests")
    args = parser.parse_args()
    recipe = json.loads(args.config.read_text())
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Use a new output directory to avoid mixing pruning runs")
    train_data = ManifestDataset(args.train_manifest)
    dev_data = ManifestDataset(args.dev_manifest)
    if len(train_data) < 10:
        parser.error("The historical attention sampling procedure requires at least 10 training utterances")
    if not 0 < recipe["keep_ratio"] < 1 or recipe["iterations"] < 1:
        parser.error("Require 0 < keep_ratio < 1 and iterations >= 1")

    import numpy as np
    import torch
    import transformers
    from jiwer import wer
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Trainer, TrainingArguments, set_seed
    from irhp.pruning import (prune_wav2vec2_attention, get_avg_attention_matrices,
                              find_heads_to_keep, cluster_and_select_heads, my_compute_loss_ctc)
    from irhp.checkpoint import save_checkpoint
    if args.device == "cpu" and not args.smoke_test:
        parser.error("CPU training is limited to --smoke-test")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The training recipe requires an NVIDIA GPU with CUDA.")
    set_seed(recipe["seed"])
    args.output_dir.mkdir(parents=True)
    run = {"recipe": recipe, "torch": torch.__version__, "transformers": transformers.__version__,
           "train_manifest": train_data.path, "dev_manifest": dev_data.path,
           "smoke_test": args.smoke_test, "selection_metric": "dev_wer"}
    (args.output_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    processor = Wav2Vec2Processor.from_pretrained(recipe["pretrained_model"])
    initial = Wav2Vec2ForCTC.from_pretrained(recipe["pretrained_model"],
                                           attn_implementation="eager", ctc_loss_reduction="mean",
                                           pad_token_id=processor.tokenizer.pad_token_id)
    initial.config.output_attentions = False
    collator = CTCDataCollator(processor)
    layers = initial.config.num_hidden_layers
    heads_per_layer = initial.config.num_attention_heads
    total_heads = layers * heads_per_layer
    remove_per_step = int(total_heads * (1 - recipe["keep_ratio"]) / recipe["iterations"])
    if remove_per_step < 1:
        raise ValueError("This recipe removes fewer than one head per iteration")
    pruned = {i: set() for i in range(layers)}
    # Preserve the original procedure: protected heads are selected once initially.
    initial.to(args.device)
    attention = get_avg_attention_matrices(initial, processor, train_data, collator,
                                           sample_size=10, already_pruned_heads=pruned)
    protected = find_heads_to_keep(attention, pruned, total_heads,
                                  SimpleNamespace(test_mode=False, distance_metric="cosine"))
    initial.cpu()
    del attention
    torch.cuda.empty_cache()

    class ResearchTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = my_compute_loss_ctc(outputs, labels, processor)
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(prediction):
        ids = prediction.predictions
        if isinstance(ids, tuple):
            ids = ids[0]
        refs = prediction.label_ids.copy()
        refs[refs == -100] = processor.tokenizer.pad_token_id
        return {"wer": wer(processor.batch_decode(refs, group_tokens=False),
                           processor.batch_decode(ids))}

    def fit(model, phase, epochs):
        training_args = TrainingArguments(
            output_dir=str(args.output_dir / phase), num_train_epochs=epochs,
            max_steps=1 if args.smoke_test else -1,
            per_device_train_batch_size=recipe["batch_size"],
            per_device_eval_batch_size=recipe["batch_size"],
            eval_strategy="epoch", save_strategy="epoch", learning_rate=1e-4,
            fp16=args.device == "cuda", use_cpu=args.device == "cpu",
            save_total_limit=1, load_best_model_at_end=True,
            metric_for_best_model="wer", greater_is_better=False,
            remove_unused_columns=False, report_to="none", seed=recipe["seed"],
            logging_strategy="epoch")
        trainer = ResearchTrainer(model=model, args=training_args, data_collator=collator,
                                  train_dataset=train_data, eval_dataset=dev_data,
                                  processing_class=processor, compute_metrics=compute_metrics,
                                  preprocess_logits_for_metrics=lambda logits, labels: logits.argmax(dim=-1))
        trainer.train()
        # This export uses the best dev-WER state loaded by Trainer.
        save_checkpoint(model, processor, args.output_dir / phase / "export", pruned,
                        {**run, "phase": phase, "protected_heads": protected})
        return trainer

    for iteration in range(recipe["iterations"]):
        phase = f"iteration-{iteration + 1:02d}"
        print(f"{phase}: reset pretrained parameters, apply current mask, fine-tune")
        model = deepcopy(initial)
        prune_wav2vec2_attention(model, pruned)
        trainer = fit(model, phase, recipe["iteration_epochs"])
        attention = get_avg_attention_matrices(model, processor, train_data, collator,
                                               sample_size=10, already_pruned_heads=pruned)
        candidates = cluster_and_select_heads(attention, n_remove=remove_per_step,
                                              init_num_total_heads=total_heads,
                                              already_pruned_heads=pruned, init_num_heads=heads_per_layer,
                                              heads_to_keep=protected)
        # The historical procedure may remove fewer heads when a layer reaches one head.
        for layer, head in sorted(candidates):
            if head not in pruned[layer] and layer * heads_per_layer + head not in protected:
                if len(pruned[layer]) < heads_per_layer - 1:
                    pruned[layer].add(head)
        (args.output_dir / phase / "next_mask.json").write_text(
            json.dumps({str(k): sorted(v) for k, v in pruned.items()}, indent=2) + "\n")
        del trainer, model, attention
        gc.collect()
        torch.cuda.empty_cache()
    final = deepcopy(initial)
    prune_wav2vec2_attention(final, pruned)
    fit(final, "final", recipe["final_epochs"])
    print(f"Checkpoint: {args.output_dir / 'final' / 'export'}")
    print(f"Heads retained: {total_heads - sum(map(len, pruned.values()))}/{total_heads}")


if __name__ == "__main__":
    main()
