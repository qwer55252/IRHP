# Reproduction guide

## Available weights

No trained IRHP checkpoint is released. Create checkpoints by running the supplied training recipes. Wav2Vec 2.0 uses the public `facebook/wav2vec2-base-100h` checkpoint; its model card identifies it as pretrained and fine-tuned on LibriSpeech 100h. The Conformer sources retain the NeMo Small/Large pretrained models. Do not interpret these public starting weights as pruned IRHP results.

## Mapping paper experiments to code

| Paper experiment | Public entry point | Status |
| --- | --- | --- |
| Table 1, Wav2Vec 2.0 IRHP clean-100 | `configs/wav2vec2-ls100.json` + `irhp.train` | Small-model CPU workflow tested; full rerun pending |
| Table 1, Wav2Vec 2.0 IRHP clean-460 | `configs/wav2vec2-ls460.json` + combined training manifest | Full rerun pending |
| Table 2, TED-LIUM IRHP | `configs/wav2vec2-tedlium.json` | Full-corpus preparation/training unverified |
| Table 3, Conformer-S/L | `configs/conformer-*-ls100.json` + `cmd/train_conformer.py` | Original NeMo attention patch required |
| Table 4, ablations | Original `IHP_*` scripts at research snapshot | Exact final-run mapping not established |
| Figure 1 | Framework diagram in the paper | Conceptual illustration |
| Figure 2, pruning-ratio sweep | Copy Wav2Vec 2.0 config and vary `keep_ratio` | `keep_ratio = 1 - removal_ratio`; rerun required |
| Baselines | Original baseline/one-shot/magnitude/L0 scripts | Preserved in Git history; not revalidated |

[Original research snapshot](https://github.com/qwer55252/IRHP/tree/b837575).

## Settings

The default Wav2Vec 2.0 recipe retains the saved script's settings: 10 iterations, 2 epochs per iteration, 50 final epochs, 10 sampled attention utterances, 10 protected heads, learning rate 1e-4, and nominal 80% head removal. The paper also mentions 100 final epochs in some experiments; an exact mapping to every table row is not recorded in the snapshot. Set `final_epochs` explicitly when investigating that setting.

The public TED-LIUM recipe uses the paper's general batch size of 4. The original TED-LIUM shell script overrides it to 12. This distinction is not silently treated as equivalent.

## Preserved research behavior

The following details are retained from `train_IRHP_libri.py`:

- The initial pretrained model is reset before each short fine-tuning phase and before final fine-tuning.
- Protected heads are chosen once from the initial model; the code does not recompute the protected set each iteration.
- Similarity is computed as `1 - cosine_distance / 2`. The paper writes `(1 + cos) / 2` while describing `cos` as a distance, so its notation differs from the code.
- A cluster representative is selected randomly. The paper describes retaining the least redundant representative.
- The number removed per iteration is floored to an integer. For 144 heads, a keep ratio of 0.2, and 10 iterations, this is 11 candidates per iteration. At least one head per layer is retained, which can further reduce actual pruning.
- The historical custom CTC objective assigns all batch items the padded logit length. This behavior is retained for traceability; it differs from length-aware CTC.

These are scientific interpretation issues, not claims that exact paper scores have been reproduced.

## Release changes affecting runs

The publication workflow intentionally separates infrastructure from the research algorithms:

- Audio is read from explicit manifests rather than machine-specific caches and dynamically loaded dataset scripts. Transcript normalization is explicit.
- The default seed is 42; the original scripts did not seed every random source explicitly. Candidate application uses a sorted order.
- **Checkpoint selection uses development WER only.** The original Wav2Vec 2.0 script selected by `eval_test_wer`; the public workflow keeps test data out of training and model selection.
- The upstream Transformers attention reshape width is updated after pruning. The old code changed Q/K/V and head counts but left `embed_dim` unchanged, which fails with unmodified upstream attention.
- Each iteration has its own output directory. The final checkpoint is saved after final fine-tuning, together with the exact head mask and processor.
- Evaluation strictly loads all weights, reconstructs structure from saved head IDs, and excludes padded logit frames from decoding. Reference labels are not CTC-collapsed.
- Smoke tests use real attention computations; the historical `test_mode` could substitute random distance matrices.

These changes mean the public workflow is not a bit-for-bit replay of the historical script. Keep the original snapshot for comparison and report observed scores rather than assuming equivalence.

## Checkpoint semantics

`final/export` is the selected final model for inference. A `checkpoint-*` directory produced by Trainer is an internal training checkpoint and does not carry the standalone pruning metadata. Use the export directory with `irhp.checkpoint.load_checkpoint`.

`iteration-XX/export/pruning.json` describes that iteration's actual model. `iteration-XX/next_mask.json` describes the mask selected for the next reset. Never apply the next mask to an earlier export's weights.

Whole-run resume is not implemented because each pruning phase changes model structure. Use a new output directory. Internal Trainer checkpoints are retained for investigation; no resume claim is made for them.

## Validation scope

The CPU tests use tiny randomly initialized models and synthetic audio. They validate structural pruning against explicitly masked dense attention, repeated original head indexing, original-position attention padding, offline checkpoint round trips, and the training/export/evaluation workflow. They do not validate paper WERs, pretrained-model download, GPU memory use, or the original Conformer patch.
