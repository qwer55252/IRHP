# Conformer experiments

The original Conformer source is preserved in English under `irhp/conformer/`. Its dataset loaders are in `data_scripts/`. The pretrained starting points remain:

| Recipe | Pretrained model | Nominal removal | Batch size | Final epochs |
| --- | --- | ---: | ---: | ---: |
| Small / LibriSpeech 100h | `nvidia/stt_en_conformer_ctc_small` | 0.5 | 32 | 100 |
| Large / LibriSpeech 100h | `stt_en_conformer_ctc_large` | 0.8 | 8 | 100 |

Both use 10 pruning iterations and 2 short fine-tuning epochs per iteration. These values come from the saved shell scripts; they are not independently revalidated.

## Required NeMo patch

The research attention hook expects `self_attn` to return `(features, attention_maps)`, with maps shaped `(batch, heads, time, time)`. This assumes a modified NeMo attention implementation. That patch and the exact original NeMo version are absent from the repository. Installing an arbitrary official NeMo release does not establish compatibility.

The preserved hook now fails with a descriptive error if this contract is not met. The Wav2Vec 2.0 route does not require this patch.

## Commands

Inspect the exact command without importing NeMo:

```bash
python cmd/train_conformer.py --config configs/conformer-small-ls100.json \
  --output-dir outputs/conformer-small-ls100 --dry-run
python cmd/train_conformer.py --config configs/conformer-large-ls100.json \
  --output-dir outputs/conformer-large-ls100 --dry-run
```

After restoring the original compatible NeMo environment and attention patch, omit `--dry-run` to launch. The source retains pretrained loading, local dataset builders, pruning, fine-tuning, and its inline evaluation. A trained IRHP Conformer checkpoint must be generated locally; none is distributed.

The dataset loaders require a Hugging Face Datasets version supporting local scripts; the original requirements list `datasets==3.2.0`. The full historical environment is retained in `docs/provenance/original-requirements.txt`, but it does not specify the missing NeMo patch. The default Wav2Vec 2.0 environment intentionally excludes NeMo.

## Output and limitations

The source saves `final_finetune/final_epoch.ckpt` and a separate `pruning.json` mask. These are **not** compatible with the Wav2Vec 2.0 checkpoint loader. Conformer reload and standalone evaluation need validation against the original attention implementation before they can be advertised as a supported route.

The migration translates the source, makes dataset-script paths independent of the working directory, removes the fixed `/workspace` tokenizer directory, creates the checkpoint parent directory, and saves pruning metadata. It does not replace the underlying pruning algorithm or claim to reproduce Table 3 without the missing environment.
