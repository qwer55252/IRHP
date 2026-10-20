# Conformer experiments

The original Conformer source is preserved in English under `irhp/conformer/`. Its dataset loaders are in `data_scripts/`. The pretrained starting points remain:

| Recipe | Pretrained model | Nominal removal | Batch size | Final epochs |
| --- | --- | ---: | ---: | ---: |
| Small / LibriSpeech 100h | `nvidia/stt_en_conformer_ctc_small` | 0.5 | 32 | 100 |
| Large / LibriSpeech 100h | `stt_en_conformer_ctc_large` | 0.8 | 8 | 100 |

Both use 10 pruning iterations and 2 short fine-tuning epochs per iteration. These values come from the saved shell scripts; they are not independently revalidated.

## Environment shared with GRP-KD

Use official **NeMo 2.5.0**, the same release selected for [GRP-KD](https://github.com/qwer55252/GRP-KD). `constraints-conformer.txt` matches its PyTorch/torchaudio 2.5.1, Lightning 2.4.0, Transformers 4.53.3, NumPy 1.26.4, and PEFT 0.15.2 constraints. These are candidate compatibility versions, not a recovered lockfile from the original paper experiments.

Create a separate environment: the default Wav2Vec 2.0 recipe uses Transformers 4.46.0.

```bash
conda create -n irhp-conformer python=3.10 -y
conda activate irhp-conformer
python -m pip install torch==2.5.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-conformer.txt
python -m pip check
python -c "import torch, nemo, lightning; print(torch.__version__, torch.cuda.is_available(), nemo.__file__, lightning.__version__)"
```

The requirements retain Datasets 3.2.0 for the historical local dataset scripts. Full dependency installation and CUDA training still need validation on the training machine.

## Required NeMo patch

The research attention hook expects `self_attn` to return `(features, attention_maps)`, with maps shaped `(batch, heads, time, time)`. This assumes a modified NeMo attention implementation. The patch is absent from the repository. Official NeMo 2.5.0 returns features (or features and streaming cache), not attention maps; see its [attention implementation](https://github.com/NVIDIA/NeMo/blob/v2.5.0/nemo/collections/asr/parts/submodules/multi_head_attention.py). Matching the GRP-KD dependency version does not supply this IRHP-specific patch. GRP-KD's historical vendored NeMo tree is incomplete and cannot substitute for an installable patched runtime.

The preserved hook now fails with a descriptive error if this contract is not met. The Wav2Vec 2.0 route does not require this patch.

## Commands

Inspect the exact command without importing NeMo:

```bash
python cmd/train_conformer.py --config configs/conformer-small-ls100.json \
  --output-dir outputs/conformer-small-ls100 --dry-run
python cmd/train_conformer.py --config configs/conformer-large-ls100.json \
  --output-dir outputs/conformer-large-ls100 --dry-run
```

After installing the environment above and providing a compatible attention-map patch, omit `--dry-run` to launch. The source retains pretrained loading, local dataset builders, pruning, fine-tuning, and its inline evaluation. A trained IRHP Conformer checkpoint must be generated locally; none is distributed.

The dataset loaders require a Hugging Face Datasets version supporting local scripts; the original requirements list `datasets==3.2.0`. The full historical environment is retained in `docs/provenance/original-requirements.txt`, but it does not specify the missing NeMo patch. The default Wav2Vec 2.0 environment intentionally excludes NeMo.

## Output and limitations

The source saves `final_finetune/final_epoch.ckpt` and a separate `pruning.json` mask. These are **not** compatible with the Wav2Vec 2.0 checkpoint loader. Conformer reload and standalone evaluation need validation against the original attention implementation before they can be advertised as a supported route.

The migration translates the source, makes dataset-script paths independent of the working directory, removes the fixed `/workspace` tokenizer directory, creates the checkpoint parent directory, and saves pruning metadata. It does not replace the underlying pruning algorithm or claim to reproduce Table 3 without validating the attention patch and full training workflow.
