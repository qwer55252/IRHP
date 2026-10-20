# IRHP

**Iterative Redundancy-Based Head Pruning for Efficient Self-Supervised Speech Recognition Models**

**ICASSP 2026 · Oral presentation**

Sang-Heon Jeong, Dong-Hyun Kim, and Joon-Hyuk Chang · Hanyang University

[Paper](https://www.cmsworkshops.com/ICASSP2026/view_paper.php?PaperNum=17724) · [Setup](docs/setup.md) · [Data](docs/data.md) · [Reproduction](docs/reproducibility.md) · [Results](docs/results.md) · [Citation](#citation)

IRHP progressively prunes redundant attention heads through repeated parameter re-initialization, short fine-tuning, attention-map analysis, and spectral clustering. This repository contains the Wav2Vec 2.0 implementation and the historical Conformer experiments.

> **Checkpoints:** Trained IRHP checkpoints are not distributed. Run pruning and fine-tuning to create your own checkpoint. The Wav2Vec 2.0 recipe starts from the public [`facebook/wav2vec2-base-100h`](https://huggingface.co/facebook/wav2vec2-base-100h) weights, downloaded automatically. Conformer experiments retain their public NeMo pretrained starting points.
>
> **Validation:** Small-model CPU tests cover structural pruning and checkpoint reload. Full paper-scale GPU runs remain pending. Conformer uses NeMo 2.5.0, aligned with GRP-KD, and requires an additional attention-map patch; see [its status](docs/conformer.md).

## Reported results

WER (%) from the paper. These are published values, not new measurements from this repository cleanup.

| Backbone | Training set | dev-clean | dev-other | test-clean | test-other |
| --- | --- | ---: | ---: | ---: | ---: |
| Wav2Vec 2.0 | LibriSpeech clean-100 | 7.80 | 23.05 | 8.21 | 23.67 |
| Wav2Vec 2.0 | LibriSpeech clean-460 | 4.86 | 15.97 | 5.07 | 16.41 |
| Conformer-S | LibriSpeech clean-100 | 5.97 | 15.40 | 6.06 | 15.03 |
| Conformer-L | LibriSpeech clean-100 | 4.89 | 13.44 | 4.99 | 13.24 |

Wav2Vec 2.0 IRHP on TED-LIUM: **19.96% dev / 22.69% test**. See [Tables 1–4](docs/results.md) for baselines and ablations.

## Installation

Use Linux with an NVIDIA GPU for paper-scale training. The original experiments used one RTX 4090. The focused environment retains the original Transformers 4.46.0 and PyTorch 2.5.1 versions.

```bash
git clone https://github.com/qwer55252/IRHP.git
cd IRHP
conda env create -f environment.yml
conda activate irhp
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python -m pip check
```

See [setup](docs/setup.md) for CPU tests and Docker. Conformer uses a separate environment and is not installed by the default requirements.

## Prepare data

Download and extract [LibriSpeech](https://www.openslr.org/12), then build manifests:

```bash
python cmd/prepare_librispeech.py --root /path/to/LibriSpeech \
  --splits train-clean-100 dev-clean dev-other test-clean test-other \
  --output-dir data/manifests
```

See [data preparation](docs/data.md) for clean-460 and TED-LIUM. Manifests contain absolute audio paths, durations, and transcripts. Audio must be mono at 16 kHz.

## Train and create an IRHP checkpoint

```bash
bash cmd/train.sh --config configs/wav2vec2-ls100.json \
  --train-manifest data/manifests/train-clean-100.jsonl \
  --dev-manifest data/manifests/dev-clean.jsonl \
  --output-dir outputs/wav2vec2-ls100
```

This runs 10 pruning iterations, 2 fine-tuning epochs per iteration, and 50 final fine-tuning epochs, following the saved Wav2Vec 2.0 training defaults. `keep_ratio=0.2` means a nominal **80% removal target**, not 20% removal. Integer rounding and the per-layer minimum mean the actual head count can differ; the checkpoint records the actual mask.

For an integration check, add `--smoke-test` and use a separate output directory. This uses one optimizer step per phase but still samples real attention maps; it does not reproduce paper scores. At least 10 training utterances are required.

A completed run produces:

```text
outputs/wav2vec2-ls100/
├── run.json
├── iteration-01/          # Short fine-tuning, export, and next_mask.json
├── ...
├── iteration-10/
└── final/
    └── export/
        ├── model.safetensors
        ├── config.json
        ├── pruning.json   # Original head IDs and run metadata
        └── ...            # Processor/tokenizer files
```

Use `final/export` for evaluation. Intermediate exports describe the mask actually applied to their weights; `next_mask.json` is the mask for the next phase. Runs use local Trainer logs and do not require a W&B account. Use a new output directory for each run.

## Evaluate

```bash
bash cmd/evaluate.sh --checkpoint outputs/wav2vec2-ls100/final/export \
  --manifest dev-clean=data/manifests/dev-clean.jsonl \
  --manifest dev-other=data/manifests/dev-other.jsonl \
  --manifest test-clean=data/manifests/test-clean.jsonl \
  --manifest test-other=data/manifests/test-other.jsonl \
  --output outputs/wav2vec2-ls100/evaluation.json

python cmd/compare_results.py --results outputs/wav2vec2-ls100/evaluation.json \
  --experiment wav2vec2-ls100
```

Evaluation reconstructs the pruned architecture from `pruning.json`, loads weights strictly, and performs greedy CTC decoding without an external language model. It loads model and processor files locally. Add `--device cpu` for CPU evaluation. JSON results include WER as a fraction, word-error counts, and model/manifest checksums.

Do not load a structurally pruned export directly with ordinary `Wav2Vec2ForCTC.from_pretrained`: the original configuration alone does not encode different head counts in each layer. Use `irhp.checkpoint.load_checkpoint`.

## Repository layout

```text
IRHP/
├── irhp/                 # Pruning, checkpoint I/O, training, evaluation, data
│   └── conformer/        # Historical NeMo experiments; separate prerequisites
├── cmd/                  # Data preparation, training, evaluation, comparison
├── configs/              # Explicit Wav2Vec 2.0 and Conformer recipes
├── data_scripts/         # Original dataset loaders used by Conformer experiments
├── docs/                 # Setup, reproduction notes, results, provenance
├── tests/                # CPU tests with synthetic audio and tiny models
├── environment.yml
├── pyproject.toml
├── CITATION.cff
└── LICENSE
```

Earlier exploratory scripts and notebooks remain in the [original research snapshot](https://github.com/qwer55252/IRHP/tree/b837575). The pre-release `main` is also retained in [a backup branch](https://github.com/qwer55252/IRHP/tree/backup/main-before-irhp-release).

## License

Original research code is released under [Apache-2.0](LICENSE). Pretrained models, datasets, and dependencies retain their own licenses and usage terms.

## Citation

```bibtex
@inproceedings{jeong2026irhp,
  title     = {Iterative Redundancy-Based Head Pruning for Efficient Self-Supervised Speech Recognition Models},
  author    = {Jeong, Sang-Heon and Kim, Dong-Hyun and Chang, Joon-Hyuk},
  booktitle = {2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2026}
}
```
