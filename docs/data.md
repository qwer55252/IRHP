# Data preparation

Each JSONL manifest row contains an existing absolute audio path, duration in seconds, and a nonempty reference:

```json
{"audio_filepath": "/data/utterance.wav", "duration": 3.2, "text": "A REFERENCE TRANSCRIPT"}
```

The loader checks manifest structure and file existence before training, and checks mono 16 kHz audio when each file is read. Transcript normalization follows the historical `clean_transcript`: remove brace/parenthesis annotations and `<sil>`, uppercase, retain A–Z/apostrophe/spaces, and collapse whitespace. Empty normalized references are rejected.

## LibriSpeech clean-100

Download official archives from [OpenSLR](https://www.openslr.org/12), extract them into one `LibriSpeech` directory, and run the preparation command in the README. Use only `train-clean-100` for training and `dev-clean` for checkpoint selection. The test subsets are used only in the standalone evaluation command.

## LibriSpeech clean-460

Extract `train-clean-360` in addition to `train-clean-100`:

```bash
python cmd/prepare_librispeech.py --root /path/to/LibriSpeech \
  --splits train-clean-100 train-clean-360 --output-dir data/manifests
cat data/manifests/train-clean-100.jsonl \
    data/manifests/train-clean-360.jsonl > data/manifests/train-clean-460.jsonl

bash cmd/train.sh --config configs/wav2vec2-ls460.json \
  --train-manifest data/manifests/train-clean-460.jsonl \
  --dev-manifest data/manifests/dev-clean.jsonl \
  --output-dir outputs/wav2vec2-ls460
```

The pretrained starting model remains `facebook/wav2vec2-base-100h`, matching the historical scripts. The larger training subset does not imply a different pretrained checkpoint.

## TED-LIUM

The historical code uses **release 1**. Obtain that release through its dataset provider and extract the directory with `train`, `dev`, and `test` subdirectories, each containing `sph` and `stm` folders.

```bash
python cmd/prepare_tedlium.py --root /path/to/TEDLIUM_release1 \
  --output-dir data/tedlium --max-seconds 10
bash cmd/train.sh --config configs/wav2vec2-tedlium.json \
  --train-manifest data/tedlium/train.jsonl \
  --dev-manifest data/tedlium/dev.jsonl \
  --output-dir outputs/wav2vec2-tedlium
bash cmd/evaluate.sh --checkpoint outputs/wav2vec2-tedlium/final/export \
  --manifest dev=data/tedlium/dev.jsonl --manifest test=data/tedlium/test.jsonl \
  --output outputs/wav2vec2-tedlium/evaluation.json
```

The converter extracts STM segments into WAV files, skips ignored scoring segments, and retains segments at most 10 seconds long. This matches the duration policy in the historical Wav2Vec 2.0 loader, but full-corpus segmentation equivalence has not been verified. The source paper does not identify the TED-LIUM release explicitly; release 1 is established from code, not inferred from the title.

The paper's general batch-size description is 4; the saved TED-LIUM shell script overrides it to 12. The public recipe uses 4 and documents this difference in the reproduction guide. Change the JSON explicitly to investigate the historical override.
