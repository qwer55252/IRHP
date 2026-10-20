# Migration from HWDHuBERT

The repository is now named IRHP. Despite its former name, the main published experiments use Wav2Vec 2.0, with separate Conformer experiments. No HuBERT result is claimed by the public recipes.

The original code remains available at commit `b837575` and the remote branch `backup/main-before-irhp-release`. No Git history is rewritten.

| Original file or command | Public location |
| --- | --- |
| `train_IRHP_libri.py` pruning functions | `irhp/pruning.py` |
| Wav2Vec 2.0 training scripts | `irhp/train.py` + `configs/wav2vec2-*.json` |
| `libri_eval.py` / `tedlium_eval.py` | `irhp/evaluate.py` + saved pruning metadata |
| `train_conformer_IRHP_libri.py` | `irhp/conformer/train.py` |
| `train_conformer_IRHP_tedlium.py` | `irhp/conformer/train_tedlium.py` |
| `pruning_utils.py` | `irhp/conformer/pruning.py` |
| Original LibriSpeech/TED-LIUM dataset loaders | `data_scripts/` |
| Full original dependencies | `docs/provenance/original-requirements.txt` |
| Baseline scripts, alternate implementations, notebooks | Original Git snapshot |

The model-selection, export, and upstream compatibility changes are documented in [reproducibility](reproducibility.md). The Conformer sources are retained for recovery of the original environment, not advertised as independently validated.

To inspect an original recipe:

```bash
git show b837575:scripts/train_libri100_redundancybased.sh
git show b837575:train_IRHP_libri.py
```

For an existing clone:

```bash
git remote set-url origin https://github.com/qwer55252/IRHP.git
```
