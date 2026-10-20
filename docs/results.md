# Published results

WER (%) transcribed from the ICASSP 2026 camera-ready paper. Lower is better. These are published reference values; no trained IRHP checkpoints are distributed and no paper-scale rerun is claimed.

## Table 1: Wav2Vec 2.0 on LibriSpeech

| Method | Training set | dev-clean | dev-other | test-clean | test-other |
| --- | --- | ---: | ---: | ---: | ---: |
| Baseline | clean-100 | 5.34 | 15.58 | 5.55 | 15.32 |
| Baseline | clean-460 | 3.84 | 12.54 | 4.05 | 12.37 |
| One-shot redundant head pruning | clean-100 | 8.92 | 25.96 | 9.33 | 26.83 |
| One-shot redundant head pruning | clean-460 | 7.40 | 21.53 | 7.56 | 21.58 |
| Iterative L0-based | clean-100 | 8.24 | 25.37 | 9.03 | 26.45 |
| Iterative L0-based | clean-460 | 4.85 | 16.65 | 5.24 | 17.17 |
| Iterative magnitude-based | clean-100 | 7.56 | 23.00 | 7.93 | 23.81 |
| Iterative magnitude-based | clean-460 | 4.75 | 16.23 | 5.10 | 16.53 |
| IRHP | clean-100 | 7.80 | 23.05 | 8.21 | 23.67 |
| IRHP | clean-460 | 4.86 | 15.97 | 5.07 | 16.41 |

## Table 2: Wav2Vec 2.0 on TED-LIUM

| Method | dev | test |
| --- | ---: | ---: |
| Baseline | 15.37 | 18.32 |
| One-shot redundant head pruning | 21.07 | 23.48 |
| Iterative L0-based | 21.38 | 23.82 |
| Iterative magnitude-based | 21.47 | 24.65 |
| IRHP | 19.96 | 22.69 |

## Table 3: Conformer on LibriSpeech clean-100

| Method | Model | dev-clean | dev-other | test-clean | test-other |
| --- | --- | ---: | ---: | ---: | ---: |
| Baseline | Small | 5.64 | 14.60 | 5.63 | 14.36 |
| Baseline | Large | 4.33 | 11.80 | 4.44 | 11.63 |
| One-shot redundant head pruning | Small | 6.93 | 19.68 | 7.13 | 19.84 |
| One-shot redundant head pruning | Large | 6.40 | 20.16 | 6.46 | 20.24 |
| Iterative L0-based | Small | 6.45 | 17.21 | 6.55 | 17.00 |
| Iterative L0-based | Large | 5.58 | 15.56 | 5.73 | 15.69 |
| Iterative magnitude-based | Small | 6.17 | 15.34 | 6.41 | 15.08 |
| Iterative magnitude-based | Large | 4.95 | 13.60 | 5.23 | 13.41 |
| IRHP | Small | 5.97 | 15.40 | 6.06 | 15.03 |
| IRHP | Large | 4.89 | 13.44 | 4.99 | 13.24 |

## Table 4: IRHP component ablation

The rows progressively add components; the first row does not represent the complete recipe.

| Components | dev-clean | dev-other | test-clean | test-other |
| --- | ---: | ---: | ---: | ---: |
| Initial pruning setup | 10.23 | 30.23 | 11.01 | 31.57 |
| + Retain low-redundancy heads | 8.63 | 25.24 | 8.91 | 26.16 |
| + More post-pruning fine-tuning | 8.10 | 24.81 | 8.53 | 25.68 |
| + Parameter re-initialization | 7.80 | 23.05 | 8.21 | 23.67 |

Figure 2's exact numerical points are not supplied as machine-readable data; they are not invented here. Run the ratio sweep with explicit configurations to produce new measurements.
