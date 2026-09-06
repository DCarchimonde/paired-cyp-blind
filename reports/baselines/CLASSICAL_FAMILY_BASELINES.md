# Day 3–5 Classical Baseline Report

Generated: 2026-08-29T01:03:15.562494+00:00  
Protocol commit: `0d259e189298aefba63ab965610e70b3998fb070`

These are challenge-mimetic family-fold results. Lower MA-ST-RAE is better; higher MCC is better. No primary-model claim is authorized by this report.

## Pooled primary endpoints

| Model | MA-ST-RAE ↓ | CYP1A2 ST-RAE | CYP2C9 ST-RAE | CYP2D6 ST-RAE | CYP3A4 ST-RAE | CYP3A4 MCC ↑ | CYP2D6 MCC ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| endpoint_mean | 1.0778 | 1.0127 | 1.0979 | 1.0237 | 1.1768 | 0.0000 | 0.0000 |
| endpoint_median | 1.0367 | 0.9700 | 1.0613 | 1.0426 | 1.0728 | 0.0000 | 0.0000 |
| ecfp4_knn | 0.9925 | 1.0256 | 1.0761 | 1.0139 | 0.8544 | 0.2738 | 0.1235 |
| ecfp4_random_forest | 0.8635 | 0.7958 | 0.9068 | 0.8761 | 0.8755 | 0.2839 | -0.0673 |
| ecfp4_lightgbm | 0.7998 | 0.8046 | 0.8241 | 0.8940 | 0.6766 | 0.2271 | 0.1921 |

## Runtime

| Model | Fits | Wall seconds |
|---|---:|---:|
| endpoint_mean | 30 | 0.0 |
| endpoint_median | 30 | 0.0 |
| ecfp4_knn | 30 | 0.6 |
| ecfp4_random_forest | 30 | 32.5 |
| ecfp4_lightgbm | 30 | 16.5 |

## Interpretation boundary

This completes only the frozen classical subset. Chemprop single-task/masked-multitask baselines and random-versus-scaffold split-gap diagnostics remain pending. The Day 13–14 GO/STOP gates compare the later latent model against the strongest completed fair baseline, not against a convenient weak model.
