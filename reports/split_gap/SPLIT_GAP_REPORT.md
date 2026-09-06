# Random vs Scaffold vs Family Split Diagnostic

Generated: 2026-08-29T01:15:07.725318+00:00  
Frozen run commit: `86b372c2b7644acc9f93f2d4f206286127b13bc5`

All schemes score the same frozen 750-molecule panel with identical model hyperparameters and endpoints. Only grouping into five folds changes. The family result is read from its previously frozen artifact, not rerun or selected post hoc.

## Split construction

| Scheme | Fold sizes | Groups | Original families spanning >1 fold |
|---|---|---:|---:|
| random | {'0': 150, '1': 150, '2': 150, '3': 150, '4': 150} | 750 | 75 |
| murcko_scaffold | {'0': 150, '1': 150, '2': 150, '3': 150, '4': 150} | 650 | 75 |
| family | {'0': 150, '1': 150, '2': 150, '3': 150, '4': 150} | 75 | 0 |

## Pooled primary scores

| Scheme | Model | MA-ST-RAE ↓ | CYP3A4 MCC ↑ | CYP2D6 MCC ↑ |
|---|---|---:|---:|---:|
| random | endpoint_mean | 1.0774 | 0.0000 | 0.0000 |
| random | endpoint_median | 1.0363 | 0.0000 | 0.0000 |
| random | ecfp4_knn | 0.9926 | 0.3521 | 0.1798 |
| random | ecfp4_random_forest | 0.8432 | 0.4055 | 0.0858 |
| random | ecfp4_lightgbm | 0.7829 | 0.4030 | 0.2259 |
| murcko_scaffold | endpoint_mean | 1.0769 | 0.0000 | 0.0000 |
| murcko_scaffold | endpoint_median | 1.0354 | 0.0000 | 0.0000 |
| murcko_scaffold | ecfp4_knn | 0.9940 | 0.2824 | 0.2183 |
| murcko_scaffold | ecfp4_random_forest | 0.8526 | 0.2864 | 0.0506 |
| murcko_scaffold | ecfp4_lightgbm | 0.7838 | 0.3392 | 0.2034 |
| family | endpoint_mean | 1.0778 | 0.0000 | 0.0000 |
| family | endpoint_median | 1.0367 | 0.0000 | 0.0000 |
| family | ecfp4_knn | 0.9925 | 0.2738 | 0.1235 |
| family | ecfp4_random_forest | 0.8635 | 0.2839 | -0.0673 |
| family | ecfp4_lightgbm | 0.7998 | 0.2271 | 0.1921 |

## Interpretation boundary

This is a validation-regime diagnostic, not a model-selection tournament. Random splitting intentionally allows members of an original analog family to appear in other folds; family splitting forbids that. Any apparent random-split gain is evidence of optimism from related-series exposure, not evidence that the underlying model improved.
