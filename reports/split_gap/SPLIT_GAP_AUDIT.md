# Split-Gap Audit

**Overall status: PASS**  
Generated: 2026-08-29T01:18:27.575235+00:00

## Optimism gaps relative to frozen family folds

Positive values mean the alternative split looks better than the family split. For direct inhibition this is `family MA-ST-RAE − alternative MA-ST-RAE`; for TDI it is `alternative MCC − family MCC`.

| Alternative | Model | Direct gap | CYP3A4 MCC gap | CYP2D6 MCC gap |
|---|---|---:|---:|---:|
| random | ecfp4_knn | -0.0002 | +0.0783 | +0.0564 |
| random | ecfp4_random_forest | +0.0203 | +0.1216 | +0.1531 |
| random | ecfp4_lightgbm | +0.0170 | +0.1759 | +0.0338 |
| murcko_scaffold | ecfp4_knn | -0.0015 | +0.0086 | +0.0948 |
| murcko_scaffold | ecfp4_random_forest | +0.0109 | +0.0024 | +0.1180 |
| murcko_scaffold | ecfp4_lightgbm | +0.0160 | +0.1122 | +0.0113 |

## What the diagnostic supports

- Direct-inhibition optimism is modest and not universal: RF/LightGBM improve by about 0.016–0.020 MA-ST-RAE under random/scaffold grouping, while kNN is essentially unchanged or slightly worse.
- Random splitting inflates TDI MCC for all three structural models. The CYP3A4 gap is +0.078 to +0.176; this supports keeping family folds primary.
- CYP2D6 gaps are positive but remain fragile because each family fold has only 5–7 positives and the family-fold baseline itself changes sign across folds.
- The fixed-panel scaffold diagnostic is not equivalent to strict scaffold exclusion over the full 6,145-molecule universe: all 75 original analog families still span folds. It is secondary evidence only.

## Checks

| Status | Critical | Check | Detail |
|---|---|---|---|
| PASS | yes | `manifest.clean_start` | git_status_at_start='' |
| PASS | yes | `manifest.scheme_and_model_set` | schemes=['random', 'murcko_scaffold'], models=['endpoint_mean', 'endpoint_median', 'ecfp4_knn', 'ecfp4_random_forest', 'ecfp4_lightgbm'] |
| PASS | yes | `manifest.hashes` | hash mismatches=[] |
| PASS | yes | `predictions.unique_keys` | duplicate scheme/model/molecule/endpoint rows=0 |
| PASS | yes | `predictions.coverage_exact` | observed=15340, expected=15340, missing=0, extra=0 |
| PASS | yes | `predictions.split_mapping` | fold mapping errors=0, group mapping errors=0 |
| PASS | yes | `splits.fixed_panel_group_integrity` | split invariant errors=0 |
| PASS | yes | `predictions.numeric_domain` | nonfinite truth/prediction cells=0, out-of-range probabilities=0 |
| PASS | yes | `predictions.train_counts_exclude_test_fold` | scheme/fold/endpoint groups with unexpected n_train=0 |
| PASS | yes | `metrics.official_parity` | maximum endpoint difference=6.661e-16 |
| PASS | yes | `metrics.family_artifact_immutable` | embedded family metrics exactly equal frozen artifact |

## Decision boundary

The split-gap diagnostic is now complete. It does not establish a new model contribution; it only shows why the frozen analog-family validation remains primary. Neural baseline completion is still required before Day 3–5 can close.
