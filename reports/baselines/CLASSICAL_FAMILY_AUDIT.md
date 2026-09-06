# Classical Family-Baseline Audit

**Overall status: PASS**  
Generated: 2026-08-29T01:07:47.678263+00:00

## Audited pooled results

| Endpoint | Best audited model | Score | Direction |
|---|---|---:|---|
| Direct MA-ST-RAE | ecfp4_lightgbm | 0.7998 | lower is better |
| CYP3A4 MCC | ecfp4_random_forest | 0.2839 | higher is better |
| CYP2D6 MCC | ecfp4_lightgbm | 0.1921 | higher is better |

## Fold-level TDI warning

The pooled MCC values are valid, but small fold-level positive counts make TDI estimates unstable, especially CYP2D6. This is disclosed now and must be handled with family-level uncertainty later.

| Isoform | Fold positive counts | Best-model fold MCC values |
|---|---|---|
| CYP3A4 | {'0': 25, '1': 28, '2': 14, '3': 36, '4': 30} | {'0': 0.5181426079664405, '1': 0.1090460762305419, '2': 0.2511479767507418, '3': 0.2094197544705749, '4': 0.2629273696056835} |
| CYP2D6 | {'0': 7, '1': 7, '2': 5, '3': 5, '4': 5} | {'0': 0.0582104127717333, '1': -0.1796053020267749, '2': 0.4274373669939289, '3': 0.3715752707301467, '4': 0.3273268353539886} |

## Checks

| Status | Critical | Check | Detail |
|---|---|---|---|
| PASS | yes | `manifest.complete_classical_set` | complete_classical_set=True |
| PASS | yes | `manifest.models_exact` | observed=['endpoint_mean', 'endpoint_median', 'ecfp4_knn', 'ecfp4_random_forest', 'ecfp4_lightgbm'], expected=['endpoint_mean', 'endpoint_median', 'ecfp4_knn', 'ecfp4_random_forest', 'ecfp4_lightgbm'] |
| PASS | yes | `manifest.clean_start` | git_status_at_start='', dirty_override=False |
| PASS | yes | `manifest.hashes` | hash mismatches=[] |
| PASS | yes | `predictions.unique_keys` | duplicate model/molecule/endpoint rows=0 |
| PASS | yes | `predictions.coverage_exact` | observed=7670, expected=7670, missing=0, extra=0 |
| PASS | yes | `predictions.fold_mapping` | fold mismatches=0 |
| PASS | yes | `predictions.family_mapping` | family mismatches=0 |
| PASS | yes | `predictions.finite_truth_and_prediction` | nonfinite cells=0 |
| PASS | yes | `predictions.classification_domain` | predicted classes=[0.0, 1.0], probability range=0.000000-0.941377 |
| PASS | yes | `predictions.regression_bounds` | nonfinite bound cells=0, inverted rows=0 |
| PASS | yes | `predictions.train_counts_exclude_test_fold` | fold/endpoint groups with unexpected n_train=0 |
| PASS | yes | `metrics.official_parity` | maximum endpoint difference=8.882e-16, macro mismatches=0 |

## Decision boundary

This PASS freezes the classical baseline evidence only. It does not complete Day 3–5, authorize the primary latent model, or establish flagship status. Neural baselines and random/scaffold split-gap diagnostics remain mandatory.
