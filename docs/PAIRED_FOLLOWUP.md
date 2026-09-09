# Paired-assay follow-up v5

Protocol `paired-assay-followup-v5-20260909`. This is an exploratory follow-up to
the inspected v4 results, not a new blind test or an automatic model promotion.
The original full model remains `paired_std`; its weak comparison with
`independent_std` is retained. Selecting the `no_std` pair for expansion after
seeing v4 is explicitly an adaptive research decision.

## Run on the same completed AutoDL instance

```bash
cd /root/autodl-tmp/paired-cyp-blind-github &&
git pull --ff-only origin main &&
bash scripts/start_paired_followup_4090.sh &&
tail -n 60 -F .runtime/paired-followup.log
```

Keep the complete original experiment directory, including the large rolling
checkpoints and graph cache. A review ZIP alone does not contain the optimizer
states needed by this launcher. No packages are installed or upgraded.

Wait for:

```text
PASS: all 50 new jobs, 10 verified v4 jobs, diagnostics and independent result audit completed.
```

Then upload `CYP_paired_v5_review.zip` from the repository root. Status is in
`.runtime/paired-followup-status.txt`. New runtime:
`.runtime/frozen-experiment/.runtime/paired-followup-v5/`.
The launcher shares the existing experiment lock, short temporary path and fixed
four-thread budget; the data loader uses zero workers. Ctrl+C in `tail` stops the
viewer; the background run continues while the instance remains running.
Use the same command to resume interrupted epochs and reuse verified completed
jobs. Do not pull updates during a run. A changed fingerprint stops and preserves
the existing work instead of silently mixing revisions.

## Why this intervention, and what the evidence does not establish

Before writing the new training protocol, all 20 v4 selected models were evaluated
on their own 5,845 training molecules and 150 internal-validation molecules, using
the uploaded weights and frozen inputs. No new outer predictions were used for
these diagnostics. Detailed tables are in
[`reports/paired_v4_training_diagnostics`](../reports/paired_v4_training_diagnostics/).
Values below are **unweighted means of five per-fold diagnostics**, not pooled
scores or independent five-seed estimates.

| Selected v4 paired model | 2D6 training AUROC | 2D6 validation AUROC | 2D6 training AP | 2D6 validation AP |
|---|---:|---:|---:|---:|
| `paired_no_std` | 0.7964 | 0.6663 | 0.5868 | 0.3792 |
| `paired_std` | 0.6655 | 0.6605 | 0.3689 | 0.3019 |

Both ranking and recall need attention. The `paired_std` fits average only 0.4
predicted-positive training rows per fold and zero validation positives at 0.5;
the reportability factor is near one for 2D6. This does not look like a broken
missing-label mask or a reportability gate suppressing every prediction.

On a fixed uniform subset of up to 256 training molecules per fold, the mean
shared-encoder gradient norm from 2D6 label BCE was 0.0798 for `paired_std`, versus
0.6699 from all other objective terms combined (0.1257 versus 0.7718 without
reported SD). These are local gradients at selected fitted weights, with dropout
off. They suggest testing objective balance; they **do not prove causal task
interference**. Gradient cosines are exported, including values that do not
support a consistent conflict hypothesis. An extra encoder is not yet justified.

For each selected paired model, a separate fixed-weight sensitivity calculation
added/removed only the training-median curve SD in the prediction integral. On
internal validation it changed no 0.5 classification decision for either enzyme.
For `paired_std`, removing this term changed probabilities by an average absolute
0.00245 for 2D6 and 0.000576 for 3A4. The different *trained* std/no_std results
therefore cannot be repaired simply by toggling this inference term. Training
weighting, residual noise and selection trajectories remain plausible contributors;
this calculation does not identify their separate causal effects.

## Registered matrix

| Conditions | Seeds | Frozen folds | New fits | Reused v4 fits |
|---|---|---:|---:|---:|
| `independent_no_std`, `paired_no_std` | 20260829–20260833 | 5 | 40 | 10 |
| `independent_no_std_2d6x2`, `paired_no_std_2d6x2` | 20260829 | 5 | 10 | 0 |

There are 60 analyzed jobs, 50 of them newly trained. All 20 original v4 jobs are
preserved, although only 10 enter the five-seed expansion. The 2d6x2 comparison is
a **one-seed pilot**, not a five-seed improved model. All seeds are reported; none
is selected for a best-seed headline. There is still one frozen five-fold family
assignment, not five independently generated splits.

New replication fits invoke the unchanged v4 `fit` function with the original
configuration and an explicit new seed. The ten modified fits use the identical
forward model, active parameters, input features, initialization, task masks,
sampling, Adam settings, maximum 100 epochs and patience 15. Only the scalar
coefficient on released-label CYP2D6 BCE changes.

Writing each endpoint's mean label BCE as B3A4 and B2D6:

```text
original classification objective = 0.5 * B3A4 + 0.5 * B2D6
modified classification objective = 0.5 * B3A4 + 1.0 * B2D6
```

The continuous likelihood and 0.2 × reportability objective are unchanged. Both
positive and negative 2D6 observations receive the same endpoint multiplier;
this is not positive-class weighting, oversampling, threshold adjustment or an
independent 2D6 output head. The coefficient is fixed at two before new outer
results exist; it was not selected from a sweep of multipliers. Since derived
labels and activities are dependent, this remains a composite objective.

The original forward probability, learned residual SD, left censoring, strict
official label rule and training-median reference-SD convention are inherited.
`no_std` removes reported curve SD only; it retains learned uncertainty and
censoring. No MSE, continuous-amplitude or Bayesian uncertainty claim is added.

## Selection and interpretation

Every condition retains the original internal-validation minimum:

```text
mean_direct_MAE_divided_by_training_SD + mean_TDI_BCE
```

Validation BCE is **not** reweighted for the new conditions. The threshold stays
0.5, selection ties retain the earliest epoch, and outer labels do not select
weights, thresholds or stopping time. Actual stopping epochs may change because
the training trajectories change; maximum budget and stopping rule do not.

Modified fits additionally record each enzyme's validation MAE, each TDI
endpoint's BCE, AP, AUROC, MCC and predicted-positive count each epoch. These are
diagnostics, not alternate checkpoint selectors. This makes a later decision
about selection rules possible without confusing the selected and final weights.

The questions are fixed before this run:

1. Does paired/no_std improve 3A4 compared with its matched independent control
   across all five seeds, and separately across the four additional seeds?
2. Does doubling 2D6 label BCE improve 2D6 at the fixed threshold within each
   architecture, and what happens to direct regression and 3A4?
3. Under the same increased 2D6 weight, does the paired architecture add value
   compared with the independent control?

All three primary endpoints (macro direct ST-RAE, 3A4 MCC, 2D6 MCC) accompany every
contrast. AP, AUROC, Brier, BCE and full confusion counts are also exported. A
recall increase alone is insufficient if false positives or other endpoints
deteriorate. There is no automatic winner or claim that the intervention worked.

## Audit, provenance and review package

The launcher verifies the exact reviewed v4 registration
`baab86d4d7c70b845fcfb17fc3005fd2c9c4d2f5127b5bc107aa27793d1995a3`, every original
job's weights/rolling checkpoint/exports, frozen data, v3 outputs, source and
environment. Data remain the same frozen TRAIN_TDI universe and family splits.
Direct-only extra data, real blinded structures and external measurements are
excluded. Old model source and old results are not overwritten.

It executes actual CPU self-tests and then four real 300-unit CUDA smoke fits on
a deterministic training subset plus internal validation. Both unchanged and
weighted variants must survive save/reload and one-epoch interruption followed
by bitwise-equivalent GPU continuation. CPU success alone does not qualify GPU
training. A failed gate prevents production from starting.

Each new job is audited using actual selected-weight inference, independently
recomputed validation selection, raw row/mask/label/CI parity, fixed classification
threshold and adaptive SciPy probability integration. Final counts are 18,408
scored prediction rows, 504 official metric rows and 144 TDI diagnostic rows.
Every condition/seed must cover the same original observed rows. Seed predictions
are **not** pooled as extra independent molecules.

`seed_summary.csv` reports mean and sample SD across seeds; SD is missing for the
one-seed pilots, rather than reported as zero. `contrasts_per_seed.csv` contains
all 24 endpoint/seed contrasts. `contrast_summary.csv` contains 15 summaries,
including a separate additional-four-seed replication view.

The 5,000 bootstrap draws resample whole families within each outer fold. The same
family draw applies to both models and all selected seeds. Official regression
denominators are recomputed after resampling; metric differences are calculated
per seed and then averaged. These are descriptive intervals conditional on
already fitted weights. They do not reproduce training uncertainty, repair
adaptive model selection or constitute fresh blind validation. No p-value,
multiplicity-adjusted discovery, significance or publication claim is emitted.

The ZIP includes 50 new selected models and all 20 original v4 selected models,
histories, individual validation/outer tables, latent outputs, code/config,
training/validation diagnostics, original full-model comparisons and audit/hash
records. The large rolling optimizer/RNG checkpoints and graph tensors remain on
the instance; hashes are recorded. They are deliberately omitted from the ZIP.
`EXPORT.json` records the completed ZIP hash and location. Archives are written
atomically and existing archives are retained with distinct names.

## Development qualification

The local development environment has Torch 2.5.1+cpu and RDKit 2025.9.6, with no
CUDA device. CPU checks do not claim a completed scientific v5 run. The launcher
performs the actual 4090 checks on the user's instance before production.

The development tests exercise actual fitting and serialization, analytic loss
and gradient effects, exact multiplier-one equivalence to original v4 training,
weighted-fit restart, a full synthetic 20-old + 50-new fit pipeline, 504-row
multi-seed metrics, ZIP contents, preservation/reuse and corruption/device gates.
Bootstrap matrix arithmetic is cross-checked against literal repeated-family
rows and sklearn/the frozen official scorer. These fixtures are marked
engineering-only and cannot be finalized as production GPU results.

Completed locally on 2026-09-09: the full synthetic integration test passed, with
all 70 actual fits, 50 new and 20 preserved weight files in the review archive,
all registered summary counts and the corruption/device guards. In addition,
each of the four production-width conditions completed three real-data CPU epochs
on the same 106-molecule training-only smoke subset, plus an interrupted/restarted
three-epoch fit, with 150 internal-validation molecules. Final weights were
bitwise equal across continuous/restarted fits. The maximum selected-model
probability discrepancy against independent adaptive integration was below
5.6e-8. No outer scoring or scientific v5 performance estimate was generated by
this qualification.
