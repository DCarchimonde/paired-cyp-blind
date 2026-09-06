# Chemprop Neural-Baseline Protocol Freeze

**Historical record, superseded on 2026-09-06.** The old tag and this record
are retained for provenance. Do not launch the old production version.
The active protocol uses unweighted masked BCE without class-balance
resampling in both TDI modes, and the tag
`neural-protocol-hardened-20260906`. See `PRE_PUSH_REVIEW_20260906.md`.

Date: 2026-08-29
Required production tag: `neural-protocol-frozen-20260829`

## Decision

The Chemprop single-task and ordinary masked-multitask baseline protocol is
ready to freeze. This is an execution-readiness decision, not a model-result or
paper-quality decision. The production jobs have not been run in the current
CPU-only environment.

## Frozen comparison

Both neural baselines use the same standardized molecules, family folds,
inner-validation rule, D-MPNN architecture, optimization schedule, task loss,
and five random seeds. The only scientific difference is task sharing.

| Item | Single-task | Masked multitask |
|---|---:|---:|
| Direct models per outer fold | 4 | 1 with 4 targets |
| TDI models per outer fold | 2 | 1 with 2 targets |
| Outer folds | 5 | 5 |
| Seeds | 5 | 5 |
| Jobs | 150 | 50 |

Total workload: 40 data specifications, 200 jobs, and exactly 15,340 retained
out-of-fold prediction rows. For outer test fold
`k`, validation is family fold `(k + 1) mod 5`; all other eligible rows are
training rows. Missing multitask targets remain missing and are masked by
Chemprop. The outer test partition is supplied only for post-fit prediction,
not for early stopping.

## Frozen implementation

- Python 3.12.13; Chemprop 2.3.1; PyTorch 2.5.1 compiled for CUDA 12.4.
- D-MPNN message width 300, depth 3, mean aggregation, ReLU, dropout 0.1.
- One-layer FFN with width 300; V2 atom featurizer.
- Batch size 64; up to 100 epochs; patience 15; two warmup epochs.
- Learning rates: initial `1e-4`, maximum `1e-3`, final `1e-4`.
- Regression: MSE loss, validation tracking by MAE.
- Classification: class-balanced BCE, validation tracking by PRC.
- Final TDI class threshold: fixed at 0.5; it is not tuned on outer folds.
- Seeds: 20260829 through 20260833.

The official Chemprop CLI parser in 2.3.1 accepts `binary-mcc` as a reported
metric but not as the tracking-metric token. PRC is therefore the frozen
early-stopping metric; final MCC is recomputed independently at the fixed 0.5
threshold. The installed package is not patched.

## Evidence completed before freeze

- Full repository test suite passed, including deterministic gzip output and
  an end-to-end synthetic 40-job collection/result-audit test.
- Synthetic Chemprop regression and classification smoke tests passed.
- Real prepared-data masked-multitask smoke tests passed for direct inhibition
  (138 test rows, four outputs) and TDI (117 rows, two outputs). They exercised
  missing labels, exact output order, finite outputs, and probability bounds.
- Temporary smoke predictions and model weights were destroyed; no smoke
  metric is a scientific result.
- Independent input audit passed all 16 critical checks: exact 40-spec matrix,
  exact 200-job matrix, file hashes, column schemas, ordered membership,
  disjoint partitions, standardized SMILES, raw-label parity, manifest counts,
  and nondegenerate TDI train/validation classes.

Partition-size ranges across specifications are 1,219–4,644 training rows,
22–140 validation rows, and 22–140 test rows. The small endpoint-specific
folds are retained rather than hidden; five seeds and fold-level reporting are
mandatory.

## Production guards

Every formal job refuses to start unless all of the following hold:

1. CUDA is available on a device whose name contains `4090` and which reports
   at least 20 GiB VRAM.
2. Chemprop, PyTorch, and compiled CUDA versions exactly match the freeze.
3. Git is clean and HEAD carries the required protocol tag.
4. Raw training data, family split, preparation manifest, and every prepared
   CSV retain their recorded SHA-256 hashes.
5. No silent CPU fallback occurs.

Each completed job records its command, runtime, environment, Git commit/tag,
input/output hashes, retained best checkpoint, prediction file, and log. The
runner is resumable through immutable `COMPLETE.json` markers. Collection is
refused until all 200 jobs are present. A separate result auditor then
reconstructs exact coverage and metrics from raw labels and verifies every job
artifact and GPU/provenance record.

## Boundary

Neural Day 3–5 is still pending until the tagged 4090 run, collection, and
independent result audit pass. Nothing in this freeze authorizes the primary
assay-structured model, a GO decision, a leaderboard claim, a Nature-family
claim, or a flagship-paper claim.
