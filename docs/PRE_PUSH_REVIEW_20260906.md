# Pre-push review and replacement neural protocol

Date: 2026-09-06. New tag: `neural-protocol-hardened-20260906`.
Previous tag: `neural-protocol-frozen-20260829` (preserved, superseded).
This is an engineering and baseline-protocol review, not a paper-readiness,
novelty, clinical, or journal-acceptance certification.

## Findings and corrections

| Finding | Evidence and correction |
|---|---|
| Missing-label class-balance error | Installed Chemprop 2.3.1 `chemprop/data/samplers.py` computes `Y.any(1)`, where NaN is truthy. `data/dataloader.py` documents class balance as single-task only. Of 4,822 eligible rows in the frozen two-endpoint TDI training table, 4,563 are partially missing and 3,632 have no observed positive but are treated as active by that sampler. Both ST and MT now use uniform seeded sampling and unweighted, missing-label-masked BCE. |
| Resume manifest drift | Re-preparing identical data used to change `generated_utc`, hence every recorded manifest hash. Preparation now verifies and reuses the exact existing manifest and CSVs. Changed inputs or missing provenance stop instead of rewriting completed-run evidence. |
| Partial/stale completion accepted | A previous marker required only a matching prediction hash. Schema v2 requires identity, tagged clean revision, locked dependency and GPU evidence, all three input files, model weights, log, raw Chemprop predictions, and normalized prediction parity. |
| CPU production escape | Individual `run-job` could produce formal markers without `--require-gpu`. It now refuses; CPU smoke tests remain diagnostic only. |
| Incorrect source attribution | Production now records starting provenance and verifies unchanged code/data at completion. Marker publication is atomic. Per-job and whole-workflow locks prevent duplicate writers. |
| Incomplete result audit | Audit now requires exact artifact sets, verifies aggregate predictions against job tables and task mapping, and rejects NaN/Inf metric substitutions and duplicate summary keys. Official metric reconstruction remains independent of the production summarizer. |
| Runtime dirties source tree | New outputs, cache, environment, status, and partial downloads are ignored. They no longer make a successful run fail its own later clean-worktree guard. Old results remain in their old namespace. |
| Terminal/session fragility | A background launcher, lock, append-only console log, exit status, pinned project-local uv bootstrap, and explicit `uv run --no-sync` provide a two-command handoff. |
| Source/structure isolation | Raw training and family split hashes are pinned in the active config. Independent input audit now additionally verifies pairwise disjoint standardized connectivity keys across train/validation/test. |

The sampling correction is a pre-production scientific protocol amendment,
not a post-score optimization. No production neural result was examined to
choose it. The folds, molecule standardization, targets, loss families,
architecture, optimizer schedule, seeds, validation rule, and fixed threshold
remain as previously specified. There is no claim that unweighted BCE is the
strongest possible TDI baseline. Its observed prevalence/calibration and
per-fold MCC must be reported even if poor. Old class-balanced and new
unweighted neural results must never be pooled.

Upstream implementation provenance: installed wheel `chemprop==2.3.1`,
locked by `uv.lock`; source paths `chemprop/data/samplers.py` and
`chemprop/data/dataloader.py`. No installed upstream package was patched.

## Verification record

| Check performed on the reviewed revision | Result |
|---|---|
| Full pytest suite | 46 passed; includes resumed and newly completed synthetic jobs, stale/CPU/tampered artifacts, NaN audit rejection, input-change-during-fit rejection, missing jobs, and launcher fail-fast/locking checks |
| Neural input audit, real frozen training data | 17/17 critical checks PASS; 40 specifications, 200 planned jobs, 15,340 expected predictions; connectivity-disjoint train/validation/test |
| Repeated real-data preparation | Byte-identical manifest; SHA-256 `bc67d8decfeb3bad10ee8cade15b01b1fe4de85710598ab2e9123aa2371ea655` unchanged on resume |
| Synthetic Chemprop smoke | Regression and classification PASS on CPU |
| Real-data Chemprop integration smoke | Three epochs on CPU, direct MT: 138 rows / 4 outputs; TDI MT: 117 rows / 2 outputs; both PASS, finite and correctly ordered outputs |
| Historical Day 1–2 audit rerun in an isolated copy | 43 PASS / 1 non-critical WARN |
| Historical classical result audit rerun in an isolated copy | 13/13 PASS; 7,670 predictions |
| Historical split-gap audit rerun in an isolated copy | 11/11 PASS; 15,340 predictions |
| Preservation check | No diff in historical split CSVs, classical/split-gap results, or Day 1–2 reports |
| Launcher bootstrap | Project-local pip target installation of uv 0.11.33 verified; no base-environment replacement |
| Static checks | Shell syntax and changed-Python-file lint checks PASS; `git diff --check` clean |

The real-data integration smoke is limited to three epochs, discards model
weights and predictions, and generates no manuscript performance metric.
Synthetic tests use explicitly synthetic GPU metadata solely to test
validators; they are not evidence of physical GPU execution.

## Preserved evidence and limits

All eight recovered historical commits and four old freeze/audit tags are
retained. Historical classical predictions and audit reports are not
overwritten by this revision. Raw source data is downloaded at a pinned
upstream commit and is not committed to GitHub. The full project is isolated
from ProteinMPNN and RACER-C.

The code review cannot establish unique scientific novelty or absence of all
bugs. The 200 production jobs still need a real RTX 4090 run and independent
final audit. The proposed assay-structured primary model, its ablations,
prospective external/blind validation, and manuscript remain future work.
Complete baseline execution is not complete paper execution.
