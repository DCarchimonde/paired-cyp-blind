# Paired-assay minimum prototype, v4

Protocol: `paired-assay-pilot-v4-20260908`. This is a prospective code/config
registration for a **post-baseline exploratory pilot**. The internal v2/v3 outer
results were already inspected. This is not a fresh blinded evaluation, a full
five-seed experiment, or evidence that the paper's main hypothesis succeeds.

## Run on the existing AutoDL instance

```bash
cd /root/autodl-tmp/paired-cyp-blind-github &&
git pull --ff-only origin main &&
bash scripts/start_paired_prototype_4090.sh &&
tail -n 60 -F .runtime/paired-prototype.log
```

The launcher reuses the completed frozen environment and the experiment lock.
It installs no packages. New artifacts live in
`.runtime/frozen-experiment/.runtime/paired-prototype-v4/`; v2/v3 remain intact.
The log reports each training epoch, then an audit phase. CPU self-tests may say
`device=cpu`; all four real-data smoke fits and all 20 pilot jobs require CUDA on
the RTX 4090. Graph preparation uses CPU and reports progress every 500 molecules.
The loader uses zero worker processes and a short temporary path.

Wait for `PASS: all 20 paired-assay pilot jobs and independent result audit completed.`
Then upload `CYP_paired_v4_review.zip` from the outer repository directory. It
includes selected model weights, training histories, validation/outer predictions,
latent outputs, configuration, code, audit, and descriptive comparisons. Large
optimizer checkpoints and graph tensors stay on the instance; their hashes are
recorded, but those two artifact types are not included in the review ZIP.

Check status with `cat .runtime/paired-prototype-status.txt`. Ctrl+C while following
the log stops the log viewer. The background process continues while the instance
stays running. If training is interrupted, use the same launcher; it resumes the
last atomic completed epoch with optimizer and random-number states. Completed
jobs are hash-checked and reused. A changed code/config/environment fingerprint
stops with existing artifacts preserved. Do not pull updates during a run.

## The four predeclared conditions

| Variant | Continuous observation model | TDI prediction |
|---|---|---|
| `independent_no_std` | Free D mean and three free T mixture locations; censored Gaussian mixture; learned residual SD | Independent conditional classifier × learned reportability |
| `independent_std` | Same; add the reported curve SD in quadrature | Same independent classifier |
| `paired_no_std` | D mean and signed zero/positive/negative shift mixture; censored Gaussian mixture; learned residual SD | Integrate the released rule over the paired observation distribution × reportability |
| `paired_std` | Same paired model; add the reported curve SD in quadrature | Same derived probability |

`paired_std` is the prototype of the intended primary model. Its primary matched
control is `independent_std`. The factorial contrasts also compare pairing without
reported SD, and reported SD within each architecture. All four have the same
custom 2D D-MPNN (300 hidden units, depth 3, mean pooling, dropout 0.1), all eight
continuous readouts, both TDI labels, reportability supervision, batch size, Adam
settings, seed, maximum epochs, loss weights, selection rule, and split. Active
parameter counts must differ by less than 1%; no dummy parameters are added.

The encoder implements directed bond messages excluding reverse edges, following
the [Chemprop D-MPNN construction](https://github.com/chemprop/chemprop/blob/v2.3.1/chemprop/nn/message_passing/base.py).
It has its own explicit feature vocabulary and is not numerically identical to
Chemprop V2. Causal attribution uses the matched v4 controls. Comparisons to v3
are descriptive because v3 used separate direct/TDI models and fewer continuous
auxiliary labels. Same-seed v3 scores are exported without selecting a best seed.

## Data and official label semantics

Only frozen `TRAIN_TDI.csv` is used: the same 6,145-compound universe as v2/v3,
with all available paired continuous arms now exposed to **every** v4 condition.
The larger direct-only table, Emax, single-dose data, external sources, and actual
blinded test structures are excluded. Raw file and split SHA256 values are fixed
in the configuration. Standardization reuses frozen RDKit code and version.

The existing 750-compound internal family panel has one five-fold assignment.
For outer fold f, validation is (f+1) modulo 5; the other panel folds and nonpanel
core train the model. These are **five folds, not five independently repeated
split designs**. The one seed is 20260829: four conditions × five folds = 20 jobs.
All readouts of a molecule stay in one partition. Connectivity identities cannot
cross core/panel/fold boundaries. No fitted graph preprocessing uses outer data.

The released rule is positive only when both numeric arms exist and
`T > max(D, 4) + log10(2)`. Numeric D below 4 uses the inferred-positive branch;
missing D is different: on an eligible released row it receives an assigned
negative label. An entirely missing TDI label is masked, never assigned negative.
The implementation reproduces every available raw label before training. These
semantics follow the [challenge assay description](https://openadmet.ghost.io/announcing-openadmets-cyp-inhibition-blind-challenge/)
and the frozen official label implementation already audited in v2/v3.

Inference must use SMILES only. Therefore an auxiliary reportability head predicts
the chance that both numeric arms are reported, conditional on TDI eligibility.
It is supervised only where a released TDI label exists, in all four models.
The released-label probability is `q_report(x) × p_positive_given_reported(x)`.
No observed missingness, true activity, per-curve SD, or assay label is fed at
inference. This factorization models an operational reporting outcome; reporting
is not itself a biochemical mechanism, and conditional independence from the
unobserved assay values is a pilot assumption that needs later sensitivity work.

## What the minimum probabilistic model actually represents

The paired model predicts D location d and three T locations
`[d, d + softplus(a), d - softplus(b)]` with softmax weights. It therefore permits
zero, positive and negative shifts. **Conditional shift magnitudes are point
masses in this prototype.** A continuous half-normal/Gamma amplitude distribution,
Bayesian parameter uncertainty, and a CYP2D6 expert branch remain unimplemented;
the prototype must not be described as the finished blueprint model.

Given a mixture state, the D and T observations have separate Gaussian errors.
Their learned residual SDs are bounded between 0.05 and 2 pIC50 units. Observed
numeric values below 4 contribute a left-censoring probability; values at or
above 4 contribute a density. Missing numeric arms contribute no loss. Treating
the floor this way is an observation-model assumption, not a claim that the
released extrapolated point estimates are exact measurements at 4.

For the `std` variants, observation variance is residual variance plus reported
curve variance. Confidence intervals are not also added as a second noise source.
The existing data contain a finite positive SD for every numeric readout; the
pilot fails rather than silently inventing a missing SD. For TDI probability
integration, a per-CYP/per-arm **training-fold median reported SD** is used at
both training and inference; per-row validation/test SD never enters prediction.
This is a pragmatic predictive-error approximation, not propagation of known
test-time curve uncertainty or an epistemic uncertainty estimate.

The `no_std` conditions remove the reported-SD contribution only. They retain
learned residual uncertainty and censoring, keeping this contrast interpretable.
It is **not** the blueprint's ordinary-MSE ablation, and it does not identify the
separate value of censoring. Uncensored/MSE sensitivity fits are deferred.

The paired classifier computes `P(T > max(D,4)+log10(2))` by a deterministic
bivariate Gaussian integration for each mixture component. A 64-node correlation
angle quadrature is differentiated in double precision. The auditor independently
integrates the direct-observation axis with SciPy adaptive quadrature, rather than
reusing the model's probability routine. Synthetic edge cases and all exported
paired validation/outer rows must agree within 0.0005 absolute probability.

The loss is an endpoint-normalized **composite objective**: mean censored
continuous NLL + mean released-label BCE + 0.2 × mean reportability BCE. The labels
derive from the continuous observations, so these terms are not independent
evidence and are not presented as a full generative likelihood. Training-count
normalization keeps task weights fixed despite sparse labels. Sampling is uniform;
there is no class weighting, threshold tuning, oversampling, or hyperparameter
search in this pilot.

## Selection, audit, and scientific interpretation

All four conditions minimize the same internal validation score:
mean direct MAE divided by training endpoint SD, plus mean TDI BCE. It does not
use outer labels, variant-specific NLL, PRC callbacks, or the eventual outer MCC.
Training SD has a fixed lower bound of 0.1 for this normalization. Selection ties
retain the earliest epoch. Maximum 100 epochs; patience 15; threshold 0.5.

Atomic rolling checkpoints contain the current model, selected model, optimizer,
history and RNG states in one transaction. `best.pt` explicitly records its epoch
and validation score. Saved-model replay must reproduce validation and outer
exports; the selected score and minimum epoch are independently checked against
history. Partial audit failure can be retried without discarding trained epochs.

The run requires a real CPU synthetic self-test (gradient finite differences,
masking, censoring, held-out-statistic isolation, actual training of every variant,
bitwise continuous-versus-resumed weights, CSV round trips, saved-model inference),
then a real three-epoch CUDA smoke run of all four variants using training/internal
validation data only. Each small fit is also interrupted after one completed epoch
and resumed, checking bitwise equality of final GPU weights against the continuous
fit; the restart check adds another three tiny epochs per variant. The GPU smoke
is an engineering gate; it does not select a
model or alter the protocol. No CPU fallback is accepted for production.

Final audit requires 20 jobs, 6,136 observed outer prediction rows, 168 official
metric rows and 48 TDI diagnostic rows; missing endpoints have no prediction rows
in the scored long table. An unscored latent table retains every panel molecule.
ST-RAE arithmetic is compared against the frozen official scorer and MCC against
an independent confusion-matrix calculation. Raw truth, confidence bounds, family
identity and same-seed v3 row coverage are checked. Summary shows every condition,
including poor results, and records TDI class counts, average precision (AP), ROC
AUC, Brier score, BCE, and confusion counts at the fixed threshold.

One-seed results cannot establish seed stability or a flagship-paper claim. A
useful direction in the matched contrasts motivates a separately frozen five-seed
expansion and family-level paired uncertainty analysis; weak TDI results or loss
of direct accuracy motivate revisiting the hypothesis. There is no automatic
model promotion, test submission, significance claim, or publication decision.

## Implementation qualification before delivery

Completed on 2026-09-08 with Python 3.12.13, PyTorch **2.5.1+cpu**, and RDKit
2025.9.6 in a separate development environment:

- The 160 random/edge-case mixture probability checks agreed with independent
  adaptive integration to a maximum absolute error of approximately 8.1e-13;
  finite-difference gradient checks passed.
- All four variants completed real synthetic training, serialization, saved-model
  replay and bitwise uninterrupted-versus-resumed final-weight checks. The paired
  fixtures selected an earlier epoch whose weights differed from the final epoch.
  Separate actual fits exercised early stopping when patience was exhausted.
- Two integration tests passed, including a **20-job synthetic CPU pipeline**
  using 80 toy molecules, two epochs and 160 hidden units. It exercised per-job
  completion, raw-row checks, 168 metric rows, ZIP verification and all 20 selected
  weight files; it rejected both a modified completed artifact and attempted
  production finalization of CPU jobs. These are engineering fixtures, not CYP
  performance results.
- The actual frozen 6,145-molecule file passed label reconstruction, graph
  standardization, connectivity and five-fold membership checks. Each fold has
  5,845 training, 150 validation and 150 outer-panel molecules. Every variant has
  1,534 scored outer rows. Actual active parameter counts are 313,234 (paired) and
  315,040 (independent), a difference below 0.6%.
- Each production-size 300-unit variant completed three CPU training epochs on a
  fixed 106-molecule training-only smoke subset. Its 150 internal-validation
  predictions passed saved-model and independent probability/selection checks.
  No outer performance was scored in this qualification run. Raw-row metadata
  for all 1,534 scored panel rows matched the uploaded v3 same-seed baseline.

The development machine had no CUDA device. **4090 training, CUDA-specific
numerics and GPU optimizer restart remain host qualification gates**, executed
by the launcher before production. CPU checks do not authorize a claim that a
GPU run or the scientific pilot has already passed.
