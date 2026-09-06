# RTX 4090 neural baseline runbook — active release

Required execution tag: `neural-protocol-hardened-20260906` inside the
automatically restored frozen checkout. The GitHub delivery commit has its own
identity; see `GITHUB_DELIVERY.md` for exact source/archive provenance.
Scope: 200 single-task/masked-multitask baseline jobs, collection, and audit.
The assay-structured primary model and blind submission are not implemented.

## Machine and disk

Use Linux/AutoDL, one RTX 4090-class card (name contains `4090`, compute
capability 8.9, at least 20 GiB visible VRAM), NVIDIA driver compatible with
the locked CUDA 12.4 PyTorch build, and at least 30 GiB free for a fresh
install. An expandable data disk is preferable. The installer uses the data
disk for its project environment, cache, downloaded Python, and temporary
files; it does not modify another project's environment.

The image's preinstalled PyTorch version is not the experiment dependency:
the isolated environment pins Python 3.12.13, Chemprop 2.3.1, and torch 2.5.1
with compiled CUDA 12.4. `python3`, pip, git, bash, flock, and nohup must be
available. The script can install uv 0.11.33 into a project-local directory.
Dependency/network failures stop the run; there is no CPU fallback.

## Two copy-paste commands

First use clones into a separate GitHub delivery directory; subsequent uses
fast-forward that directory. Existing local bundle clones remain separate.
Do not run this update while training is active. Conflicting local files are
not discarded and unrelated Git history is not force-replaced.

```bash
cd /root/autodl-tmp && if [ -d paired-cyp-blind-github/.git ]; then git -C paired-cyp-blind-github pull --ff-only origin main; else git clone https://github.com/DCarchimonde/paired-cyp-blind.git paired-cyp-blind-github; fi
```

```bash
cd /root/autodl-tmp/paired-cyp-blind-github && bash scripts/start_reviewed_neural_baselines_4090.sh
```

The entrypoint verifies the bundled freeze and restores the original commit
and tags under `.runtime/frozen-experiment`. The original launcher then uses
`nohup` and a per-directory process lock. It appends to
`.runtime/neural-run.log`, records the PID in `.runtime/runner.pid`, and
writes `.runtime/run-status.txt`. Opening those files does not need another
setup command. A second invocation during a run reports that it is active
and does not start another training process.

## Stages and recovery

The workflow installs locked dependencies, verifies the upstream file hashes,
runs the test suite, checks GPU/dependency/tag/clean-worktree provenance,
runs synthetic smoke tests, prepares or verifies 40 data specifications,
independently audits the inputs, executes the 200-job matrix, collects all
15,340 predictions, and independently recomputes the official metrics.

Re-run the second command after a stopped or failed run. Prepared manifest
bytes remain unchanged. A completed job is reused only if its exact job
identity, source revision, dependency/GPU metadata, all input/output hashes,
checkpoint, log, and original Chemprop prediction table are valid. An
unfinished job starts a new attempt; partial attempts are preserved, not
promoted to complete. Corrupt or cross-version evidence causes a stop rather
than automatic replacement. The old `reports/neural/` namespace is not used.

Do not tune epochs, seeds, folds, architecture, batch size, early stopping,
sampling, or the 0.5 TDI threshold after viewing results. Any justified change
requires another documented protocol, tag, and separate output namespace.

## What counts as complete

All 200 jobs must be validated and the final audit must report
`overall_pass: true`. Expected evidence under `reports/neural_v2/`:

- `neural_family_predictions.csv.gz`
- `neural_family_metrics.csv`
- `neural_family_seed_summary.csv`
- `neural_family_manifest.json`
- `NEURAL_RESULT_AUDIT.md` and `neural_result_audit.json`
- All `jobs/` logs, original predictions, best checkpoints, and completion manifests

The short log/status/result paths in the delivery checkout are symlinks to
the actual files inside `.runtime/frozen-experiment`. Preserve the real
`.runtime/frozen-experiment/reports/neural_v2/` directory, not just its symlink.
For any manual experiment CLI command, first enter the frozen checkout.

The final wrapper status then says PASS. Preserve this whole directory and
the tag before releasing the instance. Results are not pushed to GitHub
automatically. A terminal disconnect is tolerated; machine shutdown ends
the process and loss of the data disk loses local results.

GPU wall time and total rental cost are unmeasured. The previous 50–100-hour
estimate was a planning allowance, not a benchmark or guaranteed upper bound.
This release has CPU integration checks, not a completed RTX 4090 run.
