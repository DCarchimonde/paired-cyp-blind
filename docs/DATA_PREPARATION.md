# Data preparation and AutoDL GPU mode

The reported dependency recovery finished successfully: nine remaining large
packages were installed, all 77 installed packages passed the environment check,
and the original workflow resumed. The subsequent `Errno 99: Cannot assign
requested address` occurred inside `socket.connect()` while fetching the
Hugging Face CSV. It is a network connection failure; the traceback alone does
not establish whether the underlying cause is address selection, routing, a
proxy, or temporary network state. It is not a CUDA error.

AutoDL's [official documentation](https://www.autodl.com/docs/save_money/) allows
environment preparation and data transfer in GPU-free mode, followed by normal
GPU startup on the same instance. This project's formal baseline jobs require
the frozen RTX 4090 configuration. GPU-free mode does not run those jobs.

## Step 1: Update and prepare data in the current GPU-free session

```bash
cd /root/autodl-tmp/paired-cyp-blind-github && git pull --ff-only origin main && bash scripts/prepare_official_data.sh
```

Wait for `OFFICIAL DATA READY: 5/5 original hashes verified`.
The repository includes a compressed copy of the same five public files, with
their upstream attribution and license. This step uses the archive, validates the
original manifest and per-file hashes, and does not install dependencies or use
a GPU. Existing valid files are reused. Existing mismatched files are preserved
and cause a clear error. An active workflow is not interrupted.

## Step 2: Restart this instance normally with RTX 4090 and resume

After data preparation finishes, shut down the same instance in the AutoDL
console and select normal startup with its RTX 4090. Do not release/delete or
reset the instance. The files and environment on its data disk are retained.
Then open the terminal and run:

```bash
cd /root/autodl-tmp/paired-cyp-blind-github && UV_OFFLINE=1 bash scripts/start_reviewed_neural_baselines_4090.sh && tail -n 40 -F .runtime/neural-run.log
```

This command assumes the reported dependency recovery already passed. uv uses
the installed frozen environment offline, and the original data-fetch script
verifies the restored files locally. All original regression tests, GPU checks,
200 baseline jobs and independent audits still run. Ctrl+C exits the log viewer
while the background workflow continues.

The regular reviewed foreground/background launchers now also restore the
verified local data before entering the unchanged frozen workflow. The original
`fetch_official.py`, experimental bundle, configuration, split and labels are not
modified. Data preparation is not an experimental completion claim.

Validation includes all five actual CSV hashes, restoration/reuse without file
rewrites, rejection of conflicting files and modified manifests/archives,
workflow locking, and the original fetcher succeeding with network calls
explicitly disabled in its test. Full GPU training remains a separate execution
on the user's RTX 4090 instance.
