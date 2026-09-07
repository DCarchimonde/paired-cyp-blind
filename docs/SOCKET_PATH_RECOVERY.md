# Recover the first-job AF_UNIX pathname failure

2026-09-07. This is an execution-path correction to the delivery layer.

## Diagnosis

The user's first Chemprop job printed `OSError: AF_UNIX path too long` in
`multiprocessing.resource_sharer`, while the GPU was idle and the wrapper still
reported `RUNNING`. A queue feeder thread can fail while the parent continues
waiting for data; a live PID and the wrapper's status do not prove training progress.

The original script sets `TMPDIR` to the project's `.runtime/tmp`. In the AutoDL
delivery layout that directory is 80 bytes long. Python 3.12's usual
`/pymp-XXXXXXXX/listener-XXXXXXXX` suffix brings the socket address to 112 bytes.
Linux's pathname field is 108 bytes including a terminating null byte. See the
[Linux unix(7) manual](https://man7.org/linux/man-pages/man7/unix.7.html) and
[Python tempfile documentation](https://docs.python.org/3.12/library/tempfile.html).

Exporting a different TMPDIR before calling the old launcher is insufficient:
the frozen script assigns TMPDIR again. The reviewed launchers now invoke the
same frozen scripts through a private short symlink under `/tmp`. Bash retains
that logical path, giving Python a short literal TMPDIR. The link points to the
same data-disk checkout; actual cache, environment, data, temporary contents and
results remain on that disk. Source paths used by the scientific implementation
are still resolved to the canonical frozen root.

## Resume the affected AutoDL instance

Run these commands separately in the existing project's terminal:

```bash
cd /root/autodl-tmp/paired-cyp-blind-github && git pull --ff-only origin main
```

```bash
cd /root/autodl-tmp/paired-cyp-blind-github && bash scripts/recover_socket_error_4090.sh && tail -n 40 -F .runtime/neural-run.log
```

This particular outer-repository update is safe while the failed frozen attempt
is waiting: it does not modify the running experiment's source. The recovery
checks the recorded runner, its `run-all` ancestor chain, exact Chemprop output
directory, unfinished job, and the AF_UNIX error in that attempt's log. It stops
only that process and its verified children using stable PID handles, then waits
for the original run lock to be released. A healthy job, a completed job, an
unrelated process, stale identity or ambiguous process tree causes a refusal.
All old attempts, caches and environment files are retained; the unfinished job
starts a new attempt under the original protocol.

The first recovery release mistakenly invoked the system `python3` for process
control. The user then reported `Safe PID handles are unavailable`; that message
means the selected interpreter lacks one of the required Python APIs, and the
guard stopped recovery before any signal was sent. The corrected entrypoint uses
the already installed `.runtime/frozen-experiment/.venv/bin/python` explicitly.
It prints the interpreter/version and both API flags. All existing process
identity and PID-handle requirements remain enforced; there is no numeric-PID
or broad process-kill fallback. The system Python version was not supplied, so
the diagnosis does not assume a specific system version or a kernel failure.

The entrypoint checks a real AF_UNIX listener before launch. When torch is already
installed (as in this incident), it also transfers 64 ordered tensor values using
four DataLoader workers, under a 45-second outer timeout. It must print:

```text
IPC PREFLIGHT PASS: 4 workers transferred all 64 values in order.
```

The runtime probe writes `.runtime/frozen-experiment/.runtime/ipc-preflight.json`.
Targeted-stop evidence goes to that same runtime directory's `socket-recovery/`.
`Ctrl+C` ends the final log viewer; the background workflow continues. First-job
progress still appears in its own `attempt_*/train.log`. Completed-job transitions
appear as `[2/200]`, etc. Full success still requires all 200 jobs and the original
independent result audit, not just this IPC probe.

## Preservation and validation

- The original bundle, tag, source, dependency lock, data, folds, seeds, models,
  four-worker setting, epochs, batch size, TDI threshold and result validators
  are unchanged. The same alias correction also covers the download recovery's
  handoff back to the frozen workflow.
- Targeted standard-library regression suite: 14 passed, 1 skipped. Tests check
  the exact 112-byte old pathname, short literal Python TMPDIR, target/reuse and
  conflict handling, both actual frozen shell launchers with synthetic commands,
  failed-parent selection despite inherited worker command lines, healthy and
  completed job rejection, unrelated processes, PID reuse and foreign children.
  Interpreter tests also cover an incompatible system `python3`, a missing
  frozen interpreter, and an unavailable PID API without weakening the guard.
- The executor prohibits creating AF_UNIX sockets, so its real-listener test is
  explicitly skipped. It also virtualizes process IDs relative to `/proc` and
  has no installed PyTorch/4090. Real socket/tensor transfer and targeted process
  shutdown are therefore checked on the user's Linux instance; no local GPU
  training or real-process recovery success is claimed.
- Run the focused checks with `python3 tests/test_socket_runtime.py -v`.

The recovery is not a change to the scientific protocol or evidence that the
baseline/primary-model experiments have succeeded.
