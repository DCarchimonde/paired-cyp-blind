#!/usr/bin/env bash
set -euo pipefail

delivery_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
frozen_dir="$delivery_dir/.runtime/frozen-experiment"
python_exe="$frozen_dir/.venv/bin/python"
runtime_dir="$delivery_dir/.runtime"
run_log="$runtime_dir/paired-followup.log"
run_status="$runtime_dir/paired-followup-status.txt"
[[ -x "$python_exe" ]] || { echo 'The existing frozen Python environment is required.' >&2; exit 1; }
[[ -f "$frozen_dir/.runtime/paired-prototype-v4/RESULT_AUDIT.json" ]] || {
  echo 'Complete the reviewed v4 pilot before starting v5. Preserve the entire experiment directory.' >&2; exit 1;
}
command -v flock >/dev/null
command -v nohup >/dev/null

if [[ "${1:-}" != '--worker' ]]; then
  exec 9>"$frozen_dir/.runtime/run.lock"
  if ! flock -n 9; then
    echo 'An experiment is already running; no additional training was started.'
    echo "Follow-up log: $run_log"
    exit 0
  fi
  printf '\nPaired follow-up requested at %s\n' "$(date -u +%FT%TZ)" >> "$run_log"
  CYP_FOLLOWUP_LOCK_HELD=1 nohup bash "$delivery_dir/scripts/start_paired_followup_4090.sh" --worker \
    >> "$run_log" 2>&1 < /dev/null &
  runner_pid=$!
  printf '%s\n' "$runner_pid" > "$runtime_dir/paired-followup.pid"
  echo "Paired follow-up launched (PID $runner_pid)."
  echo "Log: $run_log"
  echo "Status: $run_status"
  echo 'Completion requires PASS after all 50 new jobs, 10 verified v4 jobs, diagnostics and independent result audit.'
  exit 0
fi

[[ "${CYP_FOLLOWUP_LOCK_HELD:-0}" == 1 ]] || { echo 'Start through the background entry point.' >&2; exit 1; }
flock -n 9
finish() {
  run_exit_code=$?
  if [[ "$run_exit_code" == 0 ]]; then
    printf 'PASS: all 50 new jobs, 10 verified v4 jobs, diagnostics and independent result audit completed.\n' > "$run_status"
  else
    printf 'FAILED (exit %s): inspect .runtime/paired-followup.log. Existing epochs and artifacts are preserved.\n' "$run_exit_code" > "$run_status"
  fi
}
trap finish EXIT
printf 'RUNNING since %s\n' "$(date -u +%FT%TZ)" > "$run_status"
export PYTHONUNBUFFERED=1
export PATH="$frozen_dir/.venv/bin:$PATH"
export PYTHONPATH="$frozen_dir/src${PYTHONPATH:+:$PYTHONPATH}"
# An explicit small thread budget is part of the follow-up protocol. No DataLoader
# workers transfer tensors, and no package in the frozen environment is modified.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
short_dir="$("$python_exe" "$delivery_dir/scripts/short_runtime.py" --root "$frozen_dir")"
export TMPDIR="$short_dir/.runtime/tmp"
cd "$frozen_dir"
"$python_exe" -u "$delivery_dir/scripts/paired_followup.py" --root "$frozen_dir" --delivery "$delivery_dir"
