#!/usr/bin/env bash
set -euo pipefail

delivery_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
frozen_dir="$delivery_dir/.runtime/frozen-experiment"
python_exe="$frozen_dir/.venv/bin/python"
runtime_dir="$delivery_dir/.runtime"
run_log="$runtime_dir/paired-prototype.log"
run_status="$runtime_dir/paired-prototype-status.txt"
[[ -x "$python_exe" ]] || { echo 'The existing frozen Python environment is required.' >&2; exit 1; }
[[ -f "$frozen_dir/.runtime/neural-prc-max-v3/neural_result_audit.json" ]] || {
  echo 'Complete the PRC-max v3 baseline before starting the paired pilot.' >&2; exit 1;
}
command -v flock >/dev/null
command -v nohup >/dev/null

if [[ "${1:-}" != '--worker' ]]; then
  exec 9>"$frozen_dir/.runtime/run.lock"
  if ! flock -n 9; then
    echo 'An experiment is already running; no additional training was started.'
    echo "Pilot log: $run_log"
    exit 0
  fi
  printf '\nPaired prototype requested at %s\n' "$(date -u +%FT%TZ)" >> "$run_log"
  CYP_PAIRED_LOCK_HELD=1 nohup bash "$delivery_dir/scripts/start_paired_prototype_4090.sh" --worker \
    >> "$run_log" 2>&1 < /dev/null &
  runner_pid=$!
  printf '%s\n' "$runner_pid" > "$runtime_dir/paired-prototype.pid"
  echo "Paired pilot launched (PID $runner_pid)."
  echo "Log: $run_log"
  echo "Status: $run_status"
  echo 'Completion requires PASS after all 20 pilot jobs and the independent result audit.'
  exit 0
fi

[[ "${CYP_PAIRED_LOCK_HELD:-0}" == 1 ]] || { echo 'Start through the background entry point.' >&2; exit 1; }
flock -n 9
finish() {
  run_exit_code=$?
  if [[ "$run_exit_code" == 0 ]]; then
    printf 'PASS: all 20 paired-assay pilot jobs and the independent result audit completed.\n' > "$run_status"
  else
    printf 'FAILED (exit %s): inspect .runtime/paired-prototype.log. Existing epochs and artifacts are preserved.\n' "$run_exit_code" > "$run_status"
  fi
}
trap finish EXIT
printf 'RUNNING since %s\n' "$(date -u +%FT%TZ)" > "$run_status"
export PYTHONUNBUFFERED=1
export PATH="$frozen_dir/.venv/bin:$PATH"
export PYTHONPATH="$frozen_dir/src${PYTHONPATH:+:$PYTHONPATH}"
# An explicit small thread budget is part of the pilot protocol. No DataLoader
# workers transfer tensors, and no package in the frozen environment is modified.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
short_dir="$("$python_exe" "$delivery_dir/scripts/short_runtime.py" --root "$frozen_dir")"
export TMPDIR="$short_dir/.runtime/tmp"
cd "$frozen_dir"
"$python_exe" -u "$delivery_dir/scripts/paired_prototype.py" --root "$frozen_dir" --delivery "$delivery_dir"
