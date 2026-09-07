#!/usr/bin/env bash
set -euo pipefail

delivery_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
frozen_dir="$delivery_dir/.runtime/frozen-experiment"
python_exe="$frozen_dir/.venv/bin/python"
runtime_dir="$delivery_dir/.runtime"
run_log="$runtime_dir/tdi-prc-repair.log"
run_status="$runtime_dir/tdi-prc-repair-status.txt"
[[ -x "$python_exe" ]] || { echo 'The completed frozen Python environment is required.' >&2; exit 1; }
[[ -f "$frozen_dir/reports/neural_v2/neural_family_manifest.json" ]] || {
  echo 'This repair requires the completed v2 result manifest.' >&2; exit 1;
}
command -v flock >/dev/null
command -v nohup >/dev/null
command -v timeout >/dev/null

if [[ "${1:-}" != '--worker' ]]; then
  exec 9>"$frozen_dir/.runtime/run.lock"
  if ! flock -n 9; then
    echo 'An experiment is already running; no additional training was started.'
    echo "Repair log: $run_log"
    exit 0
  fi
  printf '\nPRC-max repair requested at %s\n' "$(date -u +%FT%TZ)" >> "$run_log"
  CYP_TDI_LOCK_HELD=1 nohup bash "$delivery_dir/scripts/start_tdi_prc_repair_4090.sh" --worker \
    >> "$run_log" 2>&1 < /dev/null &
  runner_pid=$!
  printf '%s\n' "$runner_pid" > "$runtime_dir/tdi-prc-repair.pid"
  echo "PRC-max repair launched (PID $runner_pid)."
  echo "Log: $run_log"
  echo "Status: $run_status"
  echo 'Completion requires PASS after all 75 TDI jobs and the independent result audit.'
  exit 0
fi

[[ "${CYP_TDI_LOCK_HELD:-0}" == 1 ]] || { echo 'Use the background entry point to hold the experiment lock.' >&2; exit 1; }
flock -n 9
finish() {
  run_exit_code=$?
  if [[ "$run_exit_code" == 0 ]]; then
    printf 'PASS: 75 corrected TDI jobs + 125 verified reused regression jobs; independent audit completed.\n' > "$run_status"
  else
    printf 'FAILED (exit %s): inspect .runtime/tdi-prc-repair.log. Existing artifacts are preserved.\n' "$run_exit_code" > "$run_status"
  fi
}
trap finish EXIT
printf 'RUNNING since %s\n' "$(date -u +%FT%TZ)" > "$run_status"
export PYTHONUNBUFFERED=1
export PATH="$frozen_dir/.venv/bin:$PATH"
export PYTHONPATH="$frozen_dir/src${PYTHONPATH:+:$PYTHONPATH}"
# Invalid inherited values make libgomp fall back to its default and warn in
# every child. Unset only invalid values; preserve valid positive thread lists.
if [[ ${OMP_NUM_THREADS+x} && ! "$OMP_NUM_THREADS" =~ ^[[:space:]]*[+]?0*[1-9][0-9]*([[:space:]]*,[[:space:]]*[+]?0*[1-9][0-9]*)*[[:space:]]*$ ]]; then
  export CYP_IGNORED_OMP_NUM_THREADS="$OMP_NUM_THREADS"
  unset OMP_NUM_THREADS
  echo 'Ignoring invalid inherited OMP_NUM_THREADS; using the OpenMP default.'
fi
short_dir="$("$python_exe" "$delivery_dir/scripts/short_runtime.py" --root "$frozen_dir")"
export TMPDIR="$short_dir/.runtime/tmp"
timeout --kill-after=5s 45s "$python_exe" -u "$delivery_dir/scripts/probe_tensor_ipc.py"
cd "$frozen_dir"
"$python_exe" -u "$delivery_dir/scripts/tdi_prc_repair.py" --root "$frozen_dir" --delivery "$delivery_dir"
