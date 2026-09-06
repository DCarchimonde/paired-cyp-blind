#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
runtime_dir="$project_dir/.runtime"
mkdir -p "$runtime_dir"
command -v flock >/dev/null || { echo "Linux flock (util-linux) is required." >&2; exit 1; }
command -v nohup >/dev/null || { echo "Linux nohup is required." >&2; exit 1; }
exec 9>"$runtime_dir/run.lock"
if ! flock -n 9; then
  echo "A run is already active; no second process was started."
  echo "Log: $runtime_dir/neural-run.log"
  exit 0
fi
# The child inherits fd 9 and holds the lock until the workflow exits.
printf '\nRun requested at %s\n' "$(date -u +%FT%TZ)" >> "$runtime_dir/neural-run.log"
PAIRED_CYP_LOCK_HELD=1 nohup bash scripts/run_neural_baselines_4090.sh \
  >> "$runtime_dir/neural-run.log" 2>&1 < /dev/null &
runner_pid=$!
printf '%s\n' "$runner_pid" > "$runtime_dir/runner.pid"
echo "Background workflow launched (PID $runner_pid); launch is not a completion claim."
echo "Log: $runtime_dir/neural-run.log"
echo "Status: $runtime_dir/run-status.txt"
echo "Results after a successful final audit: $project_dir/reports/neural_v2/"
