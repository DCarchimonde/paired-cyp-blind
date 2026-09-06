#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
frozen_dir="$(bash "$project_dir/scripts/prepare_frozen_checkout.sh")"
runtime_dir="$frozen_dir/.runtime"
python_exe="$frozen_dir/.venv/bin/python"
[[ -x "$python_exe" ]] || {
  echo 'The initial installer must have created its Python environment before this recovery can run.' >&2
  exit 1
}
command -v nohup >/dev/null
exec 8>"$runtime_dir/download-recovery.lock"
if ! flock -n 8; then
  echo "A download recovery is already active. Log: $runtime_dir/neural-run.log"
  exit 0
fi
printf '\nDownload recovery requested at %s\n' "$(date -u +%FT%TZ)" >> "$runtime_dir/neural-run.log"
nohup "$python_exe" -u "$project_dir/scripts/recover_frozen_downloads.py" \
  --root "$frozen_dir" >> "$runtime_dir/neural-run.log" 2>&1 < /dev/null &
echo "Download recovery launched (PID $!); inspect the log for its outcome."
echo "Log: $runtime_dir/neural-run.log"
echo 'The recovery preserves the frozen source and cache, verifies package hashes, and resumes the original workflow after installation.'
