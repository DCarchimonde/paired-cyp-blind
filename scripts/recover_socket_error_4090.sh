#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
frozen_dir="$(bash "$project_dir/scripts/prepare_frozen_checkout.sh")"
python_exe="$frozen_dir/.venv/bin/python"
[[ -x "$python_exe" ]] || {
  echo 'The installed frozen Python environment is required for socket recovery.' >&2
  exit 1
}
# This is a targeted recovery for an already failed queue transfer. It does
# not interrupt a healthy job, clear cache, modify a freeze, or reinstall CUDA.
exec 8>"$frozen_dir/.runtime/socket-recovery.lock"
flock -n 8 || { echo 'A socket recovery is already active.'; exit 1; }
"$python_exe" "$project_dir/scripts/recover_socket_run.py" --root "$frozen_dir"
export UV_OFFLINE=1
bash "$project_dir/scripts/start_reviewed_neural_baselines_4090.sh" 8>&-
