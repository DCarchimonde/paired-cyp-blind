#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
frozen_dir="$(bash "$project_dir/scripts/prepare_frozen_checkout.sh")"
# This is a targeted recovery for an already failed queue transfer. It does
# not interrupt a healthy job, clear cache, modify a freeze, or reinstall CUDA.
exec 8>"$frozen_dir/.runtime/socket-recovery.lock"
flock -n 8 || { echo 'A socket recovery is already active.'; exit 1; }
python3 "$project_dir/scripts/recover_socket_run.py" --root "$frozen_dir"
export UV_OFFLINE=1
bash "$project_dir/scripts/start_reviewed_neural_baselines_4090.sh" 8>&-
