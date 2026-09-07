#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
frozen_dir="$(bash "$project_dir/scripts/prepare_frozen_checkout.sh")"
python3 "$project_dir/scripts/restore_official_data.py" --root "$frozen_dir" >&2
short_dir="$(python3 "$project_dir/scripts/short_runtime.py" --root "$frozen_dir")"
python_exe="$frozen_dir/.venv/bin/python"
[[ -x "$python_exe" ]] || python_exe="$(command -v python3)"
command -v timeout >/dev/null
# A real kernel/worker probe must succeed on the training host. The timeout
# bounds failures in queue threads which otherwise leave the parent waiting.
TMPDIR="$short_dir/.runtime/tmp" timeout --kill-after=5s 45s \
  "$python_exe" -u "$project_dir/scripts/probe_tensor_ipc.py" >&2
printf '%s\n' "$short_dir"
