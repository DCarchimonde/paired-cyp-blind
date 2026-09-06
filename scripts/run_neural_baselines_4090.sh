#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

runtime_dir="$project_dir/.runtime"
mkdir -p "$runtime_dir"
if [[ "${PAIRED_CYP_LOCK_HELD:-0}" != 1 ]]; then
  exec 9>"$runtime_dir/run.lock"
  flock -n 9 || { echo "Another paired-cyp run is active in this directory." >&2; exit 1; }
fi
finish() {
  run_exit_code=$?
  if [[ "$run_exit_code" == 0 ]]; then
    printf 'PASS: all 200 neural baseline jobs and the independent result audit completed.\n' | tee "$runtime_dir/run-status.txt"
  else
    printf 'FAILED (exit %s): inspect .runtime/neural-run.log; no complete-run claim is authorized.\n' "$run_exit_code" | tee "$runtime_dir/run-status.txt"
  fi
}
trap finish EXIT
printf 'RUNNING since %s\n' "$(date -u +%FT%TZ)" > "$runtime_dir/run-status.txt"
export PYTHONUNBUFFERED=1
export UV_PROJECT_ENVIRONMENT="$project_dir/.venv"
export UV_CACHE_DIR="$runtime_dir/uv-cache"
export UV_PYTHON_INSTALL_DIR="$runtime_dir/uv-python"
export TMPDIR="$runtime_dir/tmp"
mkdir -p "$TMPDIR"

command -v nvidia-smi >/dev/null || { echo "RTX 4090 NVIDIA driver is required." >&2; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
if [[ ! -x "$project_dir/.venv/bin/chemprop" ]]; then
  free_kib="$(df -Pk "$project_dir" | awk 'NR==2 {print $4}')"
  [[ "$free_kib" -ge 31457280 ]] || { echo "Need at least 30 GiB free on the project data disk for a fresh install." >&2; exit 1; }
fi
uv_exe="$(command -v uv || true)"
if [[ -z "$uv_exe" ]] || [[ "$("$uv_exe" --version)" != uv\ 0.11.33* ]]; then
  uv_exe="$runtime_dir/uv-bootstrap/bin/uv"
  if [[ ! -x "$uv_exe" ]]; then
    python3 -m pip install --disable-pip-version-check --no-deps --target "$runtime_dir/uv-bootstrap" 'uv==0.11.33'
  fi
fi
"$uv_exe" sync --frozen --extra dev --extra neural
# Explicit --no-sync keeps the optional neural/dev dependencies installed.
"$uv_exe" run --no-sync python scripts/fetch_official.py
"$uv_exe" run --no-sync pytest -q
"$uv_exe" run --no-sync python -m cyp_blind.neural_baselines --config configs/neural_baselines.yaml preflight --require-gpu
"$uv_exe" run --no-sync python -m cyp_blind.neural_baselines --config configs/neural_baselines.yaml smoke
"$uv_exe" run --no-sync python -m cyp_blind.neural_baselines --config configs/neural_baselines.yaml prepare
"$uv_exe" run --no-sync python -m cyp_blind.neural_protocol_audit --config configs/neural_baselines.yaml
"$uv_exe" run --no-sync python -m cyp_blind.neural_baselines --config configs/neural_baselines.yaml run-all --require-gpu
"$uv_exe" run --no-sync python -m cyp_blind.neural_baselines --config configs/neural_baselines.yaml collect
"$uv_exe" run --no-sync python -m cyp_blind.neural_result_audit --config configs/neural_baselines.yaml
