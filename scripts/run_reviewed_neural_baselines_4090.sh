#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
short_dir="$(bash "$project_dir/scripts/prepare_short_runtime.sh")"
exec bash "$short_dir/scripts/run_neural_baselines_4090.sh"
