#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
frozen_dir="$(bash "$project_dir/scripts/prepare_frozen_checkout.sh")"
python3 "$project_dir/scripts/restore_official_data.py" --root "$frozen_dir"
exec bash "$frozen_dir/scripts/run_neural_baselines_4090.sh"
