#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bundle="$project_dir/releases/paired-cyp-blind-reviewed-20260906.bundle"
expected_bundle_sha='1aa7a8d7ac8f734a4e758fef9232ddd98d77a711485ee6b87bc5c363e1f6e79f'
expected_commit='bc00641390c559ec696506d6fd07edb77c514c13'
expected_tag='neural-protocol-hardened-20260906'
runtime_dir="$project_dir/.runtime"
frozen_dir="$runtime_dir/frozen-experiment"

command -v flock >/dev/null || { echo 'Linux flock is required.' >&2; exit 1; }
[[ -f "$bundle" ]] || { echo 'Frozen experiment bundle is missing.' >&2; exit 1; }
observed_sha="$(sha256sum "$bundle" | cut -d ' ' -f 1)"
[[ "$observed_sha" == "$expected_bundle_sha" ]] || {
  echo 'Frozen bundle SHA256 mismatch; no experiment will be launched.' >&2
  exit 1
}
mkdir -p "$runtime_dir"
exec 8>"$runtime_dir/frozen-import.lock"
flock 8

verify_checkout() {
  local candidate="$1"
  [[ -d "$candidate/.git" ]] || { echo 'Frozen checkout is not a Git repository.' >&2; return 1; }
  [[ "$(git -C "$candidate" rev-parse HEAD)" == "$expected_commit" ]] || {
    echo 'Frozen checkout commit mismatch; existing files are preserved.' >&2; return 1;
  }
  [[ "$(git -C "$candidate" rev-parse "$expected_tag^{commit}")" == "$expected_commit" ]] || {
    echo 'Frozen protocol tag mismatch.' >&2; return 1;
  }
  [[ -z "$(git -C "$candidate" status --porcelain)" ]] || {
    echo 'Frozen checkout has source changes; existing files are preserved.' >&2; return 1;
  }
}

if [[ ! -e "$frozen_dir" ]]; then
  git -C "$project_dir" bundle verify "$bundle" >/dev/null
  stage_dir="$(mktemp -d "$runtime_dir/frozen-import.XXXXXX")"
  git clone --quiet --branch main "$bundle" "$stage_dir"
  verify_checkout "$stage_dir"
  mv "$stage_dir" "$frozen_dir"
fi
verify_checkout "$frozen_dir"

# Expose the actual execution artifacts at short, stable paths in this checkout.
mkdir -p "$frozen_dir/.runtime" "$frozen_dir/reports/neural_v2" "$project_dir/reports"
ensure_link() {
  local target="$1" destination="$2"
  if [[ -L "$destination" ]]; then
    [[ "$(readlink "$destination")" == "$target" ]] || {
      echo "Existing artifact link points elsewhere: $destination" >&2; return 1;
    }
  elif [[ -e "$destination" ]]; then
    echo "Existing artifact path will not be overwritten: $destination" >&2
    return 1
  else
    ln -s "$target" "$destination"
  fi
}
ensure_link 'frozen-experiment/.runtime/neural-run.log' "$runtime_dir/neural-run.log"
ensure_link 'frozen-experiment/.runtime/run-status.txt' "$runtime_dir/run-status.txt"
ensure_link 'frozen-experiment/.runtime/runner.pid' "$runtime_dir/runner.pid"
ensure_link '../.runtime/frozen-experiment/reports/neural_v2' "$project_dir/reports/neural_v2"
printf '%s\n' "$frozen_dir"
