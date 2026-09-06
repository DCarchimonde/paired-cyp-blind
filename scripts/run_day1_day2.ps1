$ErrorActionPreference = "Stop"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required. Install it from https://docs.astral.sh/uv/ and rerun."
}

Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    uv sync --extra dev
    uv run python scripts/fetch_official.py
    uv run pytest -q
    uv run python -m cyp_blind.audit --config configs/research_contract.yaml
    uv run python -m cyp_blind.freeze --config configs/research_contract.yaml
}
finally {
    Pop-Location
}

