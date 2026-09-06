.PHONY: setup fetch test audit freeze baselines audit-baselines split-gap audit-split-gap neural-smoke neural-integration-smoke neural-prepare audit-neural-protocol neural-preflight neural-baselines audit-neural-results day1-day2

setup:
	uv sync --extra dev

fetch:
	uv run python scripts/fetch_official.py

test:
	uv run pytest -q

audit:
	uv run python -m cyp_blind.audit --config configs/research_contract.yaml

freeze:
	uv run python -m cyp_blind.freeze --config configs/research_contract.yaml

baselines:
	uv run python -m cyp_blind.baselines --config configs/baselines.yaml

audit-baselines:
	uv run python -m cyp_blind.baseline_audit --config configs/baselines.yaml

split-gap:
	uv run python -m cyp_blind.split_gap --config configs/split_gap.yaml

audit-split-gap:
	uv run python -m cyp_blind.split_gap_audit --config configs/split_gap.yaml

neural-smoke:
	uv run python -m cyp_blind.neural_baselines --config configs/neural_baselines.yaml smoke

neural-integration-smoke:
	uv run python -m cyp_blind.neural_baselines --config configs/neural_baselines.yaml integration-smoke

neural-prepare:
	uv run python -m cyp_blind.neural_baselines --config configs/neural_baselines.yaml prepare

audit-neural-protocol:
	uv run python -m cyp_blind.neural_protocol_audit --config configs/neural_baselines.yaml

neural-preflight:
	uv run python -m cyp_blind.neural_baselines --config configs/neural_baselines.yaml preflight --require-gpu

neural-baselines:
	bash scripts/run_reviewed_neural_baselines_4090.sh

audit-neural-results:
	uv run python -m cyp_blind.neural_result_audit --config configs/neural_baselines.yaml

day1-day2: setup fetch test audit freeze
