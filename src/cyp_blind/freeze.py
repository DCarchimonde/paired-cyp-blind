from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rdkit

from .io import file_sha256, load_yaml


def _git_value(root: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/research_contract.yaml"))
    args = parser.parse_args()
    root = args.config.resolve().parents[1]
    config = load_yaml(args.config)
    audit_path = root / config["paths"]["reports"] / "day1_day2_audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError("Run the Day 1–2 audit before freezing")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("overall_pass"):
        raise RuntimeError("Refusing to freeze a failed Day 1–2 audit")

    tracked_inputs = [
        args.config.resolve(),
        root / config["paths"]["manifest"],
        root / config["paths"]["split_contract"],
        root / config["paths"]["split_output"],
        audit_path,
        root / config["paths"]["reports"] / "DAY1_DAY2_AUDIT.md",
        root / "uv.lock",
    ]
    freeze = {
        "schema_version": 1,
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "day1_day2",
        "status": "PASS",
        "git_head": _git_value(root, "rev-parse", "HEAD"),
        "git_status_porcelain": _git_value(root, "status", "--porcelain"),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "rdkit": rdkit.__version__,
        },
        "artifacts": {
            str(path.relative_to(root)): {
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in tracked_inputs
        },
        "next_stage_authorized": "Day 3–5 baselines",
        "flagship_claim_authorized": False,
    }
    output = root / config["paths"]["reports"] / "DAY1_DAY2_FREEZE.json"
    output.write_text(json.dumps(freeze, indent=2, sort_keys=True), encoding="utf-8")
    print(f"PASS frozen {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

