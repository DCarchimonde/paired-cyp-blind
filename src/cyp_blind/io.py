from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import yaml

from .constants import FILES_BY_ROLE


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected YAML object in {path}")
    return payload


def load_tables(raw_dir: Path) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for role, filename in FILES_BY_ROLE.items():
        path = raw_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing official data file: {path}")
        tables[role] = pd.read_csv(path)
    return tables

