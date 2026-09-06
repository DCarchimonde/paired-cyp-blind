from __future__ import annotations

import hashlib
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifests" / "official_dataset.yaml"
RAW = ROOT / "data" / "raw"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    source = manifest["source"]
    commit = source["commit"]
    RAW.mkdir(parents=True, exist_ok=True)

    for filename, metadata in manifest["files"].items():
        destination = RAW / filename
        expected = metadata["sha256"]
        if destination.exists() and sha256(destination) == expected:
            print(f"PASS existing {filename}")
            continue
        if destination.exists():
            raise RuntimeError(
                f"Existing raw file hash mismatch: {destination}; preserve it and restore the frozen file explicitly"
            )

        url = (
            "https://huggingface.co/datasets/openadmet/"
            f"cyp-challenge-train-test/resolve/{commit}/{filename}?download=true"
        )
        temporary = destination.with_suffix(destination.suffix + ".part")
        print(f"FETCH {filename}")
        for attempt in range(3):
            try:
                with (
                    urllib.request.urlopen(url, timeout=60) as response,
                    temporary.open("wb") as handle,
                ):
                    shutil.copyfileobj(response, handle)
                break
            except (OSError, urllib.error.URLError):
                if attempt == 2:
                    raise
                time.sleep(2 ** (attempt + 1))
        actual = sha256(temporary)
        if actual != expected:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"SHA256 mismatch for {filename}: expected {expected}, got {actual}"
            )
        temporary.replace(destination)
        print(f"PASS downloaded {filename}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
