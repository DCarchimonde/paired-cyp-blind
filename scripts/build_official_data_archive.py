"""Package the unchanged, already verified public files with their attribution."""
import argparse
import hashlib
from pathlib import Path
import zipfile

import yaml


ROOT = Path(__file__).resolve().parents[1]


def build(raw_dir: Path, output: Path) -> str:
    manifest_path = ROOT / "data/manifests/official_dataset.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    contents = {}
    for name, info in manifest["files"].items():
        payload = (raw_dir / name).read_bytes()
        if len(payload) != info["bytes"] or hashlib.sha256(payload).hexdigest() != info["sha256"]:
            raise RuntimeError(f"Original data mismatch: {name}")
        contents[name] = payload
    contents["official_dataset.yaml"] = manifest_path.read_bytes()
    contents["UPSTREAM_DATASET_README.md"] = (ROOT / "data/manifests/UPSTREAM_DATASET_README.md").read_bytes()
    contents["LICENSE.txt"] = (ROOT / "vendor/openadmet/LICENSE").read_bytes()
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, payload in sorted(contents.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 8, 28, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, payload, compresslevel=9)
    return hashlib.sha256(output.read_bytes()).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.raw_dir, args.output))
