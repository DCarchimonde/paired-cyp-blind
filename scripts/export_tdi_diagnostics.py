"""Export completed TDI training evidence without changing the frozen experiment.

Uses only the Python standard library. It does not train, predict, change a
threshold, or load model checkpoints. A new ZIP is created in the delivery repo.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile


def export(root: Path, destination: Path) -> dict:
    root = root.resolve(strict=True)
    summary = root / "reports/neural_v2"
    manifest_path = summary / "neural_family_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete") or manifest.get("jobs") != 200:
        raise RuntimeError("A complete 200-job result manifest is required.")
    selected = {
        job: digest for job, digest in manifest["completion_manifest_hashes"].items()
        if "__tdi__" in job
    }
    if len(selected) != 75:
        raise RuntimeError(f"Expected 75 completed TDI jobs; found {len(selected)}.")
    files: dict[str, dict] = {}

    def add(path: Path, expected_hash: str | None = None) -> bytes:
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise RuntimeError(f"Expected a file inside the frozen experiment: {path}")
        content = resolved.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if expected_hash is not None and digest != expected_hash:
            raise RuntimeError(f"Recorded file hash changed: {path.relative_to(root)}")
        relative = str(resolved.relative_to(root))
        files[relative] = {"sha256": digest, "size_bytes": len(content)}
        return content

    add(manifest_path)
    for relative, digest in manifest["inputs"].items():
        # Include configuration, preparation metadata and family mapping. The
        # public raw data is already in the reviewed release and need not be copied.
        if relative != "data/raw/cyp-challenge-TRAIN_TDI.csv":
            add(root / relative, digest)
    add(summary / "neural_result_audit.json")
    text_suffixes = {".csv", ".json", ".yaml", ".yml", ".toml", ".ini", ".txt", ".log"}
    for job_id, digest in sorted(selected.items()):
        job_root = summary / "jobs" / job_id
        record = json.loads(add(job_root / "COMPLETE.json", digest))
        if (record.get("status") != "PASS" or record.get("job_id") != job_id
                or record.get("job", {}).get("task_group") != "tdi"):
            raise RuntimeError(f"Unexpected completion identity: {job_id}")
        for entry in record["inputs"].values():
            add(root / entry["path"], entry["sha256"])
        for kind in ("log", "predictions", "chemprop_predictions"):
            entry = record["outputs"][kind]
            add(root / entry["path"], entry["sha256"])
        attempt = (root / record["outputs"]["log"]["path"]).resolve().parent
        if not attempt.is_relative_to(job_root.resolve()):
            raise RuntimeError(f"Unexpected attempt directory: {job_id}")
        # Preserve the successful attempt's curves/configs and TensorBoard events.
        for path in sorted(attempt.rglob("*")):
            if path.is_file() and (path.suffix.lower() in text_suffixes
                                   or path.name.startswith("events.out.tfevents")):
                add(path)
    inventory = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_git_head": manifest["git_head"],
        "tdi_jobs": len(selected),
        "files": files,
        "model_weights_included": False,
        "scientific_protocol_changed": False,
    }
    created = False
    try:
        with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            created = True
            for relative, entry in sorted(files.items()):
                content = (root / relative).read_bytes()
                if hashlib.sha256(content).hexdigest() != entry["sha256"]:
                    raise RuntimeError(f"File changed during export: {relative}")
                archive.writestr(relative, content)
            archive.writestr("TDI_EXPORT_MANIFEST.json", json.dumps(inventory, indent=2) + "\n")
    except BaseException:
        if created:
            destination.unlink(missing_ok=True)
        raise
    return {"output": str(destination.resolve()), "tdi_jobs": len(selected),
            "files": len(files), "zip_size_bytes": destination.stat().st_size}


def main() -> None:
    delivery = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=delivery / ".runtime/frozen-experiment")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    destination = args.output or delivery / "CYP_TDI_diagnostics.zip"
    if args.output is None and destination.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        destination = delivery / f"CYP_TDI_diagnostics_{stamp}.zip"
    try:
        result = export(args.root, destination)
    except (OSError, ValueError, RuntimeError, KeyError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"TDI EXPORT FAILED: {exc}\n")
    print(json.dumps(result, indent=2))
    print("PASS: TDI training evidence exported. Upload the ZIP shown above.")


if __name__ == "__main__":
    main()
