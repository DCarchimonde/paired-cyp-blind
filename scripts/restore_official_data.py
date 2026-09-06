"""Restore the pinned public CSVs from the delivery archive, without networking."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile


DELIVERY = Path(__file__).resolve().parents[1]
ARCHIVE = DELIVERY / "releases/official-data-3ac9c5d.zip"
ARCHIVE_SHA256 = "ed3b68085181651410ed8bc4f31a64b3b29f2b818e5d236c73738c5997a1e8a5"
MANIFEST_SHA256 = "b36145f84dc833bf1177987c1f8b6835c016592ec2722e95dfd6628163aeec8f"
SOURCE_COMMIT = "3ac9c5dbb83eec5780ec7fa511908698cfe1396d"
FILES = {
    "cyp-challenge-TEST-BLINDED.csv": (44935, "a342f8444a8dcb531ca12f3685293f0bd6c36ae9073f491e44a9bc1cc4b741f9"),
    "cyp-challenge-TRAIN_Emax.csv": (1130831, "482f686a9a9f9166f290e6f5ea463a99de1da478b179a88f4baef48bc66501f1"),
    "cyp-challenge-TRAIN_TDI.csv": (1218524, "b458f599a792412292664386e8f18adc5d4a4129d6bd212ae80a60fb9b96bb60"),
    "cyp-challenge-TRAIN_inhibition.csv": (653407, "b8f79addd266fb6f9f4c222c5e4e73d926362328b6a8d2841871a54e46bd2278"),
    "cyp-challenge-single-concentration-TRAIN.csv": (3619009, "cf275440aa33d10b16e7a3a19f1b196d59dc379052175ab79b47ca43db663c7a"),
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def matches(name: str, payload: bytes) -> bool:
    size, digest = FILES[name]
    return len(payload) == size and sha256(payload) == digest


def restore(root: Path, archive: Path = ARCHIVE) -> dict:
    root = root.resolve()
    manifest = root / "data/manifests/official_dataset.yaml"
    if sha256(manifest.read_bytes()) != MANIFEST_SHA256:
        raise RuntimeError("Frozen data manifest mismatch; no data will be written.")
    if sha256(archive.read_bytes()) != ARCHIVE_SHA256:
        raise RuntimeError("Public data archive SHA256 mismatch; no data will be written.")
    payloads = {}
    with zipfile.ZipFile(archive) as packed:
        expected = set(FILES) | {"LICENSE.txt", "UPSTREAM_DATASET_README.md", "official_dataset.yaml"}
        if len(packed.namelist()) != len(expected) or set(packed.namelist()) != expected:
            raise RuntimeError("Unexpected archive members.")
        if packed.read("official_dataset.yaml") != manifest.read_bytes():
            raise RuntimeError("Archive manifest differs from the frozen experiment.")
        for name in FILES:
            if packed.getinfo(name).file_size != FILES[name][0]:
                raise RuntimeError(f"Archive member size mismatch: {name}")
            payloads[name] = packed.read(name)
            if not matches(name, payloads[name]):
                raise RuntimeError(f"Archive member SHA256 mismatch: {name}")
    runtime = root / ".runtime"
    raw = root / "data/raw"
    if runtime.is_symlink() or raw.is_symlink() or (root / "data").is_symlink():
        raise RuntimeError("Data or runtime directory is a symbolic link; existing paths are preserved.")
    runtime.mkdir(exist_ok=True)
    with (runtime / "run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("An experiment or recovery is active; data preparation will not overlap it.") from error
        # Validate all existing destinations before publishing any missing file.
        reused = []
        for name in FILES:
            path = raw / name
            if path.is_symlink() or (path.exists() and (not path.is_file() or not matches(name, path.read_bytes()))):
                raise RuntimeError(f"Existing raw file mismatch; preserved without overwrite: {name}")
            if path.exists():
                reused.append(name)
        raw.mkdir(parents=True, exist_ok=True)
        restored = []
        with tempfile.TemporaryDirectory(prefix="official-data-restore-", dir=runtime) as stage_name:
            stage = Path(stage_name)
            for name, payload in payloads.items():
                if name in reused:
                    print(f"PASS existing {name}", flush=True)
                    continue
                temporary = stage / name
                temporary.write_bytes(payload)
                # Both paths are on the project disk. Never overwrite a late-arriving file.
                os.link(temporary, raw / name)
                restored.append(name)
                print(f"PASS restored {name}", flush=True)
            evidence = {
                "data_ready": True, "file_count": len(FILES),
                "source_commit": SOURCE_COMMIT, "archive_sha256": ARCHIVE_SHA256,
                "manifest_sha256": MANIFEST_SHA256, "restored": restored, "reused": reused,
                "verified_utc": datetime.now(timezone.utc).isoformat(),
                "training_completed": False,
            }
            temporary_audit = stage / "audit.json"
            temporary_audit.write_text(json.dumps(evidence, indent=2) + "\n")
            os.replace(temporary_audit, runtime / "official-data-restore.json")
    print("OFFICIAL DATA READY: 5/5 original hashes verified. No network or GPU was used.", flush=True)
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    try:
        restore(parser.parse_args().root)
    except Exception as error:
        parser.exit(1, f"DATA PREPARATION FAILED: {error}\n")
