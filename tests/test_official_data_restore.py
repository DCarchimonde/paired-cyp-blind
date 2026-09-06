import fcntl
import importlib.util
from pathlib import Path
import shutil
import urllib.request

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


restore = module(ROOT / "scripts/restore_official_data.py")


@pytest.fixture
def frozen(tmp_path: Path) -> Path:
    (tmp_path / "data/manifests").mkdir(parents=True)
    shutil.copy2(ROOT / "data/manifests/official_dataset.yaml", tmp_path / "data/manifests/official_dataset.yaml")
    return tmp_path


def test_archive_restores_exact_files_and_original_fetch_needs_no_network(frozen: Path, monkeypatch) -> None:
    result = restore.restore(frozen)
    assert result["data_ready"] and len(result["restored"]) == 5
    manifest = yaml.safe_load((frozen / "data/manifests/official_dataset.yaml").read_text())
    for name, info in manifest["files"].items():
        raw = (frozen / "data/raw" / name).read_bytes()
        assert restore.sha256(raw) == info["sha256"]
        assert len(raw) == info["bytes"]
    fetch = module(ROOT / "scripts/fetch_official.py")
    fetch.MANIFEST = frozen / "data/manifests/official_dataset.yaml"
    fetch.RAW = frozen / "data/raw"
    def forbidden(*args, **kwargs):
        raise AssertionError("Original fetch attempted networking with verified local data")
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    assert fetch.main() == 0
    before = {p.name: p.stat().st_mtime_ns for p in fetch.RAW.glob("*.csv")}
    result = restore.restore(frozen)
    assert len(result["reused"]) == 5 and result["restored"] == []
    assert before == {p.name: p.stat().st_mtime_ns for p in fetch.RAW.glob("*.csv")}


def test_existing_conflicting_file_is_preserved_before_any_other_file_is_written(frozen: Path) -> None:
    raw = frozen / "data/raw"
    raw.mkdir()
    path = raw / "cyp-challenge-TRAIN_TDI.csv"
    path.write_bytes(b"existing user file")
    with pytest.raises(RuntimeError, match="preserved without overwrite"):
        restore.restore(frozen)
    assert path.read_bytes() == b"existing user file"
    assert list(raw.iterdir()) == [path]


def test_corrupt_archive_is_rejected_before_data_is_written(frozen: Path) -> None:
    archive = frozen / "broken.zip"
    archive.write_bytes(restore.ARCHIVE.read_bytes() + b"tampered")
    with pytest.raises(RuntimeError, match="archive SHA256 mismatch"):
        restore.restore(frozen, archive)
    assert not (frozen / "data/raw").exists()


def test_modified_frozen_manifest_is_rejected(frozen: Path) -> None:
    manifest = frozen / "data/manifests/official_dataset.yaml"
    manifest.write_text(manifest.read_text() + "\n# changed\n")
    with pytest.raises(RuntimeError, match="manifest mismatch"):
        restore.restore(frozen)
    assert not (frozen / "data/raw").exists()


def test_active_workflow_lock_is_respected(frozen: Path) -> None:
    runtime = frozen / ".runtime"
    runtime.mkdir()
    with (runtime / "run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with pytest.raises(RuntimeError, match="active"):
            restore.restore(frozen)
    assert not (frozen / "data/raw").exists()
