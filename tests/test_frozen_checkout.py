"""Real Git archive-import tests, with no downloads, installations, or training."""
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
COMMIT = "bc00641390c559ec696506d6fd07edb77c514c13"
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="The AutoDL entrypoint is Linux-only")


@pytest.fixture
def delivery(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    for name in (
        "scripts/prepare_frozen_checkout.sh",
        "releases/paired-cyp-blind-reviewed-20260906.bundle",
    ):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    return tmp_path


def prepare(root: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(root / "scripts/prepare_frozen_checkout.sh")],
        capture_output=True, text=True,
    )


def test_archive_restores_exact_revision_and_reuses_checkout(delivery: Path) -> None:
    first = prepare(delivery)
    assert first.returncode == 0, first.stderr
    frozen = Path(first.stdout.strip())
    head = subprocess.check_output(["git", "-C", str(frozen), "rev-parse", "HEAD"], text=True)
    assert head.strip() == COMMIT
    before = (frozen / ".git/HEAD").stat().st_mtime_ns
    second = prepare(delivery)
    assert second.returncode == 0, second.stderr
    assert second.stdout == first.stdout
    assert (frozen / ".git/HEAD").stat().st_mtime_ns == before
    assert (delivery / "reports/neural_v2").resolve() == frozen / "reports/neural_v2"


def test_modified_bundle_stops_before_import(delivery: Path) -> None:
    bundle = delivery / "releases/paired-cyp-blind-reviewed-20260906.bundle"
    bundle.write_bytes(bundle.read_bytes() + b"corrupt")
    result = prepare(delivery)
    assert result.returncode != 0
    assert "SHA256 mismatch" in result.stderr
    assert not (delivery / ".runtime/frozen-experiment").exists()


def test_modified_frozen_source_is_preserved_and_rejected(delivery: Path) -> None:
    assert prepare(delivery).returncode == 0
    config = delivery / ".runtime/frozen-experiment/configs/neural_baselines.yaml"
    changed = config.read_text() + "\n# unauthorized protocol drift\n"
    config.write_text(changed)
    result = prepare(delivery)
    assert result.returncode != 0
    assert "source changes" in result.stderr
    assert config.read_text() == changed


def test_existing_results_directory_is_not_overwritten(delivery: Path) -> None:
    old_results = delivery / "reports/neural_v2"
    old_results.mkdir(parents=True)
    evidence = old_results / "existing.txt"
    evidence.write_text("preserve")
    result = prepare(delivery)
    assert result.returncode != 0
    assert "will not be overwritten" in result.stderr
    assert evidence.read_text() == "preserve"
