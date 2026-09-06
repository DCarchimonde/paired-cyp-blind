"""Isolated fake-worker tests; these do not launch Chemprop or assert GPU results."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="AutoDL launcher is Linux-only"
)


def _copy_script(root: Path, name: str) -> Path:
    destination = root / "scripts" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "scripts" / name, destination)
    return destination


def test_background_launcher_holds_lock_and_prevents_duplicate(tmp_path: Path) -> None:
    launcher = _copy_script(tmp_path, "start_neural_baselines_4090.sh")
    worker = tmp_path / "scripts/run_neural_baselines_4090.sh"
    worker.write_text(
        "#!/bin/bash\nset -eu\nprintf 'started\\n' >> .runtime/starts.txt\n"
        "while [ ! -f .runtime/release ]; do sleep 0.05; done\n",
        encoding="utf-8",
    )
    runtime = tmp_path / ".runtime"
    try:
        first = subprocess.run(
            ["bash", str(launcher)], capture_output=True, text=True, check=True
        )
        assert "launched" in first.stdout
        deadline = time.monotonic() + 5
        while not (runtime / "starts.txt").exists() and time.monotonic() < deadline:
            time.sleep(0.025)
        assert (runtime / "starts.txt").is_file()
        second = subprocess.run(
            ["bash", str(launcher)], capture_output=True, text=True, check=True
        )
        assert "already active" in second.stdout
        assert (runtime / "starts.txt").read_text().splitlines() == ["started"]
    finally:
        (runtime / "release").touch()
        # Ensure the fake worker exits before pytest removes its temporary directory.
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if (
                subprocess.run(
                    ["flock", "-n", str(runtime / "run.lock"), "true"]
                ).returncode
                == 0
            ):
                break
            time.sleep(0.025)
        else:
            pytest.fail("Fake launcher worker did not release its lock")


@pytest.mark.parametrize("audit_exit", [0, 42])
def test_workflow_status_and_fail_fast(tmp_path: Path, audit_exit: int) -> None:
    worker = _copy_script(tmp_path, "run_neural_baselines_4090.sh")
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    gpu = fake_bin / "nvidia-smi"
    gpu.write_text("#!/bin/bash\necho 'SYNTHETIC driver fixture'\n", encoding="utf-8")
    uv = fake_bin / "uv"
    uv.write_text(
        "#!/bin/bash\nset -eu\n"
        "if [[ \"$*\" == --version ]]; then echo 'uv 0.11.33'; exit 0; fi\n"
        "printf '%s\\n' \"$*\" >> .runtime/commands.txt\n"
        'if [[ "$*" == *cyp_blind.neural_protocol_audit* ]]; then exit '
        + str(audit_exit)
        + "; fi\n",
        encoding="utf-8",
    )
    chemprop = tmp_path / ".venv/bin/chemprop"
    chemprop.parent.mkdir(parents=True)
    chemprop.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    for path in (uv, gpu, chemprop):
        path.chmod(0o755)
    env = {**os.environ, "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"]}
    env.pop("PAIRED_CYP_LOCK_HELD", None)
    result = subprocess.run(
        ["bash", str(worker)], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert result.returncode == audit_exit, result.stdout + result.stderr
    status = (tmp_path / ".runtime/run-status.txt").read_text()
    commands = (tmp_path / ".runtime/commands.txt").read_text()
    assert "sync --frozen --extra dev --extra neural" in commands
    assert "run --no-sync pytest -q" in commands
    if audit_exit:
        assert status.startswith("FAILED (exit 42)")
        assert "run-all" not in commands
    else:
        assert status.startswith("PASS:")
        assert "run-all --require-gpu" in commands
        assert "cyp_blind.neural_result_audit" in commands.splitlines()[-1]
