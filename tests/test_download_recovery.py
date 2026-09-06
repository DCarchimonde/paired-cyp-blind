"""Recovery must preserve active experiments and bound its own installer processes."""
import fcntl
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/recover_frozen_downloads.py"
spec = importlib.util.spec_from_file_location("download_recovery", SCRIPT)
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux process handles and flock")
native_proc = pytest.mark.skipif(Path("/proc/self").resolve().name != str(os.getpid()),
                                reason="Runtime virtualizes child PIDs; native /proc handoff requires Linux host validation")


def start_workflow(root: Path, install: bool) -> subprocess.Popen:
    (root / "scripts").mkdir()
    (root / ".runtime").mkdir()
    (root / "bin").mkdir()
    (root / "bin/uv").symlink_to(sys.executable)
    (root / "sync").write_text("import time\ntime.sleep(300)\n")
    child = '"$PWD/bin/uv" sync --frozen --extra dev --extra neural' if install else 'sleep 300'
    (root / "scripts/run_neural_baselines_4090.sh").write_text(
        '#!/usr/bin/env bash\nset -e\n' + child + '\nprintf "finished\\n"\n')
    with (root / ".runtime/run.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        proc = subprocess.Popen(["bash", "scripts/run_neural_baselines_4090.sh"],
                                cwd=root, pass_fds=(handle.fileno(),), start_new_session=True,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    (root / ".runtime/runner.pid").write_text(str(proc.pid))
    children = Path(f"/proc/{proc.pid}/task/{proc.pid}/children")
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            ids = children.read_text().split()
            if ids and (not install or recovery.is_original_sync(int(ids[0]), proc.pid, root)):
                return proc
        except FileNotFoundError:
            pass
        time.sleep(0.01)
    stop_workflow(proc)
    raise AssertionError("Test child did not start")


def stop_workflow(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    proc.wait(timeout=5)


@native_proc
def test_only_verified_installer_is_stopped_and_lock_transfers(tmp_path: Path) -> None:
    if not hasattr(os, "pidfd_open"):
        pytest.skip("pidfd unavailable")
    proc = start_workflow(tmp_path, install=True)
    try:
        with recovery.take_run_lock(tmp_path) as handle:
            assert proc.wait(timeout=5) != 0
            with (tmp_path / ".runtime/run.lock").open("a+") as other:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert (tmp_path / "sync").exists()
    finally:
        stop_workflow(proc)


@native_proc
def test_active_workflow_after_installation_is_left_running(tmp_path: Path) -> None:
    proc = start_workflow(tmp_path, install=False)
    try:
        with pytest.raises(RuntimeError, match="no longer at the initial uv sync"):
            recovery.take_run_lock(tmp_path)
        assert proc.poll() is None
    finally:
        stop_workflow(proc)


@native_proc
def test_unrelated_runner_pid_is_left_running(tmp_path: Path) -> None:
    proc = start_workflow(tmp_path, install=True)
    try:
        (tmp_path / ".runtime/runner.pid").write_text(str(os.getpid()))
        with pytest.raises(RuntimeError, match="not the original installer"):
            recovery.take_run_lock(tmp_path)
        assert proc.poll() is None
    finally:
        stop_workflow(proc)


def test_stalled_recovery_command_has_a_total_time_limit(tmp_path: Path) -> None:
    output = tmp_path / "process.log"
    code = recovery.monitored(
        [sys.executable, "-u", "-c", "import os,time; print(os.getpid()); time.sleep(300)"],
        root=tmp_path, env=os.environ.copy(), timeout=0.3, label="test timeout", capture=output)
    assert code == 124
    pid = int(output.read_text().strip())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.parametrize("change", ["none", "training", "other_parent", "other_directory", "other_program"])
def test_process_identity_rejects_every_unrelated_case(tmp_path: Path, change: str) -> None:
    root = tmp_path / "project"
    root.mkdir()
    proc_root = tmp_path / "proc"
    proc = proc_root / "123"
    proc.mkdir(parents=True)
    args = ["/project/uv", "sync", "--frozen", "--extra", "dev", "--extra", "neural"]
    if change == "training":
        args = ["/project/uv", "run", "python", "train.py"]
    if change == "other_program":
        args[0] = "/project/python"
    (proc / "cmdline").write_bytes(b"\0".join(s.encode() for s in args) + b"\0")
    (proc / "status").write_text("PPid:\t" + ("999" if change == "other_parent" else "100") + "\n")
    (proc / "cwd").symlink_to(tmp_path if change == "other_directory" else root)
    assert recovery.is_original_sync(123, 100, root, proc_root) == (change == "none")


@pytest.mark.parametrize("text", ["", "thing>=1\n", "thing==1\n", "-e .\n"])
def test_unpinned_or_unhashed_export_is_rejected(text: str) -> None:
    with pytest.raises(RuntimeError):
        recovery.requirements_blocks(text)


def test_lock_export_keeps_conditional_versions_and_all_hashes() -> None:
    value = 'thing==1; python_version >= "3.12" \\\n+    --hash=sha256:' + 'a' * 64 + ' \\\n+    --hash=sha256:' + 'b' * 64 + '\n'
    blocks = recovery.requirements_blocks(value)
    assert len(blocks) == 1
    assert 'python_version >= "3.12"' in blocks[0]
    assert blocks[0].count("--hash=sha256:") == 2
