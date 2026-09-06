"""Recover a stalled installer without changing the frozen scientific checkout.

This file belongs to the delivery repository. It is intentionally outside the
frozen experiment. Registry mirrors supply files whose hashes must match uv.lock.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import tomllib


COMMIT = "bc00641390c559ec696506d6fd07edb77c514c13"
TAG = "neural-protocol-hardened-20260906"
INDEXES = (
    "https://pypi.tuna.tsinghua.edu.cn/simple/",
    "https://mirrors.aliyun.com/pypi/simple/",
)
PYPI = "https://pypi.org/simple"


def frozen_check(root: Path) -> str:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    if git("rev-parse", "HEAD") != COMMIT or git("rev-parse", f"{TAG}^{{commit}}") != COMMIT:
        raise RuntimeError("Unexpected frozen commit or tag; nothing will be repaired.")
    if git("status", "--porcelain"):
        raise RuntimeError("Frozen source has changes; existing files are preserved.")
    if sys.version_info[:3] != (3, 12, 13):
        raise RuntimeError("Run recovery with the frozen environment's Python 3.12.13.")
    return hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest()


def installer_environment(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    for key in ("UV_INDEX", "UV_EXTRA_INDEX_URL", "UV_INDEX_URL", "UV_DEFAULT_INDEX",
                "UV_NO_VERIFY_HASHES", "UV_NO_CACHE", "UV_OFFLINE", "UV_FIND_LINKS"):
        env.pop(key, None)
    runtime = root / ".runtime"
    env.update(UV_PROJECT_ENVIRONMENT=str(root / ".venv"),
               UV_CACHE_DIR=str(runtime / "uv-cache"),
               UV_PYTHON_INSTALL_DIR=str(runtime / "uv-python"),
               TMPDIR=str(runtime / "tmp"), PYTHONUNBUFFERED="1",
               UV_HTTP_TIMEOUT="30", UV_HTTP_CONNECT_TIMEOUT="10",
               UV_HTTP_RETRIES="2", UV_CONCURRENT_DOWNLOADS="4")
    (runtime / "tmp").mkdir(exist_ok=True)
    return env


def proc_args(pid: int, proc_root: Path = Path("/proc")) -> list[str]:
    return [s.decode() for s in (proc_root / str(pid) / "cmdline").read_bytes().split(b"\0") if s]


def is_original_sync(pid: int, parent: int, root: Path, proc_root: Path = Path("/proc")) -> bool:
    try:
        proc = proc_root / str(pid)
        args = proc_args(pid, proc_root)
        status = (proc / "status").read_text()
        ppid = int(re.search(r"^PPid:\s+(\d+)$", status, re.M).group(1))
        return (ppid == parent and (proc / "cwd").resolve() == root
                and Path(args[0]).name == "uv"
                and args[1:] == ["sync", "--frozen", "--extra", "dev", "--extra", "neural"])
    except (OSError, ValueError, IndexError, AttributeError):
        return False


def take_run_lock(root: Path):
    """Stop only the verified install child; never stop a training process."""
    lock = (root / ".runtime/run.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return lock
    except BlockingIOError:
        pass
    try:
        if Path("/proc/self").resolve().name != str(os.getpid()):
            raise RuntimeError("Process IDs do not match /proc on this host; no process was stopped.")
        parent = int((root / ".runtime/runner.pid").read_text().strip())
        args = proc_args(parent)
        if (Path(args[0]).name != "bash" or len(args) != 2
                or (Path("/proc") / str(parent) / "cwd").resolve() != root
                or (root / args[1]).resolve() != root / "scripts/run_neural_baselines_4090.sh"):
            raise RuntimeError("The active process is not the original installer; it will be left running.")
        children_file = Path(f"/proc/{parent}/task/{parent}/children")
        matches = [int(p) for p in children_file.read_text().split()
                   if is_original_sync(int(p), parent, root)]
        if len(matches) != 1:
            raise RuntimeError("The active workflow is no longer at the initial uv sync; it will be left running.")
        if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
            raise RuntimeError("This host lacks safe PID handles; no process was stopped.")
        pid = matches[0]
        handle = os.pidfd_open(pid)
        try:
            if not is_original_sync(pid, parent, root):
                raise RuntimeError("Installer process changed; it will be left running.")
            print(f"Stopping the verified stalled uv sync child (PID {pid}); preserving its cache.", flush=True)
            signal.pidfd_send_signal(handle, signal.SIGTERM)
        finally:
            os.close(handle)
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return lock
            except BlockingIOError:
                time.sleep(0.2)
        raise RuntimeError("The old workflow did not release its lock; recovery will not overlap it.")
    except BaseException:
        lock.close()
        raise


def monitored(command: list[str], *, root: Path, env: dict[str, str], timeout: float,
              label: str, capture: Path | None = None) -> int:
    """Bound the entire attempt, including continuously slow HTTP transfers."""
    output = capture.open("w") if capture else None
    process = subprocess.Popen(command, cwd=root, env=env, stdout=output,
                               stderr=subprocess.STDOUT, start_new_session=True)
    started = time.monotonic()
    heartbeat = started + 30
    try:
        while process.poll() is None:
            now = time.monotonic()
            if now - started >= timeout:
                print(f"{label}: reached the {timeout:g}s limit; preserving completed cache files.", flush=True)
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                return 124
            if now >= heartbeat:
                print(f"{label}: still running, {int(now - started)}s elapsed / {timeout:g}s limit.", flush=True)
                heartbeat = now + 30
            time.sleep(0.1)
        return process.returncode
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        if output:
            output.close()


def requirements_blocks(text: str) -> list[str]:
    blocks = []
    for line in text.replace("\\\n", " ").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if not re.match(r"^[A-Za-z0-9][A-Za-z0-9_.-]*==", line) or "--hash=sha256:" not in line:
            raise RuntimeError("Expected exact versions with SHA256 hashes in the frozen export.")
        blocks.append(line + "\n")
    if not blocks:
        raise RuntimeError("Frozen dependency export is empty.")
    return blocks


def recover(root: Path, lock, lock_hash: str) -> int:
    env = installer_environment(root)
    bootstrap = root / ".runtime/uv-bootstrap/bin/uv"
    uv = str(bootstrap) if bootstrap.is_file() else shutil.which("uv")
    if not uv or not subprocess.check_output([uv, "--version"], text=True).startswith("uv 0.11.33"):
        raise RuntimeError("The existing uv 0.11.33 installer is required.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{os.getpid()}"
    evidence = root / ".runtime/download-recovery" / stamp
    evidence.mkdir(parents=True)
    status = root / ".runtime/run-status.txt"
    status.write_text(f"RECOVERING dependencies since {stamp}\n")
    python = str(root / ".venv/bin/python")
    base = [uv, "--no-config"]
    exported = subprocess.check_output(base + ["export", "--frozen", "--offline", "--extra", "dev",
                "--extra", "neural", "--no-emit-project", "--no-header", "--no-annotate"],
                cwd=root, env=env, text=True, timeout=30)
    blocks = requirements_blocks(exported)
    requirements = evidence / "requirements.txt"
    requirements.write_text(exported)
    single = evidence / "one-requirement.txt"
    pip_common = ["--python", python, "--require-hashes", "--only-binary", ":all:"]
    records = []

    def reuse(index: str) -> None:
        reused = 0
        for number, block in enumerate(blocks, 1):
            single.write_text(block)
            output = evidence / f"offline-{len(records)}-{number}.log"
            code = monitored(base + ["pip", "install", *pip_common, "--offline", "--no-deps",
                             "--default-index", index, "-r", str(single)], root=root, env=env,
                             timeout=30, label="Offline cache check", capture=output)
            reused += code == 0
            if number % 20 == 0:
                print(f"Offline cache check {number}/{len(blocks)}.", flush=True)
        records.append({"phase": "offline-cache", "index": index,
                        "successful_requirement_checks": reused, "total": len(blocks)})
        print(f"Offline cache checks: {reused}/{len(blocks)} satisfied (includes platform-inactive requirements).", flush=True)

    print("Checking existing complete cache entries offline before using a mirror.", flush=True)
    # A frozen sync can cache wheels without fetching registry index metadata.
    # Use the original lock's URLs for those entries; pip's resolver cannot find
    # them offline from package names alone. Keep already installed entries.
    locked = tomllib.loads((root / "uv.lock").read_text())
    names = sorted({p["name"] for p in locked["package"] if "registry" in p["source"]})
    cached = 0
    for number, name in enumerate(names, 1):
        exclusions = [arg for other in names if other != name
                      for arg in ("--no-install-package", other)]
        code = monitored(base + ["sync", "--frozen", "--offline", "--extra", "dev", "--extra", "neural",
                         "--inexact", "--no-install-project", *exclusions], root=root, env=env,
                         timeout=30, label=f"Frozen cache check {name}",
                         capture=evidence / f"original-cache-{number}-{name}.log")
        cached += code == 0
        if number % 20 == 0:
            print(f"Frozen cache check {number}/{len(names)}.", flush=True)
    records.append({"phase": "original-frozen-cache", "successful_requirement_checks": cached,
                    "total": len(names)})
    print(f"Frozen cache checks: {cached}/{len(names)} satisfied (includes platform-inactive requirements).", flush=True)
    succeeded = None
    for index in INDEXES:
        print(f"Installing exact, hash-verified dependencies from {index}; attempt limit 15 minutes.", flush=True)
        code = monitored(base + ["pip", "sync", *pip_common, "--default-index", index,
                         str(requirements)], root=root, env=env, timeout=900,
                         label=f"Mirror {index}")
        records.append({"phase": "mirror", "index": index, "exit_code": code})
        if code == 0:
            succeeded = index
            break
        reuse(index)
    audit = {"frozen_commit": COMMIT, "uv_lock_sha256": lock_hash, "attempts": records,
             "requirements_sha256": hashlib.sha256(requirements.read_bytes()).hexdigest(),
             "recovery_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "dependency_recovery_pass": False}
    audit_path = evidence / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    if succeeded is None:
        raise RuntimeError("Both mirror attempts failed. Cache and evidence are preserved; training was not launched.")
    sync = base + ["sync", "--frozen", "--extra", "dev", "--extra", "neural"]
    code = monitored(sync + ["--offline"], root=root, env=env, timeout=120, label="Frozen offline sync")
    if code != 0:
        # The project build backend may not yet be cached; dependency versions stay locked.
        build_env = dict(env, UV_DEFAULT_INDEX=succeeded)
        code = monitored(sync, root=root, env=build_env, timeout=300, label="Frozen project installation")
        if code != 0:
            raise RuntimeError("Frozen project installation failed; training was not launched.")
    if monitored(sync + ["--offline"], root=root, env=env, timeout=120,
                 label="Final frozen offline sync") != 0:
        raise RuntimeError("Offline frozen environment verification failed.")
    if frozen_check(root) != lock_hash:
        raise RuntimeError("Frozen source or lockfile changed during recovery.")
    subprocess.run(base + ["pip", "check", "--python", python], cwd=root, env=env, check=True, timeout=30)
    audit.update(dependency_recovery_pass=True, selected_index=succeeded,
                 frozen_source_unchanged=True, offline_frozen_sync_pass=True)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(f"DEPENDENCY RECOVERY PASS. Evidence: {audit_path}", flush=True)
    print("Resuming the original frozen workflow; GPU training has not been claimed complete.", flush=True)
    # Only uv is offline. The original Python data-fetch step retains network access.
    env.update(UV_OFFLINE="1", PAIRED_CYP_LOCK_HELD="1")
    process = subprocess.Popen(["bash", "scripts/run_neural_baselines_4090.sh"], cwd=root,
                               env=env, pass_fds=(lock.fileno(),))
    (root / ".runtime/runner.pid").write_text(str(process.pid) + "\n")
    return process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    root = parser.parse_args().root.resolve()
    lock = None
    try:
        lock_hash = frozen_check(root)
        lock = take_run_lock(root)
        return recover(root, lock, lock_hash)
    except Exception as error:
        message = f"DOWNLOAD RECOVERY FAILED: {error}"
        print(message, flush=True)
        if lock is not None:
            (root / ".runtime/run-status.txt").write_text(message + "\n")
        return 1
    finally:
        if lock is not None:
            lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
