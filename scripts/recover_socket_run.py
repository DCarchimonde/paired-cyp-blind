"""Stop only a verified Chemprop attempt with the AF_UNIX pathname error."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import time


@dataclass(frozen=True)
class Process:
    pid: int
    ppid: int
    start: int
    args: tuple[str, ...]
    cwd: Path


def read_process(pid: int, proc_root: Path = Path("/proc")) -> Process | None:
    try:
        directory = proc_root / str(pid)
        if directory.stat().st_uid != os.getuid():
            return None
        fields = (directory / "stat").read_text().rsplit(") ", 1)[1].split()
        args = tuple(s.decode() for s in (directory / "cmdline").read_bytes().split(b"\0") if s)
        if not args or fields[0] == "Z":
            return None
        return Process(pid, int(fields[1]), int(fields[19]), args,
                       (directory / "cwd").resolve(strict=True))
    except (OSError, ValueError, IndexError, UnicodeError):
        return None


def snapshot() -> dict[int, Process]:
    result = {}
    for directory in Path("/proc").iterdir():
        if directory.name.isdigit():
            process = read_process(int(directory.name))
            if process is not None:
                result[process.pid] = process
    return result


def ancestors(process: Process, processes: dict[int, Process]) -> list[Process]:
    result, seen = [], {process.pid}
    parent = processes.get(process.ppid)
    while parent is not None and parent.pid not in seen:
        result.append(parent)
        seen.add(parent.pid)
        parent = processes.get(parent.ppid)
    return result


def model_output(process: Process, root: Path) -> Path | None:
    if process.cwd != root:
        return None
    args = process.args
    if not any(Path(arg).name == "chemprop" and args[i + 1:i + 2] == ("train",)
               for i, arg in enumerate(args)):
        return None
    if args.count("-o") != 1:
        return None
    try:
        output = Path(args[args.index("-o") + 1])
        if not output.is_absolute():
            output = root / output
        output = output.resolve()
        relative = output.relative_to(root / "reports/neural_v2/jobs")
        if (len(relative.parts) != 3 or relative.parts[2] != "model"
                or not relative.parts[1].startswith("attempt_")):
            return None
        return output
    except (IndexError, ValueError, OSError):
        return None


def failed_attempt(process: Process, root: Path) -> dict:
    output = model_output(process, root)
    if output is None:
        raise RuntimeError("Process is not this project's Chemprop training command.")
    attempt, job = output.parent, output.parent.parent
    if (job / "COMPLETE.json").exists():
        raise RuntimeError("The selected job already has completion evidence; preserved.")
    log = attempt / "train.log"
    if log.is_symlink():
        raise RuntimeError("Unexpected training-log symlink; preserved.")
    with log.open("rb") as stream:
        stream.seek(max(0, log.stat().st_size - 131072))
        tail = stream.read()
    if b"OSError: AF_UNIX path too long" not in tail:
        raise RuntimeError("The current attempt has no AF_UNIX pathname error; left running.")
    return {"job_id": job.name, "attempt": str(attempt.relative_to(root)),
            "log_tail_sha256": hashlib.sha256(tail).hexdigest()}


def select_attempt(root: Path, processes: dict[int, Process], runner_pid: int) -> tuple[Process, dict]:
    runner = processes.get(runner_pid)
    if (runner is None or runner.cwd != root or len(runner.args) != 2
            or Path(runner.args[0]).name != "bash"
            or (root / runner.args[1]).resolve() != root / "scripts/run_neural_baselines_4090.sh"):
        raise RuntimeError("Recorded runner identity is not current; no process was stopped.")
    matching = {p.pid: p for p in processes.values()
                if model_output(p, root) is not None
                and runner_pid in {a.pid for a in ancestors(p, processes)}}
    # DataLoader workers can inherit the same command line. Select their
    # highest matching ancestor, not each worker as a separate model run.
    parents = [p for p in matching.values()
               if not any(a.pid in matching for a in ancestors(p, processes))]
    if len(parents) != 1:
        raise RuntimeError("Expected exactly one current Chemprop attempt; nothing was stopped.")
    process = parents[0]
    if not any("cyp_blind.neural_baselines" in a.args and "run-all" in a.args
               and "--require-gpu" in a.args for a in ancestors(process, processes)):
        raise RuntimeError("The selected process is not under the frozen run-all workflow.")
    return process, failed_attempt(process, root)


def wait_handles(handles: list[int], seconds: float) -> list[int]:
    pending = set(handles)
    deadline = time.monotonic() + seconds
    while pending and time.monotonic() < deadline:
        ready, _, _ = select.select(list(pending), [], [], min(0.2, max(0, deadline - time.monotonic())))
        pending.difference_update(ready)
    return list(pending)


def native_process_ids() -> bool:
    return Path("/proc/self").resolve().name == str(os.getpid())


def stop_attempt(root: Path, processes: dict[int, Process], process: Process) -> list[int]:
    if not native_process_ids():
        raise RuntimeError("Process IDs do not match /proc on this host; no process was stopped.")
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise RuntimeError("Safe PID handles are unavailable; no process was stopped.")
    subtree = [p for p in processes.values()
               if p.pid == process.pid or process.pid in {a.pid for a in ancestors(p, processes)}]
    if any(p.cwd != root for p in subtree):
        raise RuntimeError("A child process belongs to another directory; nothing was stopped.")
    subtree.sort(key=lambda p: len(ancestors(p, processes)), reverse=True)
    handles = []
    try:
        # Open and verify every handle before sending any signal. Reused PIDs,
        # changed parentage, and commands from another project are rejected.
        for item in subtree:
            try:
                handle = os.pidfd_open(item.pid)
            except ProcessLookupError:
                continue
            handles.append((item.pid, handle))
            if read_process(item.pid) != item:
                raise RuntimeError("Process identity changed; no further action was taken.")
        failed_attempt(process, root)
        print(f"Stopping failed Chemprop PID {process.pid} and its verified workers; preserving all attempts.", flush=True)
        for _, handle in handles:
            try:
                signal.pidfd_send_signal(handle, signal.SIGTERM)
            except ProcessLookupError:
                pass
        pending = wait_handles([h for _, h in handles], 8)
        for handle in pending:
            # These are the same verified processes, addressed by stable
            # kernel handles, not a new process that reused the numeric PID.
            try:
                signal.pidfd_send_signal(handle, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if wait_handles(pending, 5):
            raise RuntimeError("A failed process did not exit; a new run will not be started.")
        return [pid for pid, _ in handles]
    finally:
        for _, handle in handles:
            os.close(handle)


def recover(root: Path) -> None:
    root = root.resolve(strict=True)
    runtime = root / ".runtime"
    with (runtime / "run.lock").open("a+") as lock:
        processes = snapshot()
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            runner = int((runtime / "runner.pid").read_text().strip())
            process, evidence = select_attempt(root, processes, runner)
            evidence["stopped_pids"] = stop_attempt(root, processes, process)
            deadline = time.monotonic() + 20
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError("The original workflow still holds its lock; restart refused.")
                    time.sleep(0.2)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            directory = runtime / "socket-recovery"
            directory.mkdir(exist_ok=True)
            evidence.update(stopped_utc=stamp, error="AF_UNIX path too long",
                            training_completed=False, attempts_preserved=True)
            (directory / f"{stamp}.json").write_text(json.dumps(evidence, indent=2) + "\n")
            (runtime / "run-status.txt").write_text("STOPPED failed AF_UNIX attempt; ready for short-path restart.\n")
        else:
            if any(model_output(p, root) is not None for p in processes.values()):
                raise RuntimeError("An untracked training process still exists; restart refused.")
        print("SOCKET RECOVERY READY: original workflow stopped; all existing results and logs retained.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    try:
        recover(parser.parse_args().root)
    except Exception as error:
        parser.exit(1, f"SOCKET RECOVERY FAILED: {error}\n")
