"""Standard-library regression tests; fake launchers are not GPU evidence."""
from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


short = module("short_runtime")
recovery = module("recover_socket_run")


class ShortRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory(prefix="cyp-socket-test-")
        self.addCleanup(self.workspace.cleanup)
        self.root = Path(self.workspace.name) / ("long-project-" * 6) / "frozen-experiment"
        (self.root / ".git").mkdir(parents=True)
        self.aliases = []
        self.addCleanup(self.clean_aliases)

    def clean_aliases(self):
        for alias in self.aliases:
            if alias.is_symlink():
                alias.unlink()
            elif alias.is_file():
                alias.unlink()
            if alias.parent.is_dir():
                alias.parent.rmdir()

    def alias(self):
        alias = short.prepare(self.root)
        self.aliases.append(alias)
        return alias

    def test_reported_path_exceeds_limit_and_alias_preserves_target(self):
        original = Path("/root/autodl-tmp/paired-cyp-blind-github/.runtime/frozen-experiment/.runtime/tmp")
        self.assertEqual(short.socket_path_bytes(original), 112)
        alias = self.alias()
        self.assertLess(short.socket_path_bytes(alias / ".runtime/tmp"), 108)
        self.assertEqual(alias.resolve(), self.root.resolve())
        self.assertEqual((alias / ".runtime/tmp").resolve(), self.root / ".runtime/tmp")

    def test_python_retains_short_literal_tmpdir_and_alias_is_reused(self):
        alias = self.alias()
        before = alias.lstat().st_mtime_ns
        self.assertEqual(short.prepare(self.root), alias)
        self.assertEqual(alias.lstat().st_mtime_ns, before)
        output = subprocess.check_output(
            [sys.executable, "-c", "import tempfile; print(tempfile.gettempdir())"],
            env={**os.environ, "TMPDIR": str(alias / ".runtime/tmp")}, text=True,
        )
        self.assertEqual(output.strip(), str(alias / ".runtime/tmp"))

    def test_conflicting_alias_is_preserved(self):
        alias = self.alias()
        alias.unlink()
        alias.write_text("existing user file")
        with self.assertRaisesRegex(RuntimeError, "points elsewhere"):
            short.prepare(self.root)
        self.assertEqual(alias.read_text(), "existing user file")

    def test_real_listener_probe_when_kernel_allows_unix_sockets(self):
        alias = self.alias()
        run = subprocess.run(
            [sys.executable, str(ROOT / "scripts/probe_tensor_ipc.py")],
            env={**os.environ, "TMPDIR": str(alias / ".runtime/tmp")},
            text=True, capture_output=True, timeout=45,
        )
        if "PermissionError: [Errno 1] Operation not permitted" in run.stderr:
            self.skipTest("This executor prohibits AF_UNIX socket creation; training host must pass the probe")
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertIn("IPC SOCKET PASS", run.stdout)

    def test_original_foreground_and_background_launchers_keep_short_tmpdir(self):
        scripts = self.root / "scripts"
        scripts.mkdir()
        for name in ["run_neural_baselines_4090.sh", "start_neural_baselines_4090.sh"]:
            shutil.copy2(ROOT / "scripts" / name, scripts / name)
        chemprop = self.root / ".venv/bin/chemprop"
        chemprop.parent.mkdir(parents=True)
        chemprop.write_text("#!/bin/sh\nexit 0\n")
        chemprop.chmod(0o755)
        fake = self.root / "fake-bin"
        fake.mkdir()
        gpu = fake / "nvidia-smi"
        gpu.write_text("#!/bin/sh\necho 'SYNTHETIC driver fixture'\n")
        gpu.chmod(0o755)
        uv = fake / "uv"
        uv.write_text(
            "#!/usr/bin/env python3\nimport json, os, sys\n"
            "if sys.argv[1:] == ['--version']: print('uv 0.11.33')\n"
            "else:\n"
            " with open('.runtime/synthetic-commands.jsonl', 'a') as f:\n"
            "  f.write(json.dumps({'args':sys.argv[1:], 'tmpdir':os.environ['TMPDIR'], 'cwd':os.getcwd()})+'\\n')\n"
        )
        uv.chmod(0o755)
        alias = self.alias()
        env = {**os.environ, "PATH": str(fake) + os.pathsep + os.environ["PATH"]}
        env.pop("PAIRED_CYP_LOCK_HELD", None)
        status = self.root / ".runtime/run-status.txt"
        for launcher in ["run_neural_baselines_4090.sh", "start_neural_baselines_4090.sh"]:
            with self.subTest(launcher=launcher):
                status.unlink(missing_ok=True)
                result = subprocess.run(["bash", str(alias / "scripts" / launcher)],
                                        env=env, text=True, capture_output=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    if status.exists() and status.read_text().startswith("PASS:"):
                        break
                    time.sleep(0.025)
                self.assertTrue(status.read_text().startswith("PASS:"))
        commands = [json.loads(line) for line in (self.root / ".runtime/synthetic-commands.jsonl").read_text().splitlines()]
        self.assertTrue(commands)
        self.assertTrue(all(r["tmpdir"] == str(alias / ".runtime/tmp") for r in commands))
        self.assertTrue(all(r["cwd"] == str(self.root) for r in commands))
        self.assertEqual(sum("run-all" in r["args"] for r in commands), 2)


class RecoverySelectionTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory(prefix="cyp-process-test-")
        self.addCleanup(self.workspace.cleanup)
        self.root = Path(self.workspace.name).resolve()
        self.job = self.root / "reports/neural_v2/jobs/single_task__direct__fixture__fold0__seed1"
        self.attempt = self.job / "attempt_20260907T120000Z"
        self.attempt.mkdir(parents=True)
        self.log = self.attempt / "train.log"
        self.log.write_text("OSError: AF_UNIX path too long\n")
        P = recovery.Process
        self.runner = P(101, 1, 1, ("bash", "scripts/run_neural_baselines_4090.sh"), self.root)
        self.driver = P(102, 101, 2, ("python", "-m", "cyp_blind.neural_baselines", "run-all", "--require-gpu"), self.root)
        self.train = P(103, 102, 3, ("python3", str(self.root / ".venv/bin/chemprop"), "train", "-o", str(self.attempt / "model")), self.root)
        self.worker = replace(self.train, pid=104, ppid=103, start=4)
        self.processes = {p.pid:p for p in [self.runner, self.driver, self.train, self.worker]}

    def test_selects_model_parent_not_four_worker_command_lines(self):
        selected, evidence = recovery.select_attempt(self.root, self.processes, self.runner.pid)
        self.assertEqual(selected.pid, self.train.pid)
        self.assertEqual(evidence["job_id"], self.job.name)

    def test_healthy_training_is_not_stopped(self):
        self.log.write_text("Epoch 1 completed\n")
        with self.assertRaisesRegex(RuntimeError, "no AF_UNIX"):
            recovery.select_attempt(self.root, self.processes, self.runner.pid)

    def test_completed_job_is_preserved(self):
        (self.job / "COMPLETE.json").write_text("{}")
        with self.assertRaisesRegex(RuntimeError, "completion evidence"):
            recovery.select_attempt(self.root, self.processes, self.runner.pid)

    def test_stale_runner_and_wrong_project_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "runner identity"):
            recovery.select_attempt(self.root, self.processes, 999)
        changed = dict(self.processes)
        changed[self.runner.pid] = replace(self.runner, cwd=Path("/another-project"))
        with self.assertRaisesRegex(RuntimeError, "runner identity"):
            recovery.select_attempt(self.root, changed, self.runner.pid)

    def test_unrelated_matching_command_is_not_selected(self):
        other = replace(self.train, pid=999, ppid=1, start=999)
        self.processes[999] = other
        selected, _ = recovery.select_attempt(self.root, self.processes, self.runner.pid)
        self.assertEqual(selected.pid, self.train.pid)

    def test_pid_reuse_causes_no_signal(self):
        with patch.object(recovery, "native_process_ids", return_value=True), \
             patch.object(recovery.os, "pidfd_open", return_value=99), \
             patch.object(recovery, "read_process", return_value=None), \
             patch.object(recovery.os, "close"), \
             patch.object(recovery.signal, "pidfd_send_signal") as send:
            with self.assertRaisesRegex(RuntimeError, "identity changed"):
                recovery.stop_attempt(self.root, self.processes, self.train)
            send.assert_not_called()

    def test_foreign_child_cwd_causes_no_signal(self):
        self.processes[self.worker.pid] = replace(self.worker, cwd=Path("/another-project"))
        with patch.object(recovery, "native_process_ids", return_value=True), \
             patch.object(recovery.signal, "pidfd_send_signal") as send:
            with self.assertRaisesRegex(RuntimeError, "another directory"):
                recovery.stop_attempt(self.root, self.processes, self.train)
            send.assert_not_called()


if __name__ == "__main__":
    unittest.main()
