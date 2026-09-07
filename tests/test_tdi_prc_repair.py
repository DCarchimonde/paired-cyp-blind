"""Regression checks for direction, missing-label scoring, and artifact identity.

The production launcher additionally requires a real Torch/Lightning callback
self-test; these CPU-light tests do not claim to exercise GPU model training.
"""
import importlib.util
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


repair = load("tdi_repair_under_test", ROOT / "scripts/tdi_prc_repair.py")
wrapper = load("prc_wrapper_under_test", ROOT / "scripts/chemprop_prc_max.py")
checkpoint_audit = load("checkpoint_audit_under_test", ROOT / "scripts/tdi_checkpoint_audit.py")


class DirectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def curve(self, scores, chosen):
        log = self.root / "attempt/train.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("Restoring states from the checkpoint path at /tmp/checkpoints/"
                       f"best-epoch={chosen}-val_prc=0.00.ckpt\n")
        metrics = log.parent / "model/model_0/trainer_logs/version_0/metrics.csv"
        metrics.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"epoch": range(len(scores)), "val/prc": scores,
                      "train_loss_epoch": [.6 - .01 * i for i in range(len(scores))]}).to_csv(metrics, index=False)
        return log

    def test_original_min_checkpoint_is_rejected_by_max_audit(self):
        log = self.curve([.2, .8], 0)
        self.assertEqual(repair.selection_audit(log, metric="prc", mode="min", patience=1)["chosen_epoch"], 0)
        with self.assertRaisesRegex(RuntimeError, "Wrong checkpoint direction"):
            repair.selection_audit(log, metric="prc", mode="max", patience=1)

    def test_highest_prc_and_correct_early_stopping_are_accepted(self):
        log = self.curve([.2, .8, .4], 1)
        result = repair.selection_audit(log, metric="prc", mode="max", patience=1)
        self.assertEqual(result["chosen_epoch"], 1)
        self.assertEqual(result["final_epoch"], 2)

    def test_wrong_early_stopping_and_missing_epoch_are_rejected(self):
        log = self.curve([.2, .8, .4, .3], 1)
        with self.assertRaisesRegex(RuntimeError, "Early stopping"):
            repair.selection_audit(log, metric="prc", mode="max", patience=1)
        path = next(log.parent.rglob("metrics.csv"))
        d = pd.read_csv(path); d.loc[1, "epoch"] = 8; d.to_csv(path, index=False)
        with self.assertRaisesRegex(RuntimeError, "Incomplete epoch coverage"):
            repair.selection_audit(log, metric="prc", mode="max", patience=1)

    def test_direction_changes_only_prc_class_metadata(self):
        class PRC:
            higher_is_better = None

        class MAE:
            higher_is_better = False

        nn = ModuleType("chemprop.nn")
        nn.MetricRegistry = {"prc": PRC, "mae": MAE}
        metrics = ModuleType("chemprop.nn.metrics")
        metrics.BinaryAUPRC = PRC
        with patch.dict(sys.modules, {"chemprop.nn": nn, "chemprop.nn.metrics": metrics}):
            result = wrapper.correct_direction()
        self.assertIsNone(result["before"])
        self.assertIs(PRC.higher_is_better, True)
        self.assertIs(MAE.higher_is_better, False)

    def test_masked_multitask_prc_and_row_identity(self):
        val = self.root / "val.csv"
        pred = self.root / "pred.csv"
        pd.DataFrame({"SMILES": ["C", "CC", "CCC", "CCCC"],
                      "A": [0, 1, 1, 0], "B": [None, 0, 1, 0]}).to_csv(val, index=False)
        p = pd.DataFrame({"SMILES": ["C", "CC", "CCC", "CCCC"],
                          "A": [.1, .9, .8, .2], "B": [.7, .1, .8, .2]})
        p.to_csv(pred, index=False)
        self.assertAlmostEqual(repair.validation_prc(val, pred, ("A", "B")), 1.0)
        p.iloc[::-1].to_csv(pred, index=False)
        with self.assertRaisesRegex(RuntimeError, "order/SMILES"):
            repair.validation_prc(val, pred, ("A", "B"))

    def test_recorded_hash_and_directory_boundaries(self):
        target = self.root / "data.csv"; target.write_text("original")
        artifact = repair.entry(self.root, target)
        self.assertEqual(repair.recorded_file(self.root, artifact), target)
        allowed = self.root / "allowed"; allowed.mkdir()
        with self.assertRaisesRegex(RuntimeError, "Unexpected artifact path"):
            repair.recorded_file(self.root, artifact, within=allowed)
        target.write_text("changed")
        with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
            repair.recorded_file(self.root, artifact)

    def test_missing_or_nonfinite_training_loss_is_rejected(self):
        log = self.curve([.2, .8, .4], 1)
        path = next(log.parent.rglob("metrics.csv"))
        d = pd.read_csv(path); d.loc[1, "train_loss_epoch"] = float("nan"); d.to_csv(path, index=False)
        with self.assertRaisesRegex(RuntimeError, "Missing/nonfinite training epoch loss"):
            repair.selection_audit(log, metric="prc", mode="max", patience=1)


class LastCheckpointTests(unittest.TestCase):
    def saved(self, epoch=1, step=2, weight=.64):
        return {"epoch": epoch, "global_step": step, "pytorch-lightning_version": "2.6.5",
                "state_dict": {"weight": np.array(weight)}}

    def audit(self, best, last):
        return checkpoint_audit.audit_last_checkpoint(best, last, {"chosen_epoch": 1, "final_epoch": 16},
                                                       tensor_equal=np.array_equal)

    def test_last_topk_epoch_is_valid_even_when_training_continues(self):
        observed = self.audit(self.saved(), self.saved())
        self.assertEqual(observed["last_saved_epoch"], 1)
        self.assertEqual(observed["final_trained_epoch"], 16)
        self.assertTrue(observed["last_weights_match_best"])

    def test_wrong_epoch_step_version_or_weights_are_rejected(self):
        for change in ({"epoch": 0}, {"epoch": 16}, {"global_step": 3},
                       {"pytorch-lightning_version": "2.5.0"}, {"state_dict": {"weight": np.array(.3)}}):
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                self.audit(self.saved(), {**self.saved(), **change})


class RevisionRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output = self.root / "runtime"
        self.output.mkdir()
        self.original = {"protocol": {"protocol_id": repair.PROTOCOL_ID},
                         "delivery_commit": repair.LEGACY_COMMIT, "package_environment": {"pinned": True},
                         "files": {"frozen": {"input.csv": "unchanged"}, "repair": dict(repair.LEGACY_CODE_HASHES)}}
        self.old_fp = repair.identity_fingerprint(self.original)
        self.registration = self.output / "RUN_REGISTRATION.json"
        repair.write_json(self.registration, {"registered_utc": "original time", "run_fingerprint": self.old_fp,
                                               **self.original})
        self.current = deepcopy(self.original)
        self.current["delivery_commit"] = "new-auditor-commit"
        self.current["files"]["repair"]["scripts/tdi_prc_repair.py"] = "new-auditor-hash"
        self.current["files"]["repair"]["scripts/start_tdi_prc_repair_4090.sh"] = "new-launcher-hash"
        self.current["files"]["repair"]["scripts/tdi_checkpoint_audit.py"] = "new-self-test-hash"

    def tearDown(self):
        self.temp.cleanup()

    def test_recognized_revision_preserves_original_registration_and_is_repeatable(self):
        original_bytes = self.registration.read_bytes()
        fp, accepted = repair.register_revision(self.output, self.current)
        self.assertEqual(accepted[self.old_fp], self.original)
        self.assertEqual(accepted[fp], self.current)
        self.assertEqual(self.registration.read_bytes(), original_bytes)
        revision = self.output / "registrations" / f"{fp}.json"
        revision_bytes = revision.read_bytes()
        self.assertEqual(repair.register_revision(self.output, self.current), (fp, accepted))
        self.assertEqual(revision.read_bytes(), revision_bytes)

    def test_training_wrapper_inputs_protocol_and_packages_cannot_change(self):
        changed = []
        for kind in ("wrapper", "inputs", "protocol", "packages"):
            identity = deepcopy(self.current)
            if kind == "wrapper":
                identity["files"]["repair"]["scripts/chemprop_prc_max.py"] = "different-training-code"
            elif kind == "inputs":
                identity["files"]["frozen"]["input.csv"] = "different-data"
            elif kind == "protocol":
                identity["protocol"]["new_threshold"] = .4
            else:
                identity["package_environment"]["pinned"] = False
            changed.append((kind, identity))
        for kind, identity in changed:
            with self.subTest(kind=kind), self.assertRaisesRegex(RuntimeError, "changed training inputs/packages"):
                repair.register_revision(self.output, identity)
        self.assertFalse((self.output / "registrations").exists())

    def test_forged_registration_is_rejected(self):
        value = json.loads(self.registration.read_text())
        value["files"]["repair"]["scripts/tdi_prc_repair.py"] = "modified"
        repair.write_json(self.registration, value)
        with self.assertRaisesRegex(RuntimeError, "fingerprint is invalid"):
            repair.register_revision(self.output, self.current)

    def test_saved_configuration_allows_only_retention_and_output_change(self):
        original = self.root / "original.toml"
        current = self.root / "current.toml"
        original.write_text("output-dir = /original/model\nremove-checkpoints = true\nloss-function = bce\nepochs = 100\n")
        current.write_text(f"output-dir = {self.root}/model\nloss-function = bce\nepochs = 100\n")
        repair.verify_saved_config(current, original, self.root)
        current.write_text(current.read_text().replace("loss-function = bce", "loss-function = different"))
        with self.assertRaisesRegex(RuntimeError, "configuration differs"):
            repair.verify_saved_config(current, original, self.root)

    def fixture_attempt(self):
        @dataclass
        class Job:
            job_id: str = "first_tdi_job"
            targets: tuple = ("A",)
            seed: int = 20260829
        job = Job()
        job_root = self.output / "jobs" / job.job_id
        attempt = job_root / "attempt_original"
        model = attempt / "model/model_0"
        checkpoints = model / "checkpoints"
        checkpoints.mkdir(parents=True)
        (checkpoints / "best-epoch=1-val_prc=0.80.ckpt").write_bytes(b"best checkpoint fixture")
        (checkpoints / "last.ckpt").write_bytes(b"last checkpoint fixture")
        (model / "best.pt").write_bytes(b"model fixture")
        (attempt / "train.log").write_text("Restoring states from the checkpoint path at /tmp/best-epoch=1-val_prc=0.80.ckpt\n")
        metrics = model / "trainer_logs/version_0/metrics.csv"
        metrics.parent.mkdir(parents=True)
        pd.DataFrame({"epoch": range(17), "val/prc": [.2, .8] + [.4] * 15,
                      "train_loss_epoch": [.6] * 17}).to_csv(metrics, index=False)
        prediction = pd.DataFrame({"SMILES": ["C", "CC"], "A": [.1, .9]})
        prediction.to_csv(model / "test_predictions.csv", index=False)
        prediction.to_csv(attempt / "predictions.csv", index=False)
        prediction.to_csv(attempt / "validation_predictions.csv", index=False)
        (attempt / "validation.log").write_text("completed inference fixture")
        (attempt / "model/config.toml").write_text(f"output-dir = {attempt}/model\nloss-function = bce\n")
        old_attempt = self.root / "old" / job.job_id / "attempt"
        (old_attempt / "model").mkdir(parents=True)
        (old_attempt / "train.log").write_text("original log")
        (old_attempt / "model/config.toml").write_text("output-dir = /old/model\nremove-checkpoints = true\nloss-function = bce\n")
        (old_attempt.parent / "COMPLETE.json").write_text("old completion fixture")
        inputs = {}
        for member in ("train", "val", "test"):
            path = self.root / f"{member}.csv"
            prediction.to_csv(path, index=False)
            inputs[member] = repair.entry(self.root, path)
        old = {"outputs": {"log": repair.entry(self.root, old_attempt / "train.log")}, "inputs": inputs}
        repair.write_json(self.output / "gpu_preflight.json", {"environment": {"fixture": True}})
        return job, job_root, attempt, old, prediction

    def test_recovery_replays_inference_preserves_artifacts_and_does_not_invent_duration(self):
        _, registrations = repair.register_revision(self.output, self.current)
        job, job_root, attempt, old, prediction = self.fixture_attempt()
        before = {str(p): repair.sha(p) for p in attempt.rglob("*") if p.is_file()}
        calls = []
        def predict(command, **kwargs):
            self.assertEqual(command[1], "predict")
            calls.append(command)
            prediction.to_csv(Path(command[command.index("-o") + 1]), index=False)
        neural = SimpleNamespace(build_chemprop_command=lambda *a, **k: ["chemprop", "train", "--remove-checkpoints"])
        with patch.object(repair.subprocess, "run", side_effect=predict):
            result = repair.recover_legacy_candidate(self.root, self.output, job, old, job_root, neural,
                                                     {"paths": {"job_root": "old"}}, self.root, registrations)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["status"], "PENDING_AUDIT")
        self.assertEqual(result["run_fingerprint"], self.old_fp)
        self.assertIsNone(result["runtime_seconds"])
        self.assertTrue(result["recovery"]["wall_duration_unknown"])
        self.assertFalse((job_root / "COMPLETE.json").exists())
        self.assertEqual(json.loads((attempt / "CANDIDATE.json").read_text()), result)
        self.assertEqual(before, {path: repair.sha(Path(path)) for path in before})

    def test_recovery_rejects_prediction_mismatch_without_marking_complete(self):
        _, registrations = repair.register_revision(self.output, self.current)
        job, job_root, attempt, old, prediction = self.fixture_attempt()
        def wrong_predict(command, **kwargs):
            prediction.assign(A=[.8, .1]).to_csv(Path(command[command.index("-o") + 1]), index=False)
        with patch.object(repair.subprocess, "run", side_effect=wrong_predict), self.assertRaises(AssertionError):
            repair.recover_legacy_candidate(self.root, self.output, job, old, job_root, None,
                                            {"paths": {"job_root": "old"}}, self.root, registrations)
        self.assertFalse((attempt / "CANDIDATE.json").exists())
        self.assertFalse((job_root / "COMPLETE.json").exists())


if __name__ == "__main__":
    unittest.main()
