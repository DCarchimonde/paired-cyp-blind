"""Regression checks for direction, missing-label scoring, and artifact identity.

The production launcher additionally requires a real Torch/Lightning callback
self-test; these CPU-light tests do not claim to exercise GPU model training.
"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


repair = load("tdi_repair_under_test", ROOT / "scripts/tdi_prc_repair.py")
wrapper = load("prc_wrapper_under_test", ROOT / "scripts/chemprop_prc_max.py")


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


if __name__ == "__main__":
    unittest.main()
