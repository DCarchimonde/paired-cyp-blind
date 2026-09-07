"""Explicit, process-local PRC direction correction for the recorded v3 rerun.

The installed package and frozen v2 checkout are not edited. The PRC formula,
loss, sampler, architecture, and thresholds retain their original definitions.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys


def environment() -> dict:
    expected = {"chemprop": "2.3.1", "torch": "2.5.1",
                "torchmetrics": "1.9.0", "lightning": "2.6.5"}
    observed = {name: importlib.metadata.version(name) for name in expected}
    if observed != expected or platform.python_version() != "3.12.13":
        raise RuntimeError(f"Unexpected frozen dependencies: {observed}, Python {platform.python_version()}")
    import chemprop.cli.train as train
    import chemprop.nn.metrics as metrics
    curves = importlib.import_module("torchmetrics.classification.precision_recall_curve")
    return {"versions": observed, "python": platform.python_version(),
            "package_source_sha256": {
                module.__name__: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                for module in (train, metrics, curves)}}


def correct_direction() -> dict:
    from chemprop.nn import MetricRegistry
    from chemprop.nn.metrics import BinaryAUPRC
    metric = MetricRegistry["prc"]
    before = metric.higher_is_better
    if metric is not BinaryAUPRC or before not in (None, True):
        raise RuntimeError(f"Unexpected PRC implementation/direction: {metric}, {before}")
    metric.higher_is_better = True
    if MetricRegistry["mae"].higher_is_better is not False:
        raise RuntimeError("Unexpected MAE direction; regression cannot be carried forward.")
    return {"metric": "prc", "before": before, "higher_is_better": True,
            "checkpoint_mode": "max", "early_stopping_mode": "max"}


def self_test(output: Path) -> None:
    """Exercise real Lightning checkpoints and early stopping on known scores."""
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from chemprop.nn import MetricRegistry

    info = environment()
    if MetricRegistry["prc"].higher_is_better is not None:
        raise RuntimeError("The recorded upstream PRC defect was not reproduced in this process.")
    output.mkdir(parents=True, exist_ok=False)

    class KnownCurve(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

        def training_step(self, batch, batch_idx):
            return self.weight.square()

        def validation_step(self, batch, batch_idx):
            score = [0.2, 0.8, 0.4, 0.3][self.current_epoch]
            self.log("val/prc", torch.tensor(score), on_step=False, on_epoch=True, batch_size=1)

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.0)

    def run_case(label):
        mode = "max" if MetricRegistry["prc"].higher_is_better else "min"
        checkpoint = ModelCheckpoint(dirpath=output / label, monitor="val/prc",
                                     mode=mode, filename="best-{epoch}", save_last=True)
        stopping = EarlyStopping(monitor="val/prc", mode=mode, patience=1, min_delta=0.0)
        trainer = pl.Trainer(accelerator="cpu", devices=1, max_epochs=4, logger=False,
                             callbacks=[checkpoint, stopping], num_sanity_val_steps=0,
                             enable_progress_bar=False, enable_model_summary=False,
                             deterministic=True)
        loader = DataLoader(TensorDataset(torch.zeros(1)), batch_size=1, num_workers=0)
        trainer.fit(KnownCurve(), loader, loader)
        saved = torch.load(checkpoint.best_model_path, map_location="cpu", weights_only=False)
        return {"mode": mode, "selected_epoch": int(saved["epoch"]),
                "selected_score": float(checkpoint.best_model_score),
                "stopped_epoch": int(stopping.stopped_epoch)}

    old = run_case("original_min")
    correction = correct_direction()
    fixed = run_case("corrected_max")
    if (old["selected_epoch"], old["stopped_epoch"], fixed["selected_epoch"], fixed["stopped_epoch"]) != (0, 1, 1, 2):
        raise RuntimeError(f"Checkpoint/early stopping behavioral test failed: {old}, {fixed}")
    metric = MetricRegistry["prc"]()
    metric.update(torch.tensor([[.05], [.9], [.8], [.1]]),
                  torch.tensor([[0.], [1.], [1.], [0.]]), torch.ones((4, 1), dtype=torch.bool))
    if abs(float(metric.compute()) - 1.0) > 1e-6:
        raise RuntimeError("PRC formula self-test failed.")
    if environment() != info:
        raise RuntimeError("Installed package source changed during the direction test.")
    result = {"overall_pass": True, "environment": info, "correction": correction,
              "original": old, "corrected": fixed, "prc_perfect_ranking": 1.0}
    (output / "self_test.json").write_text(json.dumps(result, indent=2) + "\n")
    print("PRC_SELF_TEST_PASS " + json.dumps(result), flush=True)


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--self-test":
        self_test(Path(sys.argv[2]))
        return
    args = sys.argv[1:]
    if not args or args[0] != "train":
        raise RuntimeError("This entry point is only for the registered TDI training repair.")
    for flag, value in [("-t", "classification"), ("-l", "bce"), ("--tracking-metric", "prc")]:
        if args.count(flag) != 1 or args[args.index(flag) + 1] != value:
            raise RuntimeError(f"Expected {flag} {value}")
    if "--class-balance" in args:
        raise RuntimeError("The v3 direction-only correction retains uniform sampling.")
    info = environment()
    correction = correct_direction()
    print("PRC_DIRECTION_FIX " + json.dumps({**info, **correction}), flush=True)
    from chemprop.cli.main import main as chemprop_main
    chemprop_main()


if __name__ == "__main__":
    main()
