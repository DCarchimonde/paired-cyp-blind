"""Check the actual save_last lifecycle in pinned Lightning 2.6.5.

In this version, epoch-end save_last follows a successful top-k save. It is
therefore the latest *saved* epoch, not necessarily the final trained epoch.
No callback or training behavior is changed here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


def audit_last_checkpoint(best: dict, last: dict, selection: dict, *, tensor_equal=None) -> dict:
    if tensor_equal is None:
        import torch
        tensor_equal = torch.equal
    chosen, final = selection["chosen_epoch"], selection["final_epoch"]
    observed = {"best_epoch": int(best["epoch"]), "last_saved_epoch": int(last["epoch"]),
                "final_trained_epoch": final, "best_global_step": int(best["global_step"]),
                "last_global_step": int(last["global_step"])}
    if (best.get("pytorch-lightning_version") != "2.6.5"
            or last.get("pytorch-lightning_version") != "2.6.5"
            or not 0 <= chosen <= final
            or observed["best_epoch"] != chosen or observed["last_saved_epoch"] != chosen
            or observed["best_global_step"] <= 0
            or observed["last_global_step"] != observed["best_global_step"]):
        raise RuntimeError(f"Unexpected Lightning 2.6.5 last-checkpoint metadata: {observed}")
    if (set(best["state_dict"]) != set(last["state_dict"])
            or any(not tensor_equal(value, last["state_dict"][key])
                   for key, value in best["state_dict"].items())):
        raise RuntimeError(f"Last checkpoint weights differ from the latest top-k save: {observed}")
    return {**observed, "last_semantics": "latest_top_k_save_lightning_2.6.5",
            "last_weights_match_best": True}


def self_test(output: Path) -> None:
    """Use different epoch weights and exercise fit -> best restore -> predict."""
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    import lightning.pytorch.callbacks.model_checkpoint as checkpoint_module
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    if pl.__version__ != "2.6.5":
        raise RuntimeError("This lifecycle check is specific to Lightning 2.6.5.")
    output.mkdir(parents=True, exist_ok=False)

    class KnownCurve(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def training_step(self, batch, batch_idx):
            return self.weight.square()

        def validation_step(self, batch, batch_idx):
            self.log("val/prc", torch.tensor([.2, .8, .4, .3][self.current_epoch]),
                     on_step=False, on_epoch=True, batch_size=1)

        def predict_step(self, batch, batch_idx):
            return self.weight.detach().clone()

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=.1)

    checkpoint = ModelCheckpoint(dirpath=output / "checkpoints", monitor="val/prc", mode="max",
                                 save_last=True, filename="best-{epoch}")
    stopping = EarlyStopping(monitor="val/prc", mode="max", patience=1, min_delta=0.)
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_epochs=4, logger=False,
                         callbacks=[checkpoint, stopping], num_sanity_val_steps=0,
                         enable_progress_bar=False, enable_model_summary=False, deterministic=True)
    loader = DataLoader(TensorDataset(torch.zeros(1)), batch_size=1, num_workers=0)
    model = KnownCurve()
    trainer.fit(model, loader, loader)
    best = torch.load(checkpoint.best_model_path, map_location="cpu", weights_only=False)
    last_path = Path(checkpoint.last_model_path)
    last_digest = hashlib.sha256(last_path.read_bytes()).hexdigest()
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    if int(best["epoch"]) != 1 or int(stopping.stopped_epoch) != 2:
        raise RuntimeError("Unexpected selected/stopped epochs in the lifecycle test.")
    evidence = audit_last_checkpoint(best, last, {"chosen_epoch": 1, "final_epoch": 2})
    if torch.equal(model.weight.detach(), best["state_dict"]["weight"]):
        raise RuntimeError("The fixture must distinguish final weights from best weights.")
    trainer.predict(dataloaders=loader, ckpt_path="best", weights_only=False)
    if (any(not torch.equal(value, model.state_dict()[key]) for key, value in best["state_dict"].items())
            or hashlib.sha256(last_path.read_bytes()).hexdigest() != last_digest):
        raise RuntimeError("Best prediction restore changed the last file or restored different weights.")
    result = {"overall_pass": True, **evidence, "prediction_restores_best": True,
              "final_weights_differ_from_best": True,
              "model_checkpoint_source_sha256": hashlib.sha256(Path(checkpoint_module.__file__).read_bytes()).hexdigest()}
    (output / "self_test.json").write_text(json.dumps(result, indent=2) + "\n")
    print("LAST_CHECKPOINT_SELF_TEST_PASS " + json.dumps(result), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: tdi_checkpoint_audit.py OUTPUT_DIRECTORY")
    self_test(Path(sys.argv[1]))
