"""Actual weighting, training, checkpoint and restart tests for v5; no mock fits."""
from pathlib import Path
import json
import tempfile

import numpy as np
import torch

from paired_model import (AssayModel, batch_graphs, fit, load_selected, predict, seed_everything,
                          tensor_targets, training_loss as base_loss, training_statistics)
from paired_followup_model import fit_weighted, training_loss
from paired_self_test import self_test as base_self_test, toy_data
from paired_prototype import audit_checkpoint, latent_frame, write_json


def self_test(output: Path, config_path: Path, result_path: Path | None = None):
    output.mkdir(parents=True, exist_ok=True)
    original = base_self_test(output / "original_v4", config_path)
    base = json.loads(config_path.read_text())
    small = {**base, "hidden_dim": 24, "max_epochs": 3, "patience": 3, "batch_size": 16, "cpu_threads": 1}
    raw, family, table, graphs, arrays = toy_data()
    parts = {"train": np.arange(48), "val": np.arange(48, 64), "test": np.arange(64, 80)}
    stats = training_statistics(arrays, parts["train"])
    results = []
    with tempfile.TemporaryDirectory(prefix="actual-", dir=output) as temporary:
        temporary = Path(temporary)
        for variant in ("independent_no_std", "paired_no_std"):
            seed_everything(11, 1)
            cfg = {**small, "tdi_2d6_loss_multiplier": 2.0}
            model = AssayModel(cfg, variant, stats)
            model.eval()
            target = tensor_targets(arrays, np.arange(16), "cpu")
            pred = model(batch_graphs(graphs[:16], "cpu"))
            old, new = base_loss(pred, target, model, stats), training_loss(pred, target, model, stats)
            p = pred["probability"][:, 1].detach().numpy().astype(float)
            y = target["labels"][:, 1].numpy().astype(float)
            mask = target["label_mask"][:, 1].numpy()
            extra = float((-(y * np.log(p) + (1-y) * np.log1p(-p)) * mask).sum() * stats["n_train"] / (16 * stats["label_counts"][1]) / 2)
            np.testing.assert_allclose(float((new - old).detach()), extra, atol=3e-7, rtol=2e-6)
            go = torch.autograd.grad(old, pred["probability"], retain_graph=True)[0]
            gn = torch.autograd.grad(new, pred["probability"], retain_graph=True)[0]
            torch.testing.assert_close(gn[:, 0], go[:, 0], atol=0, rtol=0)
            torch.testing.assert_close(gn[:, 1], 2 * go[:, 1], atol=0, rtol=0)
            if not torch.equal(gn[~target["label_mask"]], torch.zeros_like(gn[~target["label_mask"]])):
                raise AssertionError("Missing labels received a gradient.")
            # Multiplier 1 must have the exact original training trajectory,
            # including selected weights, despite additional diagnostic logging.
            _, original_state = fit(small, variant, stats, graphs, arrays, parts["train"], parts["val"], 11,
                                    temporary / (variant + "-original"), "equivalence", "cpu")
            _, copied_state = fit_weighted({**small, "tdi_2d6_loss_multiplier": 1.0}, variant, stats, graphs, arrays,
                                           parts["train"], parts["val"], 11, temporary / (variant + "-copy"), "equivalence", "cpu")
            for key in original_state["model"]:
                if not torch.equal(original_state["model"][key], copied_state["model"][key]):
                    raise AssertionError("Copied fit changed the original trajectory at multiplier 1.")
            if (original_state["best_epoch"], original_state["best_score"]) != (copied_state["best_epoch"], copied_state["best_score"]):
                raise AssertionError("Copied fit changed the original checkpoint selection.")
            directory = temporary / (variant + "-weighted")
            model, state = fit_weighted(cfg, variant, stats, graphs, arrays, parts["train"], parts["val"], 11,
                                         directory, "weighted", "cpu")
            restart_dir = directory / "restart"
            fit_weighted(cfg, variant, stats, graphs, arrays, parts["train"], parts["val"], 11, restart_dir,
                         "weighted", "cpu", stop_after_epoch=0)
            _, resumed = fit_weighted(cfg, variant, stats, graphs, arrays, parts["train"], parts["val"], 11,
                                      restart_dir, "weighted", "cpu")
            for key in state["model"]:
                if not torch.equal(state["model"][key], resumed["model"][key]):
                    raise AssertionError("Weighted fit restart changed final weights.")
            if state["best_score"] != resumed["best_score"] or state["best_epoch"] != resumed["best_epoch"]:
                raise AssertionError("Weighted restart changed selected epoch.")
            for part in ("val", "test"):
                out = predict(model, graphs, parts[part], cfg["batch_size"], "cpu")
                latent_frame(out, table, parts[part]).to_csv(directory / (part + "_latents.csv"), index=False)
            audit_checkpoint(directory, "weighted", stats, cfg, model, graphs, arrays, parts, table, "cpu")
            loaded, saved = load_selected(directory / "best.pt", "weighted", "cpu")
            if saved["config"]["tdi_2d6_loss_multiplier"] != 2:
                raise AssertionError("Selected checkpoint lost the intervention configuration.")
            for parameter in loaded.parameters():
                if not torch.isfinite(parameter).all():
                    raise AssertionError("Weighted training produced non-finite weights.")
            results.append({"variant": variant, "multiplier": 2, "epochs": len(state["history"]),
                            "best_epoch": state["best_epoch"], "mask_and_analytic_loss_pass": True,
                            "only_2D6_label_probability_gradient_doubled": True,
                            "multiplier_one_matches_original_final_weights_bitwise": True,
                            "restart_final_weights_bitwise_equal": True, "saved_model_audit_pass": True})
    result = {"overall_pass": True, "device": "cpu", "engineering_only": True,
              "original_v4_self_test": original, "weighted_variants": results}
    if result_path is not None:
        write_json(result_path, result)
    print("V5_CPU_SELF_TEST_PASS: analytic loss, masks, original-fit equivalence, actual weighted training and restart.", flush=True)
    return result
