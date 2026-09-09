"""Descriptive v4 training/internal-validation diagnostics; never select on outer rows."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

from paired_model import (CLASS_ISOS, ISOS, AssayModel, assay_probability, batch_graphs,
                          observation_nll, predict, tensor_targets, training_statistics)
from paired_prototype import job_id, partitions, sha, write_json


def classification_rows(pred, arrays, indices, common):
    rows = []
    for k, iso in enumerate(CLASS_ISOS):
        mask = arrays["label_mask"][indices, k]
        y = arrays["labels"][indices, k][mask].astype(int)
        p = pred["probability"][mask, k].astype(float).clip(1e-6, 1 - 1e-6)
        if not len(y):
            raise ValueError("Diagnostic endpoint is empty.")
        decision = p >= .5
        rows.append({**common, "endpoint": iso + "_is_TDI", "n": len(y), "positives": int(y.sum()),
                     "predicted_positive": int(decision.sum()), "TP": int((decision & (y == 1)).sum()),
                     "FP": int((decision & (y == 0)).sum()), "MCC": float(matthews_corrcoef(y, decision)),
                     "AUROC": float(roc_auc_score(y, p)) if len(set(y)) == 2 else np.nan,
                     "AP": float(average_precision_score(y, p)) if y.sum() else np.nan,
                     "BCE": float(-(y * np.log(p) + (1 - y) * np.log1p(-p)).mean()),
                     "p_mean": float(p.mean()), "p_max": float(p.max()),
                     "p_positive_mean": float(p[y == 1].mean()) if y.sum() else np.nan,
                     "p_negative_mean": float(p[y == 0].mean()) if (y == 0).sum() else np.nan,
                     "reportability_mean": float(pred["reportability"][mask, k].mean())})
    return rows


def gradient_diagnostics(model, graphs, arrays, train, statistics, *, sample_size=256):
    """Fixed uniform training subset, dropout off, gradients accumulated before cosine.

    These local gradients of already selected fits are descriptive, not proof of
    causal interference. Full-training denominators retain the actual loss weights.
    No validation/outer labels enter this calculation.
    """
    chosen = np.random.default_rng(20260909).permutation(train)[:sample_size]
    device = next(model.parameters()).device
    parameters = list(model.encoder.parameters())
    terms = ("continuous_2D6", "continuous_other", "classification_2D6", "classification_3A4", "reportability")
    gradients = {name: torch.zeros(sum(p.numel() for p in parameters), dtype=torch.float64) for name in terms}
    totals = dict.fromkeys(terms, 0.0)
    model.eval()
    for start in range(0, len(chosen), 64):
        ix = chosen[start:start + 64]
        target = tensor_targets(arrays, ix, device)
        out = model(batch_graphs([graphs[int(i)] for i in ix], device))
        nll = observation_nll(out, target, model.use_std, model.config["pic50_floor"])
        n = statistics["n_train"] / len(chosen)
        continuous = nll.sum(0) * n / nll.new_tensor(statistics["continuous_counts"]) / 8
        bce = F.binary_cross_entropy(out["probability"], target["labels"], reduction="none")
        cls = (bce * target["label_mask"]).sum(0) * n / nll.new_tensor(statistics["label_counts"]) / 2
        report = F.binary_cross_entropy_with_logits(out["reportability_logits"], target["reported"], reduction="none")
        reporting = ((report * target["label_mask"]).sum(0) * n / nll.new_tensor(statistics["label_counts"])).mean()
        losses = {"continuous_2D6": continuous[2].sum() * model.config["continuous_loss_weight"],
                  "continuous_other": continuous[[0, 1, 3]].sum() * model.config["continuous_loss_weight"],
                  "classification_2D6": cls[1] * model.config["classification_loss_weight"],
                  "classification_3A4": cls[0] * model.config["classification_loss_weight"],
                  "reportability": reporting * model.config["reportability_loss_weight"]}
        for name, loss in losses.items():
            grad = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
            flat = torch.cat([(g if g is not None else torch.zeros_like(p)).detach().flatten().cpu().double()
                              for g, p in zip(grad, parameters, strict=True)])
            gradients[name] += flat
            totals[name] += float(loss.detach())
    all_other = sum(gradients[name] for name in terms if name != "classification_2D6")
    gradients["all_except_classification_2D6"] = all_other
    reference = gradients["classification_2D6"]
    rows = []
    for name, vector in gradients.items():
        denominator = float(reference.norm() * vector.norm())
        rows.append({"term": name, "training_sample_n": len(chosen), "weighted_loss": totals.get(name, np.nan),
                     "encoder_gradient_norm": float(vector.norm()),
                     "cosine_to_2D6_classification": float(reference.dot(vector)) / denominator if denominator else np.nan})
    return rows


def run_diagnostics(v4: Path, output: Path, raw, family, arrays, graphs, *, device="cpu", gradient_sample=256):
    """Read selected v4 weights, evaluate TRAIN and VAL only, leave v4 untouched."""
    registration = json.loads((v4 / "RUN_REGISTRATION.json").read_text())
    config = registration["identity"]["config"]
    output.mkdir(parents=True, exist_ok=True)
    scores, losses, sensitivity, gradients, selection, inputs = [], [], [], [], [], {}
    for fold in config["folds"]:
        parts = partitions(raw, family, fold)
        stats = training_statistics(arrays, parts["train"])
        for variant in config["variants"]:
            name = job_id(variant, fold, config["seeds"][0])
            directory = v4 / "jobs" / name
            record = json.loads((directory / "COMPLETE.json").read_text())
            inputs[name] = sha(directory / "best.pt")
            if inputs[name] != record["files"]["best.pt"]:
                raise AssertionError("V4 diagnostic checkpoint changed: " + name)
            saved = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
            if saved["identity"] != record["identity"] or saved["statistics"] != stats or saved["config"] != config:
                raise AssertionError("V4 diagnostic checkpoint identity mismatch.")
            model = AssayModel(config, variant, stats).to(device)
            model.load_state_dict(saved["state_dict"])
            common = {"variant": variant, "fold": fold, "seed": config["seeds"][0]}
            print("DIAGNOSE training/internal validation " + name, flush=True)
            for part in ("train", "val"):
                ix = parts[part]
                pred = predict(model, graphs, ix, config["batch_size"], device)
                scores.extend(classification_rows(pred, arrays, ix, {**common, "partition": part}))
                target = tensor_targets(arrays, ix, "cpu")
                tensors = {key: torch.from_numpy(value) for key, value in pred.items()}
                nll = observation_nll(tensors, target, model.use_std, config["pic50_floor"]).numpy()
                for k, iso in enumerate(ISOS):
                    for a, arm in enumerate(("direct", "treated")):
                        mask = arrays["mask"][ix, k, a]
                        noise = pred["sd"][mask, k, a]
                        measured = arrays["std"][ix, k, a][mask]
                        losses.append({**common, "partition": part, "iso": iso, "arm": arm, "n": int(mask.sum()),
                                       "NLL": float(nll[mask, k, a].mean()),
                                       "residual_sd_median": float(np.median(noise)),
                                       "reported_sd_median": float(np.median(measured)),
                                       "reported_variance_fraction_median": float(np.median(measured**2 / (measured**2 + noise**2)))})
                if model.paired:
                    # A fixed-weight sensitivity experiment: only reference SD at
                    # inference is toggled. It is not a retrained model comparison.
                    ref = torch.tensor(stats["reference_std"], dtype=torch.float32)
                    altered_sd = (tensors["sd"].square() + (0 if model.use_std else ref.square())).sqrt()
                    p = assay_probability(tensors["d"][:, [3, 2]], tensors["centers_t"][:, [3, 2]],
                                          altered_sd[:, [3, 2], 0], altered_sd[:, [3, 2], 1],
                                          tensors["weights"][:, [3, 2]], model.nodes.cpu(), model.quadrature_weights.cpu())
                    altered = (p * tensors["reportability"]).clamp(1e-6, 1 - 1e-6).numpy()
                    for k, iso in enumerate(CLASS_ISOS):
                        mask = arrays["label_mask"][ix, k]
                        delta = altered[mask, k] - pred["probability"][mask, k]
                        sensitivity.append({**common, "partition": part, "endpoint": iso + "_is_TDI",
                                            "change": "remove_reference_std" if model.use_std else "add_reference_std",
                                            "mean_absolute_probability_change": float(np.abs(delta).mean()),
                                            "max_absolute_probability_change": float(np.abs(delta).max()),
                                            "decisions_changed_at_half": int(((altered[mask, k] >= .5) != (pred["probability"][mask, k] >= .5)).sum())})
            gradients.extend({**common, **row} for row in gradient_diagnostics(model, graphs, arrays, parts["train"], stats,
                                                                              sample_size=gradient_sample))
            history = pd.read_csv(directory / "history.csv")
            selection.append({**common, "epochs": len(history), "selected_epoch": saved["best_epoch"],
                              "minimum_direct_component_epoch": int(history.direct_scaled_mae.idxmin()),
                              "minimum_TDI_component_epoch": int(history.tdi_bce.idxmin()),
                              "note": "component minima are diagnostics; original composite selection is unchanged"})
            del model
    tables = {"classification_train_val.csv": scores, "observation_losses_train_val.csv": losses,
              "fixed_weight_std_sensitivity.csv": sensitivity, "training_gradient_diagnostics.csv": gradients,
              "selection_diagnostics.csv": selection}
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(output / name, index=False)
    result = {"complete": True, "v4_registration": registration["fingerprint"], "selected_weight_sha256": inputs,
              "partitions_evaluated": ["train", "val"], "outer_predictions_used": False,
              "threshold_tuned": False, "gradient_sample": gradient_sample, "device": str(device),
              "outputs": {name: sha(output / name) for name in tables},
              "interpretation": "Descriptive fitted-model diagnostics; local gradients do not establish causal interference."}
    write_json(output / "DIAGNOSTIC_AUDIT.json", result)
    return result
