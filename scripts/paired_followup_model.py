"""One fixed v5 intervention: double CYP2D6 label BCE, preserve the v4 forward model.

The fit lifecycle is copied from frozen v4 so that v4 source fingerprints remain
unchanged. v5 adds per-endpoint validation logging; selection remains the original
unweighted composite score. Both classes receive the same endpoint multiplier.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score
from paired_model import (AssayModel, CLASS_ISOS, ISOS, atomic_torch_save, batch_graphs,
                          predict, restore_rng, rng_state, seed_everything, tensor_targets,
                          training_loss as original_training_loss, validation_score)


def training_loss(pred, target, model, statistics):
    multiplier = model.config["tdi_2d6_loss_multiplier"]
    if multiplier not in (1.0, 2.0):
        raise ValueError("Only the registered multiplier 2 and equivalence-test multiplier 1 are supported.")
    loss = original_training_loss(pred, target, model, statistics)
    if multiplier == 1:
        return loss
    bce = F.binary_cross_entropy(pred["probability"][:, 1], target["labels"][:, 1], reduction="none")
    extra = ((bce * target["label_mask"][:, 1]).sum() * statistics["n_train"]
             / (len(target["y"]) * statistics["label_counts"][1]) / 2)
    loss = loss + model.config["classification_loss_weight"] * (multiplier - 1) * extra
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite weighted training loss.")
    return loss


def endpoint_validation_diagnostics(pred, arrays, indices, statistics):
    result = {}
    for k, iso in enumerate(ISOS):
        mask = arrays["mask"][indices, k, 0]
        result[iso + "_scaled_mae"] = float(np.abs(pred["d"][mask, k] - arrays["y"][indices, k, 0][mask]).mean()
                                                  / statistics["direct_scale"][k])
    for k, iso in enumerate(CLASS_ISOS):
        mask = arrays["label_mask"][indices, k]
        y = arrays["labels"][indices, k][mask].astype(int)
        p = pred["probability"][mask, k].astype(float).clip(1e-6, 1 - 1e-6)
        if len(set(y)) != 2:
            raise ValueError("Both validation classes are required for registered endpoint diagnostics.")
        result.update({iso + "_BCE": float(-(y * np.log(p) + (1-y) * np.log1p(-p)).mean()),
                       iso + "_AP": float(average_precision_score(y, p)),
                       iso + "_AUROC": float(roc_auc_score(y, p)),
                       iso + "_MCC": float(matthews_corrcoef(y, p >= .5)),
                       iso + "_predicted_positive": int((p >= .5).sum())})
    return result


def fit_weighted(config: dict, variant: str, statistics: dict, graphs: list, arrays: dict,
        train_index, val_index, seed: int, output: Path, identity: str, device="cpu",
        stop_after_epoch: int | None = None) -> tuple[AssayModel, dict]:
    """Resume at an atomic completed epoch, including optimizer and all RNG states.

    Each rolling checkpoint contains both current and selected model states, so a
    crash between separate best/last writes cannot create an inconsistent pair.
    """
    seed_everything(seed, config["cpu_threads"])
    output.mkdir(parents=True, exist_ok=True)
    model = AssayModel(config, variant, statistics).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    path = output / "resume.pt"
    state = None
    start, history, best_score, best_epoch, best_state = 0, [], float("inf"), -1, None
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["identity"] != identity or state["statistics"] != statistics:
            raise RuntimeError("Resume identity/statistics changed; checkpoint preserved.")
        model.load_state_dict(state["model"])
        # Optimizer.load_state_dict follows each parameter's device and keeps
        # non-capturable Adam step counters on CPU, as required by PyTorch 2.5.
        optimizer.load_state_dict(state["optimizer"])
        start, history = state["epoch"] + 1, state["history"]
        best_score, best_epoch, best_state = state["best_score"], state["best_epoch"], state["best_state"]
        restore_rng(state["rng"])
        print(f"RESUME {variant}: {start} completed epochs; best={best_epoch}", flush=True)
    for epoch in range(start, config["max_epochs"]):
        if best_epoch >= 0 and epoch - 1 - best_epoch >= config["patience"]:
            break
        import time
        started = time.monotonic()
        model.train()
        shuffled = np.random.permutation(train_index)
        loss_sum = 0.0
        for offset in range(0, len(shuffled), config["batch_size"]):
            idx = shuffled[offset:offset + config["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_graphs([graphs[int(i)] for i in idx], device))
            loss = training_loss(pred, tensor_targets(arrays, idx, device), model, statistics)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
            if not torch.isfinite(norm):
                raise FloatingPointError("Non-finite gradients.")
            optimizer.step()
            loss_sum += float(loss.detach()) * len(idx)
        val = predict(model, graphs, val_index, config["batch_size"], device)
        score, components = validation_score(val, arrays, val_index, statistics)
        components.update(endpoint_validation_diagnostics(val, arrays, val_index, statistics))
        if score < best_score:
            best_score, best_epoch = score, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        history.append({"epoch": epoch, "train_loss": loss_sum / len(train_index), "selection_score": score,
                        **components, "seconds": time.monotonic() - started})
        state = {"identity": identity, "statistics": statistics, "epoch": epoch,
                 "model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": rng_state(),
                 "history": history, "best_score": best_score, "best_epoch": best_epoch, "best_state": best_state}
        atomic_torch_save(state, path)
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        print(f"  epoch {epoch + 1}/{config['max_epochs']} loss={history[-1]['train_loss']:.5f} "
              f"val={score:.5f} best_epoch={best_epoch + 1} seconds={history[-1]['seconds']:.1f}", flush=True)
        if stop_after_epoch is not None and epoch >= stop_after_epoch:
            return model, state
    if state is None or best_state is None:
        raise RuntimeError("No completed training epoch.")
    # A state_dict contains live tensor references. Freeze the final state before
    # loading the selected weights into the same module.
    state["model"] = {key: value.detach().cpu().clone() for key, value in state["model"].items()}
    # Never assume the final state is the selected state.
    model.load_state_dict(best_state)
    atomic_torch_save({"identity": identity, "config": config, "variant": variant, "statistics": statistics,
                       "best_epoch": best_epoch, "best_score": best_score, "state_dict": best_state}, output / "best.pt")
    pd.DataFrame(history).to_csv(output / "history.csv", index=False)
    return model, state

