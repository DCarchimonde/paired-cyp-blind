"""Small, explicit D-MPNN and paired-assay model; no Lightning callbacks.

The paired arm is a minimum three-state hurdle: conditional magnitudes are
learned point masses, with Gaussian observation error. It is not the final
continuous-amplitude/Bayesian model described in the project blueprint.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from rdkit import Chem

ISOS = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
CLASS_ISOS = ("CYP3A4", "CYP2D6")
CLASS_INDEX = (3, 2)
DIRECT = tuple(f"{x}_pIC50_direct_inhibition" for x in ISOS)
TREATED = tuple(f"{x}_pIC50_TDI_condition" for x in ISOS)
LABELS = tuple(f"{x}_is_TDI" for x in CLASS_ISOS)


def seed_everything(seed: int, threads: int = 4) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def one_hot(value, choices) -> list[float]:
    return [float(value == x) for x in choices] + [float(value not in choices)]


def molecular_graph(smiles: str) -> dict[str, torch.Tensor]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"Invalid molecular graph: {smiles}")
    atoms = []
    for a in mol.GetAtoms():
        atoms.append(
            one_hot(a.GetAtomicNum(), [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53])
            + one_hot(a.GetTotalDegree(), list(range(6)))
            + one_hot(a.GetFormalCharge(), [-2, -1, 0, 1, 2])
            + one_hot(a.GetTotalNumHs(), list(range(5)))
            + one_hot(int(a.GetHybridization()), [1, 2, 3, 4, 6, 7])
            + one_hot(int(a.GetChiralTag()), [0, 1, 2, 3])
            + [float(a.GetIsAromatic()), float(a.IsInRing()), a.GetMass() * 0.01]
        )
    edges, features, reverse = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        b = (one_hot(float(bond.GetBondTypeAsDouble()), [1.0, 2.0, 3.0, 1.5])
             + [float(bond.GetIsConjugated()), float(bond.IsInRing())]
             + one_hot(int(bond.GetStereo()), list(range(6))))
        k = len(edges)
        edges.extend([(i, j), (j, i)])
        features.extend([b, b])
        reverse.extend([k + 1, k])
    return {
        "atoms": torch.tensor(atoms, dtype=torch.float32),
        "bonds": torch.tensor(features, dtype=torch.float32).reshape(-1, 14),
        "edges": torch.tensor(edges, dtype=torch.long).reshape(-1, 2),
        "reverse": torch.tensor(reverse, dtype=torch.long),
    }


def batch_graphs(graphs: list[dict], device: str | torch.device) -> dict:
    atoms, bonds, edges, reverse, membership = [], [], [], [], []
    na = nb = 0
    for k, g in enumerate(graphs):
        atoms.append(g["atoms"])
        bonds.append(g["bonds"])
        edges.append(g["edges"] + na)
        reverse.append(g["reverse"] + nb)
        membership.append(torch.full((len(g["atoms"]),), k, dtype=torch.long))
        na += len(g["atoms"])
        nb += len(g["bonds"])
    return {"atoms": torch.cat(atoms).to(device), "bonds": torch.cat(bonds).to(device),
            "edges": torch.cat(edges).to(device), "reverse": torch.cat(reverse).to(device),
            "membership": torch.cat(membership).to(device), "size": len(graphs)}


class DirectedEncoder(nn.Module):
    """Directed bond messages exclude the reverse edge; atom means pool graphs.

    This implementation shares the D-MPNN construction, but uses its own explicit
    feature vocabulary. It does not claim numerical identity to Chemprop V2.
    All four v4 variants use this exact same encoder.
    """
    def __init__(self, hidden: int, depth: int, dropout: float):
        super().__init__()
        atom_dim = molecular_graph("C")["atoms"].shape[1]
        self.input = nn.Linear(atom_dim + 14, hidden, bias=False)
        self.message = nn.Linear(hidden, hidden, bias=False)
        self.atom = nn.Linear(atom_dim + hidden, hidden)
        self.readout = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(dropout)
        self.depth = depth

    def forward(self, g: dict) -> torch.Tensor:
        source, destination = g["edges"].unbind(1)
        initial = F.relu(self.input(torch.cat([g["atoms"][source], g["bonds"]], 1)))
        hidden = initial
        for _ in range(self.depth - 1):
            incoming = hidden.new_zeros((len(g["atoms"]), hidden.shape[1]))
            incoming.index_add_(0, destination, hidden)
            hidden = self.drop(F.relu(initial + self.message(incoming[source] - hidden[g["reverse"]])))
        incoming = hidden.new_zeros((len(g["atoms"]), hidden.shape[1]))
        incoming.index_add_(0, destination, hidden)
        atom = self.drop(F.relu(self.atom(torch.cat([g["atoms"], incoming], 1))))
        pooled = atom.new_zeros((g["size"], atom.shape[1]))
        pooled.index_add_(0, g["membership"], atom)
        counts = torch.bincount(g["membership"], minlength=g["size"]).to(atom.dtype)
        return self.drop(F.relu(self.readout(pooled / counts[:, None])))


def assay_probability(mu_d, centers_t, sd_d, sd_t, weights, nodes, quadrature_weights,
                      floor: float = 4.0, shift: float = math.log10(2.0)):
    """P(T > max(D, floor) + shift), conditional on both arms being reported.

    A bivariate Gaussian tail is integrated over the correlation angle (Plackett
    identity). Double precision prevents cancellation in the negative-correlation
    tail. No observed assay result or per-row uncertainty is an inference input.
    """
    dtype = mu_d.dtype
    md, mt, sd, st = [x.double() for x in (mu_d, centers_t, sd_d, sd_t)]
    a = ((floor - md) / sd)[..., None]
    delta_sd = (sd.square() + st.square()).sqrt()[..., None]
    b = (shift - (mt - md[..., None])) / delta_sd
    angle = torch.asin(-sd[..., None] / delta_sd)
    theta = angle[..., None] * nodes.double()
    sine = theta.sin()
    exponent = -(a[..., None].square() - 2 * a[..., None] * b[..., None] * sine
                 + b[..., None].square()) / (2 * theta.cos().square())
    integral = angle / (2 * math.pi) * (exponent.exp() * quadrature_weights.double()).sum(-1)
    joint_tail = torch.special.ndtr(-a) * torch.special.ndtr(-b) + integral
    floor_tail = torch.special.ndtr(a) * torch.special.ndtr((mt - floor - shift) / st[..., None])
    component = (joint_tail + floor_tail).clamp(0.0, 1.0)
    return (weights.double() * component).sum(-1).to(dtype)


class AssayModel(nn.Module):
    def __init__(self, config: dict, variant: str, statistics: dict):
        super().__init__()
        if variant not in ("independent_no_std", "independent_std", "paired_no_std", "paired_std"):
            raise ValueError(variant)
        self.config, self.variant = config, variant
        self.paired = variant.startswith("paired_")
        self.use_std = not variant.endswith("no_std")
        h = config["hidden_dim"]
        self.encoder = DirectedEncoder(h, config["depth"], config["dropout"])
        # d, two signed amplitudes OR three free T locations, three mixture logits,
        # and separate D/T residual SDs. Independent controls have 4 extra location
        # outputs and 2 classifier outputs, all active (no dummy parameters).
        self.width = 8 if self.paired else 9
        self.observation = nn.Linear(h, 4 * self.width)
        self.reportability = nn.Linear(h, 2)
        self.classifier = None if self.paired else nn.Linear(h, 2)
        self.register_buffer("reference_std", torch.tensor(statistics["reference_std"], dtype=torch.float32))
        nodes, weights = np.polynomial.legendre.leggauss(config["quadrature_nodes"])
        self.register_buffer("nodes", torch.tensor((nodes + 1) / 2, dtype=torch.float64))
        self.register_buffer("quadrature_weights", torch.tensor(weights / 2, dtype=torch.float64))
        self._initialize(statistics)

    def _initialize(self, statistics: dict) -> None:
        nn.init.normal_(self.observation.weight, std=0.01)
        nn.init.normal_(self.reportability.weight, std=0.01)
        bias = torch.zeros(4, self.width)
        bias[:, 0] = torch.tensor(statistics["initial_d"])
        if self.paired:
            bias[:, 1:3] = math.log(math.expm1(0.6))
            start = 3
        else:
            bias[:, 1:4] = torch.tensor(statistics["initial_t"])[:, None] + torch.tensor([0.0, 0.6, -0.6])
            start = 4
        bias[:, start:start + 3] = torch.tensor([0.7, 0.15, 0.15]).log()
        lo, hi = self.config["residual_sd_min"], self.config["residual_sd_max"]
        scaled = (0.5 - lo) / (hi - lo)
        bias[:, start + 3:] = math.log(scaled / (1 - scaled))
        with torch.no_grad():
            self.observation.bias.copy_(bias.flatten())
            self.reportability.bias.copy_(torch.logit(torch.tensor(statistics["reportability_prior"])))
            if self.classifier is not None:
                nn.init.normal_(self.classifier.weight, std=0.01)
                self.classifier.bias.copy_(torch.logit(torch.tensor(statistics["conditional_positive_prior"])))

    def forward(self, graph: dict) -> dict[str, torch.Tensor]:
        h = self.encoder(graph)
        out = self.observation(h).reshape(-1, 4, self.width)
        d = out[:, :, 0]
        if self.paired:
            t = d[..., None] + torch.stack([torch.zeros_like(d), F.softplus(out[:, :, 1]),
                                            -F.softplus(out[:, :, 2])], -1)
            start = 3
        else:
            t, start = out[:, :, 1:4], 4
        weights = out[:, :, start:start + 3].softmax(-1)
        lo, hi = self.config["residual_sd_min"], self.config["residual_sd_max"]
        sd = lo + (hi - lo) * out[:, :, start + 3:].sigmoid()
        reference = self.reference_std if self.use_std else torch.zeros_like(self.reference_std)
        total_sd = (sd.square() + reference.square()).sqrt()
        reported_logits = self.reportability(h)
        reported = reported_logits.sigmoid()
        if self.paired:
            conditional = assay_probability(d[:, CLASS_INDEX], t[:, CLASS_INDEX],
                                            total_sd[:, CLASS_INDEX, 0], total_sd[:, CLASS_INDEX, 1],
                                            weights[:, CLASS_INDEX], self.nodes, self.quadrature_weights,
                                            self.config["pic50_floor"], math.log10(self.config["fold_shift"]))
        else:
            conditional = self.classifier(h).sigmoid()
        return {"d": d, "centers_t": t, "weights": weights, "sd": sd, "total_sd": total_sd,
                "t": (weights * t).sum(-1), "reportability_logits": reported_logits,
                "reportability": reported, "conditional": conditional,
                "probability": (reported * conditional).clamp(1e-6, 1 - 1e-6)}


def arrays_from_raw(raw: pd.DataFrame) -> dict[str, np.ndarray]:
    # Do not fill missing targets before capturing masks. Release-assigned negative
    # labels remain observed labels; entirely missing TDI labels remain masked.
    y = np.stack([raw[list(DIRECT)].to_numpy(float), raw[list(TREATED)].to_numpy(float)], -1)
    std = np.stack([raw[[x + "_std" for x in DIRECT]].to_numpy(float),
                    raw[[x + "_std" for x in TREATED]].to_numpy(float)], -1)
    labels = raw[list(LABELS)].map(lambda x: np.nan if pd.isna(x) else float(x in (True, 1, "True"))).to_numpy(float)
    mask = np.isfinite(y)
    label_mask = np.isfinite(labels)
    if not np.isfinite(std[mask]).all() or np.any(std[mask] <= 0):
        raise ValueError("Every numeric observation must have a positive finite reported SD.")
    reported = mask[:, CLASS_INDEX, :].all(-1).astype(float)
    # Separate implementation of the exact released rule, checked against labels.
    yd, yt = y[:, CLASS_INDEX, 0], y[:, CLASS_INDEX, 1]
    derived = reported.astype(bool) & (yt > np.maximum(yd, 4.0) + math.log10(2.0))
    if not np.array_equal(derived[label_mask].astype(float), labels[label_mask]):
        raise ValueError("Official label derivation differs from raw labels.")
    return {"y": np.nan_to_num(y).astype(np.float32), "std": np.nan_to_num(std).astype(np.float32),
            "mask": mask, "labels": np.nan_to_num(labels).astype(np.float32),
            "label_mask": label_mask, "reported": reported.astype(np.float32)}


def training_statistics(arrays: dict, indices: np.ndarray) -> dict:
    y, mask, std = [arrays[k][indices] for k in ("y", "mask", "std")]
    initial, reference, scale, counts = [], [], [], []
    for iso in range(4):
        iv, rv, cv = [], [], []
        for arm in range(2):
            values = y[:, iso, arm][mask[:, iso, arm]]
            if not len(values):
                raise ValueError("Training partition has an empty continuous endpoint.")
            iv.append(float(np.maximum(values, 4.0).mean()))
            rv.append(float(np.median(std[:, iso, arm][mask[:, iso, arm]])))
            cv.append(len(values))
        initial.append(iv)
        reference.append(rv)
        counts.append(cv)
        scale.append(max(float(y[:, iso, 0][mask[:, iso, 0]].std()), 0.1))
    report_prior, conditional, label_counts = [], [], []
    for k in range(2):
        observed = arrays["label_mask"][indices, k]
        reported = arrays["reported"][indices, k][observed]
        labels = arrays["labels"][indices, k][observed]
        if not len(labels) or not reported.sum():
            raise ValueError("Training partition has no eligible paired labels.")
        report_prior.append(float(np.clip(reported.mean(), 0.01, 0.99)))
        conditional.append(float(np.clip(labels[reported.astype(bool)].mean(), 0.01, 0.99)))
        label_counts.append(int(len(labels)))
    return {"initial_d": [x[0] for x in initial], "initial_t": [x[1] for x in initial],
            "reference_std": reference, "direct_scale": scale, "continuous_counts": counts,
            "label_counts": label_counts, "n_train": int(len(indices)),
            "reportability_prior": report_prior, "conditional_positive_prior": conditional}


def tensor_targets(arrays: dict, indices, device) -> dict:
    return {k: torch.as_tensor(v[indices], device=device) for k, v in arrays.items()}


def observation_nll(pred: dict, target: dict, use_std: bool, floor: float) -> torch.Tensor:
    y, mask = target["y"], target["mask"]
    # Substitute safe values before evaluating CDFs, then mask the resulting loss.
    y = torch.where(mask, y, torch.full_like(y, floor))
    assay_sd = target["std"] if use_std else torch.zeros_like(target["std"])
    sd = (pred["sd"].square() + assay_sd.square()).sqrt()

    def log_normal(value, mean, noise):
        z = (value - mean) / noise
        log_density = -0.5 * z.square() - noise.log() - 0.5 * math.log(2 * math.pi)
        log_censored = torch.special.log_ndtr((floor - mean) / noise)
        return torch.where(value < floor, log_censored, log_density)

    d = log_normal(y[:, :, 0], pred["d"], sd[:, :, 0])
    t_components = log_normal(y[:, :, 1, None], pred["centers_t"], sd[:, :, 1, None])
    t = torch.logsumexp(pred["weights"].clamp_min(1e-12).log() + t_components, -1)
    return torch.where(mask, -torch.stack([d, t], -1), torch.zeros_like(y))


def training_loss(pred: dict, target: dict, model: AssayModel, statistics: dict):
    # Whole-training counts make each mini-batch an unbiased estimate of the
    # equal-endpoint objective; sparse labels do not change task weights per batch.
    n, batch = statistics["n_train"], len(target["y"])
    cont = observation_nll(pred, target, model.use_std, model.config["pic50_floor"])
    cont_count = cont.new_tensor(statistics["continuous_counts"])
    continuous = (cont.sum(0) * n / (batch * cont_count)).mean()
    class_loss = F.binary_cross_entropy(pred["probability"], target["labels"], reduction="none")
    report_loss = F.binary_cross_entropy_with_logits(pred["reportability_logits"], target["reported"], reduction="none")
    label_count = cont.new_tensor(statistics["label_counts"])
    mask = target["label_mask"]
    classification = ((class_loss * mask).sum(0) * n / (batch * label_count)).mean()
    reporting = ((report_loss * mask).sum(0) * n / (batch * label_count)).mean()
    c = model.config
    loss = (c["continuous_loss_weight"] * continuous + c["classification_loss_weight"] * classification
            + c["reportability_loss_weight"] * reporting)
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite training objective.")
    return loss


@torch.no_grad()
def predict(model: AssayModel, graphs: list[dict], indices, batch_size: int, device) -> dict:
    model.eval()
    outputs: dict[str, list] = {}
    for start in range(0, len(indices), batch_size):
        batch = indices[start:start + batch_size]
        pred = model(batch_graphs([graphs[int(i)] for i in batch], device))
        for key, value in pred.items():
            outputs.setdefault(key, []).append(value.cpu())
    return {key: torch.cat(values).numpy() for key, values in outputs.items()}


def validation_score(pred: dict, arrays: dict, indices, statistics: dict) -> tuple[float, dict]:
    errors, bces = [], []
    for k in range(4):
        mask = arrays["mask"][indices, k, 0]
        if not mask.any():
            raise ValueError("Empty validation regression endpoint.")
        errors.append(float(np.abs(pred["d"][mask, k] - arrays["y"][indices, k, 0][mask]).mean()
                            / statistics["direct_scale"][k]))
    for k in range(2):
        mask = arrays["label_mask"][indices, k]
        if not mask.any():
            raise ValueError("Empty validation classification endpoint.")
        labels = arrays["labels"][indices, k][mask].astype(float)
        p = pred["probability"][mask, k].astype(float).clip(1e-6, 1 - 1e-6)
        bces.append(float(-(labels * np.log(p) + (1 - labels) * np.log1p(-p)).mean()))
    score = float(np.mean(errors) + np.mean(bces))
    if not np.isfinite(score):
        raise FloatingPointError("Non-finite validation selection score.")
    return score, {"direct_scaled_mae": float(np.mean(errors)), "tdi_bce": float(np.mean(bces))}


def atomic_torch_save(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def fit(config: dict, variant: str, statistics: dict, graphs: list, arrays: dict,
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


def load_selected(path: Path, identity: str, device="cpu") -> tuple[AssayModel, dict]:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved["identity"] != identity:
        raise RuntimeError("Selected model identity mismatch.")
    model = AssayModel(saved["config"], saved["variant"], saved["statistics"]).to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, saved
