"""Registered 20-job exploratory paired-assay pilot; preserves all v2/v3 results."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import torch

from paired_model import (AssayModel, CLASS_INDEX, CLASS_ISOS, DIRECT, ISOS, LABELS,
                          arrays_from_raw, atomic_torch_save, fit, load_selected,
                          molecular_graph, predict, training_statistics, validation_score)
from paired_audit import audit_latents, audit_raw_predictions, independent_validation_score, summarize

CODE_FILES = ("configs/paired_prototype.json", "scripts/paired_model.py", "scripts/paired_audit.py",
              "scripts/paired_prototype.py", "scripts/paired_self_test.py", "scripts/start_paired_prototype_4090.sh",
              "scripts/short_runtime.py", "docs/PAIRED_PROTOTYPE.md")
RUNTIME = ".runtime/paired-prototype-v4"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def git(root: Path, *args) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def jobs(config):
    return [(v, int(f), int(s)) for s in config["seeds"] for f in config["folds"] for v in config["variants"]]


def job_id(variant, fold, seed):
    return f"{variant}__fold{fold}__seed{seed}"


def read_inputs(root: Path, config: dict):
    for name in ("training", "split"):
        if sha(root / config[name + "_file"]) != config[name + "_sha256"]:
            raise RuntimeError(f"Frozen {name} hash mismatch.")
    raw = pd.read_csv(root / config["training_file"])
    family = pd.read_csv(root / config["split_file"])
    if len(raw) != 6145 or len(family) != 750 or raw.Molecule_Name.duplicated().any() or family.Molecule_Name.duplicated().any():
        raise RuntimeError("Unexpected raw/split row identity.")
    if not set(family.Molecule_Name).issubset(set(raw.Molecule_Name)) or set(family.outer_fold) != set(range(5)):
        raise RuntimeError("Family fold membership mismatch.")
    if family.groupby("family_id").outer_fold.nunique().max() != 1:
        raise RuntimeError("A family spans multiple outer folds.")
    return raw, family, arrays_from_raw(raw)


def partitions(raw, family, fold):
    mapped = raw.Molecule_Name.map(family.set_index("Molecule_Name").outer_fold).fillna(-1).to_numpy(int)
    val_fold = (fold + 1) % 5
    parts = {"train": np.flatnonzero((mapped != fold) & (mapped != val_fold)),
             "val": np.flatnonzero(mapped == val_fold), "test": np.flatnonzero(mapped == fold)}
    if sum(map(len, parts.values())) != len(raw) or any(not len(v) for v in parts.values()):
        raise RuntimeError("Partition coverage mismatch.")
    return parts


def graph_cache(root, config, raw, family, output):
    sys.path.insert(0, str(root / "src"))
    from cyp_blind.chem import standardize_smiles
    cache, index, marker = output / "graphs.pt", output / "molecules.csv", output / "GRAPH_CACHE.json"
    identity = {"training_sha256": config["training_sha256"], "split_sha256": config["split_sha256"],
                "rdkit": importlib.metadata.version("rdkit"),
                "features_sha256": sha(Path(__file__).with_name("paired_model.py")),
                "standardization_sha256": sha(root / "src/cyp_blind/chem.py")}
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved["identity"] != identity or saved["graphs_sha256"] != sha(cache) or saved["index_sha256"] != sha(index):
            raise RuntimeError("Graph-cache identity changed; artifacts preserved.")
        table = pd.read_csv(index, keep_default_na=False)
        if table.Molecule_Name.tolist() != raw.Molecule_Name.tolist():
            raise RuntimeError("Cached graph identities/order changed.")
        return torch.load(cache, map_location="cpu", weights_only=True), table
    print("Preparing molecular graphs from the frozen TRAIN_TDI file (CPU)...", flush=True)
    graphs, rows = [], []
    for i, row in enumerate(raw.itertuples(index=False)):
        record = standardize_smiles(row.SMILES)
        graphs.append(molecular_graph(record.canonical_smiles))
        rows.append({"Molecule_Name": row.Molecule_Name, "SMILES": record.canonical_smiles,
                     "connectivity_key": record.connectivity_key})
        if (i + 1) % 500 == 0:
            print(f"  graphs {i + 1}/{len(raw)}", flush=True)
    table = pd.DataFrame(rows).merge(family[["Molecule_Name", "family_id", "outer_fold"]], on="Molecule_Name", how="left", validate="one_to_one")
    table["outer_fold"] = table.outer_fold.fillna(-1).astype(int)
    table["family_id"] = table.family_id.fillna("nonpanel_core")
    # Exact same connectivity cannot leak between core and held-out panel/folds.
    if table.groupby("connectivity_key").outer_fold.nunique().max() != 1:
        raise RuntimeError("Connectivity identity crosses frozen partitions; stop for split review.")
    joined = table.merge(family[["Molecule_Name", "connectivity_key"]], on="Molecule_Name", suffixes=("", "_frozen"))
    if not joined.connectivity_key.eq(joined.connectivity_key_frozen).all():
        raise RuntimeError("Standardization differs from frozen family identities.")
    atomic_torch_save(graphs, cache)
    table.to_csv(index, index=False)
    write_json(marker, {"identity": identity, "graphs_sha256": sha(cache), "index_sha256": sha(index)})
    return graphs, table


def preflight(root, delivery, config):
    if git(root, "rev-parse", "HEAD") != config["frozen_commit"]:
        raise RuntimeError("Unexpected frozen scientific checkout.")
    tracked = list(CODE_FILES)
    if git(delivery, "status", "--porcelain", "--", *tracked):
        raise RuntimeError("Pilot code/config has local changes; preserve them and use the committed revision.")
    versions = {name: importlib.metadata.version(name) for name in ("torch", "rdkit", "numpy", "pandas", "scipy", "scikit-learn")}
    if platform.python_version() != "3.12.13" or versions["torch"].split("+")[0] != "2.5.1" or versions["rdkit"] != "2025.9.6":
        raise RuntimeError(f"Expected existing frozen Python/Torch/RDKit environment, got {versions}.")
    if not torch.cuda.is_available() or torch.version.cuda != "12.4":
        raise RuntimeError("Pinned CUDA 12.4 Torch build and an active GPU are required.")
    gpu = torch.cuda.get_device_properties(0)
    if "4090" not in gpu.name or (gpu.major, gpu.minor) != (8, 9) or gpu.total_memory < 20 * 2 ** 30:
        raise RuntimeError("This entry point requires the reviewed RTX 4090 host.")
    old = root / ".runtime/neural-prc-max-v3"
    manifest = json.loads((old / "neural_family_manifest.json").read_text())
    audit = json.loads((old / "neural_result_audit.json").read_text())
    if (manifest.get("complete") is not True or manifest.get("jobs") != 200 or manifest.get("prediction_rows") != 15340
            or audit.get("overall_pass") is not True or audit["summary"].get("rerun_tdi_jobs") != 75):
        raise RuntimeError("The completed and audited v3 baseline is required.")
    frozen = {config["training_file"]: config["training_sha256"], config["split_file"]: config["split_sha256"]}
    for name in ("src/cyp_blind/chem.py", "src/cyp_blind/__init__.py", "vendor/openadmet/custom_scoring_functions.py"):
        expected = subprocess.check_output(["git", "-C", str(root), "show", config["frozen_commit"] + ":" + name])
        if hashlib.sha256(expected).hexdigest() != sha(root / name):
            raise RuntimeError("Frozen dependency modified: " + name)
        frozen[name] = sha(root / name)
    for relative, expected in manifest["outputs"].items():
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(old.resolve()) or sha(path) != expected:
            raise RuntimeError("V3 baseline output integrity check failed: " + relative)
        frozen[relative] = expected
    for name in ("neural_family_manifest.json", "neural_result_audit.json"):
        frozen[str((old / name).relative_to(root))] = sha(old / name)
    identity = {"config": config, "delivery_commit": git(delivery, "rev-parse", "HEAD"),
                "code": {p: sha(delivery / p) for p in CODE_FILES}, "frozen_inputs": frozen,
                "environment": {"python": platform.python_version(), "versions": versions, "cuda": torch.version.cuda,
                                "gpu": gpu.name, "capability": [gpu.major, gpu.minor]}}
    verify_sources(root, delivery, identity)
    return identity


def verify_sources(root, delivery, identity):
    for base, name in ((root, "frozen_inputs"), (delivery, "code")):
        for relative, expected in identity[name].items():
            if sha(base / relative) != expected:
                raise RuntimeError("A registered input/source changed during the run: " + relative)


def latent_frame(pred, table, indices):
    frame = table.iloc[indices][["Molecule_Name", "SMILES"]].reset_index(drop=True).copy()
    columns = {}
    for k, iso in enumerate(ISOS):
        columns[iso + "_d"] = pred["d"][:, k]
        columns[iso + "_t_mean"] = pred["t"][:, k]
        for j in range(3):
            columns[f"{iso}_t{j}"] = pred["centers_t"][:, k, j]
            columns[f"{iso}_w{j}"] = pred["weights"][:, k, j]
        for j, arm in enumerate(("d", "t")):
            columns[f"{iso}_residual_sd_{arm}"] = pred["sd"][:, k, j]
            columns[f"{iso}_total_sd_{arm}"] = pred["total_sd"][:, k, j]
    for k, iso in enumerate(CLASS_ISOS):
        for key in ("reportability", "conditional", "probability"):
            columns[f"{iso}_{key}"] = pred[key][:, k]
    return pd.concat([frame, pd.DataFrame(columns)], axis=1)


def prediction_frame(pred, raw, table, indices, variant, fold, seed):
    rows = []
    for j, idx in enumerate(indices):
        source, meta = raw.iloc[idx], table.iloc[idx]
        common = {"model": "v4_" + variant, "variant": variant, "seed": seed, "fold": fold,
                  "Molecule_Name": source.Molecule_Name, "family_id": meta.family_id}
        for k, endpoint in enumerate(DIRECT):
            if pd.notna(source[endpoint]):
                rows.append({**common, "endpoint": endpoint, "task": "regression", "y_true": float(source[endpoint]),
                             "y_true_lower": float(source[endpoint + "_conf_low"]), "y_true_upper": float(source[endpoint + "_conf_high"]),
                             "y_pred": float(pred["d"][j, k]), "probability": np.nan})
        for k, endpoint in enumerate(LABELS):
            if pd.notna(source[endpoint]):
                probability = float(pred["probability"][j, k])
                rows.append({**common, "endpoint": endpoint, "task": "classification", "y_true": int(source[endpoint]),
                             "y_true_lower": np.nan, "y_true_upper": np.nan,
                             "y_pred": int(probability >= 0.5), "probability": probability})
    return pd.DataFrame(rows)


def audit_checkpoint(directory, identity, statistics, config, model, graphs, arrays, parts, table, device):
    best = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
    last = torch.load(directory / "resume.pt", map_location="cpu", weights_only=False)
    history = pd.read_csv(directory / "history.csv")
    if best["identity"] != identity or last["identity"] != identity or best["statistics"] != statistics:
        raise AssertionError("Checkpoint identity/statistics mismatch.")
    if list(history.epoch) != list(range(len(history))) or last["epoch"] != len(history) - 1:
        raise AssertionError("Last completed epoch differs from training history.")
    if not np.isfinite(history.to_numpy(float)).all() or len(history) > config["max_epochs"]:
        raise AssertionError("Invalid training history.")
    selected = int(np.argmin(history.selection_score.to_numpy()))
    if best["best_epoch"] != selected or last["best_epoch"] != selected:
        raise AssertionError("Selected checkpoint did not minimize validation score.")
    np.testing.assert_allclose(best["best_score"], history.selection_score.iloc[selected], atol=1e-12, rtol=1e-12)
    if len(history) < config["max_epochs"] and len(history) - 1 - selected < config["patience"]:
        raise AssertionError("Training ended before its registered stopping condition.")
    for epoch in range(len(history) - 1):
        best_so_far = int(np.argmin(history.selection_score.to_numpy()[:epoch + 1]))
        if epoch - best_so_far >= config["patience"]:
            raise AssertionError("Training continued after patience was exhausted.")
    for key, value in best["state_dict"].items():
        if not torch.equal(value, last["best_state"][key]):
            raise AssertionError("Exported best weights differ from the recorded selected state.")
    reloaded, _ = load_selected(directory / "best.pt", identity, device)
    for partition in ("val", "test"):
        original = predict(model, graphs, parts[partition], config["batch_size"], device)
        replay = predict(reloaded, graphs, parts[partition], config["batch_size"], device)
        for key in original:
            np.testing.assert_allclose(original[key], replay[key], atol=1e-7, rtol=1e-6)
        expected = latent_frame(replay, table, parts[partition])
        actual = pd.read_csv(directory / (partition + "_latents.csv"))
        pd.testing.assert_frame_equal(expected, actual, check_dtype=False, atol=1e-7, rtol=1e-6)
        if partition == "val":
            score, _ = validation_score(replay, arrays, parts["val"], statistics)
            np.testing.assert_allclose(score, best["best_score"], atol=2e-6, rtol=2e-6)
    return {"best_epoch": selected, "last_completed_epoch": len(history) - 1,
            "best_validation_score": best["best_score"], "saved_model_replay_pass": True}


def run_job(root, delivery, config, registration, variant, fold, seed, graphs, table, raw, family, arrays, device="cuda:0"):
    output = root / RUNTIME
    name = job_id(variant, fold, seed)
    directory = output / "jobs" / name
    parts = partitions(raw, family, fold)
    statistics = training_statistics(arrays, parts["train"])
    ident = fingerprint({"registration": registration["fingerprint"], "job": name, "statistics": statistics})
    complete = directory / "COMPLETE.json"
    if complete.exists():
        saved = json.loads(complete.read_text())
        if saved["identity"] != ident:
            raise RuntimeError("Existing job identity mismatch.")
        for relative, expected in saved["files"].items():
            if sha(directory / relative) != expected:
                raise RuntimeError("Completed pilot artifact changed: " + name + "/" + relative)
        print("REUSE verified " + name, flush=True)
        return saved
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "TRAINING.json", {"identity": ident, "variant": variant, "fold": fold, "seed": seed,
                                            "statistics": statistics, "partitions": {k: len(v) for k, v in parts.items()},
                                            "device": device, "registration": registration["fingerprint"]})
    model, state = fit(config, variant, statistics, graphs, arrays, parts["train"], parts["val"], seed,
                       directory, ident, device)
    write_json(directory / "TRAINED.json", {"identity": ident, "best_sha256": sha(directory / "best.pt"),
                                           "resume_sha256": sha(directory / "resume.pt"), "best_epoch": state["best_epoch"]})
    print("  AUDIT selected weights, validation replay, label rules and exported predictions...", flush=True)
    audit_rows = {}
    for partition in ("val", "test"):
        pred = predict(model, graphs, parts[partition], config["batch_size"], device)
        latent_frame(pred, table, parts[partition]).to_csv(directory / (partition + "_latents.csv"), index=False)
        prediction_frame(pred, raw, table, parts[partition], variant, fold, seed).to_csv(directory / (partition + "_predictions.csv"), index=False)
        latent = pd.read_csv(directory / (partition + "_latents.csv"))
        prediction = pd.read_csv(directory / (partition + "_predictions.csv"))
        audit_raw_predictions(prediction, latent, raw, family, parts[partition], variant, fold, seed)
        audit_rows[partition] = audit_latents(latent, variant, config["quadrature_audit_atol"])
    selection = audit_checkpoint(directory, ident, statistics, config, model, graphs, arrays, parts, table, device)
    independent_score = independent_validation_score(pd.read_csv(directory / "val_latents.csv"), raw,
                                                     parts["train"], parts["val"], statistics)
    np.testing.assert_allclose(independent_score, selection["best_validation_score"], atol=2e-6, rtol=2e-6)
    verify_sources(root, delivery, registration["identity"])
    filenames = ["TRAINING.json", "TRAINED.json", "best.pt", "resume.pt", "history.csv",
                 "val_latents.csv", "test_latents.csv", "val_predictions.csv", "test_predictions.csv"]
    saved = {"identity": ident, "job_id": name, "variant": variant, "fold": fold, "seed": seed,
             "complete": True, "selection": selection, "prediction_audit": audit_rows,
             "parameters": sum(p.numel() for p in model.parameters()), "device": device,
             "files": {p: sha(directory / p) for p in filenames}}
    write_json(complete, saved)
    del model, state
    torch.cuda.empty_cache()
    return saved


def real_data_smoke(root, config, registration, graphs, table, raw, family, arrays):
    output = root / RUNTIME / "smoke"
    marker = output / "PASS.json"
    if marker.exists():
        prior = json.loads(marker.read_text())
        if prior["registration"] != registration["fingerprint"]:
            raise RuntimeError("Smoke-test registration changed.")
        print("REUSE prior real GPU smoke PASS for this exact registration.", flush=True)
        return
    parts = partitions(raw, family, 0)
    chosen = set()
    for k in range(4):
        for a in range(2):
            for below in (False, True):
                eligible = arrays["mask"][parts["train"], k, a]
                eligible &= (arrays["y"][parts["train"], k, a] < 4) == below
                chosen.update(parts["train"][eligible][:12].tolist())
    for k in range(2):
        for value in (0, 1):
            eligible = arrays["label_mask"][parts["train"], k] & (arrays["labels"][parts["train"], k] == value)
            chosen.update(parts["train"][eligible][:16].tolist())
    train_index = np.array(sorted(chosen), dtype=int)
    statistics = training_statistics(arrays, train_index)
    small = {**config, "max_epochs": 3, "patience": 3}
    results = []
    for variant in config["variants"]:
        print(f"REAL GPU SMOKE {variant}: {len(train_index)} training molecules, 3 epochs; internal validation only.", flush=True)
        directory = output / variant
        ident = fingerprint({"registration": registration["fingerprint"], "smoke": variant})
        model, state = fit(small, variant, statistics, graphs, arrays, train_index, parts["val"], 20260829,
                           directory, ident, "cuda:0")
        restarted_dir = directory / "restart"
        fit(small, variant, statistics, graphs, arrays, train_index, parts["val"], 20260829,
            restarted_dir, ident, "cuda:0", stop_after_epoch=0)
        resumed, resumed_state = fit(small, variant, statistics, graphs, arrays, train_index, parts["val"], 20260829,
                                     restarted_dir, ident, "cuda:0")
        for key in state["model"]:
            if not torch.equal(state["model"][key], resumed_state["model"][key]):
                raise AssertionError("GPU epoch restart changed final model weights: " + variant)
        if state["best_epoch"] != resumed_state["best_epoch"] or state["best_score"] != resumed_state["best_score"]:
            raise AssertionError("GPU restart changed checkpoint selection.")
        before = predict(model, graphs, parts["val"], small["batch_size"], "cuda:0")
        loaded, _ = load_selected(directory / "best.pt", ident, "cuda:0")
        after = predict(loaded, graphs, parts["val"], small["batch_size"], "cuda:0")
        for key in before:
            np.testing.assert_allclose(before[key], after[key], atol=1e-7, rtol=1e-6)
        path = directory / "validation_latents.csv"
        latent_frame(after, table, parts["val"]).to_csv(path, index=False)
        audit = audit_latents(pd.read_csv(path), variant, config["quadrature_audit_atol"])
        results.append({"variant": variant, "epochs": len(state["history"]), "device": "cuda:0",
                        "restart_final_weights_bitwise_equal": True, **audit})
        del model, loaded, state, resumed, resumed_state
        torch.cuda.empty_cache()
    write_json(marker, {"overall_pass": True, "registration": registration["fingerprint"], "results": results})
    print("REAL_GPU_SMOKE_PASS: all four variants trained, resumed, reloaded and passed probability audit.", flush=True)


def finish_run(root, delivery, config, registration, raw, family, arrays, table, records, *, engineering_only=False):
    output = root / RUNTIME
    # The engineering-only path is used by the synthetic CPU integration test;
    # the production CLI never enables it and never accepts a CPU smoke marker.
    required_device = "cpu" if engineering_only else "cuda:0"
    cpu_test = json.loads((output / "CPU_SELF_TEST.json").read_text())
    smoke = json.loads((output / "smoke/PASS.json").read_text())
    if cpu_test.get("overall_pass") is not True or smoke.get("overall_pass") is not True:
        raise RuntimeError("Missing successful real training self-tests.")
    if (smoke.get("registration") != registration["fingerprint"]
            or {r["variant"] for r in smoke["results"]} != set(config["variants"])
            or any(r.get("device") != required_device for r in smoke["results"])
            or any(r.get("device") != required_device for r in records)):
        raise RuntimeError("Self-test/job device or registration mismatch.")
    frames = []
    for variant, fold, seed in jobs(config):
        directory = output / "jobs" / job_id(variant, fold, seed)
        record = json.loads((directory / "COMPLETE.json").read_text())
        for relative, expected in record["files"].items():
            if sha(directory / relative) != expected:
                raise RuntimeError("Completed artifact changed during final audit.")
        pred = pd.read_csv(directory / "test_predictions.csv")
        latent = pd.read_csv(directory / "test_latents.csv")
        audit_raw_predictions(pred, latent, raw, family, partitions(raw, family, fold)["test"], variant, fold, seed)
        frames.append(pred)
    predictions = pd.concat(frames, ignore_index=True)
    keys = ["model", "seed", "Molecule_Name", "endpoint"]
    if len(records) != config["expected_jobs"] or len(predictions) != config["expected_prediction_rows"] or predictions.duplicated(keys).any():
        raise RuntimeError("Pilot final job/prediction coverage mismatch.")
    counts = [record["parameters"] for record in records]
    if max(counts) / min(counts) > 1.01:
        raise RuntimeError("Active parameter budgets differ by more than 1%.")
    predictions.to_csv(output / "predictions.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    roundtrip = pd.read_csv(output / "predictions.csv.gz")
    pd.testing.assert_frame_equal(roundtrip, predictions, check_dtype=False, atol=1e-10, rtol=1e-10)
    metrics, diagnostics = summarize(roundtrip, root / "vendor/openadmet/custom_scoring_functions.py")
    if len(metrics) != 168 or len(diagnostics) != 48:
        raise RuntimeError("Expected 168 metric rows and 48 classification diagnostic rows.")
    metrics.to_csv(output / "metrics.csv", index=False)
    diagnostics.to_csv(output / "tdi_diagnostics.csv", index=False)
    primary = metrics[metrics.scope.eq("pooled") & metrics.endpoint.isin(["MA", *LABELS])].copy()
    primary.to_csv(output / "primary_summary.csv", index=False)
    contrasts = []
    for treatment, control, question in (("paired_std", "independent_std", "paired_structure_with_std"),
                                        ("paired_no_std", "independent_no_std", "paired_structure_without_std"),
                                        ("paired_std", "paired_no_std", "reported_std_in_paired"),
                                        ("independent_std", "independent_no_std", "reported_std_in_independent")):
        for endpoint in ("MA", *LABELS):
            left = primary[primary.model.eq("v4_" + treatment) & primary.endpoint.eq(endpoint)].value.item()
            right = primary[primary.model.eq("v4_" + control) & primary.endpoint.eq(endpoint)].value.item()
            contrasts.append({"question": question, "endpoint": endpoint, "treatment": treatment, "control": control,
                              "treatment_value": left, "control_value": right, "raw_difference": left - right,
                              "benefit_direction_difference": right - left if endpoint == "MA" else left - right,
                              "interpretation": "exploratory one-seed contrast; no significance or promotion claim"})
    pd.DataFrame(contrasts).to_csv(output / "core_contrasts.csv", index=False)
    # Same-seed v3 context is descriptive: its two separate task-group models did
    # not receive all eight continuous arms, unlike every matched v4 control.
    v3dir = root / ".runtime/neural-prc-max-v3"
    baseline_predictions = pd.read_csv(v3dir / "neural_family_predictions.csv.gz")
    expected = baseline_predictions[baseline_predictions.seed.eq(config["seeds"][0]) & baseline_predictions["mode"].eq("masked_multitask")]
    metadata = ["Molecule_Name", "endpoint", "fold", "family_id", "y_true", "y_true_lower", "y_true_upper"]
    for _, block in predictions.groupby("variant"):
        pd.testing.assert_frame_equal(block[metadata].sort_values(metadata[:2]).reset_index(drop=True),
                                      expected[metadata].sort_values(metadata[:2]).reset_index(drop=True),
                                      check_dtype=False, atol=1e-10, rtol=1e-10)
    baseline = pd.read_csv(v3dir / "neural_family_metrics.csv")
    baseline = baseline[baseline.seed.eq(config["seeds"][0]) & baseline.scope.eq("pooled") & baseline.endpoint.isin(["MA", *LABELS])].copy()
    pd.concat([primary, baseline], ignore_index=True).to_csv(output / "same_seed_baseline_context.csv", index=False)
    verify_sources(root, delivery, registration["identity"])
    files = ["predictions.csv.gz", "metrics.csv", "tdi_diagnostics.csv", "primary_summary.csv", "core_contrasts.csv", "same_seed_baseline_context.csv", "molecules.csv", "GRAPH_CACHE.json",
             "CPU_SELF_TEST.json", "smoke/PASS.json"]
    audit = {"overall_pass": True, "protocol_id": config["protocol_id"], "registration": registration["fingerprint"],
             "summary": {"completed_jobs": len(records), "prediction_rows": len(predictions), "metric_rows": len(metrics), "seeds": config["seeds"]},
             "checks": ["raw_label_rule_and_missing_masks", "frozen_family_and_connectivity_membership",
                        "training_only_statistics", "real_cpu_self_test",
                        "real_four_variant_cpu_fixture" if engineering_only else "real_four_variant_gpu_smoke",
                        "active_parameter_budget_within_one_percent", "validation_only_minimum_selection",
                        "best_weights_reloaded_predictions_reproduced", "adaptive_probability_integration",
                        "all_observed_outer_rows_exactly_once", "official_metrics_independently_recomputed",
                        "v3_same_seed_row_and_raw_truth_parity", "v3_outputs_and_frozen_sources_unchanged"],
             "outputs": {name: sha(output / name) for name in files},
             "jobs": {record["job_id"]: {"complete_sha256": sha(output / "jobs" / record["job_id"] / "COMPLETE.json"),
                                        "parameters": record["parameters"], "selection": record["selection"]} for record in records},
             "outer_baseline_results_seen": True, "pilot_results_are_exploratory": True,
             "engineering_only": engineering_only, "scientific_results_generated": not engineering_only,
             "gpu_training_qualified": not engineering_only,
             "flagship_claim_authorized": False, "independent_blind_validation_completed": False}
    write_json(output / "RESULT_AUDIT.json", audit)
    target_dir = output if engineering_only else delivery
    target_name = "CYP_paired_v4_engineering_test" if engineering_only else "CYP_paired_v4_review"
    target = target_dir / (target_name + ".zip")
    if target.exists():
        target = target_dir / (target_name + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".zip")
    with zipfile.ZipFile(target, "x", zipfile.ZIP_DEFLATED) as archive:
        for name in files + ["RESULT_AUDIT.json", "RUN_REGISTRATION.json"]:
            archive.write(output / name, name)
        for relative in CODE_FILES:
            archive.write(delivery / relative, "source/" + relative)
        for record in records:
            for name in ("COMPLETE.json", "TRAINING.json", "TRAINED.json", "history.csv", "best.pt",
                         "val_predictions.csv", "test_predictions.csv", "val_latents.csv", "test_latents.csv"):
                path = output / "jobs" / record["job_id"] / name
                archive.write(path, str(path.relative_to(output)))
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Review archive CRC verification failed.")
    print(primary.to_string(index=False), flush=True)
    if engineering_only:
        print(f"ENGINEERING_PIPELINE_PASS: synthetic CPU fixture only; no scientific result.\nTest ZIP: {target}", flush=True)
    else:
        print(f"PASS: all 20 paired-assay pilot jobs and independent result audit completed.\nReview ZIP: {target}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--delivery", type=Path, required=True)
    args = parser.parse_args()
    root, delivery = args.root.resolve(), args.delivery.resolve()
    config = json.loads((delivery / "configs/paired_prototype.json").read_text())
    if (config["pic50_floor"] != 4.0 or config["fold_shift"] != 2.0 or config["threshold"] != 0.5
            or config["construction_data"] != "training_only" or config["test_structures_allowed"] is not False
            or config["selection"] != "min(mean_direct_MAE_divided_by_training_SD + mean_TDI_BCE)"):
        raise RuntimeError("This implementation requires its registered assay/selection rules.")
    if len(jobs(config)) != config["expected_jobs"]:
        raise RuntimeError("Job matrix differs from registered counts.")
    output = root / RUNTIME
    identity = preflight(root, delivery, config)
    registration = {"identity": identity, "fingerprint": fingerprint(identity)}
    marker = output / "RUN_REGISTRATION.json"
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved != registration:
            raise RuntimeError("Registered code/data/environment changed. Existing pilot is preserved; use a separately reviewed protocol.")
    else:
        write_json(marker, registration)
    print("PREFLIGHT_PASS: frozen inputs, completed v3 outputs, pinned environment, and RTX 4090.", flush=True)
    from paired_self_test import self_test
    self_test(output / "cpu_self_test", delivery / "configs/paired_prototype.json", output / "CPU_SELF_TEST.json")
    raw, family, arrays = read_inputs(root, config)
    graphs, table = graph_cache(root, config, raw, family, output)
    real_data_smoke(root, config, registration, graphs, table, raw, family, arrays)
    records = []
    for k, (variant, fold, seed) in enumerate(jobs(config), 1):
        print(f"[{k}/{config['expected_jobs']}] {job_id(variant, fold, seed)}", flush=True)
        verify_sources(root, delivery, identity)
        records.append(run_job(root, delivery, config, registration, variant, fold, seed, graphs, table, raw, family, arrays))
    print("All training jobs completed; checking final coverage and official metrics...", flush=True)
    finish_run(root, delivery, config, registration, raw, family, arrays, table, records)


if __name__ == "__main__":
    main()
