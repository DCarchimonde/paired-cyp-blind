"""Rerun the 75 affected TDI jobs with explicit PRC maximization.

Reuses verified v2 regression artifacts with their original provenance. All new
artifacts live in a separate ignored runtime directory; the v2 source, data,
completed jobs and summaries remain available for historical reproduction.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import zipfile

import numpy as np
import pandas as pd


PROTOCOL_ID = "neural-prc-max-v3-20260907"
CODE_FILES = ("scripts/chemprop_prc_max.py", "scripts/tdi_prc_repair.py",
              "scripts/start_tdi_prc_repair_4090.sh", "scripts/tdi_checkpoint_audit.py",
              "configs/tdi_prc_repair.json")
LEGACY_COMMIT = "6b10538bc6d2595d6eddbf383cf852ebba24766f"
LEGACY_CODE_HASHES = {
    "scripts/chemprop_prc_max.py": "49fad40ad6ca895f6dd1fa518f4e73c3a5477c539ad4b8529e3413ea36c78606",
    "scripts/tdi_prc_repair.py": "3679aceeac6e989e8c4abd7b34317dfda2ddaa2d858f1d177fa0cbb8ef146f25",
    "scripts/start_tdi_prc_repair_4090.sh": "98b5af7de9d4d594409d4de84c8a2a5cd987b5a0ce6b26db05e7966249f1999f",
    "configs/tdi_prc_repair.json": "bfb6fbb913f53702116860a9b3df8e64482c0f4f0df2dd914c97864c0a1e1c0e",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def recorded_file(root: Path, entry: dict, *, within: Path | None = None) -> Path:
    path = (root / entry["path"]).resolve(strict=True)
    if not path.is_relative_to((within or root).resolve()) or not path.is_file():
        raise RuntimeError(f"Unexpected artifact path: {entry['path']}")
    if sha(path) != entry["sha256"]:
        raise RuntimeError(f"Artifact hash mismatch: {entry['path']}")
    return path


def entry(root: Path, path: Path) -> dict:
    return {"path": str(path.resolve().relative_to(root)), "sha256": sha(path)}


def identity_fingerprint(identity: dict) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def register_revision(output: Path, identity: dict) -> tuple[str, dict]:
    """Keep the original registration and record this auditor/code revision separately."""
    fingerprint = identity_fingerprint(identity)
    path = output / "RUN_REGISTRATION.json"
    if not path.exists():
        write_json(path, {"registered_utc": utc(), "run_fingerprint": fingerprint, **identity})
    original = json.loads(path.read_text())
    before = {key: original[key] for key in ("protocol", "files", "delivery_commit", "package_environment")}
    old_fingerprint = identity_fingerprint(before)
    if original.get("run_fingerprint") != old_fingerprint:
        raise RuntimeError("Original run registration fingerprint is invalid.")
    same_code = before["files"]["repair"] == identity["files"]["repair"]
    known_legacy = (before["delivery_commit"] == LEGACY_COMMIT
                    and before["files"]["repair"] == LEGACY_CODE_HASHES
                    and identity["files"]["repair"]["scripts/chemprop_prc_max.py"]
                    == LEGACY_CODE_HASHES["scripts/chemprop_prc_max.py"])
    if (before["protocol"] != identity["protocol"]
            or before["files"]["frozen"] != identity["files"]["frozen"]
            or before["package_environment"] != identity["package_environment"]
            or not (same_code or known_legacy)):
        raise RuntimeError("Unrecognized repair revision or changed training inputs/packages; files preserved.")
    revision = output / "registrations" / f"{fingerprint}.json"
    if revision.exists():
        saved = json.loads(revision.read_text())
        if any(saved.get(key) != value for key, value in identity.items()):
            raise RuntimeError("Existing audit revision identity differs.")
        if saved.get("original_registration_sha256") != sha(path):
            raise RuntimeError("Original run registration changed after the audit revision.")
    else:
        write_json(revision, {"registered_utc": utc(), "run_fingerprint": fingerprint, **identity,
                              "original_registration_sha256": sha(path),
                              "reason": "Audit actual Lightning 2.6.5 last-save semantics; preserve training provenance"})
    return fingerprint, {old_fingerprint: before, fingerprint: identity}


def attempt_artifacts(log_path: Path, selection: dict) -> dict[str, Path]:
    attempt = log_path.parent
    model = attempt / "model/model_0"
    best = list((model / "checkpoints").glob("best-epoch=*-val_prc=*.ckpt"))
    if len(best) != 1:
        raise RuntimeError(f"Expected one retained best checkpoint: {attempt}")
    paths = {"log": log_path, "config": attempt / "model/config.toml",
             "training_metrics": Path(selection["training_metrics_path"]), "best_model": model / "best.pt",
             "best_checkpoint": best[0], "last_checkpoint": model / "checkpoints/last.ckpt",
             "chemprop_predictions": model / "test_predictions.csv", "predictions": attempt / "predictions.csv",
             "validation_predictions": attempt / "validation_predictions.csv", "validation_log": attempt / "validation.log"}
    if any(not path.is_file() for path in paths.values()):
        raise RuntimeError(f"Incomplete post-training artifacts: {attempt}")
    return paths


def prediction_command(root: Path, data: Path, model: Path, output: Path) -> list[str]:
    return [str(root / ".venv/bin/chemprop"), "predict", "-q", "-i", str(data), "-o", str(output),
            "--model-path", str(model), "-s", "SMILES", "--accelerator", "gpu", "--devices", "1",
            "--num-workers", "4", "--batch-size", "64"]


def verify_saved_config(path: Path, original_path: Path, attempt: Path) -> None:
    # Chemprop writes configargparse key=value syntax (not standards-compliant TOML).
    def read(source):
        rows = [line.split("=", 1) for line in source.read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#")]
        parsed = {key.strip(): value.strip() for key, value in rows}
        if len(parsed) != len(rows):
            raise RuntimeError(f"Duplicate saved configuration key: {source}")
        return parsed
    current, original = read(path), read(original_path)
    if Path(current.pop("output-dir")).resolve() != (attempt / "model").resolve():
        raise RuntimeError("Recovered model output directory differs.")
    original.pop("output-dir")
    if current.pop("remove-checkpoints", "false") != "false":
        raise RuntimeError("The repaired run must retain checkpoints.")
    if original.pop("remove-checkpoints", None) != "true" or current != original:
        raise RuntimeError("Recovered configuration differs from the direction-only repair.")


def recover_legacy_candidate(root, output, job, old, job_root, neural, config, delivery, registrations):
    """Adopt only a complete pre-fix attempt; rerun inference, never infer its duration."""
    origins = [(fp, ident) for fp, ident in registrations.items()
               if ident["delivery_commit"] == LEGACY_COMMIT and ident["files"]["repair"] == LEGACY_CODE_HASHES]
    if not origins:
        return None
    eligible = [p for p in sorted(job_root.glob("attempt_*")) if p.is_dir()
                and not (p / "STARTED.json").exists() and not (p / "CANDIDATE.json").exists()
                and (p / "model/model_0/checkpoints/last.ckpt").is_file()
                and (p / "validation_predictions.csv").is_file() and (p / "validation.log").is_file()]
    if not eligible:
        return None
    if len(eligible) != 1:
        raise RuntimeError(f"Multiple historical complete attempts; refusing to choose by their scores: {job.job_id}")
    attempt = eligible[0]
    origin_fp, origin = origins[0]
    selection = selection_audit(attempt / "train.log", metric="prc", mode="max")
    paths = attempt_artifacts(attempt / "train.log", selection)
    old_config = (root / old["outputs"]["log"]["path"]).parent / "model/config.toml"
    verify_saved_config(paths["config"], old_config, attempt)
    outputs = {key: entry(root, path) for key, path in paths.items()}
    evidence_dir = attempt / datetime.now(timezone.utc).strftime("recovery_%Y%m%dT%H%M%S%fZ")
    evidence_dir.mkdir()
    snapshot = {"captured_utc": utc(), "run_fingerprint": origin_fp, "outputs": outputs,
                "registration": entry(root, output / "RUN_REGISTRATION.json"),
                "gpu_preflight": entry(root, output / "gpu_preflight.json"),
                "original_config": entry(root, old_config),
                "runtime_seconds": None, "duration_note": "Not persisted by the interrupted legacy auditor"}
    write_json(evidence_dir / "snapshot.json", snapshot)
    commands, rechecks = {}, {}
    print(f"RECOVER existing trained attempt; checking saved-model inference: {job.job_id}", flush=True)
    for member, old_key in (("val", "validation_predictions"), ("test", "chemprop_predictions")):
        result = evidence_dir / f"{member}_predictions.csv"
        log_path = evidence_dir / f"{member}_predict.log"
        commands[member] = prediction_command(root, root / old["inputs"][member]["path"], paths["best_model"], result)
        with log_path.open("w") as log:
            subprocess.run(commands[member], cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
        pd.testing.assert_frame_equal(pd.read_csv(result), pd.read_csv(paths[old_key]),
                                      check_dtype=False, atol=1e-7, rtol=1e-6)
        rechecks[member] = entry(root, result)
        rechecks[member + "_log"] = entry(root, log_path)
    for artifact in outputs.values():
        recorded_file(root, artifact, within=job_root)
    job_data = asdict(job)
    job_data["targets"] = list(job.targets)
    record = {"schema_version": 1, "status": "PENDING_AUDIT", "protocol_id": PROTOCOL_ID,
              "job_id": job.job_id, "job": job_data, "run_fingerprint": origin_fp,
              "runtime_seconds": None, "command": fixed_command(neural, config, job,
                  {key: root / value["path"] for key, value in old["inputs"].items()}, attempt, delivery),
              "validation_command": prediction_command(root, root / old["inputs"]["val"]["path"],
                                                        paths["best_model"], paths["validation_predictions"]),
              "inputs": old["inputs"], "outputs": outputs,
              "environment": json.loads(recorded_file(root, snapshot["gpu_preflight"]).read_text())["environment"],
              "execution": {"accelerator": "gpu", "devices": "1"}, "package_environment": origin["package_environment"],
              "selection": selection, "source_v2_complete_sha256": sha(root / config["paths"]["job_root"] / job.job_id / "COMPLETE.json"),
              "recovery": {"kind": "legacy_post_training_audit_failure", "snapshot": entry(root, evidence_dir / "snapshot.json"),
                           "inference_rechecks": rechecks, "inference_commands": commands,
                           "inference_runtime_environment": {key: os.environ.get(key) for key in
                               ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CYP_IGNORED_OMP_NUM_THREADS")},
                           "training_command_reconstructed": True, "wall_duration_unknown": True}}
    write_json(attempt / "CANDIDATE.json", record)
    return record


def input_audit(root: Path, config: dict, *, task_group: str | None = None) -> dict:
    """Read-only check of exact membership, raw labels, classes and file hashes."""
    raw = pd.read_csv(root / config["inputs"]["training"]).set_index("Molecule_Name")
    family = pd.read_csv(root / config["inputs"]["family_split"]).set_index("Molecule_Name")
    manifest = json.loads((root / config["paths"]["prepared_data"] / "preparation_manifest.json").read_text())
    if raw.index.has_duplicates or family.index.has_duplicates:
        raise RuntimeError("Duplicate raw/family molecule identifiers.")
    specs = partitions = 0
    for spec in manifest["data_specs"].values():
        if task_group and spec["task_group"] != task_group:
            continue
        specs += 1
        eligible = raw[raw[spec["targets"]].notna().any(axis=1)]
        assigned = family.outer_fold.reindex(eligible.index)
        masks = {"test": assigned.eq(spec["fold"]), "val": assigned.eq((spec["fold"] + 1) % 5),
                 "train": ~assigned.isin([spec["fold"], (spec["fold"] + 1) % 5])}
        for member, artifact in spec["files"].items():
            frame = pd.read_csv(recorded_file(root, artifact))
            partitions += 1
            expected_names = eligible.index[masks[member]].tolist()
            if (frame.Molecule_Name.tolist() != expected_names
                    or frame.columns.tolist() != ["Molecule_Name", "SMILES", *spec["targets"]]
                    or len(frame) != spec["counts"][member]):
                raise RuntimeError(f"Prepared membership/schema differs: {artifact['path']}")
            for target in spec["targets"]:
                values = raw.loc[frame.Molecule_Name, target]
                if spec["task_group"] == "tdi":
                    values = values.astype("boolean").astype("Float64").to_numpy(float, na_value=np.nan)
                else:
                    values = values.to_numpy(float)
                if not np.allclose(frame[target], values, equal_nan=True, rtol=1e-12, atol=1e-12):
                    raise RuntimeError(f"Prepared labels differ: {artifact['path']}/{target}")
                if spec["task_group"] == "tdi" and member in ("train", "val"):
                    if set(frame[target].dropna().tolist()) != {0., 1.}:
                        raise RuntimeError(f"Degenerate training/validation classes: {artifact['path']}/{target}")
    return {"overall_pass": True, "data_specs": specs, "partitions": partitions,
            "exact_ordered_membership": True, "raw_labels": True, "file_hashes": True,
            "tdi_training_validation_two_classes": True, "read_only": True}


def selection_audit(log_path: Path, *, metric: str, mode: str,
                    epochs: int = 100, patience: int = 15) -> dict:
    if mode not in ("min", "max"):
        raise ValueError("An explicit min/max direction is required.")
    log = log_path.read_text(encoding="utf-8")
    matches = re.findall(r"Restoring states from the checkpoint path at .*best-epoch=(\d+)-val_"
                         + re.escape(metric) + r"=", log)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one restored checkpoint: {log_path}")
    chosen = int(matches[0])
    paths = list(log_path.parent.glob("model/model_0/trainer_logs/version_*/metrics.csv"))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one CSV training curve: {log_path}")
    frame = pd.read_csv(paths[0])
    curve = frame.dropna(subset=["val/" + metric]).set_index("epoch")["val/" + metric]
    if curve.empty or curve.index.has_duplicates or not np.isfinite(curve).all():
        raise RuntimeError(f"Invalid validation curve: {paths[0]}")
    train = frame.dropna(subset=["train_loss_epoch"]).set_index("epoch")["train_loss_epoch"]
    if not train.index.equals(curve.index) or not np.isfinite(train).all():
        raise RuntimeError(f"Missing/nonfinite training epoch loss: {paths[0]}")
    if list(curve.index) != list(range(int(curve.index.max()) + 1)) or chosen not in curve.index:
        raise RuntimeError(f"Incomplete epoch coverage: {paths[0]}")
    optimum = float(curve.max() if mode == "max" else curve.min())
    if not np.isclose(float(curve.loc[chosen]), optimum, atol=1e-9, rtol=1e-8):
        raise RuntimeError(f"Wrong checkpoint direction: {log_path}: epoch {chosen}, "
                           f"value {curve.loc[chosen]}, required {mode}={optimum}")
    final_epoch = int(curve.index.max())
    if final_epoch != min(epochs - 1, chosen + patience):
        raise RuntimeError(f"Early stopping does not match the selected optimum: {log_path}")
    return {"metric": metric, "mode": mode, "chosen_epoch": chosen,
            "chosen_value": float(curve.loc[chosen]), "min_value": float(curve.min()),
            "max_value": float(curve.max()), "final_epoch": final_epoch,
            "training_metrics_path": str(paths[0]), "selection_verified": True}


def validation_prc(val_path: Path, predictions_path: Path, targets: tuple[str, ...]) -> float:
    from sklearn.metrics import auc, precision_recall_curve
    truth = pd.read_csv(val_path)
    prediction = pd.read_csv(predictions_path)
    if len(truth) != len(prediction) or not truth.SMILES.equals(prediction.SMILES):
        raise RuntimeError("Validation prediction order/SMILES mismatch.")
    y = truth[list(targets)].to_numpy(float)
    p = prediction[list(targets)].to_numpy(float)
    mask = np.isfinite(y)
    if not np.isfinite(p).all() or not ((0 <= p) & (p <= 1)).all():
        raise RuntimeError("Invalid validation probabilities.")
    # Matches the frozen Chemprop PRC: flatten observed multitask labels and
    # integrate the precision-recall curve. This is not average_precision_score.
    precision, recall, _ = precision_recall_curve(y[mask].astype(int), p[mask])
    return float(auc(recall, precision))


def fixed_command(neural, config: dict, job, prepared: dict, attempt: Path,
                  delivery: Path) -> list[str]:
    original = neural.build_chemprop_command(
        config, job, train_path=prepared["train"], val_path=prepared["val"],
        test_path=prepared["test"], output_dir=attempt / "model", accelerator="gpu", devices="1")
    # Retain best and last checkpoints for independent epoch/weight inspection.
    original.remove("--remove-checkpoints")
    return [sys.executable, str(delivery / "scripts/chemprop_prc_max.py"), *original[1:]]


def verify_fixed(root: Path, record: dict, job, old: dict, registrations: dict,
                 new_job_root: Path, neural, config: dict, command: list[str]) -> pd.DataFrame:
    import torch
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from chemprop_prc_max import environment as package_environment
    expected_job = asdict(job)
    expected_job["targets"] = list(job.targets)
    if (record.get("schema_version") != 1 or record.get("status") != "PASS"
            or record.get("job_id") != job.job_id or record.get("job") != expected_job
            or record.get("protocol_id") != PROTOCOL_ID
            or record.get("run_fingerprint") not in registrations
            or record.get("command") != command or record.get("inputs") != old["inputs"]):
        raise RuntimeError(f"Invalid revised completion identity: {job.job_id}")
    required = {"log", "training_metrics", "best_model", "best_checkpoint", "last_checkpoint",
                "chemprop_predictions", "predictions", "validation_predictions", "validation_log", "config"}
    if set(record.get("outputs", {})) != required:
        raise RuntimeError(f"Incomplete revised artifacts: {job.job_id}")
    recovery = record.get("recovery")
    if recovery:
        origin = registrations[record["run_fingerprint"]]
        if (origin["delivery_commit"] != LEGACY_COMMIT or origin["files"]["repair"] != LEGACY_CODE_HASHES
                or recovery.get("kind") != "legacy_post_training_audit_failure"
                or recovery.get("wall_duration_unknown") is not True or record.get("runtime_seconds") is not None):
            raise RuntimeError(f"Invalid historical recovery identity: {job.job_id}")
        snapshot = json.loads(recorded_file(root, recovery["snapshot"], within=new_job_root).read_text())
        if snapshot["outputs"] != record["outputs"] or snapshot["run_fingerprint"] != record["run_fingerprint"]:
            raise RuntimeError(f"Historical recovery snapshot differs: {job.job_id}")
        for key in ("registration", "gpu_preflight", "original_config"):
            recorded_file(root, snapshot[key])
        for artifact in recovery["inference_rechecks"].values():
            recorded_file(root, artifact, within=new_job_root)
    elif (not isinstance(record.get("runtime_seconds"), (float, int))
          or not np.isfinite(record["runtime_seconds"]) or record["runtime_seconds"] <= 0):
        raise RuntimeError(f"Invalid training duration: {job.job_id}")
    inputs = {key: recorded_file(root, value) for key, value in record["inputs"].items()}
    outputs = {key: recorded_file(root, value, within=new_job_root) for key, value in record["outputs"].items()}
    if recovery:
        for member, original_key in (("val", "validation_predictions"), ("test", "chemprop_predictions")):
            replay = recorded_file(root, recovery["inference_rechecks"][member], within=new_job_root)
            if recovery["inference_commands"][member] != prediction_command(root, inputs[member], outputs["best_model"], replay):
                raise RuntimeError(f"Historical inference command differs: {job.job_id}")
            pd.testing.assert_frame_equal(pd.read_csv(replay), pd.read_csv(outputs[original_key]),
                                          check_dtype=False, atol=1e-7, rtol=1e-6)
    flight = record.get("environment", {})
    if (flight.get("cuda_available") is not True or "4090" not in str(flight.get("device_name"))
            or flight.get("compiled_cuda") != "12.4" or flight.get("compute_capability") != "8.9"
            or float(flight.get("vram_gib", 0)) < 20 or record.get("execution") != {"accelerator": "gpu", "devices": "1"}):
        raise RuntimeError(f"Invalid GPU evidence: {job.job_id}")
    selection = selection_audit(outputs["log"], metric="prc", mode="max")
    markers = [line.split(" ", 1)[1] for line in outputs["log"].read_text().splitlines()
               if line.startswith("PRC_DIRECTION_FIX ")]
    if len(markers) != 1:
        raise RuntimeError(f"Missing explicit PRC correction evidence: {job.job_id}")
    marker = json.loads(markers[0])
    actual_package = package_environment()
    if (record.get("package_environment") != actual_package
            or any(marker.get(key) != value for key, value in actual_package.items())
            or marker.get("before") is not None or marker.get("higher_is_better") is not True
            or marker.get("checkpoint_mode") != "max" or marker.get("early_stopping_mode") != "max"):
        raise RuntimeError(f"Package source or PRC correction evidence differs: {job.job_id}")
    if record.get("selection") != selection:
        raise RuntimeError(f"Recorded selection differs from the actual training curve: {job.job_id}")
    if record.get("source_v2_complete_sha256") != sha(root / config["paths"]["job_root"] / job.job_id / "COMPLETE.json"):
        raise RuntimeError(f"Original completion reference changed: {job.job_id}")
    checkpoint = torch.load(outputs["best_checkpoint"], map_location="cpu", weights_only=False)
    if int(checkpoint["epoch"]) != selection["chosen_epoch"]:
        raise RuntimeError(f"Saved checkpoint epoch differs from restored/logged epoch: {job.job_id}")
    exported = torch.load(outputs["best_model"], map_location="cpu", weights_only=False)
    saved_state, exported_state = checkpoint["state_dict"], exported["state_dict"]
    if (exported.get("output_columns") != list(job.targets)
            or set(saved_state) != set(exported_state)
            or any(not torch.equal(value, exported_state[key]) for key, value in saved_state.items())):
        raise RuntimeError(f"Exported model differs from the selected checkpoint: {job.job_id}")
    last = torch.load(outputs["last_checkpoint"], map_location="cpu", weights_only=False)
    from tdi_checkpoint_audit import audit_last_checkpoint
    last_evidence = audit_last_checkpoint(checkpoint, last, selection)
    if record.get("last_checkpoint_audit") not in (None, last_evidence):
        raise RuntimeError(f"Recorded last-checkpoint evidence differs: {job.job_id}")
    callbacks = checkpoint["callbacks"]
    for callback in [ModelCheckpoint(monitor="val/prc", mode="max"),
                     EarlyStopping(monitor="val/prc", mode="max", patience=15)]:
        if callback.state_key not in callbacks:
            raise RuntimeError(f"Checkpoint lacks the expected max callback state: {job.job_id}")
    best_key = ModelCheckpoint(monitor="val/prc", mode="max").state_key
    if not np.isclose(float(callbacks[best_key]["best_model_score"]), selection["chosen_value"], atol=1e-8):
        raise RuntimeError(f"Checkpoint callback score mismatch: {job.job_id}")
    observed_prc = validation_prc(inputs["val"], outputs["validation_predictions"], job.targets)
    if not np.isclose(observed_prc, selection["chosen_value"], atol=2e-5, rtol=1e-5):
        raise RuntimeError(f"Best model validation PRC cannot be reproduced: {job.job_id}: "
                           f"{observed_prc} != {selection['chosen_value']}")
    expected = neural._validate_job_predictions(root, config, job, inputs["test"], outputs["chemprop_predictions"])
    observed = pd.read_csv(outputs["predictions"])
    pd.testing.assert_frame_equal(observed, expected, check_dtype=False, atol=1e-10, rtol=1e-8)
    record["last_checkpoint_audit"] = last_evidence
    return observed


def verify_sources(root: Path, delivery: Path, files: dict) -> None:
    for namespace, mapping in files.items():
        base = root if namespace == "frozen" else delivery
        for relative, digest in mapping.items():
            if sha(base / relative) != digest:
                raise RuntimeError(f"Run source/input changed: {namespace}/{relative}")


def run(root: Path, delivery: Path) -> None:
    root = root.resolve(strict=True)
    delivery = delivery.resolve(strict=True)
    sys.path.insert(0, str(root / "src"))
    from cyp_blind import neural_baselines as neural
    from cyp_blind.io import load_yaml
    from chemprop_prc_max import environment as package_environment

    protocol = json.loads((delivery / "configs/tdi_prc_repair.json").read_text())
    config_path = root / "configs/neural_baselines.yaml"
    config = load_yaml(config_path)
    if (protocol["protocol_id"] != PROTOCOL_ID or protocol["checkpoint_mode"] != "max"
            or protocol["early_stopping_mode"] != "max" or protocol["classification_threshold"] != 0.5
            or protocol["tdi_jobs_to_rerun"] != 75 or protocol["direct_jobs_to_verify_and_reuse"] != 125
            or protocol["change_loss_sampling_architecture_or_seeds"] is not False
            or sha(config_path) != protocol["frozen_v2_config_sha256"]
            or sha(root / config["paths"]["prepared_data"] / "preparation_manifest.json") != protocol["frozen_preparation_sha256"]
            or git(root, "rev-parse", "HEAD") != protocol["frozen_v2_commit"]):
        raise RuntimeError("The reviewed v2 data/configuration or direction-only repair contract changed.")
    # Require these exact repair files to have been committed. Other user files,
    # including downloaded ZIPs, do not make the recorded code ambiguous.
    for relative in CODE_FILES:
        committed = subprocess.check_output(["git", "-C", str(delivery), "show", "HEAD:" + relative])
        if committed != (delivery / relative).read_bytes():
            raise RuntimeError(f"Uncommitted repair source: {relative}")
    flight = neural.preflight(config_path, require_gpu=True)
    if not flight["overall_pass"]:
        raise RuntimeError(f"Frozen GPU/dependency preflight failed: {flight['checks']}")
    protocol_check = input_audit(root, config)
    if not protocol_check["overall_pass"] or protocol_check["data_specs"] != 40:
        raise RuntimeError("Frozen data/partition audit failed.")
    original_manifest_path = root / "reports/neural_v2/neural_family_manifest.json"
    original_manifest = json.loads(original_manifest_path.read_text())
    if original_manifest.get("jobs") != 200 or original_manifest.get("complete") is not True:
        raise RuntimeError("The completed reviewed v2 run is required.")
    for relative, digest in {**original_manifest["inputs"], **original_manifest["outputs"]}.items():
        if sha(root / relative) != digest:
            raise RuntimeError(f"Original summary/input hash mismatch: {relative}")
    job_list = neural.neural_jobs(config)
    old_records, direct_frames, legacy_selection = {}, [], []
    print("Verifying all 200 original jobs and checkpoint directions...", flush=True)
    for index, job in enumerate(job_list):
        complete = root / config["paths"]["job_root"] / job.job_id / "COMPLETE.json"
        if sha(complete) != original_manifest["completion_manifest_hashes"][job.job_id]:
            raise RuntimeError(f"Original completion changed: {job.job_id}")
        old = neural.validate_completion(config_path, job)
        old_records[job.job_id] = old
        metric = "mae" if job.task_group == "direct" else "prc"
        selection = selection_audit(root / old["outputs"]["log"]["path"], metric=metric, mode="min")
        legacy_selection.append({"job_id": job.job_id, "task_group": job.task_group, **selection})
        if job.task_group == "direct":
            direct_frames.append(pd.read_csv(root / old["outputs"]["predictions"]["path"]))
        if (index + 1) % 25 == 0:
            print(f"Verified original jobs: {index + 1}/200", flush=True)
    tdi_jobs = [job for job in job_list if job.task_group == "tdi"]
    if len(tdi_jobs) != 75 or len(direct_frames) != 125:
        raise RuntimeError("Unexpected job matrix.")
    frozen_files = {**original_manifest["inputs"], **original_manifest["outputs"],
                    str(original_manifest_path.relative_to(root)): sha(original_manifest_path)}
    frozen_files.update({str((root / config["paths"]["job_root"] / job_id / "COMPLETE.json").relative_to(root)): digest
                         for job_id, digest in original_manifest["completion_manifest_hashes"].items()})
    files = {"frozen": frozen_files, "repair": {p: sha(delivery / p) for p in CODE_FILES}}
    identity = {"protocol": protocol, "files": files, "delivery_commit": git(delivery, "rev-parse", "HEAD"),
                "package_environment": package_environment()}
    output = root / ".runtime/neural-prc-max-v3"
    output.mkdir(parents=True, exist_ok=True)
    fingerprint, registrations = register_revision(output, identity)
    pd.DataFrame(legacy_selection).to_csv(output / "legacy_selection_audit.csv", index=False)
    write_json(output / "input_audit.json", protocol_check)
    if not (output / "gpu_preflight.json").exists():
        write_json(output / "gpu_preflight.json", flight)
    test_dir = output / "self_tests" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    print("Running real checkpoint + early-stopping self-test before production...", flush=True)
    subprocess.run([sys.executable, str(delivery / "scripts/chemprop_prc_max.py"), "--self-test", str(test_dir)],
                   cwd=root, check=True)
    self_test = json.loads((test_dir / "self_test.json").read_text())
    if self_test.get("overall_pass") is not True:
        raise RuntimeError("Real callback self-test did not pass.")
    print("Checking Lightning 2.6.5 last-checkpoint lifecycle before production...", flush=True)
    lifecycle_dir = test_dir / "last_checkpoint_lifecycle"
    subprocess.run([sys.executable, str(delivery / "scripts/tdi_checkpoint_audit.py"), str(lifecycle_dir)],
                   cwd=root, check=True)
    lifecycle = json.loads((lifecycle_dir / "self_test.json").read_text())
    if lifecycle.get("overall_pass") is not True:
        raise RuntimeError("Real last-checkpoint lifecycle self-test did not pass.")
    self_test["last_checkpoint_lifecycle"] = lifecycle
    write_json(output / "callback_self_test.json", self_test)
    new_records = {}
    for index, job in enumerate(tdi_jobs):
        verify_sources(root, delivery, files)
        old = old_records[job.job_id]
        prepared = {key: recorded_file(root, value) for key, value in old["inputs"].items()}
        job_root = output / "jobs" / job.job_id
        complete = job_root / "COMPLETE.json"
        if complete.exists():
            record = json.loads(complete.read_text())
            attempt = (root / record["outputs"]["log"]["path"]).parent
            command = fixed_command(neural, config, job, prepared, attempt, delivery)
            verify_fixed(root, record, job, old, registrations, job_root, neural, config, command)
            new_records[job.job_id] = record
            print(f"[{index + 1}/75] REUSE verified {job.job_id}", flush=True)
            continue
        candidates = list(job_root.glob("attempt_*/CANDIDATE.json"))
        if len(candidates) > 1:
            raise RuntimeError(f"Multiple pending candidates; no score-based choice is allowed: {job.job_id}")
        candidate = json.loads(candidates[0].read_text()) if candidates else recover_legacy_candidate(
            root, output, job, old, job_root, neural, config, delivery, registrations)
        if candidate is not None:
            if candidate.get("status") != "PENDING_AUDIT":
                raise RuntimeError(f"Unexpected pending candidate status: {job.job_id}")
            record = {**candidate, "status": "PASS", "completed_utc": utc(), "audit_fingerprint": fingerprint}
            attempt = (root / record["outputs"]["log"]["path"]).parent
            command = fixed_command(neural, config, job, prepared, attempt, delivery)
            verify_sources(root, delivery, files)
            verify_fixed(root, record, job, old, registrations, job_root, neural, config, command)
            write_json(complete, record)
            new_records[job.job_id] = record
            print(f"[{index + 1}/75] RECOVERED PASS {job.job_id}; existing training reused", flush=True)
            continue
        attempt = job_root / datetime.now(timezone.utc).strftime("attempt_%Y%m%dT%H%M%S%fZ")
        attempt.mkdir(parents=True, exist_ok=False)
        command = fixed_command(neural, config, job, prepared, attempt, delivery)
        env = os.environ.copy()
        env.update(PYTHONHASHSEED=str(job.seed), CUBLAS_WORKSPACE_CONFIG=":4096:8")
        runtime_environment = {key: env.get(key) for key in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CYP_IGNORED_OMP_NUM_THREADS",
            "PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG")}
        write_json(attempt / "STARTED.json", {"started_utc": utc(), "run_fingerprint": fingerprint,
                   "command": command, "runtime_environment": runtime_environment})
        print(f"[{index + 1}/75] TRAIN PRC=max {job.job_id}", flush=True)
        started = time.perf_counter()
        log_path = attempt / "train.log"
        with log_path.open("w") as log:
            subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        duration = time.perf_counter() - started
        selection = selection_audit(log_path, metric="prc", mode="max")
        model_dir = attempt / "model/model_0"
        best_model = model_dir / "best.pt"
        best_checkpoints = list((model_dir / "checkpoints").glob("best-epoch=*-val_prc=*.ckpt"))
        if len(best_checkpoints) != 1:
            raise RuntimeError(f"Expected one retained best checkpoint: {job.job_id}")
        prediction = neural._validate_job_predictions(root, config, job, prepared["test"], model_dir / "test_predictions.csv")
        prediction.to_csv(attempt / "predictions.csv", index=False)
        val_output = attempt / "validation_predictions.csv"
        val_command = prediction_command(root, prepared["val"], best_model, val_output)
        with (attempt / "validation.log").open("w") as log:
            subprocess.run(val_command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        targets = attempt_artifacts(log_path, selection)
        job_data = asdict(job)
        job_data["targets"] = list(job.targets)
        record = {"schema_version": 1, "status": "PENDING_AUDIT", "protocol_id": PROTOCOL_ID,
                  "job_id": job.job_id, "job": job_data, "run_fingerprint": fingerprint,
                  "runtime_seconds": duration, "command": command, "runtime_environment": runtime_environment,
                  "validation_command": val_command, "inputs": old["inputs"],
                  "outputs": {key: entry(root, path) for key, path in targets.items()},
                  "environment": flight["environment"], "execution": {"accelerator": "gpu", "devices": "1"},
                  "package_environment": identity["package_environment"],
                  "selection": selection, "source_v2_complete_sha256": original_manifest["completion_manifest_hashes"][job.job_id]}
        write_json(attempt / "CANDIDATE.json", record)
        record.update(status="PASS", completed_utc=utc(), audit_fingerprint=fingerprint)
        verify_sources(root, delivery, files)
        verify_fixed(root, record, job, old, registrations, job_root, neural, config, command)
        write_json(complete, record)
        new_records[job.job_id] = record
        print(f"[{index + 1}/75] PASS epoch={selection['chosen_epoch']} validation_PRC={selection['chosen_value']:.6f}", flush=True)
    collect(root, delivery, output, config, neural, job_list, old_records, new_records,
            fingerprint, registrations, files, original_manifest, self_test)


def collect(root, delivery, output, config, neural, jobs, old_records, new_records,
            fingerprint, registrations, files, original_manifest, self_test):
    from cyp_blind.neural_result_audit import _independent_official_metrics, _load_official_soft_threshold_rae
    frames, provenance, selections = [], {}, []
    print("Independently auditing corrected jobs and the combined 200-job baseline...", flush=True)
    for job in jobs:
        old = neural.validate_completion(root / "configs/neural_baselines.yaml", job)
        if job.task_group == "direct":
            frames.append(pd.read_csv(root / old["outputs"]["predictions"]["path"]))
            provenance[job.job_id] = {"origin": "v2_regression_unchanged", "git_head": old["git_head"],
                                     "completion_sha256": original_manifest["completion_manifest_hashes"][job.job_id]}
            continue
        record = new_records[job.job_id]
        attempt = (root / record["outputs"]["log"]["path"]).parent
        prepared = {key: root / value["path"] for key, value in old["inputs"].items()}
        command = fixed_command(neural, config, job, prepared, attempt, delivery)
        frames.append(verify_fixed(root, record, job, old, registrations, output / "jobs" / job.job_id, neural, config, command))
        provenance[job.job_id] = {"origin": "v3_prc_max_rerun", "run_fingerprint": record["run_fingerprint"],
                                 "training_delivery_commit": registrations[record["run_fingerprint"]]["delivery_commit"],
                                 "audit_fingerprint": fingerprint, "recovered_after_audit_failure": bool(record.get("recovery")),
                                 "completion_sha256": sha(output / "jobs" / job.job_id / "COMPLETE.json")}
        selections.append({"job_id": job.job_id, **record["selection"], **record["last_checkpoint_audit"]})
    prediction = pd.concat(frames, ignore_index=True)
    original = pd.read_csv(root / "reports/neural_v2/neural_family_predictions.csv.gz")
    keys = ["mode", "seed", "Molecule_Name", "endpoint"]
    if len(prediction) != 15340 or prediction.duplicated(keys).any():
        raise RuntimeError("Revised prediction coverage/count mismatch.")
    metadata = keys + ["model", "fold", "family_id", "task", "y_true", "y_true_lower", "y_true_upper"]
    pd.testing.assert_frame_equal(prediction[metadata].sort_values(keys).reset_index(drop=True),
                                  original[metadata].sort_values(keys).reset_index(drop=True),
                                  check_dtype=False, atol=1e-12, rtol=1e-12)
    pd.testing.assert_frame_equal(prediction[prediction.task.eq("regression")].sort_values(keys).reset_index(drop=True),
                                  original[original.task.eq("regression")].sort_values(keys).reset_index(drop=True),
                                  check_dtype=False, atol=1e-12, rtol=1e-12)
    official = _load_official_soft_threshold_rae(root)
    metric_frames = []
    for seed, group in prediction.groupby("seed", sort=True):
        observed = neural.summarize_predictions(group)
        expected = _independent_official_metrics(group, official)
        pd.testing.assert_frame_equal(observed, expected, check_dtype=False, atol=1e-10, rtol=1e-8)
        observed.insert(0, "seed", int(seed))
        metric_frames.append(observed)
    metrics = pd.concat(metric_frames, ignore_index=True)
    primary = metrics[metrics.scope.eq("pooled") & metrics.endpoint.isin(["MA", "CYP3A4_is_TDI", "CYP2D6_is_TDI"])]
    summary = primary.groupby(["model", "endpoint", "metric"]).value.agg(["mean", "std", "median", "min", "max", "count"]).reset_index()
    if len(metrics) != 420 or len(summary) != 6:
        raise RuntimeError("Revised metric/summary row count mismatch.")
    paths = [output / "neural_family_predictions.csv.gz", output / "neural_family_metrics.csv", output / "neural_family_seed_summary.csv"]
    neural._write_deterministic_csv_gzip(prediction, paths[0])
    metrics.to_csv(paths[1], index=False)
    summary.to_csv(paths[2], index=False)
    for path, expected in zip(paths, [prediction, metrics, summary], strict=True):
        pd.testing.assert_frame_equal(pd.read_csv(path), expected, check_dtype=False, atol=1e-10, rtol=1e-8)
    pd.DataFrame(selections).to_csv(output / "corrected_selection_audit.csv", index=False)
    verify_sources(root, delivery, files)
    manifest = {"schema_version": 3, "protocol_id": PROTOCOL_ID, "run_fingerprint": fingerprint,
                "generated_utc": utc(), "complete": True, "jobs": 200, "rerun_tdi_jobs": 75,
                "reused_direct_jobs": 125, "prediction_rows": 15340, "expected_prediction_rows": 15340,
                "seeds": config["seeds"], "files": files, "job_provenance": provenance,
                "training_registrations": registrations,
                "outputs": {str(p.relative_to(root)): sha(p) for p in paths},
                "flagship_claim_authorized": False, "outer_results_seen_before_correction": True}
    write_json(output / "neural_family_manifest.json", manifest)
    checks = ["original_200_artifacts_and_frozen_inputs", "original_regression_min_mae_selection",
              "actual_callback_direction_self_test", "corrected_75_identity_and_hashes",
              "actual_last_checkpoint_lifecycle_self_test", "corrected_75_last_save_metadata_and_weights",
              "corrected_75_required_gpu", "corrected_75_max_prc_checkpoint_selection",
              "corrected_75_max_prc_early_stopping", "corrected_75_checkpoint_callback_state",
              "corrected_75_exported_weights_match_selected_checkpoint",
              "corrected_75_validation_prc_recomputed_from_saved_model",
              "predictions_match_job_artifacts_and_raw_labels", "prediction_coverage_exact",
              "regression_predictions_unchanged", "official_420_metrics_recomputed",
              "six_seed_summaries_recomputed", "source_and_input_hashes_unchanged"]
    audit = {"overall_pass": True, "protocol_id": PROTOCOL_ID, "generated_utc": utc(),
             "checks": [{"check_id": check, "status": "PASS", "critical": True} for check in checks],
             "summary": {"completed_jobs": 200, "rerun_tdi_jobs": 75, "reused_direct_jobs": 125,
                         "prediction_rows": 15340, "metric_rows": 420, "seed_summary_rows": 6},
             "flagship_claim_authorized": False, "callback_self_test": self_test}
    write_json(output / "neural_result_audit.json", audit)
    report = "# Corrected neural baseline audit\n\nPASS: 75 PRC-max TDI reruns + 125 verified unchanged regression jobs.\n\n"
    report += "The v2 TDI jobs selected minimum PRC and remain historical implementation-defective artifacts.\n"
    report += "The repaired run does not establish a primary-model, blind-test, or flagship-paper claim.\n\n"
    report += "\n".join("- PASS: " + check for check in checks) + "\n"
    (output / "NEURAL_RESULT_AUDIT.md").write_text(report)
    target = delivery / "CYP_neural_v3_review.zip"
    if target.exists():
        target = delivery / ("CYP_neural_v3_review_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".zip")
    names = [p.name for p in paths] + ["neural_family_manifest.json", "neural_result_audit.json",
              "NEURAL_RESULT_AUDIT.md", "legacy_selection_audit.csv", "corrected_selection_audit.csv",
              "callback_self_test.json", "RUN_REGISTRATION.json"]
    with zipfile.ZipFile(target, "x", zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(output / name, name)
        for path in sorted((output / "registrations").glob("*.json")):
            archive.write(path, str(path.relative_to(output)))
    print(summary.to_string(index=False), flush=True)
    print(f"PASS: 75 corrected TDI jobs, 125 reused regression jobs, and result audit completed.\nReview ZIP: {target}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--delivery", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.delivery)


if __name__ == "__main__":
    main()
