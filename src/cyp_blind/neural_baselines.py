from __future__ import annotations

import argparse
import gzip
import importlib.metadata
import io
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import summarize_predictions
from .chem import standardize_smiles
from .io import file_sha256, load_yaml


@dataclass(frozen=True)
class DataSpec:
    mode: str
    task_group: str
    endpoint: str | None
    fold: int
    targets: tuple[str, ...]

    @property
    def key(self) -> str:
        endpoint = self.endpoint or "all"
        return f"{self.mode}__{self.task_group}__{endpoint}__fold{self.fold}"


@dataclass(frozen=True)
class NeuralJob:
    mode: str
    task_group: str
    endpoint: str | None
    fold: int
    seed: int
    targets: tuple[str, ...]

    @property
    def data_key(self) -> str:
        endpoint = self.endpoint or "all"
        return f"{self.mode}__{self.task_group}__{endpoint}__fold{self.fold}"

    @property
    def job_id(self) -> str:
        endpoint = self.endpoint or "all"
        return (
            f"{self.mode}__{self.task_group}__{endpoint}__"
            f"fold{self.fold}__seed{self.seed}"
        )

    @property
    def model_name(self) -> str:
        return f"chemprop_{self.mode}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(root: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", *args], cwd=root, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _atomic_json(path: Path, payload: dict) -> None:
    """Publish a marker only after the entire JSON has reached the filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_deterministic_csv_gzip(frame: pd.DataFrame, path: Path) -> None:
    """Write a gzip-compressed CSV without path or wall-clock metadata."""
    with path.open("wb") as raw_file:
        with gzip.GzipFile(
            filename="", fileobj=raw_file, mode="wb", mtime=0
        ) as compressed_file:
            with io.TextIOWrapper(
                compressed_file, encoding="utf-8", newline=""
            ) as text_file:
                frame.to_csv(text_file, index=False)


def data_specifications(config: dict) -> list[DataSpec]:
    folds = int(config["cross_validation"]["outer_folds"])
    specs: list[DataSpec] = []
    for fold in range(folds):
        for task_group in ("direct", "tdi"):
            single_targets = tuple(config["models"]["single_task"][task_group])
            for endpoint in single_targets:
                specs.append(
                    DataSpec(
                        mode="single_task",
                        task_group=task_group,
                        endpoint=endpoint,
                        fold=fold,
                        targets=(endpoint,),
                    )
                )
            multitask_targets = tuple(config["models"]["masked_multitask"][task_group])
            specs.append(
                DataSpec(
                    mode="masked_multitask",
                    task_group=task_group,
                    endpoint=None,
                    fold=fold,
                    targets=multitask_targets,
                )
            )
    if len(specs) != int(config["expected"]["data_specs"]):
        raise AssertionError("Unexpected neural data-spec count")
    return specs


def neural_jobs(config: dict) -> list[NeuralJob]:
    specs = data_specifications(config)
    jobs = [
        NeuralJob(
            mode=spec.mode,
            task_group=spec.task_group,
            endpoint=spec.endpoint,
            fold=spec.fold,
            seed=int(seed),
            targets=spec.targets,
        )
        for seed in config["seeds"]
        for spec in specs
    ]
    if len(jobs) != int(config["expected"]["jobs"]):
        raise AssertionError("Unexpected neural job count")
    if len({job.job_id for job in jobs}) != len(jobs):
        raise AssertionError("Neural job identifiers are not unique")
    return jobs


def split_membership(
    molecule_names: pd.Series,
    family_fold_by_name: dict[str, int],
    *,
    test_fold: int,
    validation_fold: int,
) -> pd.Series:
    membership = pd.Series("train", index=molecule_names.index, dtype="string")
    fold_values = molecule_names.map(family_fold_by_name)
    membership.loc[fold_values.eq(validation_fold)] = "val"
    membership.loc[fold_values.eq(test_fold)] = "test"
    return membership


def _normalise_targets(
    training: pd.DataFrame, targets: tuple[str, ...], task_group: str
) -> None:
    if task_group == "tdi":
        for target in targets:
            values = training[target].astype("boolean")
            training[target] = values.astype("Float64")
    else:
        for target in targets:
            training[target] = pd.to_numeric(training[target], errors="raise")


def _verify_frozen_inputs(root: Path, config: dict) -> None:
    for name, expected in config.get("input_sha256", {}).items():
        if file_sha256(root / config["inputs"][name]) != expected:
            raise RuntimeError(f"Frozen source input hash mismatch: {name}")


def prepare_data(config_path: Path) -> dict:
    config_path = config_path.resolve()
    root = config_path.parents[1]
    config = load_yaml(config_path)
    _verify_frozen_inputs(root, config)
    training_path = root / config["inputs"]["training"]
    family_path = root / config["inputs"]["family_split"]
    prepared_root = root / config["paths"]["prepared_data"]
    manifest_path = prepared_root / "preparation_manifest.json"
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if set(payload.get("data_specs", {})) != {
            spec.key for spec in data_specifications(config)
        }:
            raise RuntimeError(
                "Existing preparation matrix differs; preserve it and use a new protocol namespace"
            )
        # Critically, do not change generated_utc or the manifest hash on resume.
        for spec in data_specifications(config):
            job = NeuralJob(**asdict(spec), seed=int(config["seeds"][0]))
            _prepared_files(root, config, job, config_path=config_path)
        print(f"Reusing verified preparation manifest at {manifest_path}")
        return payload
    if any((root / config["paths"]["job_root"]).glob("*/COMPLETE.json")):
        raise RuntimeError(
            "Completed jobs exist but their preparation manifest is missing; restore the original manifest"
        )
    training = pd.read_csv(training_path)
    family = pd.read_csv(family_path)
    if training["Molecule_Name"].duplicated().any():
        raise ValueError("Training Molecule_Name must be unique")
    if family["Molecule_Name"].duplicated().any():
        raise ValueError("Family panel Molecule_Name must be unique")
    if not set(family["Molecule_Name"]).issubset(set(training["Molecule_Name"])):
        raise ValueError(
            "Family panel contains molecules outside the training universe"
        )
    observed_folds = set(family["outer_fold"].astype(int))
    expected_folds = set(range(int(config["cross_validation"]["outer_folds"])))
    if observed_folds != expected_folds:
        raise ValueError(
            f"Family panel folds differ from protocol: {observed_folds} != {expected_folds}"
        )
    records = [standardize_smiles(value) for value in training["SMILES"]]
    training = training.copy()
    training["SMILES"] = [record.canonical_smiles for record in records]
    family_fold_by_name = family.set_index("Molecule_Name")["outer_fold"].to_dict()
    prepared_root.mkdir(parents=True, exist_ok=True)
    specs_payload: dict[str, dict] = {}
    validation_offset = int(config["cross_validation"]["inner_validation_offset"])
    fold_count = int(config["cross_validation"]["outer_folds"])

    for spec in data_specifications(config):
        validation_fold = (spec.fold + validation_offset) % fold_count
        frame = training[["Molecule_Name", "SMILES", *spec.targets]].copy()
        _normalise_targets(frame, spec.targets, spec.task_group)
        frame["_membership"] = split_membership(
            frame["Molecule_Name"],
            family_fold_by_name,
            test_fold=spec.fold,
            validation_fold=validation_fold,
        )
        frame = frame[frame[list(spec.targets)].notna().any(axis=1)].copy()
        spec_dir = prepared_root / spec.key
        spec_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        counts: dict[str, int] = {}
        target_counts: dict[str, dict[str, int]] = {}
        for membership in ("train", "val", "test"):
            output = spec_dir / f"{membership}.csv"
            subset = frame[frame["_membership"].eq(membership)].drop(
                columns="_membership"
            )
            subset.to_csv(output, index=False)
            paths[membership] = output
            counts[membership] = int(len(subset))
            target_counts[membership] = {
                target: int(subset[target].notna().sum()) for target in spec.targets
            }
        if not all(counts[membership] > 0 for membership in ("train", "val", "test")):
            raise RuntimeError(f"Empty neural data partition for {spec.key}: {counts}")
        if spec.task_group == "tdi":
            for membership in ("train", "val"):
                subset = frame[frame["_membership"].eq(membership)]
                for target in spec.targets:
                    classes = set(subset[target].dropna().astype(int))
                    if classes != {0, 1}:
                        raise RuntimeError(
                            f"Degenerate {membership} classes for {spec.key}/{target}: "
                            f"{sorted(classes)}"
                        )
        specs_payload[spec.key] = {
            **asdict(spec),
            "validation_fold": validation_fold,
            "counts": counts,
            "target_counts": target_counts,
            "files": {
                membership: {
                    "path": str(path.relative_to(root)),
                    "sha256": file_sha256(path),
                    "bytes": path.stat().st_size,
                }
                for membership, path in paths.items()
            },
        }

    payload = {
        "schema_version": 1,
        "generated_utc": _utc_now(),
        "config_sha256": file_sha256(config_path),
        "training_sha256": file_sha256(training_path),
        "family_split_sha256": file_sha256(family_path),
        "standardization": "RDKit Cleanup + FragmentParent + Uncharger + canonical isomeric SMILES",
        "data_specs": specs_payload,
    }
    _atomic_json(manifest_path, payload)
    print(
        f"Prepared {len(specs_payload)} neural data specifications at {prepared_root}"
    )
    return payload


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def preflight(config_path: Path, *, require_gpu: bool) -> dict:
    config_path = config_path.resolve()
    root = config_path.parents[1]
    config = load_yaml(config_path)
    checks: list[dict] = []

    def add(check_id: str, passed: bool, detail: str, critical: bool = True) -> None:
        checks.append(
            {
                "check_id": check_id,
                "status": "PASS" if passed else ("FAIL" if critical else "WARN"),
                "critical": critical,
                "detail": detail,
            }
        )

    chemprop_version = _package_version("chemprop")
    torch_distribution_version = _package_version("torch")
    add(
        "dependencies.python",
        platform.python_version() == str(config["dependencies"]["python"]),
        f"observed={platform.python_version()}, expected={config['dependencies']['python']}",
    )
    add(
        "dependencies.chemprop",
        chemprop_version == str(config["dependencies"]["chemprop"]),
        f"observed={chemprop_version}, expected={config['dependencies']['chemprop']}",
    )
    add(
        "dependencies.torch",
        torch_distribution_version == str(config["dependencies"]["torch"]),
        f"observed={torch_distribution_version}, expected={config['dependencies']['torch']}",
    )
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        compiled_cuda = torch.version.cuda
        add(
            "dependencies.compiled_cuda",
            compiled_cuda == str(config["dependencies"]["compiled_cuda"]),
            f"observed={compiled_cuda}, expected={config['dependencies']['compiled_cuda']}",
        )
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
            compute_capability = ".".join(
                str(value) for value in torch.cuda.get_device_capability(0)
            )
            cudnn_version = torch.backends.cudnn.version()
        else:
            device_name = None
            total_gib = 0.0
            compute_capability = None
            cudnn_version = None
    except Exception as error:
        cuda_available = False
        compiled_cuda = None
        device_name = None
        total_gib = 0.0
        compute_capability = None
        cudnn_version = None
        add("dependencies.torch_import", False, repr(error))

    driver_query = (
        subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if shutil.which("nvidia-smi")
        else None
    )
    driver_version = (
        driver_query.stdout.splitlines()[0].strip()
        if driver_query is not None
        and driver_query.returncode == 0
        and driver_query.stdout.strip()
        else None
    )

    gpu_config = config["production_gpu"]
    add(
        "gpu.cuda_available",
        cuda_available,
        f"cuda_available={cuda_available}",
        critical=require_gpu,
    )
    if require_gpu:
        required_name = str(gpu_config["required_name_contains"])
        add(
            "gpu.device_name",
            device_name is not None and required_name.lower() in device_name.lower(),
            f"observed={device_name!r}, required substring={required_name!r}",
        )
        add(
            "gpu.memory",
            total_gib >= float(gpu_config["minimum_vram_gib"]),
            f"observed={total_gib:.2f} GiB, minimum={gpu_config['minimum_vram_gib']} GiB",
        )
        add(
            "gpu.compute_capability",
            compute_capability == str(gpu_config["required_compute_capability"]),
            f"observed={compute_capability!r}, "
            f"required={gpu_config['required_compute_capability']!r}",
        )
    git_head = _git(root, "rev-parse", "HEAD")
    git_status = _git(root, "status", "--porcelain")
    git_tags = (_git(root, "tag", "--points-at", "HEAD") or "").splitlines()
    add(
        "provenance.git_repository",
        git_head is not None,
        f"git_head={git_head!r}",
        critical=require_gpu,
    )
    provenance = config["provenance"]
    if require_gpu and bool(provenance["require_clean_git_at_production_start"]):
        add(
            "provenance.clean_worktree",
            git_status == "",
            f"git_status={git_status!r}",
        )
    required_tag = str(provenance["required_protocol_tag"])
    if require_gpu:
        add(
            "provenance.protocol_tag",
            required_tag in git_tags,
            f"observed_tags={git_tags}, required={required_tag!r}",
        )
    add(
        "inputs.training",
        (root / config["inputs"]["training"]).is_file(),
        str(root / config["inputs"]["training"]),
    )
    add(
        "inputs.family_split",
        (root / config["inputs"]["family_split"]).is_file(),
        str(root / config["inputs"]["family_split"]),
    )
    try:
        _verify_frozen_inputs(root, config)
        add("inputs.frozen_hashes", True, "Source hashes match the committed protocol")
    except (OSError, RuntimeError) as error:
        add("inputs.frozen_hashes", False, str(error))
    overall_pass = not any(check["status"] == "FAIL" for check in checks)
    return {
        "generated_utc": _utc_now(),
        "overall_pass": overall_pass,
        "require_gpu": require_gpu,
        "environment": {
            "python": platform.python_version(),
            "chemprop": chemprop_version,
            "torch": torch_distribution_version,
            "compiled_cuda": compiled_cuda,
            "cuda_available": cuda_available,
            "device_name": device_name,
            "vram_gib": total_gib,
            "compute_capability": compute_capability,
            "cudnn_version": cudnn_version,
            "nvidia_driver": driver_version,
            "git_head": git_head,
            "git_status": git_status,
            "git_tags_at_head": git_tags,
        },
        "checks": checks,
    }


def build_chemprop_command(
    config: dict,
    job: NeuralJob,
    *,
    train_path: Path,
    val_path: Path,
    test_path: Path,
    output_dir: Path,
    accelerator: str,
    devices: str,
    num_workers: int | None = None,
    epochs: int | None = None,
) -> list[str]:
    architecture = config["architecture"]
    optimization = config["optimization"]
    task_config = optimization[
        "regression" if job.task_group == "direct" else "classification"
    ]
    command = [
        shutil.which("chemprop") or "chemprop",
        "train",
        "-q",
        "-i",
        str(train_path),
        str(val_path),
        str(test_path),
        "-o",
        str(output_dir),
        "-s",
        "SMILES",
        "--target-columns",
        *job.targets,
        "--ignore-columns",
        "Molecule_Name",
        "-t",
        str(task_config["task_type"]),
        "-l",
        str(task_config["loss"]),
        "--metrics",
        *[str(value) for value in task_config["metrics"]],
        "--tracking-metric",
        str(task_config["tracking_metric"]),
        "--message-hidden-dim",
        str(architecture["message_hidden_dim"]),
        "--depth",
        str(architecture["depth"]),
        "--dropout",
        str(architecture["dropout"]),
        "--aggregation",
        str(architecture["aggregation"]),
        "--activation",
        str(architecture["activation"]),
        "--ffn-hidden-dim",
        str(architecture["ffn_hidden_dim"]),
        "--ffn-num-layers",
        str(architecture["ffn_num_layers"]),
        "--multi-hot-atom-featurizer-mode",
        str(architecture["atom_featurizer"]),
        "--ensemble-size",
        str(architecture["ensemble_size"]),
        "--batch-size",
        str(optimization["batch_size"]),
        "--num-workers",
        str(optimization["num_workers"] if num_workers is None else num_workers),
        "--epochs",
        str(optimization["epochs"] if epochs is None else epochs),
        "--patience",
        str(optimization["patience"]),
        "--min-delta",
        str(optimization["min_delta"]),
        "--warmup-epochs",
        str(optimization["warmup_epochs"]),
        "--init-lr",
        str(optimization["init_lr"]),
        "--max-lr",
        str(optimization["max_lr"]),
        "--final-lr",
        str(optimization["final_lr"]),
        "--pytorch-seed",
        str(job.seed),
        "--data-seed",
        str(job.seed),
        "--accelerator",
        accelerator,
        "--devices",
        devices,
        "--show-individual-scores",
        "--remove-checkpoints",
    ]
    if job.task_group == "tdi" and bool(task_config["class_balance"]):
        command.append("--class-balance")
    return command


def _prepared_files(
    root: Path,
    config: dict,
    job: NeuralJob,
    *,
    config_path: Path,
) -> dict[str, Path]:
    _verify_frozen_inputs(root, config)
    base = root / config["paths"]["prepared_data"] / job.data_key
    paths = {
        membership: base / f"{membership}.csv"
        for membership in ("train", "val", "test")
    }
    if not all(path.is_file() for path in paths.values()):
        raise FileNotFoundError(
            f"Prepared data missing for {job.data_key}; run neural prepare first"
        )
    manifest_path = base.parent / "preparation_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Neural preparation manifest is missing; run neural prepare first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    top_level_hashes = {
        "config": (manifest["config_sha256"], file_sha256(config_path)),
        "training": (
            manifest["training_sha256"],
            file_sha256(root / config["inputs"]["training"]),
        ),
        "family_split": (
            manifest["family_split_sha256"],
            file_sha256(root / config["inputs"]["family_split"]),
        ),
    }
    changed = [
        name for name, hashes in top_level_hashes.items() if hashes[0] != hashes[1]
    ]
    if changed:
        raise RuntimeError(
            f"Prepared-data provenance changed ({changed}); restore frozen inputs or use a new protocol namespace"
        )
    spec_manifest = manifest["data_specs"].get(job.data_key)
    if spec_manifest is None:
        raise RuntimeError(f"Preparation manifest lacks {job.data_key}")
    expected_spec = asdict(
        DataSpec(job.mode, job.task_group, job.endpoint, job.fold, job.targets)
    )
    expected_spec["targets"] = list(job.targets)
    if any(spec_manifest.get(key) != value for key, value in expected_spec.items()):
        raise RuntimeError(
            f"Preparation specification identity changed: {job.data_key}"
        )
    if set(spec_manifest.get("files", {})) != set(paths):
        raise RuntimeError(f"Preparation partition set changed: {job.data_key}")
    for membership, path in paths.items():
        if spec_manifest["files"][membership]["path"] != str(path.relative_to(root)):
            raise RuntimeError(
                f"Preparation partition path changed: {job.data_key}/{membership}"
            )
        expected_hash = spec_manifest["files"][membership]["sha256"]
        observed_hash = file_sha256(path)
        if observed_hash != expected_hash:
            raise RuntimeError(
                f"Prepared {membership} hash changed for {job.data_key}: "
                f"{observed_hash} != {expected_hash}"
            )
    return paths


def _validate_job_predictions(
    root: Path,
    config: dict,
    job: NeuralJob,
    test_path: Path,
    chemprop_predictions_path: Path,
) -> pd.DataFrame:
    test = pd.read_csv(test_path)
    predicted = pd.read_csv(chemprop_predictions_path)
    if len(test) != len(predicted):
        raise AssertionError("Chemprop prediction row count does not match test input")
    if not test["SMILES"].astype(str).eq(predicted["SMILES"].astype(str)).all():
        raise AssertionError("Chemprop prediction order/SMILES differs from test input")
    raw = pd.read_csv(root / config["inputs"]["training"]).set_index("Molecule_Name")
    family = pd.read_csv(root / config["inputs"]["family_split"]).set_index(
        "Molecule_Name"
    )
    threshold = float(
        config["optimization"]["classification"]["fixed_decision_threshold"]
    )
    rows: list[dict] = []
    for row_index, test_row in test.iterrows():
        molecule_name = str(test_row["Molecule_Name"])
        for endpoint in job.targets:
            if pd.isna(test_row[endpoint]):
                continue
            prediction = float(predicted.at[row_index, endpoint])
            if not np.isfinite(prediction):
                raise AssertionError("Chemprop produced a non-finite prediction")
            task = "regression" if job.task_group == "direct" else "classification"
            probability = np.nan
            y_pred = prediction
            lower = np.nan
            upper = np.nan
            if task == "classification":
                if not 0 <= prediction <= 1:
                    raise AssertionError(
                        "Chemprop classification probability is out of range"
                    )
                probability = prediction
                y_pred = float(prediction >= threshold)
                y_true = float(
                    pd.Series([test_row[endpoint]]).astype("boolean").iloc[0]
                )
            else:
                y_true = float(test_row[endpoint])
                lower = float(raw.at[molecule_name, f"{endpoint}_conf_low"])
                upper = float(raw.at[molecule_name, f"{endpoint}_conf_high"])
            rows.append(
                {
                    "model": job.model_name,
                    "mode": job.mode,
                    "seed": job.seed,
                    "fold": job.fold,
                    "Molecule_Name": molecule_name,
                    "family_id": family.at[molecule_name, "family_id"],
                    "endpoint": endpoint,
                    "task": task,
                    "y_true": y_true,
                    "y_true_lower": lower,
                    "y_true_upper": upper,
                    "y_pred": y_pred,
                    "probability": probability,
                }
            )
    return pd.DataFrame(rows)


def validate_completion(config_path: Path, job: NeuralJob) -> dict:
    """Fail closed on stale, partial, CPU, cross-revision, or altered results."""
    config_path = config_path.resolve()
    root = config_path.parents[1]
    config = load_yaml(config_path)
    job_root = root / config["paths"]["job_root"] / job.job_id
    complete = json.loads((job_root / "COMPLETE.json").read_text(encoding="utf-8"))
    expected_job = asdict(job)
    expected_job["targets"] = list(job.targets)
    if (
        complete.get("schema_version") != 2
        or complete.get("status") != "PASS"
        or complete.get("job_id") != job.job_id
        or complete.get("job") != expected_job
    ):
        raise RuntimeError(f"Completion identity/schema mismatch: {job.job_id}")
    prepared = _prepared_files(root, config, job, config_path=config_path)
    required_tag = str(config["provenance"]["required_protocol_tag"])
    head = _git(root, "rev-parse", "HEAD")
    if (
        not head
        or complete.get("git_head") != head
        or _git(root, "rev-parse", required_tag + "^{commit}") != head
        or required_tag not in complete.get("git_tags_at_head", [])
        or _git(root, "status", "--porcelain") != ""
        or complete.get("config_sha256") != file_sha256(config_path)
        or complete.get("preparation_manifest_sha256")
        != file_sha256(
            root / config["paths"]["prepared_data"] / "preparation_manifest.json"
        )
    ):
        raise RuntimeError(f"Completion frozen provenance mismatch: {job.job_id}")
    environment = complete.get("environment", {})
    gpu = config["production_gpu"]
    if (
        environment.get("cuda_available") is not True
        or str(gpu["required_name_contains"]).lower()
        not in str(environment.get("device_name")).lower()
        or not float(environment.get("vram_gib", 0)) >= float(gpu["minimum_vram_gib"])
        or environment.get("compute_capability")
        != str(gpu["required_compute_capability"])
        or environment.get("git_head") != head
        or environment.get("git_status") != ""
        or complete.get("execution")
        != {"accelerator": "gpu", "devices": str(gpu["devices"])}
        or any(
            environment.get(name) != str(version)
            for name, version in config["dependencies"].items()
        )
    ):
        raise RuntimeError(
            f"Completion GPU/dependency provenance mismatch: {job.job_id}"
        )
    runtime = float(complete.get("runtime_seconds", float("nan")))
    if not np.isfinite(runtime) or runtime < 0:
        raise RuntimeError(f"Invalid runtime: {job.job_id}")
    if set(complete.get("inputs", {})) != {"train", "val", "test"}:
        raise RuntimeError(f"Completion missing input artifacts: {job.job_id}")
    for membership, path in prepared.items():
        expected = {"path": str(path.relative_to(root)), "sha256": file_sha256(path)}
        if complete["inputs"][membership] != expected:
            raise RuntimeError(
                f"Completion input hash/path mismatch: {job.job_id}/{membership}"
            )
    required_outputs = {"predictions", "chemprop_predictions", "best_model", "log"}
    if set(complete.get("outputs", {})) != required_outputs:
        raise RuntimeError(f"Completion missing output artifacts: {job.job_id}")
    for name, entry in complete["outputs"].items():
        path = (root / entry["path"]).resolve()
        if (
            not path.is_relative_to(job_root.resolve())
            or not path.is_file()
            or file_sha256(path) != entry["sha256"]
        ):
            raise RuntimeError(
                f"Completion artifact hash/path mismatch: {job.job_id}/{name}"
            )
    expected_predictions = _validate_job_predictions(
        root,
        config,
        job,
        prepared["test"],
        root / complete["outputs"]["chemprop_predictions"]["path"],
    )
    observed_predictions = pd.read_csv(
        root / complete["outputs"]["predictions"]["path"]
    )
    pd.testing.assert_frame_equal(
        observed_predictions,
        expected_predictions,
        check_dtype=False,
        check_exact=False,
        rtol=0,
        atol=1e-12,
    )
    return complete


def run_job(config_path: Path, job: NeuralJob, *, require_gpu: bool) -> dict:
    if not require_gpu:
        raise RuntimeError(
            "Production run-job requires explicit --require-gpu; use smoke for CPU diagnostics"
        )
    import fcntl  # Production execution is Linux/AutoDL only.

    config_path = config_path.resolve()
    config = load_yaml(config_path)
    job_root = config_path.parents[1] / config["paths"]["job_root"] / job.job_id
    job_root.mkdir(parents=True, exist_ok=True)
    with (job_root / "job.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Job is already running: {job.job_id}") from error
        return _run_gpu_job(config_path, job)


def _run_gpu_job(config_path: Path, job: NeuralJob) -> dict:
    config_path = config_path.resolve()
    root = config_path.parents[1]
    config = load_yaml(config_path)
    job_root = root / config["paths"]["job_root"] / job.job_id
    complete_path = job_root / "COMPLETE.json"
    if complete_path.is_file():
        return validate_completion(config_path, job)

    flight = preflight(config_path, require_gpu=True)
    if not flight["overall_pass"]:
        raise RuntimeError(
            f"Neural preflight failed for {job.job_id}: {flight['checks']}"
        )
    prepared = _prepared_files(root, config, job, config_path=config_path)
    start_config_hash = file_sha256(config_path)
    start_preparation_hash = file_sha256(
        root / config["paths"]["prepared_data"] / "preparation_manifest.json"
    )
    start_input_hashes = {
        membership: {"path": str(path.relative_to(root)), "sha256": file_sha256(path)}
        for membership, path in prepared.items()
    }
    attempt_name = datetime.now(timezone.utc).strftime("attempt_%Y%m%dT%H%M%S%fZ")
    attempt_dir = job_root / attempt_name
    model_dir = attempt_dir / "model"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    accelerator = "gpu"
    devices = str(config["production_gpu"]["devices"])
    command = build_chemprop_command(
        config,
        job,
        train_path=prepared["train"],
        val_path=prepared["val"],
        test_path=prepared["test"],
        output_dir=model_dir,
        accelerator=accelerator,
        devices=devices,
    )
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = str(job.seed)
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    log_path = attempt_dir / "train.log"
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        result = subprocess.run(
            command,
            cwd=root,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        failure = {
            "job": asdict(job),
            "job_id": job.job_id,
            "status": "FAIL",
            "returncode": result.returncode,
            "runtime_seconds": elapsed,
            "command": command,
            "log": str(log_path.relative_to(root)),
        }
        (attempt_dir / "FAILED.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True), encoding="utf-8"
        )
        raise RuntimeError(f"Chemprop job failed: {job.job_id}; see {log_path}")

    chemprop_prediction_path = model_dir / "model_0" / "test_predictions.csv"
    if not chemprop_prediction_path.is_file():
        raise FileNotFoundError(
            f"Chemprop did not write expected predictions: {chemprop_prediction_path}"
        )
    predictions = _validate_job_predictions(
        root, config, job, prepared["test"], chemprop_prediction_path
    )
    prediction_path = attempt_dir / "predictions.csv"
    predictions.to_csv(prediction_path, index=False)
    best_model = model_dir / "model_0" / "best.pt"
    if not best_model.is_file():
        raise FileNotFoundError(f"Chemprop did not write best model: {best_model}")
    # Attribute results to the code and data at launch, not just at completion.
    _prepared_files(root, config, job, config_path=config_path)
    if (
        _git(root, "rev-parse", "HEAD") != flight["environment"]["git_head"]
        or _git(root, "status", "--porcelain") != ""
        or file_sha256(config_path) != start_config_hash
        or file_sha256(
            root / config["paths"]["prepared_data"] / "preparation_manifest.json"
        )
        != start_preparation_hash
        or any(
            file_sha256(prepared[name]) != entry["sha256"]
            for name, entry in start_input_hashes.items()
        )
    ):
        raise RuntimeError(
            f"Code/data changed during training; no completion marker written: {job.job_id}"
        )
    complete = {
        "schema_version": 2,
        "completed_utc": _utc_now(),
        "status": "PASS",
        "job": asdict(job),
        "job_id": job.job_id,
        "git_head": flight["environment"]["git_head"],
        "git_tags_at_head": flight["environment"]["git_tags_at_head"],
        "config_sha256": start_config_hash,
        "preparation_manifest_sha256": start_preparation_hash,
        "runtime_seconds": elapsed,
        "command": command,
        "execution": {"accelerator": accelerator, "devices": devices},
        "environment": flight["environment"],
        "inputs": start_input_hashes,
        "outputs": {
            "predictions": {
                "path": str(prediction_path.relative_to(root)),
                "sha256": file_sha256(prediction_path),
            },
            "chemprop_predictions": {
                "path": str(chemprop_prediction_path.relative_to(root)),
                "sha256": file_sha256(chemprop_prediction_path),
            },
            "best_model": {
                "path": str(best_model.relative_to(root)),
                "sha256": file_sha256(best_model),
            },
            "log": {
                "path": str(log_path.relative_to(root)),
                "sha256": file_sha256(log_path),
            },
        },
    }
    job_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(complete_path, complete)
    return complete


def smoke_test(config_path: Path) -> dict:
    config_path = config_path.resolve()
    root = config_path.parents[1]
    fixtures = root / "tests/fixtures"
    results: dict[str, dict] = {}
    cases = {
        "regression": {
            "train": fixtures / "chemprop_smoke_train.csv",
            "val": fixtures / "chemprop_smoke_val.csv",
            "test": fixtures / "chemprop_smoke_test.csv",
            "task_type": "regression",
            "loss": "mse",
            "metrics": ["mae"],
            "tracking": "mae",
            "class_balance": False,
        },
        "classification": {
            "train": fixtures / "chemprop_smoke_classification_train.csv",
            "val": fixtures / "chemprop_smoke_classification_val.csv",
            "test": fixtures / "chemprop_smoke_classification_test.csv",
            "task_type": "classification",
            "loss": "bce",
            "metrics": ["prc", "binary-mcc"],
            "tracking": "prc",
            "class_balance": True,
        },
    }
    for case_name, case in cases.items():
        with tempfile.TemporaryDirectory(
            prefix=f"cyp-neural-smoke-{case_name}-"
        ) as temp:
            output = Path(temp) / "model"
            command = [
                shutil.which("chemprop") or "chemprop",
                "train",
                "-q",
                "-i",
                str(case["train"]),
                str(case["val"]),
                str(case["test"]),
                "-o",
                str(output),
                "-s",
                "SMILES",
                "--target-columns",
                "target",
                "--ignore-columns",
                "Molecule_Name",
                "-t",
                str(case["task_type"]),
                "-l",
                str(case["loss"]),
                "--metrics",
                *case["metrics"],
                "--tracking-metric",
                str(case["tracking"]),
                "--epochs",
                "3",
                "--warmup-epochs",
                "1",
                "--patience",
                "1",
                "--accelerator",
                "cpu",
                "--devices",
                "1",
                "--message-hidden-dim",
                "32",
                "--depth",
                "2",
                "--dropout",
                "0",
                "--aggregation",
                "mean",
                "--ffn-hidden-dim",
                "32",
                "--ffn-num-layers",
                "1",
                "--batch-size",
                "4",
                "--num-workers",
                "0",
                "--pytorch-seed",
                "20260829",
                "--remove-checkpoints",
            ]
            if case["class_balance"]:
                command.append("--class-balance")
            process = subprocess.run(
                command,
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
            )
            prediction_path = output / "model_0" / "test_predictions.csv"
            passed = process.returncode == 0 and prediction_path.is_file()
            detail = ""
            if passed:
                frame = pd.read_csv(prediction_path)
                expected = pd.read_csv(case["test"])
                passed = (
                    len(frame) == 2
                    and np.isfinite(frame["target"]).all()
                    and frame["SMILES"]
                    .astype(str)
                    .equals(expected["SMILES"].astype(str))
                )
                if case_name == "classification":
                    passed = passed and frame["target"].between(0, 1).all()
                detail = f"rows={len(frame)}, range={frame['target'].min():.6f}-{frame['target'].max():.6f}"
            else:
                detail = process.stderr[-1000:]
            results[case_name] = {
                "status": "PASS" if passed else "FAIL",
                "detail": detail,
            }
    payload = {
        "schema_version": 1,
        "generated_utc": _utc_now(),
        "overall_pass": all(result["status"] == "PASS" for result in results.values()),
        "cases": results,
        "environment": preflight(config_path, require_gpu=False)["environment"],
        "scientific_results_generated": False,
    }
    output_dir = root / load_yaml(config_path)["paths"]["summary_root"]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "NEURAL_SMOKE.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def real_data_integration_smoke(config_path: Path) -> dict:
    """Exercise masked multitask training on prepared data without retaining results."""
    config_path = config_path.resolve()
    root = config_path.parents[1]
    config = load_yaml(config_path)
    flight = preflight(config_path, require_gpu=False)
    if not flight["overall_pass"]:
        raise RuntimeError(
            f"CPU integration-smoke preflight failed: {flight['checks']}"
        )
    jobs = [
        job
        for job in neural_jobs(config)
        if job.mode == "masked_multitask"
        and job.fold == 0
        and job.seed == int(config["seeds"][0])
    ]
    if {job.task_group for job in jobs} != {"direct", "tdi"} or len(jobs) != 2:
        raise AssertionError(
            "Expected one direct and one TDI masked-multitask smoke job"
        )
    results: dict[str, dict] = {}
    for job in jobs:
        prepared = _prepared_files(root, config, job, config_path=config_path)
        with tempfile.TemporaryDirectory(
            prefix=f"cyp-neural-real-smoke-{job.task_group}-"
        ) as temp:
            output = Path(temp) / "model"
            command = build_chemprop_command(
                config,
                job,
                train_path=prepared["train"],
                val_path=prepared["val"],
                test_path=prepared["test"],
                output_dir=output,
                accelerator="cpu",
                devices="1",
                num_workers=0,
                epochs=3,
            )
            environment = os.environ.copy()
            environment["PYTHONHASHSEED"] = str(job.seed)
            started = time.perf_counter()
            process = subprocess.run(
                command,
                cwd=root,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            elapsed = time.perf_counter() - started
            prediction_path = output / "model_0" / "test_predictions.csv"
            passed = process.returncode == 0 and prediction_path.is_file()
            detail = ""
            prediction_range: dict[str, list[float]] = {}
            if passed:
                predicted = pd.read_csv(prediction_path)
                expected = pd.read_csv(prepared["test"])
                passed = (
                    len(predicted) == len(expected)
                    and predicted["SMILES"]
                    .astype(str)
                    .equals(expected["SMILES"].astype(str))
                    and set(job.targets).issubset(predicted.columns)
                    and np.isfinite(predicted[list(job.targets)].to_numpy(float)).all()
                )
                if job.task_group == "tdi":
                    passed = (
                        passed
                        and predicted[list(job.targets)]
                        .apply(lambda column: column.between(0, 1).all())
                        .all()
                    )
                if passed:
                    # Reuse the production validator, but discard all derived rows.
                    _validate_job_predictions(
                        root, config, job, prepared["test"], prediction_path
                    )
                prediction_range = {
                    target: [
                        float(predicted[target].min()),
                        float(predicted[target].max()),
                    ]
                    for target in job.targets
                }
                detail = f"rows={len(predicted)}, targets={len(job.targets)}"
            else:
                detail = process.stderr[-1000:]
            results[job.task_group] = {
                "status": "PASS" if passed else "FAIL",
                "detail": detail,
                "runtime_seconds": elapsed,
                "prediction_range": prediction_range,
            }
    payload = {
        "schema_version": 1,
        "generated_utc": _utc_now(),
        "overall_pass": all(result["status"] == "PASS" for result in results.values()),
        "cases": results,
        "environment": flight["environment"],
        "epochs": 3,
        "predictions_retained": False,
        "model_weights_retained": False,
        "scientific_metrics_generated": False,
        "flagship_claim_authorized": False,
    }
    output_dir = root / config["paths"]["summary_root"]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "NEURAL_REAL_DATA_SMOKE.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def job_status(config_path: Path) -> dict:
    config_path = config_path.resolve()
    root = config_path.parents[1]
    config = load_yaml(config_path)
    jobs = neural_jobs(config)
    job_root = root / config["paths"]["job_root"]
    completed = []
    invalid = {}
    for job in jobs:
        if (job_root / job.job_id / "COMPLETE.json").is_file():
            try:
                validate_completion(config_path, job)
                completed.append(job.job_id)
            except (
                OSError,
                ValueError,
                RuntimeError,
                KeyError,
                AssertionError,
            ) as error:
                invalid[job.job_id] = str(error)
    failed_attempts = (
        list(job_root.glob("*/attempt_*/FAILED.json")) if job_root.exists() else []
    )
    return {
        "expected": len(jobs),
        "completed": len(completed),
        "remaining": len(jobs) - len(completed),
        "failed_attempts": len(failed_attempts),
        "invalid_completions": invalid,
        "complete_job_ids": completed,
    }


def collect_results(config_path: Path) -> dict:
    config_path = config_path.resolve()
    root = config_path.parents[1]
    config = load_yaml(config_path)
    jobs = neural_jobs(config)
    job_root = root / config["paths"]["job_root"]
    frames: list[pd.DataFrame] = []
    completion_hashes: dict[str, str] = {}
    missing: list[str] = []
    total_runtime = 0.0
    for job in jobs:
        complete_path = job_root / job.job_id / "COMPLETE.json"
        if not complete_path.is_file():
            missing.append(job.job_id)
            continue
        complete = validate_completion(config_path, job)
        prediction_path = root / complete["outputs"]["predictions"]["path"]
        frame = pd.read_csv(prediction_path)
        frames.append(frame)
        completion_hashes[job.job_id] = file_sha256(complete_path)
        total_runtime += float(complete["runtime_seconds"])
    if missing:
        raise RuntimeError(
            f"Cannot collect incomplete neural run; missing {len(missing)} jobs"
        )
    predictions = pd.concat(frames, ignore_index=True)
    key_columns = ["mode", "seed", "Molecule_Name", "endpoint"]
    if predictions.duplicated(key_columns).any():
        raise AssertionError("Duplicate neural prediction keys across collected jobs")
    training = pd.read_csv(root / config["inputs"]["training"]).set_index(
        "Molecule_Name"
    )
    family = pd.read_csv(root / config["inputs"]["family_split"])
    endpoints = [
        *config["models"]["single_task"]["direct"],
        *config["models"]["single_task"]["tdi"],
    ]
    expected_prediction_keys = {
        (mode, int(seed), str(name), endpoint)
        for mode in ("single_task", "masked_multitask")
        for seed in config["seeds"]
        for name in family["Molecule_Name"]
        for endpoint in endpoints
        if pd.notna(training.at[name, endpoint])
    }
    frozen_expected_count = int(config["expected"]["predictions"])
    if len(expected_prediction_keys) != frozen_expected_count:
        raise AssertionError(
            "Frozen neural prediction count changed: "
            f"derived={len(expected_prediction_keys)}, expected={frozen_expected_count}"
        )
    observed_prediction_keys = set(
        predictions[key_columns].itertuples(index=False, name=None)
    )
    if observed_prediction_keys != expected_prediction_keys:
        raise AssertionError(
            "Collected neural prediction coverage mismatch: "
            f"missing={len(expected_prediction_keys - observed_prediction_keys)}, "
            f"extra={len(observed_prediction_keys - expected_prediction_keys)}"
        )

    metric_frames: list[pd.DataFrame] = []
    for seed, group in predictions.groupby("seed", sort=True):
        metrics = summarize_predictions(group)
        metrics.insert(0, "seed", int(seed))
        metric_frames.append(metrics)
    metrics = pd.concat(metric_frames, ignore_index=True)
    primary = metrics[
        metrics["scope"].eq("pooled")
        & metrics["metric"].isin(["MA-ST-RAE", "MCC"])
        & (
            metrics["endpoint"].eq("MA")
            | metrics["endpoint"].isin(config["models"]["masked_multitask"]["tdi"])
        )
    ]
    summary = (
        primary.groupby(["model", "endpoint", "metric"])["value"]
        .agg(["mean", "std", "median", "min", "max", "count"])
        .reset_index()
    )
    summary_root = root / config["paths"]["summary_root"]
    summary_root.mkdir(parents=True, exist_ok=True)
    predictions_path = summary_root / "neural_family_predictions.csv.gz"
    _write_deterministic_csv_gzip(predictions, predictions_path)
    metrics_path = summary_root / "neural_family_metrics.csv"
    summary_path = summary_root / "neural_family_seed_summary.csv"
    metrics.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    manifest = {
        "schema_version": 1,
        "generated_utc": _utc_now(),
        "stage": "day3_day5_neural_baselines",
        "complete": True,
        "jobs": len(jobs),
        "prediction_rows": len(predictions),
        "expected_prediction_rows": len(expected_prediction_keys),
        "seeds": list(config["seeds"]),
        "total_runtime_seconds": total_runtime,
        "git_head": _git(root, "rev-parse", "HEAD"),
        "completion_manifest_hashes": completion_hashes,
        "inputs": {
            str(path.relative_to(root)): file_sha256(path)
            for path in [
                config_path,
                root / config["inputs"]["training"],
                root / config["inputs"]["family_split"],
                root / config["paths"]["prepared_data"] / "preparation_manifest.json",
                root / "vendor/openadmet/custom_scoring_functions.py",
            ]
        },
        "outputs": {
            str(path.relative_to(root)): file_sha256(path)
            for path in [predictions_path, metrics_path, summary_path]
        },
        "flagship_claim_authorized": False,
    }
    manifest_path = summary_root / "neural_family_manifest.json"
    _atomic_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/neural_baselines.yaml")
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--require-gpu", action="store_true")
    subparsers.add_parser("smoke")
    subparsers.add_parser("integration-smoke")
    subparsers.add_parser("status")
    subparsers.add_parser("list-jobs")
    run_job_parser = subparsers.add_parser("run-job")
    run_job_parser.add_argument("--index", type=int, required=True)
    run_job_parser.add_argument("--require-gpu", action="store_true")
    run_all_parser = subparsers.add_parser("run-all")
    run_all_parser.add_argument("--require-gpu", action="store_true")
    subparsers.add_parser("collect")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_yaml(config_path)

    if args.command == "prepare":
        payload = prepare_data(config_path)
        print(json.dumps({"data_specs": len(payload["data_specs"])}, indent=2))
        return 0
    if args.command == "preflight":
        payload = preflight(config_path, require_gpu=args.require_gpu)
        print(json.dumps(payload, indent=2))
        return 0 if payload["overall_pass"] else 1
    if args.command == "smoke":
        payload = smoke_test(config_path)
        print(json.dumps(payload, indent=2))
        return 0 if payload["overall_pass"] else 1
    if args.command == "integration-smoke":
        payload = real_data_integration_smoke(config_path)
        print(json.dumps(payload, indent=2))
        return 0 if payload["overall_pass"] else 1
    if args.command == "status":
        print(json.dumps(job_status(config_path), indent=2))
        return 0
    jobs = neural_jobs(config)
    if args.command == "list-jobs":
        print(
            json.dumps(
                [
                    {"index": index, **asdict(job), "job_id": job.job_id}
                    for index, job in enumerate(jobs)
                ],
                indent=2,
            )
        )
        return 0
    if args.command == "run-job":
        if args.index < 0 or args.index >= len(jobs):
            raise IndexError(f"Job index out of range: {args.index}")
        payload = run_job(config_path, jobs[args.index], require_gpu=args.require_gpu)
        print(
            json.dumps(
                {"job_id": payload["job_id"], "status": payload["status"]}, indent=2
            )
        )
        return 0
    if args.command == "run-all":
        if not args.require_gpu:
            raise RuntimeError("Full neural run requires explicit --require-gpu")
        flight = preflight(config_path, require_gpu=True)
        if not flight["overall_pass"]:
            print(json.dumps(flight, indent=2))
            return 1
        for index, job in enumerate(jobs):
            print(f"[{index + 1}/{len(jobs)}] {job.job_id}", flush=True)
            run_job(config_path, job, require_gpu=True)
        return 0
    if args.command == "collect":
        payload = collect_results(config_path)
        print(json.dumps(payload, indent=2))
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
