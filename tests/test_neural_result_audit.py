import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from cyp_blind.io import file_sha256, load_yaml
from cyp_blind.neural_baselines import (
    collect_results,
    neural_jobs,
    prepare_data,
    validate_completion,
    job_status,
    run_job,
)
from cyp_blind.neural_result_audit import audit_neural_results


DIRECT = [
    "CYP1A2_pIC50_direct_inhibition",
    "CYP2C9_pIC50_direct_inhibition",
    "CYP2D6_pIC50_direct_inhibition",
    "CYP3A4_pIC50_direct_inhibition",
]
TDI = ["CYP3A4_is_TDI", "CYP2D6_is_TDI"]


def _minimal_config() -> dict:
    config = load_yaml(
        Path(__file__).resolve().parents[1] / "configs/neural_baselines.yaml"
    )
    config.pop("input_sha256")  # This fixture uses synthetic, not challenge, molecules.
    config.update(
        {
            "inputs": {
                "training": "data/raw/train.csv",
                "family_split": "data/splits/family.csv",
            },
            "paths": {
                "prepared_data": "data/derived/neural",
                "job_root": "reports/neural/jobs",
                "summary_root": "reports/neural",
            },
            "provenance": {"required_protocol_tag": "test-protocol-tag"},
            "seeds": [7],
            "expected": {"data_specs": 40, "jobs": 40, "predictions": 120},
        }
    )
    return config


@pytest.fixture
def completed_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # Synthetic metadata tests validators, never asserts that GPU training occurred.
    def fake_git(root, *args):
        if args[0] == "status":
            return ""
        if args[0] == "tag":
            return "test-protocol-tag"
        return "test-commit"

    monkeypatch.setattr("cyp_blind.neural_baselines._git", fake_git)
    root = tmp_path / "project"
    config_path = root / "configs/neural_baselines.yaml"
    training_path = root / "data/raw/train.csv"
    family_path = root / "data/splits/family.csv"
    preparation_path = root / "data/derived/neural/preparation_manifest.json"
    for path in (config_path, training_path, family_path, preparation_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    official_source = (
        Path(__file__).resolve().parents[1]
        / "vendor/openadmet/custom_scoring_functions.py"
    )
    official_destination = root / "vendor/openadmet/custom_scoring_functions.py"
    official_destination.parent.mkdir(parents=True, exist_ok=True)
    official_destination.write_text(
        official_source.read_text(encoding="utf-8"), encoding="utf-8"
    )
    config = _minimal_config()
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    training_rows = []
    family_rows = []
    for fold in range(5):
        for offset in range(2):
            index = 2 * fold + offset
            name = f"mol_{index:02d}"
            row = {"Molecule_Name": name, "SMILES": "C" * (index + 1)}
            for endpoint_index, endpoint in enumerate(DIRECT):
                value = 3.0 + 0.1 * index + 0.01 * endpoint_index
                row[endpoint] = value
                row[f"{endpoint}_conf_low"] = value - 0.01
                row[f"{endpoint}_conf_high"] = value + 0.01
            for endpoint in TDI:
                row[endpoint] = offset
            training_rows.append(row)
            family_rows.append(
                {
                    "Molecule_Name": name,
                    "family_id": f"family_{fold}",
                    "outer_fold": fold,
                }
            )
    training = pd.DataFrame(training_rows)
    family = pd.DataFrame(family_rows)
    training.to_csv(training_path, index=False)
    family.to_csv(family_path, index=False)
    preparation = prepare_data(config_path)

    raw = training.set_index("Molecule_Name")
    family_indexed = family.set_index("Molecule_Name")
    for job in neural_jobs(config):
        prediction_rows = []
        test_names = family.loc[
            family["outer_fold"].eq(job.fold), "Molecule_Name"
        ].tolist()
        for name in test_names:
            for endpoint in job.targets:
                task = "regression" if job.task_group == "direct" else "classification"
                truth = float(raw.at[name, endpoint])
                if task == "regression":
                    lower = float(raw.at[name, f"{endpoint}_conf_low"])
                    upper = float(raw.at[name, f"{endpoint}_conf_high"])
                    prediction = truth
                    probability = float("nan")
                else:
                    lower = float("nan")
                    upper = float("nan")
                    probability = 0.9 if truth else 0.1
                    prediction = float(probability >= 0.5)
                prediction_rows.append(
                    {
                        "model": job.model_name,
                        "mode": job.mode,
                        "seed": job.seed,
                        "fold": job.fold,
                        "Molecule_Name": name,
                        "family_id": family_indexed.at[name, "family_id"],
                        "endpoint": endpoint,
                        "task": task,
                        "y_true": truth,
                        "y_true_lower": lower,
                        "y_true_upper": upper,
                        "y_pred": prediction,
                        "probability": probability,
                    }
                )
        job_root = root / config["paths"]["job_root"] / job.job_id
        job_root.mkdir(parents=True, exist_ok=True)
        prediction_path = job_root / "predictions.csv"
        pd.DataFrame(prediction_rows).to_csv(prediction_path, index=False)
        inputs = preparation["data_specs"][job.data_key]["files"]
        chemprop_path = job_root / "chemprop_predictions.csv"
        chemprop_frame = pd.read_csv(root / inputs["test"]["path"]).drop(
            columns="Molecule_Name"
        )
        if job.task_group == "tdi":
            for endpoint in job.targets:
                chemprop_frame[endpoint] = chemprop_frame[endpoint].map(
                    {0: 0.1, 1: 0.9}
                )
        chemprop_frame.to_csv(chemprop_path, index=False)
        best_model = job_root / "best.pt"
        best_model.write_bytes(b"SYNTHETIC FIXTURE - NOT A TRAINED MODEL")
        log_path = job_root / "train.log"
        log_path.write_text("Synthetic validator fixture only\n", encoding="utf-8")
        complete = {
            "schema_version": 2,
            "status": "PASS",
            "job_id": job.job_id,
            "job": {
                "mode": job.mode,
                "task_group": job.task_group,
                "endpoint": job.endpoint,
                "fold": job.fold,
                "seed": job.seed,
                "targets": list(job.targets),
            },
            "runtime_seconds": 0.01,
            "git_head": "test-commit",
            "git_tags_at_head": ["test-protocol-tag"],
            "config_sha256": file_sha256(config_path),
            "preparation_manifest_sha256": file_sha256(preparation_path),
            "environment": {
                **config["dependencies"],
                "git_head": "test-commit",
                "git_status": "",
                "cuda_available": True,
                "device_name": "NVIDIA GeForce RTX 4090",
                "vram_gib": 23.5,
                "compute_capability": "8.9",
            },
            "execution": {"accelerator": "gpu", "devices": "1"},
            "inputs": {
                key: {"path": entry["path"], "sha256": entry["sha256"]}
                for key, entry in inputs.items()
            },
            "outputs": {
                key: {"path": str(path.relative_to(root)), "sha256": file_sha256(path)}
                for key, path in {
                    "predictions": prediction_path,
                    "chemprop_predictions": chemprop_path,
                    "best_model": best_model,
                    "log": log_path,
                }.items()
            },
        }
        (job_root / "COMPLETE.json").write_text(
            json.dumps(complete, indent=2, sort_keys=True), encoding="utf-8"
        )
    return config_path


def test_collect_and_independent_result_audit_end_to_end(completed_run: Path) -> None:
    config_path = completed_run
    manifest = collect_results(config_path)
    assert manifest["jobs"] == 40
    audit = audit_neural_results(config_path)
    assert audit["overall_pass"] is True, audit["checks"]
    assert audit["summary"]["prediction_rows"] == 120


def test_prepare_resume_preserves_manifest_bytes(completed_run: Path) -> None:
    config = load_yaml(completed_run)
    manifest = (
        completed_run.parents[1]
        / config["paths"]["prepared_data"]
        / "preparation_manifest.json"
    )
    before = manifest.read_bytes()
    prepare_data(completed_run)
    prepare_data(completed_run)
    assert manifest.read_bytes() == before
    collect_results(completed_run)
    assert audit_neural_results(completed_run)["overall_pass"]


def test_prepare_refuses_changed_inputs(completed_run: Path) -> None:
    config = load_yaml(completed_run)
    root = completed_run.parents[1]
    manifest = root / config["paths"]["prepared_data"] / "preparation_manifest.json"
    before = manifest.read_bytes()
    training_path = root / config["inputs"]["training"]
    training_path.write_text(training_path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="provenance changed"):
        prepare_data(completed_run)
    assert manifest.read_bytes() == before


@pytest.mark.parametrize(
    "defect",
    [
        "missing_model",
        "log_hash",
        "cpu",
        "old_commit",
        "config_hash",
        "input_hash",
        "old_schema",
        "raw_predictions",
    ],
)
def test_invalid_completion_is_not_resumed_or_collected(
    completed_run: Path, defect: str
) -> None:
    config = load_yaml(completed_run)
    root = completed_run.parents[1]
    job = neural_jobs(config)[0]
    marker = root / config["paths"]["job_root"] / job.job_id / "COMPLETE.json"
    complete = json.loads(marker.read_text())
    if defect == "missing_model":
        del complete["outputs"]["best_model"]
    elif defect == "log_hash":
        (root / complete["outputs"]["log"]["path"]).write_text(
            "changed", encoding="utf-8"
        )
    elif defect == "cpu":
        complete["environment"]["cuda_available"] = False
    elif defect == "old_commit":
        complete["git_head"] = "another-revision"
    elif defect == "config_hash":
        complete["config_sha256"] = "wrong"
    elif defect == "input_hash":
        complete["inputs"]["train"]["sha256"] = "wrong"
    elif defect == "old_schema":
        complete["schema_version"] = 1
    else:
        path = root / complete["outputs"]["chemprop_predictions"]["path"]
        frame = pd.read_csv(path)
        frame.loc[0, job.targets[0]] += 0.5
        frame.to_csv(path, index=False)
        complete["outputs"]["chemprop_predictions"]["sha256"] = file_sha256(path)
    marker.write_text(json.dumps(complete), encoding="utf-8")
    with pytest.raises((RuntimeError, AssertionError)):
        validate_completion(completed_run, job)
    with pytest.raises((RuntimeError, AssertionError)):
        collect_results(completed_run)
    assert job.job_id in job_status(completed_run)["invalid_completions"]


@pytest.mark.parametrize(
    "artifact,column,check_id",
    [
        ("neural_family_metrics.csv", "value", "metrics.recomputed_parity"),
        ("neural_family_seed_summary.csv", "mean", "summary.recomputed_parity"),
    ],
)
def test_nan_cannot_pass_independent_audit(
    completed_run: Path, artifact: str, column: str, check_id: str
) -> None:
    manifest = collect_results(completed_run)
    root = completed_run.parents[1]
    path = root / "reports/neural" / artifact
    frame = pd.read_csv(path)
    frame.loc[0, column] = float("nan")
    frame.to_csv(path, index=False)
    manifest["outputs"][str(path.relative_to(root))] = file_sha256(path)
    (root / "reports/neural/neural_family_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    audit = audit_neural_results(completed_run)
    assert not audit["overall_pass"]
    assert (
        next(check for check in audit["checks"] if check["check_id"] == check_id)[
            "status"
        ]
        == "FAIL"
    )


def test_summary_must_match_individual_job_predictions(completed_run: Path) -> None:
    manifest = collect_results(completed_run)
    root = completed_run.parents[1]
    path = root / "reports/neural/neural_family_predictions.csv.gz"
    frame = pd.read_csv(path)
    frame.loc[0, "y_pred"] += 0.1
    frame.to_csv(path, index=False, compression="gzip")
    manifest["outputs"][str(path.relative_to(root))] = file_sha256(path)
    (root / "reports/neural/neural_family_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    audit = audit_neural_results(completed_run)
    assert not audit["overall_pass"]
    assert (
        next(
            check
            for check in audit["checks"]
            if check["check_id"] == "predictions.match_job_artifacts"
        )["status"]
        == "FAIL"
    )


def test_valid_completed_job_is_resumed_without_training(
    completed_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_yaml(completed_run)
    job = neural_jobs(config)[0]

    def no_new_training(*args, **kwargs):
        pytest.fail("Resume attempted a new training preflight")

    monkeypatch.setattr("cyp_blind.neural_baselines.preflight", no_new_training)
    assert run_job(completed_run, job, require_gpu=True)["status"] == "PASS"


def test_collection_refuses_missing_job(completed_run: Path) -> None:
    config = load_yaml(completed_run)
    job = neural_jobs(config)[-1]
    marker = (
        completed_run.parents[1]
        / config["paths"]["job_root"]
        / job.job_id
        / "COMPLETE.json"
    )
    marker.rename(marker.with_name("SAVED_COMPLETE.json"))
    with pytest.raises(RuntimeError, match="missing 1 jobs"):
        collect_results(completed_run)


@pytest.mark.parametrize("change_input_during_fit", [False, True])
def test_new_job_marker_requires_unchanged_start_provenance(
    completed_run: Path,
    monkeypatch: pytest.MonkeyPatch,
    change_input_during_fit: bool,
) -> None:
    """A fake trainer exercises marker publication without any actual GPU claim."""
    config = load_yaml(completed_run)
    root = completed_run.parents[1]
    job = neural_jobs(config)[0]
    marker = root / config["paths"]["job_root"] / job.job_id / "COMPLETE.json"
    saved = json.loads(marker.read_text())
    marker.rename(marker.with_name("SAVED_COMPLETE.json"))
    environment = {
        **saved["environment"],
        "git_tags_at_head": saved["git_tags_at_head"],
    }
    monkeypatch.setattr(
        "cyp_blind.neural_baselines.preflight",
        lambda *a, **kw: {
            "overall_pass": True,
            "environment": environment,
        },
    )

    def fake_trainer(command, **kwargs):
        model_dir = Path(command[command.index("-o") + 1]) / "model_0"
        model_dir.mkdir(parents=True)
        source = root / saved["outputs"]["chemprop_predictions"]["path"]
        (model_dir / "test_predictions.csv").write_bytes(source.read_bytes())
        (model_dir / "best.pt").write_bytes(b"SYNTHETIC - NOT TRAINED")
        kwargs["stdout"].write("Synthetic training process\n")
        if change_input_during_fit:
            path = root / saved["inputs"]["train"]["path"]
            path.write_text(path.read_text() + "\n", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("cyp_blind.neural_baselines.subprocess.run", fake_trainer)
    if change_input_during_fit:
        with pytest.raises(RuntimeError, match="hash changed"):
            run_job(completed_run, job, require_gpu=True)
        assert not marker.exists()
    else:
        run_job(completed_run, job, require_gpu=True)
        assert validate_completion(completed_run, job)["schema_version"] == 2
