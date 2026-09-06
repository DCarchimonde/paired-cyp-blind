import gzip
from pathlib import Path

import pandas as pd
import pytest

from cyp_blind.io import load_yaml
from cyp_blind.neural_baselines import (
    build_chemprop_command,
    data_specifications,
    neural_jobs,
    split_membership,
    run_job,
    _write_deterministic_csv_gzip,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_yaml(ROOT / "configs/neural_baselines.yaml")


def test_neural_job_matrix_is_complete_and_unique() -> None:
    specs = data_specifications(CONFIG)
    jobs = neural_jobs(CONFIG)
    assert len(specs) == 40
    assert len(jobs) == 200
    assert len({job.job_id for job in jobs}) == 200
    assert {job.seed for job in jobs} == set(CONFIG["seeds"])
    assert {job.fold for job in jobs} == set(range(5))


def test_split_membership_keeps_nonpanel_core_in_training() -> None:
    names = pd.Series(["core", "test", "val", "other"])
    folds = {"test": 2, "val": 3, "other": 4}
    membership = split_membership(names, folds, test_fold=2, validation_fold=3)
    assert membership.tolist() == ["train", "test", "val", "train"]


def test_chemprop_command_has_explicit_three_way_split_and_seed() -> None:
    job = neural_jobs(CONFIG)[0]
    command = build_chemprop_command(
        CONFIG,
        job,
        train_path=Path("train.csv"),
        val_path=Path("val.csv"),
        test_path=Path("test.csv"),
        output_dir=Path("output"),
        accelerator="gpu",
        devices="1",
    )
    assert command[command.index("-i") + 1 : command.index("-i") + 4] == [
        "train.csv",
        "val.csv",
        "test.csv",
    ]
    assert command[command.index("--pytorch-seed") + 1] == str(job.seed)
    assert command[command.index("--accelerator") + 1] == "gpu"
    assert "--remove-checkpoints" in command


def test_deterministic_gzip_has_identical_bytes(tmp_path: Path) -> None:
    frame = pd.DataFrame({"name": ["a", "b"], "value": [1.25, 2.5]})
    first = tmp_path / "first.csv.gz"
    second = tmp_path / "second.csv.gz"
    _write_deterministic_csv_gzip(frame, first)
    _write_deterministic_csv_gzip(frame, second)
    assert first.read_bytes() == second.read_bytes()
    with gzip.open(first, "rt", encoding="utf-8") as handle:
        restored = pd.read_csv(handle)
    pd.testing.assert_frame_equal(restored, frame)


def test_production_run_job_refuses_cpu_before_writing() -> None:
    with pytest.raises(RuntimeError, match="requires explicit --require-gpu"):
        run_job(
            ROOT / "configs/neural_baselines.yaml",
            neural_jobs(CONFIG)[0],
            require_gpu=False,
        )


def test_tdi_modes_do_not_use_nan_unsafe_class_balance() -> None:
    assert CONFIG["optimization"]["classification"]["class_balance"] is False
    for job in neural_jobs(CONFIG):
        if job.task_group != "tdi":
            continue
        command = build_chemprop_command(
            CONFIG,
            job,
            train_path=Path("train.csv"),
            val_path=Path("val.csv"),
            test_path=Path("test.csv"),
            output_dir=Path("output"),
            accelerator="gpu",
            devices="1",
        )
        assert "--class-balance" not in command
