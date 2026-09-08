"""Real Torch integration tests; optional locally, mandatory via host self-test."""
from pathlib import Path
import json
import shutil
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("rdkit")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from paired_self_test import self_test, toy_data
from paired_prototype import (RUNTIME, finish_run, fingerprint, job_id, jobs, run_job, sha, write_json)
from paired_audit import summarize


def test_actual_training_gradients_masking_serialization_and_restart(tmp_path):
    result = self_test(tmp_path, ROOT / "configs/paired_prototype.json")
    assert result["overall_pass"]
    assert len(result["variants"]) == 4


def test_complete_actual_twenty_job_cpu_pipeline_and_review_archive(tmp_path):
    config = json.loads((ROOT / "configs/paired_prototype.json").read_text())
    config.update(protocol_id="engineering-synthetic-fixture-only", hidden_dim=160,
                  max_epochs=2, patience=2, batch_size=16, cpu_threads=1, seeds=[11])
    raw, family, table, graphs, arrays = toy_data()
    config["expected_prediction_rows"] = int((arrays["mask"][:, :, 0].sum() + arrays["label_mask"].sum()) * 4)
    output = tmp_path / RUNTIME
    output.mkdir(parents=True)
    raw.to_csv(tmp_path / "raw.csv", index=False)
    family.to_csv(tmp_path / "folds.csv", index=False)
    official = tmp_path / "vendor/openadmet/custom_scoring_functions.py"
    official.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "vendor/openadmet/custom_scoring_functions.py", official)
    identity = {"config": config, "code": {"scripts/paired_model.py": sha(ROOT / "scripts/paired_model.py")},
                "frozen_inputs": {"raw.csv": sha(tmp_path / "raw.csv"), "folds.csv": sha(tmp_path / "folds.csv")}}
    registration = {"identity": identity, "fingerprint": fingerprint(identity)}
    write_json(output / "RUN_REGISTRATION.json", registration)
    table.to_csv(output / "molecules.csv", index=False)
    write_json(output / "GRAPH_CACHE.json", {"synthetic_fixture": True, "n_graphs": len(graphs)})
    cpu = self_test(tmp_path / "self_test", ROOT / "configs/paired_prototype.json", output / "CPU_SELF_TEST.json")
    write_json(output / "smoke/PASS.json", {"overall_pass": True, "registration": registration["fingerprint"],
                                          "results": [{**r, "device": "cpu"} for r in cpu["variants"]]})
    records = []
    for variant, fold, seed in jobs(config):
        records.append(run_job(tmp_path, ROOT, config, registration, variant, fold, seed,
                               graphs, table, raw, family, arrays, device="cpu"))
    # The old-baseline schema is represented by actual fitted toy predictions.
    # It is explicitly a synthetic fixture, never copied into real v2/v3 results.
    baseline = pd.concat([pd.read_csv(output / "jobs" / job_id("independent_std", f, 11) / "test_predictions.csv") for f in range(5)])
    baseline["mode"] = "masked_multitask"
    baseline["model"] = "chemprop_masked_multitask"
    baseline_dir = tmp_path / ".runtime/neural-prc-max-v3"
    baseline_dir.mkdir(parents=True)
    baseline.to_csv(baseline_dir / "neural_family_predictions.csv.gz", index=False)
    baseline_metrics, _ = summarize(baseline, official)
    baseline_metrics.to_csv(baseline_dir / "neural_family_metrics.csv", index=False)
    finish_run(tmp_path, ROOT, config, registration, raw, family, arrays, table, records, engineering_only=True)
    audit = json.loads((output / "RESULT_AUDIT.json").read_text())
    assert audit["overall_pass"] and audit["engineering_only"]
    assert not audit["scientific_results_generated"] and not audit["gpu_training_qualified"]
    assert audit["summary"]["completed_jobs"] == 20
    assert audit["summary"]["metric_rows"] == 168
    with zipfile.ZipFile(output / "CYP_paired_v4_engineering_test.zip") as archive:
        assert archive.testzip() is None
        assert len([n for n in archive.namelist() if n.endswith("/best.pt")]) == 20
    variant, fold, seed = jobs(config)[0]
    reused = run_job(tmp_path, ROOT, config, registration, variant, fold, seed, graphs, table, raw, family, arrays, device="cpu")
    assert reused == records[0]
    # Production finalization must reject actual CPU jobs, even if every numeric
    # check passed. This protects the CUDA gate from an accidental fallback.
    with pytest.raises(RuntimeError, match="device or registration"):
        finish_run(tmp_path, ROOT, config, registration, raw, family, arrays, table, records)
    path = output / "jobs" / job_id(variant, fold, seed) / "test_predictions.csv"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(RuntimeError, match="artifact changed"):
        run_job(tmp_path, ROOT, config, registration, variant, fold, seed, graphs, table, raw, family, arrays, device="cpu")
