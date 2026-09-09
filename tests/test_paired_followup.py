"""Actual 20-old + 50-new fit pipeline, multi-seed aggregation and preservation."""
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
import paired_prototype as v4
import paired_followup as v5
from paired_audit import summarize
from paired_followup_diagnostics import run_diagnostics
from paired_followup_self_test import self_test
from paired_followup_summary import family_samples
from paired_self_test import toy_data


def test_actual_seventy_fit_pipeline_archive_preservation_and_guards(tmp_path):
    base = json.loads((ROOT / "configs/paired_prototype.json").read_text())
    base.update(protocol_id="engineering-only-v4-fixture", hidden_dim=160, max_epochs=2,
                patience=2, batch_size=16, cpu_threads=1, seeds=[11])
    config = json.loads((ROOT / "configs/paired_followup.json").read_text())
    config.update(protocol_id="engineering-only-v5-fixture", seeds=[11, 12, 13, 14, 15],
                  reused_seed=11, intervention_seed=11, family_bootstrap_draws=40, diagnostic_gradient_sample=32)
    raw, family, table, graphs, arrays = toy_data()
    per_variant = int(arrays["mask"][:, :, 0].sum() + arrays["label_mask"].sum())
    base["expected_prediction_rows"] = per_variant * 4
    config["expected_prediction_rows"] = per_variant * 12
    v5.validate_config(config, base)
    official = tmp_path / "vendor/openadmet/custom_scoring_functions.py"
    official.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "vendor/openadmet/custom_scoring_functions.py", official)
    raw.to_csv(tmp_path / "raw.csv", index=False)
    family.to_csv(tmp_path / "folds.csv", index=False)
    old, output = tmp_path / v4.RUNTIME, tmp_path / v5.RUNTIME
    old.mkdir(parents=True)
    output.mkdir(parents=True)
    cpu = self_test(output / "cpu_self_test", ROOT / "configs/paired_prototype.json", output / "CPU_SELF_TEST.json")
    identity = {"config": base, "code": {"scripts/paired_model.py": v4.sha(ROOT / "scripts/paired_model.py")},
                "frozen_inputs": {"raw.csv": v4.sha(tmp_path / "raw.csv"), "folds.csv": v4.sha(tmp_path / "folds.csv")}}
    old_registration = {"identity": identity, "fingerprint": v4.fingerprint(identity)}
    v4.write_json(old / "RUN_REGISTRATION.json", old_registration)
    table.to_csv(old / "molecules.csv", index=False)
    v4.write_json(old / "GRAPH_CACHE.json", {"engineering_only": True, "n_graphs": len(graphs)})
    v4.write_json(old / "CPU_SELF_TEST.json", cpu["original_v4_self_test"])
    v4.write_json(old / "smoke/PASS.json", {"overall_pass": True, "registration": old_registration["fingerprint"],
                                          "results": [{**r, "device": "cpu"} for r in cpu["original_v4_self_test"]["variants"]]})
    original = [v4.run_job(tmp_path, ROOT, base, old_registration, variant, fold, seed, graphs, table,
                           raw, family, arrays, device="cpu") for variant, fold, seed in v4.jobs(base)]
    baseline = pd.concat([pd.read_csv(old / "jobs" / v4.job_id("independent_std", fold, 11) / "test_predictions.csv") for fold in range(5)])
    baseline["mode"], baseline["model"] = "masked_multitask", "chemprop_masked_multitask"
    baseline_dir = tmp_path / ".runtime/neural-prc-max-v3"
    baseline_dir.mkdir(parents=True)
    baseline.to_csv(baseline_dir / "neural_family_predictions.csv.gz", index=False)
    baseline_metrics, _ = summarize(baseline, official)
    baseline_metrics.to_csv(baseline_dir / "neural_family_metrics.csv", index=False)
    v4.finish_run(tmp_path, ROOT, base, old_registration, raw, family, arrays, table, original, engineering_only=True)
    config["expected_v4_registration"] = old_registration["fingerprint"]
    _, _, original_hashes = v5.verify_v4(old, ROOT, config["expected_v4_registration"], engineering_only=True)
    ident = {"config": config, "base_training_config": base, "code": {name: v4.sha(ROOT / name) for name in v5.CODE_FILES},
             "frozen_inputs": identity["frozen_inputs"], "v4_registration": old_registration["fingerprint"], "v4_files": original_hashes}
    registration = {"identity": ident, "fingerprint": v4.fingerprint(ident)}
    v4.write_json(output / "RUN_REGISTRATION.json", registration)
    table.to_csv(output / "molecules.csv", index=False)
    smoke = v5.smoke_test(tmp_path, config, base, registration, raw, family, arrays, graphs, table, device="cpu")
    assert len(smoke["results"]) == 4 and all(r["restart_final_weights_bitwise_equal"] for r in smoke["results"])
    run_diagnostics(old, output / "diagnostics", raw, family, arrays, graphs, device="cpu", gradient_sample=32)
    records = [v5.run_job(tmp_path, ROOT, config, base, registration, spec, graphs, table, raw, family, arrays,
                          device="cpu") for spec in v5.job_matrix(config)]
    audit = v5.finish(tmp_path, ROOT, config, base, registration, raw, family, arrays, table, records, engineering_only=True)
    assert audit["overall_pass"] and audit["engineering_only"] and not audit["scientific_results_generated"]
    assert audit["summary"]["new_jobs"] == 50 and audit["summary"]["reused_v4_jobs"] == 10
    assert audit["summary"]["metric_rows"] == 504 and audit["summary"]["diagnostic_rows"] == 144
    v5.checked_files(old, original_hashes)
    with zipfile.ZipFile(output / "CYP_paired_v5_engineering_test.zip") as archive:
        assert archive.testzip() is None
        assert len([n for n in archive.namelist() if n.startswith("jobs/") and n.endswith("/best.pt")]) == 50
        assert len([n for n in archive.namelist() if n.startswith("v4_preserved/jobs/") and n.endswith("/best.pt")]) == 20
        assert not any(n.endswith("/resume.pt") for n in archive.namelist())
        assert "v4_preserved/core_contrasts.csv" in archive.namelist()
    summary = pd.read_csv(output / "seed_summary.csv")
    assert set(summary.n_seeds) == {1, 4, 5}
    assert len(pd.read_csv(output / "contrasts_per_seed.csv")) == 24
    assert len(pd.read_csv(output / "contrast_summary.csv")) == 15
    with pytest.raises(RuntimeError, match="device or registration"):
        v5.finish(tmp_path, ROOT, config, base, registration, raw, family, arrays, table, records)
    # Bootstrap implementation is checked by literal family resampling followed
    # by row-level sklearn/official arithmetic, rather than by matrix formulas.
    from sklearn.metrics import matthews_corrcoef
    pred = pd.read_csv(output / "predictions.csv.gz")
    samples, _ = family_samples(pred, family, config)
    families = family[["family_id", "outer_fold"]].drop_duplicates().sort_values(["outer_fold", "family_id"]).reset_index(drop=True)
    rng = np.random.default_rng(config["family_bootstrap_seed"])
    draws = []
    for _, group in families.groupby("outer_fold", sort=True):
        draws.append(rng.choice(group.index.to_numpy(), size=(config["family_bootstrap_draws"], len(group)), replace=True))
    import importlib.util
    spec = importlib.util.spec_from_file_location("row_bootstrap_official", official)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    for iteration in range(6):
        chosen = [families.family_id.iloc[i] for group in draws for i in group[iteration]]
        group = pred[pred.model.eq("v5_paired_no_std") & pred.seed.eq(12)]
        resampled = pd.concat([group[group.family_id.eq(name)] for name in chosen])
        direct = []
        for endpoint, block in resampled.groupby("endpoint"):
            if block.task.iloc[0] == "regression":
                direct.append(scorer.rae_soft_threshold_absolute_error(block.y_true.to_numpy(), block.y_pred.to_numpy(),
                              y_true_upper=block.y_true_upper.to_numpy(), y_true_lower=block.y_true_lower.to_numpy()))
            else:
                np.testing.assert_allclose(samples[("v5_paired_no_std", 12, endpoint)][iteration],
                                           matthews_corrcoef(block.y_true, block.y_pred), atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(samples[("v5_paired_no_std", 12, "MA")][iteration], np.mean(direct), atol=1e-12, rtol=1e-12)
    spec = next(x for x in v5.job_matrix(config) if x["origin"] == "new")
    assert v5.run_job(tmp_path, ROOT, config, base, registration, spec, graphs, table, raw, family, arrays, device="cpu") == records[10]
    path = v5.source_directory(tmp_path, spec) / "test_predictions.csv"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(RuntimeError, match="artifact changed"):
        v5.run_job(tmp_path, ROOT, config, base, registration, spec, graphs, table, raw, family, arrays, device="cpu")
    old_weight = old / "jobs" / v4.job_id("paired_no_std", 0, 11) / "best.pt"
    old_weight.write_bytes(old_weight.read_bytes() + b"altered")
    with pytest.raises(RuntimeError, match="artifact changed"):
        v5.verify_v4(old, ROOT, config["expected_v4_registration"], engineering_only=True)
