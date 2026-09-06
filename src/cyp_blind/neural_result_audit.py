from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

from .io import file_sha256, load_yaml
from .neural_baselines import neural_jobs, validate_completion


def _official_metric_row(
    *,
    model: str,
    scope: str,
    fold: int | None,
    task: str,
    endpoint: str,
    metric: str,
    value: float,
    n: int,
    positives_true: int | None = None,
    positives_predicted: int | None = None,
) -> dict:
    return {
        "model": model,
        "scope": scope,
        "fold": fold,
        "task": task,
        "endpoint": endpoint,
        "metric": metric,
        "value": float(value),
        "n": int(n),
        "positives_true": positives_true,
        "positives_predicted": positives_predicted,
    }


def _load_official_soft_threshold_rae(root: Path) -> Callable[..., float]:
    path = root / "vendor/openadmet/custom_scoring_functions.py"
    spec = importlib.util.spec_from_file_location("openadmet_official_metrics", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load official metric implementation from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.rae_soft_threshold_absolute_error


def _independent_official_metrics(
    predictions: pd.DataFrame,
    official_soft_threshold_rae: Callable[..., float],
) -> pd.DataFrame:
    """Recompute metrics without calling the production summarizer."""
    rows: list[dict] = []
    scopes: list[tuple[str, int | None, pd.DataFrame]] = [("pooled", None, predictions)]
    scopes.extend(
        ("fold", int(fold), group)
        for fold, group in predictions.groupby("fold", sort=True)
    )
    for scope, fold, scoped in scopes:
        for (model, task, endpoint), group in scoped.groupby(
            ["model", "task", "endpoint"], sort=True
        ):
            if task == "regression":
                value = official_soft_threshold_rae(
                    group["y_true"].to_numpy(float),
                    group["y_pred"].to_numpy(float),
                    y_true_upper=group["y_true_upper"].to_numpy(float),
                    y_true_lower=group["y_true_lower"].to_numpy(float),
                )
                rows.append(
                    _official_metric_row(
                        model=str(model),
                        scope=scope,
                        fold=fold,
                        task=str(task),
                        endpoint=str(endpoint),
                        metric="ST-RAE",
                        value=float(value),
                        n=len(group),
                    )
                )
            else:
                y_true = group["y_true"].to_numpy(int)
                y_pred = group["y_pred"].to_numpy(int)
                rows.append(
                    _official_metric_row(
                        model=str(model),
                        scope=scope,
                        fold=fold,
                        task=str(task),
                        endpoint=str(endpoint),
                        metric="MCC",
                        value=float(matthews_corrcoef(y_true, y_pred)),
                        n=len(group),
                        positives_true=int(y_true.sum()),
                        positives_predicted=int(y_pred.sum()),
                    )
                )

    endpoint_metrics = pd.DataFrame(rows)
    direct = endpoint_metrics[
        endpoint_metrics["task"].eq("regression")
        & endpoint_metrics["metric"].eq("ST-RAE")
    ]
    macro_rows: list[dict] = []
    for (model, scope, fold), group in direct.groupby(
        ["model", "scope", "fold"], dropna=False
    ):
        macro_rows.append(
            _official_metric_row(
                model=str(model),
                scope=str(scope),
                fold=None if pd.isna(fold) else int(fold),
                task="regression",
                endpoint="MA",
                metric="MA-ST-RAE",
                value=float(group["value"].mean()),
                n=int(group["n"].sum()),
            )
        )
    return pd.DataFrame(rows + macro_rows)


def _render_report(payload: dict) -> str:
    status = "PASS" if payload["overall_pass"] else "FAIL"
    lines = [
        "# Neural Family-Baseline Result Audit",
        "",
        f"**Overall status: {status}**  ",
        f"Generated: {payload['generated_utc']}",
        "",
        f"Audited {payload['summary']['prediction_rows']:,} predictions from "
        f"{payload['summary']['completed_jobs']} completed Chemprop jobs.",
        "",
        "## Checks",
        "",
        "| Status | Critical | Check | Detail |",
        "|---|---|---|---|",
    ]
    for check in payload["checks"]:
        critical = "yes" if check["critical"] else "no"
        detail = str(check["detail"]).replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {check['status']} | {critical} | `{check['check_id']}` | {detail} |"
        )
    lines.extend(
        [
            "",
            "## Decision boundary",
            "",
            "This PASS would complete only the Chemprop baseline evidence. It does not "
            "authorize the assay-structured primary model, the 14-day GO decision, a blind "
            "challenge claim, or a flagship-paper claim.",
            "",
        ]
    )
    return "\n".join(lines)


def audit_neural_results(config_path: Path) -> dict:
    config_path = config_path.resolve()
    root = config_path.parents[1]
    config = load_yaml(config_path)
    official_soft_threshold_rae = _load_official_soft_threshold_rae(root)
    summary_root = root / config["paths"]["summary_root"]
    predictions_path = summary_root / "neural_family_predictions.csv.gz"
    metrics_path = summary_root / "neural_family_metrics.csv"
    seed_summary_path = summary_root / "neural_family_seed_summary.csv"
    manifest_path = summary_root / "neural_family_manifest.json"
    predictions = pd.read_csv(predictions_path)
    metrics = pd.read_csv(metrics_path)
    seed_summary = pd.read_csv(seed_summary_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    training = pd.read_csv(root / config["inputs"]["training"])
    family = pd.read_csv(root / config["inputs"]["family_split"])
    raw = training.set_index("Molecule_Name")
    family_indexed = family.set_index("Molecule_Name")
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

    output_mismatches = [
        relative
        for relative, expected_hash in manifest["outputs"].items()
        if file_sha256(root / relative) != expected_hash
    ]
    input_mismatches = [
        relative
        for relative, expected_hash in manifest["inputs"].items()
        if file_sha256(root / relative) != expected_hash
    ]
    add(
        "manifest.complete",
        manifest["complete"] is True
        and manifest["jobs"] == int(config["expected"]["jobs"])
        and manifest["prediction_rows"] == len(predictions)
        and manifest["expected_prediction_rows"]
        == int(config["expected"]["predictions"])
        and manifest["seeds"] == list(config["seeds"]),
        f"complete={manifest['complete']}, jobs={manifest['jobs']}",
    )
    add(
        "manifest.input_hashes",
        not input_mismatches,
        f"mismatches={input_mismatches}",
    )
    add(
        "manifest.output_hashes",
        not output_mismatches,
        f"mismatches={output_mismatches}",
    )

    jobs = neural_jobs(config)
    required_input_paths = {
        str(path.relative_to(root))
        for path in (
            config_path,
            root / config["inputs"]["training"],
            root / config["inputs"]["family_split"],
            root / config["paths"]["prepared_data"] / "preparation_manifest.json",
            root / "vendor/openadmet/custom_scoring_functions.py",
        )
    }
    add(
        "manifest.artifact_sets_exact",
        set(manifest["inputs"]) == required_input_paths
        and set(manifest["outputs"])
        == {
            str(path.relative_to(root))
            for path in (predictions_path, metrics_path, seed_summary_path)
        }
        and set(manifest["completion_manifest_hashes"]) == {job.job_id for job in jobs},
        "Required inputs, summaries, and all completion markers must be listed",
    )
    job_root = root / config["paths"]["job_root"]
    completion_missing: list[str] = []
    completion_identity: list[str] = []
    completion_hashes: list[str] = []
    completion_provenance: list[str] = []
    completion_gpu: list[str] = []
    completion_validation: list[str] = []
    job_prediction_frames: list[pd.DataFrame] = []
    git_heads: set[str] = set()
    preparation_manifest_path = (
        root / config["paths"]["prepared_data"] / "preparation_manifest.json"
    )
    required_tag = str(config["provenance"]["required_protocol_tag"])
    for job in jobs:
        complete_path = job_root / job.job_id / "COMPLETE.json"
        if not complete_path.is_file():
            completion_missing.append(job.job_id)
            continue
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        try:
            validate_completion(config_path, job)
            job_prediction_frames.append(
                pd.read_csv(root / complete["outputs"]["predictions"]["path"])
            )
        except (OSError, ValueError, RuntimeError, KeyError, AssertionError) as error:
            completion_validation.append(f"{job.job_id}: {error}")
        expected_job = asdict(job)
        observed_job = complete.get("job", {})
        if (
            complete.get("status") != "PASS"
            or complete.get("job_id") != job.job_id
            or observed_job.get("mode") != expected_job["mode"]
            or observed_job.get("task_group") != expected_job["task_group"]
            or observed_job.get("endpoint") != expected_job["endpoint"]
            or int(observed_job.get("fold", -1)) != expected_job["fold"]
            or int(observed_job.get("seed", -1)) != expected_job["seed"]
            or list(observed_job.get("targets", [])) != list(expected_job["targets"])
        ):
            completion_identity.append(job.job_id)
        observed_complete_hash = file_sha256(complete_path)
        if (
            manifest["completion_manifest_hashes"].get(job.job_id)
            != observed_complete_hash
        ):
            completion_hashes.append(f"{job.job_id}/COMPLETE.json")
        for output_name, output in complete.get("outputs", {}).items():
            path = root / output["path"]
            if not path.is_file() or file_sha256(path) != output["sha256"]:
                completion_hashes.append(f"{job.job_id}/{output_name}")
        git_head = str(complete.get("git_head"))
        git_heads.add(git_head)
        if (
            complete.get("config_sha256") != file_sha256(config_path)
            or complete.get("preparation_manifest_sha256")
            != file_sha256(preparation_manifest_path)
            or required_tag not in complete.get("git_tags_at_head", [])
        ):
            completion_provenance.append(job.job_id)
        environment = complete.get("environment", {})
        device_name = str(environment.get("device_name"))
        if (
            environment.get("cuda_available") is not True
            or str(config["production_gpu"]["required_name_contains"]).lower()
            not in device_name.lower()
            or float(environment.get("vram_gib", 0))
            < float(config["production_gpu"]["minimum_vram_gib"])
            or str(environment.get("compute_capability"))
            != str(config["production_gpu"]["required_compute_capability"])
        ):
            completion_gpu.append(job.job_id)
    add(
        "jobs.all_complete",
        not completion_missing,
        f"missing={len(completion_missing)}",
    )
    add(
        "jobs.identity",
        not completion_identity,
        f"mismatches={len(completion_identity)}",
    )
    add(
        "jobs.artifact_hashes",
        not completion_hashes,
        f"mismatches={len(completion_hashes)}",
    )
    add(
        "jobs.frozen_provenance",
        not completion_provenance
        and len(git_heads) == 1
        and manifest["git_head"] in git_heads,
        f"mismatches={len(completion_provenance)}, git_heads={sorted(git_heads)}",
    )
    add(
        "jobs.required_gpu",
        not completion_gpu,
        f"mismatches={len(completion_gpu)}",
    )
    add(
        "jobs.full_completion_validation",
        not completion_validation,
        f"mismatches={len(completion_validation)}; examples={completion_validation[:2]}",
    )
    concatenation_matches = False
    if len(job_prediction_frames) == len(jobs):
        try:
            pd.testing.assert_frame_equal(
                predictions,
                pd.concat(job_prediction_frames, ignore_index=True),
                check_dtype=False,
                check_exact=False,
                rtol=0,
                atol=1e-12,
            )
            concatenation_matches = True
        except AssertionError:
            pass
    add(
        "predictions.match_job_artifacts",
        concatenation_matches,
        "Summary predictions must reproduce the ordered completed-job prediction tables",
    )

    endpoints = [
        *config["models"]["single_task"]["direct"],
        *config["models"]["single_task"]["tdi"],
    ]
    expected_keys = {
        (mode, int(seed), str(name), endpoint)
        for mode in ("single_task", "masked_multitask")
        for seed in config["seeds"]
        for name in family["Molecule_Name"]
        for endpoint in endpoints
        if pd.notna(raw.at[name, endpoint])
    }
    key_columns = ["mode", "seed", "Molecule_Name", "endpoint"]
    actual_keys = set(predictions[key_columns].itertuples(index=False, name=None))
    duplicate_rows = int(predictions.duplicated(key_columns).sum())
    add(
        "predictions.coverage_exact",
        actual_keys == expected_keys
        and duplicate_rows == 0
        and len(expected_keys) == int(config["expected"]["predictions"]),
        f"rows={len(predictions)}, expected={len(expected_keys)}, duplicates={duplicate_rows}, "
        f"missing={len(expected_keys - actual_keys)}, extra={len(actual_keys - expected_keys)}",
    )
    expected_model = predictions["mode"].map(
        {
            "single_task": "chemprop_single_task",
            "masked_multitask": "chemprop_masked_multitask",
        }
    )
    add(
        "predictions.model_mapping",
        expected_model.notna().all() and predictions["model"].eq(expected_model).all(),
        f"mismatches={int((~predictions['model'].eq(expected_model)).sum())}",
    )
    expected_tasks = predictions["endpoint"].map(
        {
            **{
                endpoint: "regression"
                for endpoint in config["models"]["single_task"]["direct"]
            },
            **{
                endpoint: "classification"
                for endpoint in config["models"]["single_task"]["tdi"]
            },
        }
    )
    add(
        "predictions.task_mapping",
        bool(predictions["task"].eq(expected_tasks).all()),
        "Direct endpoints are regression; TDI endpoints are classification",
    )
    expected_folds = predictions["Molecule_Name"].map(family_indexed["outer_fold"])
    expected_families = predictions["Molecule_Name"].map(family_indexed["family_id"])
    fold_mismatches = int(
        (predictions["fold"].astype(int) != expected_folds.astype(int)).sum()
    )
    family_mismatches = int((predictions["family_id"] != expected_families).sum())
    add(
        "predictions.fold_family_mapping",
        fold_mismatches == 0 and family_mismatches == 0,
        f"fold_mismatches={fold_mismatches}, family_mismatches={family_mismatches}",
    )

    truth_mismatches = 0
    bound_mismatches = 0
    for row in predictions.itertuples(index=False):
        expected_truth = raw.at[row.Molecule_Name, row.endpoint]
        if row.task == "classification":
            expected_truth = float(
                pd.Series([expected_truth]).astype("boolean").iloc[0]
            )
        else:
            expected_truth = float(expected_truth)
        if not np.isclose(float(row.y_true), expected_truth, rtol=0, atol=1e-12):
            truth_mismatches += 1
        if row.task == "regression":
            expected_lower = float(
                raw.at[row.Molecule_Name, f"{row.endpoint}_conf_low"]
            )
            expected_upper = float(
                raw.at[row.Molecule_Name, f"{row.endpoint}_conf_high"]
            )
            if not (
                np.isclose(float(row.y_true_lower), expected_lower, rtol=0, atol=1e-12)
                and np.isclose(
                    float(row.y_true_upper), expected_upper, rtol=0, atol=1e-12
                )
            ):
                bound_mismatches += 1
    add(
        "predictions.truth_bounds_raw_parity",
        truth_mismatches == 0 and bound_mismatches == 0,
        f"truth_mismatches={truth_mismatches}, bound_mismatches={bound_mismatches}",
    )
    finite_core = np.isfinite(predictions[["y_true", "y_pred"]].to_numpy(float)).all()
    classification = predictions[predictions["task"].eq("classification")]
    regression = predictions[predictions["task"].eq("regression")]
    threshold = float(
        config["optimization"]["classification"]["fixed_decision_threshold"]
    )
    classification_ok = (
        np.isfinite(classification["probability"]).all()
        and classification["probability"].between(0, 1).all()
        and classification["y_pred"]
        .eq(classification["probability"].ge(threshold).astype(float))
        .all()
    )
    regression_ok = (
        regression["probability"].isna().all()
        and np.isfinite(
            regression[["y_true_lower", "y_true_upper"]].to_numpy(float)
        ).all()
        and regression["y_true_lower"].le(regression["y_true_upper"]).all()
    )
    add(
        "predictions.numeric_domains",
        bool(finite_core and classification_ok and regression_ok),
        f"finite_core={finite_core}, classification_ok={classification_ok}, "
        f"regression_ok={regression_ok}",
    )

    expected_metric_frames: list[pd.DataFrame] = []
    for seed, group in predictions.groupby("seed", sort=True):
        frame = _independent_official_metrics(group, official_soft_threshold_rae)
        frame.insert(0, "seed", int(seed))
        expected_metric_frames.append(frame)
    expected_metrics = pd.concat(expected_metric_frames, ignore_index=True)
    metric_keys = ["seed", "model", "scope", "fold", "task", "endpoint", "metric"]
    observed_metrics = metrics.copy()
    expected_metrics_compare = expected_metrics.copy()
    for frame in (observed_metrics, expected_metrics_compare):
        frame["fold"] = frame["fold"].fillna(-1).astype(int)
    merged_metrics = expected_metrics_compare.merge(
        observed_metrics,
        on=metric_keys,
        how="outer",
        suffixes=("_expected", "_observed"),
        indicator=True,
    )
    metric_value_diff = (
        merged_metrics["value_expected"] - merged_metrics["value_observed"]
    ).abs()
    positives_true_match = (
        merged_metrics["positives_true_expected"]
        .fillna(-1)
        .eq(merged_metrics["positives_true_observed"].fillna(-1))
    )
    positives_predicted_match = (
        merged_metrics["positives_predicted_expected"]
        .fillna(-1)
        .eq(merged_metrics["positives_predicted_observed"].fillna(-1))
    )
    metric_mismatches = int(
        (
            merged_metrics["_merge"].ne("both")
            | ~metric_value_diff.le(1e-12)
            | ~np.isfinite(merged_metrics["value_observed"])
            | merged_metrics["n_expected"].ne(merged_metrics["n_observed"])
            | ~positives_true_match
            | ~positives_predicted_match
        ).sum()
    )
    add(
        "metrics.recomputed_parity",
        metric_mismatches == 0
        and len(metrics) == len(expected_metrics)
        and not observed_metrics.duplicated(metric_keys).any(),
        f"mismatches={metric_mismatches}, rows={len(metrics)}",
    )

    primary = expected_metrics[
        expected_metrics["scope"].eq("pooled")
        & expected_metrics["metric"].isin(["MA-ST-RAE", "MCC"])
        & (
            expected_metrics["endpoint"].eq("MA")
            | expected_metrics["endpoint"].isin(
                config["models"]["masked_multitask"]["tdi"]
            )
        )
    ]
    expected_summary = (
        primary.groupby(["model", "endpoint", "metric"])["value"]
        .agg(["mean", "std", "median", "min", "max", "count"])
        .reset_index()
    )
    summary_keys = ["model", "endpoint", "metric"]
    merged_summary = expected_summary.merge(
        seed_summary,
        on=summary_keys,
        how="outer",
        suffixes=("_expected", "_observed"),
        indicator=True,
    )
    summary_mismatch = merged_summary["_merge"].ne("both")
    for column in ("mean", "std", "median", "min", "max"):
        expected_values = merged_summary[f"{column}_expected"].to_numpy(float)
        observed_values = merged_summary[f"{column}_observed"].to_numpy(float)
        # std is undefined only for the one-seed synthetic fixture, not production.
        summary_mismatch |= ~np.isclose(
            expected_values,
            observed_values,
            rtol=0,
            atol=1e-12,
            equal_nan=True,
        )
        summary_mismatch |= np.isinf(observed_values)
    summary_mismatch |= merged_summary["count_expected"].ne(
        merged_summary["count_observed"]
    )
    add(
        "summary.recomputed_parity",
        not summary_mismatch.any()
        and len(seed_summary) == len(expected_summary)
        and not seed_summary.duplicated(summary_keys).any(),
        f"mismatches={int(summary_mismatch.sum())}, rows={len(seed_summary)}",
    )

    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "overall_pass": not any(check["status"] == "FAIL" for check in checks),
        "checks": checks,
        "summary": {
            "completed_jobs": len(jobs) - len(completion_missing),
            "prediction_rows": len(predictions),
            "expected_prediction_rows": int(config["expected"]["predictions"]),
            "metric_rows": len(metrics),
            "seed_summary_rows": len(seed_summary),
        },
        "flagship_claim_authorized": False,
    }
    json_path = summary_root / "neural_result_audit.json"
    report_path = summary_root / "NEURAL_RESULT_AUDIT.md"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    report_path.write_text(_render_report(payload), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/neural_baselines.yaml")
    )
    args = parser.parse_args()
    payload = audit_neural_results(args.config)
    print(json.dumps(payload, indent=2))
    return 0 if payload["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
