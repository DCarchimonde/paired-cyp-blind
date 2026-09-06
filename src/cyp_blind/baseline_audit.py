from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

from vendor.openadmet.custom_scoring_functions import (
    rae_soft_threshold_absolute_error as official_soft_threshold_rae,
)

from .audit import Auditor
from .constants import DIRECT_ISOFORMS, TDI_ISOFORMS, direct_value_column, tdi_label_column
from .io import file_sha256, load_yaml


def _render_markdown(payload: dict) -> str:
    pooled = payload["pooled_primary"]
    lines = [
        "# Classical Family-Baseline Audit",
        "",
        f"**Overall status: {'PASS' if payload['overall_pass'] else 'FAIL'}**  ",
        f"Generated: {payload['generated_utc']}",
        "",
        "## Audited pooled results",
        "",
        "| Endpoint | Best audited model | Score | Direction |",
        "|---|---|---:|---|",
        f"| Direct MA-ST-RAE | {pooled['direct']['model']} | {pooled['direct']['score']:.4f} | lower is better |",
        f"| CYP3A4 MCC | {pooled['CYP3A4']['model']} | {pooled['CYP3A4']['score']:.4f} | higher is better |",
        f"| CYP2D6 MCC | {pooled['CYP2D6']['model']} | {pooled['CYP2D6']['score']:.4f} | higher is better |",
        "",
        "## Fold-level TDI warning",
        "",
        "The pooled MCC values are valid, but small fold-level positive counts make TDI estimates unstable, especially CYP2D6. This is disclosed now and must be handled with family-level uncertainty later.",
        "",
        "| Isoform | Fold positive counts | Best-model fold MCC values |",
        "|---|---|---|",
    ]
    for isoform in TDI_ISOFORMS:
        warning = payload["tdi_fold_diagnostics"][isoform]
        lines.append(
            f"| {isoform} | {warning['positive_counts']} | {warning['best_model_fold_mcc']} |"
        )

    lines.extend(
        [
            "",
            "## Checks",
            "",
            "| Status | Critical | Check | Detail |",
            "|---|---|---|---|",
        ]
    )
    for check in payload["checks"]:
        detail = check["detail"].replace("|", "\\|")
        lines.append(
            f"| {check['status']} | {'yes' if check['critical'] else 'no'} | "
            f"`{check['check_id']}` | {detail} |"
        )
    lines.extend(
        [
            "",
            "## Decision boundary",
            "",
            "This PASS freezes the classical baseline evidence only. It does not complete Day 3–5, authorize the primary latent model, or establish flagship status. Neural baselines and random/scaffold split-gap diagnostics remain mandatory.",
            "",
        ]
    )
    return "\n".join(lines)


def run_baseline_audit(config_path: Path) -> dict:
    config_path = config_path.resolve()
    root = config_path.parents[1]
    config = load_yaml(config_path)
    output_dir = root / "reports/baselines"
    predictions_path = output_dir / "classical_family_predictions.csv"
    metrics_path = output_dir / "classical_family_metrics.csv"
    runtimes_path = output_dir / "classical_family_runtimes.csv"
    report_path = output_dir / "CLASSICAL_FAMILY_BASELINES.md"
    manifest_path = output_dir / "classical_family_manifest.json"
    required_paths = [
        predictions_path,
        metrics_path,
        runtimes_path,
        report_path,
        manifest_path,
    ]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing classical baseline artifacts: {missing}")

    predictions = pd.read_csv(predictions_path)
    metrics = pd.read_csv(metrics_path)
    training = pd.read_csv(root / "data/raw/cyp-challenge-TRAIN_TDI.csv")
    split = pd.read_csv(root / config["evaluation"]["split"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    auditor = Auditor()

    auditor.add(
        "manifest.complete_classical_set",
        manifest["complete_classical_set"] is True,
        f"complete_classical_set={manifest['complete_classical_set']}",
    )
    expected_models = list(config["stage_completion"]["classical_models_required"])
    auditor.add(
        "manifest.models_exact",
        manifest["models_run"] == expected_models,
        f"observed={manifest['models_run']}, expected={expected_models}",
    )
    auditor.add(
        "manifest.clean_start",
        manifest["git_status_at_start"] == "" and manifest["dirty_override"] is False,
        (
            f"git_status_at_start={manifest['git_status_at_start']!r}, "
            f"dirty_override={manifest['dirty_override']}"
        ),
    )

    bad_hashes: list[str] = []
    for relative_path, expected_hash in manifest["inputs"].items():
        if file_sha256(root / relative_path) != expected_hash:
            bad_hashes.append(relative_path)
    for relative_path, expected_hash in manifest["outputs"].items():
        if file_sha256(root / relative_path) != expected_hash:
            bad_hashes.append(relative_path)
    auditor.add(
        "manifest.hashes",
        not bad_hashes,
        f"hash mismatches={bad_hashes}",
    )

    prediction_key_columns = ["model", "Molecule_Name", "endpoint"]
    duplicate_predictions = int(predictions.duplicated(prediction_key_columns).sum())
    auditor.add(
        "predictions.unique_keys",
        duplicate_predictions == 0,
        f"duplicate model/molecule/endpoint rows={duplicate_predictions}",
    )

    name_to_training_index = pd.Series(
        training.index, index=training["Molecule_Name"]
    ).to_dict()
    split_index = split.set_index("Molecule_Name")
    expected_keys = {
        (model, molecule_name, endpoint)
        for model in expected_models
        for molecule_name in split["Molecule_Name"]
        for endpoint in [
            *[direct_value_column(isoform) for isoform in DIRECT_ISOFORMS],
            *[tdi_label_column(isoform) for isoform in TDI_ISOFORMS],
        ]
        if pd.notna(training.at[name_to_training_index[molecule_name], endpoint])
    }
    actual_keys = set(predictions[prediction_key_columns].itertuples(index=False, name=None))
    auditor.add(
        "predictions.coverage_exact",
        actual_keys == expected_keys,
        (
            f"observed={len(actual_keys)}, expected={len(expected_keys)}, "
            f"missing={len(expected_keys - actual_keys)}, extra={len(actual_keys - expected_keys)}"
        ),
    )

    fold_mismatches = sum(
        int(row.fold) != int(split_index.at[row.Molecule_Name, "outer_fold"])
        for row in predictions.itertuples()
    )
    family_mismatches = sum(
        row.family_id != split_index.at[row.Molecule_Name, "family_id"]
        for row in predictions.itertuples()
    )
    auditor.add(
        "predictions.fold_mapping",
        fold_mismatches == 0,
        f"fold mismatches={fold_mismatches}",
    )
    auditor.add(
        "predictions.family_mapping",
        family_mismatches == 0,
        f"family mismatches={family_mismatches}",
    )

    finite_truth_prediction = np.isfinite(
        predictions[["y_true", "y_pred"]].to_numpy(dtype=float)
    ).all()
    auditor.add(
        "predictions.finite_truth_and_prediction",
        bool(finite_truth_prediction),
        f"nonfinite cells={int((~np.isfinite(predictions[['y_true', 'y_pred']].to_numpy(dtype=float))).sum())}",
    )
    classification = predictions[predictions["task"].eq("classification")]
    regression = predictions[predictions["task"].eq("regression")]
    classification_valid = (
        set(classification["y_pred"].unique()).issubset({0.0, 1.0})
        and np.isfinite(classification["probability"]).all()
        and classification["probability"].between(0, 1).all()
    )
    auditor.add(
        "predictions.classification_domain",
        bool(classification_valid),
        (
            f"predicted classes="
            f"{[float(value) for value in sorted(classification['y_pred'].unique())]}, "
            f"probability range={classification['probability'].min():.6f}-"
            f"{classification['probability'].max():.6f}"
        ),
    )
    bounds_valid = (
        np.isfinite(regression[["y_true_lower", "y_true_upper"]]).all().all()
        and regression["y_true_lower"].le(regression["y_true_upper"]).all()
    )
    auditor.add(
        "predictions.regression_bounds",
        bool(bounds_valid),
        (
            f"nonfinite bound cells={int((~np.isfinite(regression[['y_true_lower', 'y_true_upper']])).sum().sum())}, "
            f"inverted rows={int(regression['y_true_lower'].gt(regression['y_true_upper']).sum())}"
        ),
    )

    train_count_mismatches = 0
    for (fold, endpoint), group in predictions.groupby(["fold", "endpoint"]):
        values = training[endpoint]
        labeled_total = int(values.notna().sum())
        fold_names = set(split.loc[split["outer_fold"].eq(fold), "Molecule_Name"])
        labeled_test = int(
            training.loc[training["Molecule_Name"].isin(fold_names), endpoint].notna().sum()
        )
        expected_train = labeled_total - labeled_test
        observed_train_counts = set(group["n_train"].astype(int))
        if observed_train_counts != {expected_train}:
            train_count_mismatches += 1
    auditor.add(
        "predictions.train_counts_exclude_test_fold",
        train_count_mismatches == 0,
        f"fold/endpoint groups with unexpected n_train={train_count_mismatches}",
    )

    maximum_metric_difference = 0.0
    endpoint_metric_rows = metrics[
        metrics["metric"].isin(["ST-RAE", "MCC"])
        & metrics["endpoint"].ne("MA")
    ]
    for row in endpoint_metric_rows.itertuples():
        group = predictions[
            predictions["model"].eq(row.model)
            & predictions["endpoint"].eq(row.endpoint)
        ]
        if row.scope == "fold":
            group = group[group["fold"].eq(int(row.fold))]
        if row.metric == "ST-RAE":
            recomputed = float(
                official_soft_threshold_rae(
                    group["y_true"].to_numpy(),
                    group["y_pred"].to_numpy(),
                    y_true_upper=group["y_true_upper"].to_numpy(),
                    y_true_lower=group["y_true_lower"].to_numpy(),
                )
            )
        else:
            recomputed = float(
                matthews_corrcoef(
                    group["y_true"].astype(int), group["y_pred"].astype(int)
                )
            )
        maximum_metric_difference = max(
            maximum_metric_difference, abs(recomputed - float(row.value))
        )

    macro_mismatches = 0
    macro_rows = metrics[metrics["metric"].eq("MA-ST-RAE")]
    for row in macro_rows.itertuples():
        endpoints = metrics[
            metrics["model"].eq(row.model)
            & metrics["scope"].eq(row.scope)
            & metrics["metric"].eq("ST-RAE")
        ]
        if row.scope == "fold":
            endpoints = endpoints[endpoints["fold"].eq(int(row.fold))]
        expected_macro = float(endpoints["value"].mean())
        if abs(expected_macro - float(row.value)) > 1e-12:
            macro_mismatches += 1
    auditor.add(
        "metrics.official_parity",
        maximum_metric_difference < 1e-12 and macro_mismatches == 0,
        (
            f"maximum endpoint difference={maximum_metric_difference:.3e}, "
            f"macro mismatches={macro_mismatches}"
        ),
    )

    pooled = metrics[metrics["scope"].eq("pooled")]
    direct_rows = pooled[pooled["metric"].eq("MA-ST-RAE")]
    best_direct = direct_rows.loc[direct_rows["value"].idxmin()]
    pooled_primary = {
        "direct": {"model": best_direct["model"], "score": float(best_direct["value"])}
    }
    tdi_fold_diagnostics: dict[str, dict] = {}
    for isoform in TDI_ISOFORMS:
        endpoint = tdi_label_column(isoform)
        endpoint_rows = pooled[
            pooled["endpoint"].eq(endpoint) & pooled["metric"].eq("MCC")
        ]
        best = endpoint_rows.loc[endpoint_rows["value"].idxmax()]
        pooled_primary[isoform] = {
            "model": best["model"],
            "score": float(best["value"]),
        }
        best_fold_rows = metrics[
            metrics["scope"].eq("fold")
            & metrics["endpoint"].eq(endpoint)
            & metrics["metric"].eq("MCC")
            & metrics["model"].eq(best["model"])
        ].sort_values("fold")
        one_model_predictions = classification[
            classification["model"].eq(expected_models[0])
            & classification["endpoint"].eq(endpoint)
        ]
        positive_counts = {
            str(int(fold)): int(group["y_true"].sum())
            for fold, group in one_model_predictions.groupby("fold", sort=True)
        }
        tdi_fold_diagnostics[isoform] = {
            "best_model": str(best["model"]),
            "positive_counts": positive_counts,
            "best_model_fold_mcc": {
                str(int(row.fold)): float(row.value)
                for row in best_fold_rows.itertuples()
            },
        }

    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "overall_pass": auditor.passed,
        "source_manifest_sha256": file_sha256(manifest_path),
        "prediction_rows": int(len(predictions)),
        "pooled_primary": pooled_primary,
        "tdi_fold_diagnostics": tdi_fold_diagnostics,
        "checks": [asdict(check) for check in auditor.checks],
        "flagship_claim_authorized": False,
    }
    json_path = output_dir / "classical_family_audit.json"
    markdown_path = output_dir / "CLASSICAL_FAMILY_AUDIT.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(_render_markdown(payload), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/baselines.yaml"))
    args = parser.parse_args()
    payload = run_baseline_audit(args.config)
    print(
        json.dumps(
            {
                "overall_pass": payload["overall_pass"],
                "checks": len(payload["checks"]),
                "prediction_rows": payload["prediction_rows"],
            },
            indent=2,
        )
    )
    return 0 if payload["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
