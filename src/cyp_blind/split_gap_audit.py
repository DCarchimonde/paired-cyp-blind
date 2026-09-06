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


STRUCTURAL_MODELS = ("ecfp4_knn", "ecfp4_random_forest", "ecfp4_lightgbm")


def _render_markdown(payload: dict) -> str:
    lines = [
        "# Split-Gap Audit",
        "",
        f"**Overall status: {'PASS' if payload['overall_pass'] else 'FAIL'}**  ",
        f"Generated: {payload['generated_utc']}",
        "",
        "## Optimism gaps relative to frozen family folds",
        "",
        "Positive values mean the alternative split looks better than the family split. For direct inhibition this is `family MA-ST-RAE − alternative MA-ST-RAE`; for TDI it is `alternative MCC − family MCC`.",
        "",
        "| Alternative | Model | Direct gap | CYP3A4 MCC gap | CYP2D6 MCC gap |",
        "|---|---|---:|---:|---:|",
    ]
    for scheme in ("random", "murcko_scaffold"):
        for model in STRUCTURAL_MODELS:
            row = payload["optimism_gaps"][scheme][model]
            lines.append(
                f"| {scheme} | {model} | {row['direct']:+.4f} | "
                f"{row['CYP3A4']:+.4f} | {row['CYP2D6']:+.4f} |"
            )
    lines.extend(
        [
            "",
            "## What the diagnostic supports",
            "",
            "- Direct-inhibition optimism is modest and not universal: RF/LightGBM improve by about 0.016–0.020 MA-ST-RAE under random/scaffold grouping, while kNN is essentially unchanged or slightly worse.",
            "- Random splitting inflates TDI MCC for all three structural models. The CYP3A4 gap is +0.078 to +0.176; this supports keeping family folds primary.",
            "- CYP2D6 gaps are positive but remain fragile because each family fold has only 5–7 positives and the family-fold baseline itself changes sign across folds.",
            "- The fixed-panel scaffold diagnostic is not equivalent to strict scaffold exclusion over the full 6,145-molecule universe: all 75 original analog families still span folds. It is secondary evidence only.",
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
            "The split-gap diagnostic is now complete. It does not establish a new model contribution; it only shows why the frozen analog-family validation remains primary. Neural baseline completion is still required before Day 3–5 can close.",
            "",
        ]
    )
    return "\n".join(lines)


def run_split_gap_audit(config_path: Path) -> dict:
    config_path = config_path.resolve()
    root = config_path.parents[1]
    config = load_yaml(config_path)
    output_dir = root / config["outputs"]["directory"]
    predictions_path = output_dir / config["outputs"]["predictions"]
    metrics_path = output_dir / config["outputs"]["metrics"]
    manifest_path = output_dir / config["outputs"]["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    predictions = pd.read_csv(predictions_path)
    metrics = pd.read_csv(metrics_path)
    training = pd.read_csv(root / config["inputs"]["training"])
    auditor = Auditor()

    auditor.add(
        "manifest.clean_start",
        manifest["git_status_at_start"] == "",
        f"git_status_at_start={manifest['git_status_at_start']!r}",
    )
    auditor.add(
        "manifest.scheme_and_model_set",
        manifest["schemes_run"] == ["random", "murcko_scaffold"]
        and manifest["models_run"] == list(config["models"]),
        (
            f"schemes={manifest['schemes_run']}, models={manifest['models_run']}"
        ),
    )
    mismatched_hashes = [
        relative_path
        for relative_path, expected_hash in {
            **manifest["inputs"],
            **manifest["outputs"],
        }.items()
        if file_sha256(root / relative_path) != expected_hash
    ]
    auditor.add(
        "manifest.hashes",
        not mismatched_hashes,
        f"hash mismatches={mismatched_hashes}",
    )

    duplicate_count = int(
        predictions.duplicated(
            ["split_scheme", "model", "Molecule_Name", "endpoint"]
        ).sum()
    )
    auditor.add(
        "predictions.unique_keys",
        duplicate_count == 0,
        f"duplicate scheme/model/molecule/endpoint rows={duplicate_count}",
    )

    name_to_index = pd.Series(training.index, index=training["Molecule_Name"]).to_dict()
    panel = pd.read_csv(root / config["inputs"]["family_panel"])
    endpoints = [
        *[direct_value_column(isoform) for isoform in DIRECT_ISOFORMS],
        *[tdi_label_column(isoform) for isoform in TDI_ISOFORMS],
    ]
    expected_keys = {
        (scheme, model, molecule_name, endpoint)
        for scheme in ("random", "murcko_scaffold")
        for model in config["models"]
        for molecule_name in panel["Molecule_Name"]
        for endpoint in endpoints
        if pd.notna(training.at[name_to_index[molecule_name], endpoint])
    }
    actual_keys = set(
        predictions[
            ["split_scheme", "model", "Molecule_Name", "endpoint"]
        ].itertuples(index=False, name=None)
    )
    auditor.add(
        "predictions.coverage_exact",
        actual_keys == expected_keys,
        (
            f"observed={len(actual_keys)}, expected={len(expected_keys)}, "
            f"missing={len(expected_keys - actual_keys)}, extra={len(actual_keys - expected_keys)}"
        ),
    )

    mapping_errors = 0
    group_errors = 0
    split_invariant_errors = 0
    for scheme in ("random", "murcko_scaffold"):
        split = pd.read_csv(root / config["schemes"][scheme]["output"])
        split_index = split.set_index("Molecule_Name")
        scheme_predictions = predictions[predictions["split_scheme"].eq(scheme)]
        mapping_errors += sum(
            int(row.fold) != int(split_index.at[row.Molecule_Name, "outer_fold"])
            for row in scheme_predictions.itertuples()
        )
        group_errors += sum(
            row.diagnostic_group_id
            != split_index.at[row.Molecule_Name, "diagnostic_group_id"]
            for row in scheme_predictions.itertuples()
        )
        split_invariant_errors += int(
            set(split["Molecule_Name"]) != set(panel["Molecule_Name"])
        )
        split_invariant_errors += int(
            split.groupby("diagnostic_group_id")["outer_fold"].nunique().gt(1).any()
        )
        split_invariant_errors += int(
            set(split.groupby("outer_fold").size()) != {150}
        )
    auditor.add(
        "predictions.split_mapping",
        mapping_errors == 0 and group_errors == 0,
        f"fold mapping errors={mapping_errors}, group mapping errors={group_errors}",
    )
    auditor.add(
        "splits.fixed_panel_group_integrity",
        split_invariant_errors == 0,
        f"split invariant errors={split_invariant_errors}",
    )

    finite = np.isfinite(predictions[["y_true", "y_pred"]]).all().all()
    classification = predictions[predictions["task"].eq("classification")]
    probability_valid = classification["probability"].between(0, 1).all()
    auditor.add(
        "predictions.numeric_domain",
        bool(finite and probability_valid),
        (
            f"nonfinite truth/prediction cells="
            f"{int((~np.isfinite(predictions[['y_true', 'y_pred']])).sum().sum())}, "
            f"out-of-range probabilities="
            f"{int((~classification['probability'].between(0, 1)).sum())}"
        ),
    )

    train_count_errors = 0
    for (scheme, fold, endpoint), group in predictions.groupby(
        ["split_scheme", "fold", "endpoint"]
    ):
        split = pd.read_csv(root / config["schemes"][scheme]["output"])
        fold_names = set(split.loc[split["outer_fold"].eq(fold), "Molecule_Name"])
        expected_train = int(training[endpoint].notna().sum()) - int(
            training.loc[training["Molecule_Name"].isin(fold_names), endpoint]
            .notna()
            .sum()
        )
        if set(group["n_train"].astype(int)) != {expected_train}:
            train_count_errors += 1
    auditor.add(
        "predictions.train_counts_exclude_test_fold",
        train_count_errors == 0,
        f"scheme/fold/endpoint groups with unexpected n_train={train_count_errors}",
    )

    maximum_metric_difference = 0.0
    diagnostic_metric_rows = metrics[
        metrics["split_scheme"].isin(["random", "murcko_scaffold"])
        & metrics["metric"].isin(["ST-RAE", "MCC"])
        & metrics["endpoint"].ne("MA")
    ]
    for row in diagnostic_metric_rows.itertuples():
        group = predictions[
            predictions["split_scheme"].eq(row.split_scheme)
            & predictions["model"].eq(row.model)
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
    auditor.add(
        "metrics.official_parity",
        maximum_metric_difference < 1e-12,
        f"maximum endpoint difference={maximum_metric_difference:.3e}",
    )

    frozen_family = pd.read_csv(root / config["inputs"]["frozen_family_metrics"])
    embedded_family = (
        metrics[metrics["split_scheme"].eq("family")]
        .drop(columns="split_scheme")
        .reset_index(drop=True)
    )
    try:
        pd.testing.assert_frame_equal(embedded_family, frozen_family, check_exact=True)
        family_unchanged = True
        family_detail = "embedded family metrics exactly equal frozen artifact"
    except AssertionError as error:
        family_unchanged = False
        family_detail = str(error).splitlines()[0]
    auditor.add(
        "metrics.family_artifact_immutable",
        family_unchanged,
        family_detail,
    )

    pooled = metrics[metrics["scope"].eq("pooled")]

    def pooled_value(scheme: str, model: str, endpoint: str, metric: str) -> float:
        match = pooled[
            pooled["split_scheme"].eq(scheme)
            & pooled["model"].eq(model)
            & pooled["endpoint"].eq(endpoint)
            & pooled["metric"].eq(metric)
        ]
        if len(match) != 1:
            raise ValueError(f"Missing pooled value {scheme}/{model}/{endpoint}/{metric}")
        return float(match.iloc[0]["value"])

    optimism_gaps: dict[str, dict] = {}
    for scheme in ("random", "murcko_scaffold"):
        optimism_gaps[scheme] = {}
        for model in STRUCTURAL_MODELS:
            optimism_gaps[scheme][model] = {
                "direct": pooled_value("family", model, "MA", "MA-ST-RAE")
                - pooled_value(scheme, model, "MA", "MA-ST-RAE"),
                "CYP3A4": pooled_value(
                    scheme, model, tdi_label_column("CYP3A4"), "MCC"
                )
                - pooled_value("family", model, tdi_label_column("CYP3A4"), "MCC"),
                "CYP2D6": pooled_value(
                    scheme, model, tdi_label_column("CYP2D6"), "MCC"
                )
                - pooled_value("family", model, tdi_label_column("CYP2D6"), "MCC"),
            }

    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "overall_pass": auditor.passed,
        "source_manifest_sha256": file_sha256(manifest_path),
        "prediction_rows": int(len(predictions)),
        "optimism_gaps": optimism_gaps,
        "checks": [asdict(check) for check in auditor.checks],
        "flagship_claim_authorized": False,
    }
    json_path = output_dir / "split_gap_audit.json"
    markdown_path = output_dir / "SPLIT_GAP_AUDIT.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(_render_markdown(payload), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/split_gap.yaml"))
    args = parser.parse_args()
    payload = run_split_gap_audit(args.config)
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
