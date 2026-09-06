from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
import rdkit
import sklearn

from .baselines import _fingerprint_matrix, _fit_predict, summarize_predictions
from .chem import morgan_fingerprints, standardize_smiles
from .constants import DIRECT_ISOFORMS, TDI_ISOFORMS, direct_value_column, tdi_label_column
from .io import file_sha256, load_yaml


def _git(root: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _scaffold_tie_key(scaffold: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{scaffold}".encode()).hexdigest()


def build_diagnostic_splits(root: Path, config: dict) -> dict[str, pd.DataFrame]:
    family_path = root / config["inputs"]["family_panel"]
    training_path = root / config["inputs"]["training"]
    family = pd.read_csv(family_path)
    training = pd.read_csv(training_path)[["Molecule_Name", "SMILES"]]
    panel = family.merge(training, on="Molecule_Name", validate="one_to_one")
    records = [standardize_smiles(value) for value in panel["SMILES"]]
    panel["diagnostic_connectivity_key"] = [record.connectivity_key for record in records]
    panel["diagnostic_murcko_scaffold"] = [record.murcko_scaffold for record in records]
    if not panel["connectivity_key"].eq(panel["diagnostic_connectivity_key"]).all():
        raise AssertionError("Frozen family connectivity keys disagree with current standardization")

    fold_count = int(config["folds"])
    random_config = config["schemes"]["random"]
    connectivity_units = sorted(panel["diagnostic_connectivity_key"].unique())
    rng = np.random.default_rng(int(random_config["seed"]))
    shuffled_units = np.asarray(connectivity_units, dtype=object)
    rng.shuffle(shuffled_units)
    random_fold_by_unit = {
        str(unit): int(rank % fold_count) for rank, unit in enumerate(shuffled_units)
    }
    random_split = panel.copy()
    random_split["outer_fold"] = random_split["diagnostic_connectivity_key"].map(
        random_fold_by_unit
    )
    random_split["split_scheme"] = "random"
    random_split["diagnostic_group_id"] = random_split["diagnostic_connectivity_key"]

    scaffold_config = config["schemes"]["murcko_scaffold"]
    scaffold_groups = [
        (str(scaffold), list(indices))
        for scaffold, indices in panel.groupby(
            "diagnostic_murcko_scaffold", sort=True
        ).groups.items()
    ]
    scaffold_groups.sort(
        key=lambda item: (
            -len(item[1]),
            _scaffold_tie_key(item[0], int(scaffold_config["seed"])),
        )
    )
    fold_loads = [0] * fold_count
    scaffold_fold_by_group: dict[str, int] = {}
    for scaffold, indices in scaffold_groups:
        fold = min(range(fold_count), key=lambda candidate: (fold_loads[candidate], candidate))
        scaffold_fold_by_group[scaffold] = fold
        fold_loads[fold] += len(indices)
    scaffold_split = panel.copy()
    scaffold_split["outer_fold"] = scaffold_split["diagnostic_murcko_scaffold"].map(
        scaffold_fold_by_group
    )
    scaffold_split["split_scheme"] = "murcko_scaffold"
    scaffold_split["diagnostic_group_id"] = scaffold_split[
        "diagnostic_murcko_scaffold"
    ]

    output_columns = [
        "Molecule_Name",
        "connectivity_key",
        "family_id",
        "outer_fold",
        "split_scheme",
        "diagnostic_group_id",
        "diagnostic_murcko_scaffold",
    ]
    outputs = {
        "random": random_split[output_columns].sort_values(
            ["outer_fold", "diagnostic_group_id", "Molecule_Name"], kind="mergesort"
        ),
        "murcko_scaffold": scaffold_split[output_columns].sort_values(
            ["outer_fold", "diagnostic_group_id", "Molecule_Name"], kind="mergesort"
        ),
    }
    for scheme, frame in outputs.items():
        if set(frame["Molecule_Name"]) != set(family["Molecule_Name"]):
            raise AssertionError(f"{scheme} changed the frozen 750-molecule panel")
        if frame["Molecule_Name"].duplicated().any():
            raise AssertionError(f"{scheme} assigned a molecule more than once")
        if frame["outer_fold"].nunique() != fold_count:
            raise AssertionError(f"{scheme} did not populate all folds")
        if frame.groupby("connectivity_key")["outer_fold"].nunique().gt(1).any():
            raise AssertionError(f"{scheme} split a connectivity unit across folds")
    if scaffold_split.groupby("diagnostic_murcko_scaffold")["outer_fold"].nunique().gt(1).any():
        raise AssertionError("Murcko scaffold split placed one scaffold in multiple folds")
    if set(outputs["random"].groupby("outer_fold").size()) != {150}:
        raise AssertionError("Random diagnostic must have exactly 150 panel molecules per fold")
    return {key: value.reset_index(drop=True) for key, value in outputs.items()}


def write_diagnostic_splits(root: Path, config: dict) -> dict[str, pd.DataFrame]:
    splits = build_diagnostic_splits(root, config)
    for scheme, frame in splits.items():
        output = root / config["schemes"][scheme]["output"]
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)
    return splits


def _run_predictions(
    training: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    baseline_config: dict,
    model_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = [standardize_smiles(value) for value in training["SMILES"]]
    fp_config = baseline_config["fingerprint"]
    fingerprints = morgan_fingerprints(
        [record.canonical_smiles for record in records],
        radius=int(fp_config["radius"]),
        n_bits=int(fp_config["n_bits"]),
        include_chirality=bool(fp_config["include_chirality"]),
    )
    x = _fingerprint_matrix(fingerprints)
    name_to_index = pd.Series(training.index, index=training["Molecule_Name"]).to_dict()
    endpoints = [
        (direct_value_column(isoform), "regression") for isoform in DIRECT_ISOFORMS
    ] + [(tdi_label_column(isoform), "classification") for isoform in TDI_ISOFORMS]
    threshold = float(baseline_config["evaluation"]["classification_threshold"])
    base_seed = int(baseline_config["random_seed"])
    prediction_rows: list[dict] = []
    runtime_rows: list[dict] = []

    for scheme, split in splits.items():
        split_index = split.set_index("Molecule_Name")
        for fold in range(5):
            fold_names = split.loc[split["outer_fold"].eq(fold), "Molecule_Name"].tolist()
            test_base = np.zeros(len(training), dtype=bool)
            test_base[[name_to_index[name] for name in fold_names]] = True
            train_base = ~test_base
            for endpoint_index, (endpoint, task) in enumerate(endpoints):
                if task == "classification":
                    values = training[endpoint].astype("boolean")
                    numeric_values = values.astype("Float64").to_numpy(
                        dtype=float, na_value=np.nan
                    )
                else:
                    numeric_values = pd.to_numeric(
                        training[endpoint], errors="coerce"
                    ).to_numpy(float)
                labeled = np.isfinite(numeric_values)
                train_indices = np.flatnonzero(train_base & labeled)
                test_indices = np.flatnonzero(test_base & labeled)
                y_train = numeric_values[train_indices]
                if not len(test_indices):
                    raise RuntimeError(f"{scheme} fold {fold} has no test rows for {endpoint}")
                if task == "classification" and len(np.unique(y_train)) != 2:
                    raise RuntimeError(
                        f"{scheme} fold {fold} has degenerate training labels for {endpoint}"
                    )

                for model_name in model_names:
                    started = time.perf_counter()
                    prediction, probability, max_similarity = _fit_predict(
                        model_name,
                        baseline_config["models"][model_name],
                        task=task,
                        x_train=x[train_indices],
                        x_test=x[test_indices],
                        y_train=y_train,
                        fingerprints=fingerprints,
                        train_indices=train_indices,
                        test_indices=test_indices,
                        threshold=threshold,
                        seed=base_seed + 100 * fold + endpoint_index,
                    )
                    runtime_rows.append(
                        {
                            "split_scheme": scheme,
                            "model": model_name,
                            "fold": fold,
                            "endpoint": endpoint,
                            "seconds": time.perf_counter() - started,
                            "n_train": len(train_indices),
                            "n_test": len(test_indices),
                        }
                    )
                    for offset, index in enumerate(test_indices):
                        molecule_name = str(training.at[index, "Molecule_Name"])
                        lower = np.nan
                        upper = np.nan
                        if task == "regression":
                            lower = float(training.at[index, f"{endpoint}_conf_low"])
                            upper = float(training.at[index, f"{endpoint}_conf_high"])
                        prediction_rows.append(
                            {
                                "split_scheme": scheme,
                                "model": model_name,
                                "fold": fold,
                                "Molecule_Name": molecule_name,
                                "family_id": split_index.at[molecule_name, "family_id"],
                                "diagnostic_group_id": split_index.at[
                                    molecule_name, "diagnostic_group_id"
                                ],
                                "endpoint": endpoint,
                                "task": task,
                                "y_true": float(numeric_values[index]),
                                "y_true_lower": lower,
                                "y_true_upper": upper,
                                "y_pred": float(prediction[offset]),
                                "probability": float(probability[offset]),
                                "max_train_tanimoto": float(max_similarity[offset]),
                                "n_train": len(train_indices),
                            }
                        )
    return pd.DataFrame(prediction_rows), pd.DataFrame(runtime_rows)


def _render_report(comparison: pd.DataFrame, split_summary: dict, manifest: dict) -> str:
    pooled = comparison[comparison["scope"].eq("pooled")]
    lines = [
        "# Random vs Scaffold vs Family Split Diagnostic",
        "",
        f"Generated: {manifest['generated_utc']}  ",
        f"Frozen run commit: `{manifest['git_head_at_start']}`",
        "",
        "All schemes score the same frozen 750-molecule panel with identical model hyperparameters and endpoints. Only grouping into five folds changes. The family result is read from its previously frozen artifact, not rerun or selected post hoc.",
        "",
        "## Split construction",
        "",
        "| Scheme | Fold sizes | Groups | Original families spanning >1 fold |",
        "|---|---|---:|---:|",
    ]
    for scheme, summary in split_summary.items():
        lines.append(
            f"| {scheme} | {summary['fold_sizes']} | {summary['groups']} | "
            f"{summary['families_spanning_folds']} |"
        )
    lines.extend(
        [
            "",
            "## Pooled primary scores",
            "",
            "| Scheme | Model | MA-ST-RAE ↓ | CYP3A4 MCC ↑ | CYP2D6 MCC ↑ |",
            "|---|---|---:|---:|---:|",
        ]
    )

    def score(scheme: str, model: str, endpoint: str, metric: str) -> float:
        match = pooled[
            pooled["split_scheme"].eq(scheme)
            & pooled["model"].eq(model)
            & pooled["endpoint"].eq(endpoint)
            & pooled["metric"].eq(metric)
        ]
        if len(match) != 1:
            raise ValueError(f"Missing metric {scheme}/{model}/{endpoint}/{metric}")
        return float(match.iloc[0]["value"])

    for scheme in ["random", "murcko_scaffold", "family"]:
        for model in manifest["models_run"]:
            lines.append(
                f"| {scheme} | {model} | {score(scheme, model, 'MA', 'MA-ST-RAE'):.4f} | "
                f"{score(scheme, model, tdi_label_column('CYP3A4'), 'MCC'):.4f} | "
                f"{score(scheme, model, tdi_label_column('CYP2D6'), 'MCC'):.4f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This is a validation-regime diagnostic, not a model-selection tournament. Random splitting intentionally allows members of an original analog family to appear in other folds; family splitting forbids that. Any apparent random-split gain is evidence of optimism from related-series exposure, not evidence that the underlying model improved.",
            "",
        ]
    )
    return "\n".join(lines)


def run_split_gap(config_path: Path) -> dict:
    config_path = config_path.resolve()
    root = config_path.parents[1]
    config = load_yaml(config_path)
    status_at_start = _git(root, "status", "--porcelain")
    if status_at_start:
        raise RuntimeError("Refusing split-gap run from a dirty Git worktree")
    baseline_config_path = root / config["inputs"]["baseline_config"]
    baseline_config = load_yaml(baseline_config_path)
    splits = build_diagnostic_splits(root, config)
    training = pd.read_csv(root / config["inputs"]["training"])
    model_names = list(config["models"])
    predictions, runtimes = _run_predictions(
        training, splits, baseline_config, model_names
    )
    metric_frames = []
    for scheme, group in predictions.groupby("split_scheme", sort=False):
        frame = summarize_predictions(group)
        frame.insert(0, "split_scheme", scheme)
        metric_frames.append(frame)
    diagnostic_metrics = pd.concat(metric_frames, ignore_index=True)
    family_metrics = pd.read_csv(root / config["inputs"]["frozen_family_metrics"])
    family_metrics.insert(0, "split_scheme", "family")
    comparison = pd.concat([diagnostic_metrics, family_metrics], ignore_index=True)

    split_summary: dict[str, dict] = {}
    family_panel = pd.read_csv(root / config["inputs"]["family_panel"])
    family_frame = family_panel.copy()
    family_frame["diagnostic_group_id"] = family_frame["family_id"]
    all_splits = {**splits, "family": family_frame}
    for scheme, frame in all_splits.items():
        family_folds = frame.groupby("family_id")["outer_fold"].nunique()
        split_summary[scheme] = {
            "fold_sizes": {
                str(int(key)): int(value)
                for key, value in frame.groupby("outer_fold").size().items()
            },
            "groups": int(frame["diagnostic_group_id"].nunique()),
            "families_spanning_folds": int(family_folds.gt(1).sum()),
        }

    output_dir = root / config["outputs"]["directory"]
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / config["outputs"]["predictions"]
    metrics_path = output_dir / config["outputs"]["metrics"]
    runtimes_path = output_dir / config["outputs"]["runtimes"]
    report_path = output_dir / config["outputs"]["report"]
    manifest_path = output_dir / config["outputs"]["manifest"]
    predictions.to_csv(predictions_path, index=False)
    comparison.to_csv(metrics_path, index=False)
    runtimes.to_csv(runtimes_path, index=False)
    manifest = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "stage": config["stage"],
        "git_head_at_start": _git(root, "rev-parse", "HEAD"),
        "git_status_at_start": status_at_start,
        "models_run": model_names,
        "schemes_run": list(splits),
        "same_panel": True,
        "prediction_rows": int(len(predictions)),
        "split_summary": split_summary,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "rdkit": rdkit.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
        },
        "inputs": {
            str(path.relative_to(root)): file_sha256(path)
            for path in [
                config_path,
                baseline_config_path,
                root / config["inputs"]["training"],
                root / config["inputs"]["family_panel"],
                root / config["inputs"]["frozen_family_metrics"],
                *[
                    root / config["schemes"][scheme]["output"]
                    for scheme in splits
                ],
            ]
        },
        "flagship_claim_authorized": False,
    }
    report_path.write_text(
        _render_report(comparison, split_summary, manifest), encoding="utf-8"
    )
    manifest["outputs"] = {
        str(path.relative_to(root)): file_sha256(path)
        for path in [predictions_path, metrics_path, runtimes_path, report_path]
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/split_gap.yaml"))
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    root = config_path.parents[1]
    config = load_yaml(config_path)
    if args.build_only:
        splits = write_diagnostic_splits(root, config)
        print(
            json.dumps(
                {
                    scheme: {
                        "rows": len(frame),
                        "fold_sizes": frame.groupby("outer_fold").size().to_dict(),
                        "groups": int(frame["diagnostic_group_id"].nunique()),
                    }
                    for scheme, frame in splits.items()
                },
                indent=2,
            )
        )
        return 0
    manifest = run_split_gap(config_path)
    print(
        json.dumps(
            {
                "prediction_rows": manifest["prediction_rows"],
                "schemes_run": manifest["schemes_run"],
                "models_run": manifest["models_run"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
