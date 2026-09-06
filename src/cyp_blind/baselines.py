from __future__ import annotations

import argparse
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
from lightgbm import LGBMClassifier, LGBMRegressor
from rdkit import DataStructs
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from .chem import morgan_fingerprints, standardize_smiles
from .constants import DIRECT_ISOFORMS, TDI_ISOFORMS, direct_value_column, tdi_label_column
from .io import file_sha256, load_yaml
from .metrics import macro_st_rae, mcc, soft_threshold_rae


def _git(root: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _fingerprint_matrix(fingerprints: list[DataStructs.ExplicitBitVect]) -> np.ndarray:
    if not fingerprints:
        raise ValueError("fingerprints must not be empty")
    matrix = np.zeros((len(fingerprints), fingerprints[0].GetNumBits()), dtype=np.uint8)
    for row, fingerprint in zip(matrix, fingerprints, strict=True):
        DataStructs.ConvertToNumpyArray(fingerprint, row)
    return matrix


def _knn_predict(
    fingerprints: list[DataStructs.ExplicitBitVect],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    y_train: np.ndarray,
    *,
    neighbors: int,
    similarity_power: float,
    fallback: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(train_indices) != len(y_train):
        raise ValueError("train indices and labels must have equal length")
    if neighbors < 1:
        raise ValueError("neighbors must be positive")
    train_fingerprints = [fingerprints[index] for index in train_indices]
    predictions: list[float] = []
    maxima: list[float] = []
    for index in test_indices:
        similarities = np.asarray(
            DataStructs.BulkTanimotoSimilarity(fingerprints[index], train_fingerprints),
            dtype=float,
        )
        k = min(neighbors, len(similarities))
        top = np.argpartition(similarities, -k)[-k:]
        top = top[np.argsort(-similarities[top], kind="mergesort")]
        weights = np.power(similarities[top], similarity_power)
        if float(weights.sum()) <= 0:
            predictions.append(float(fallback))
        else:
            predictions.append(float(np.average(y_train[top], weights=weights)))
        maxima.append(float(similarities[top[0]]))
    return np.asarray(predictions), np.asarray(maxima)


def _fit_predict(
    model_name: str,
    model_config: dict,
    *,
    task: str,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    fingerprints: list[DataStructs.ExplicitBitVect],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    threshold: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kind = model_config["kind"]
    max_similarity = np.full(len(test_indices), np.nan, dtype=float)

    if kind == "constant":
        statistic = model_config["statistic"]
        value = float(np.mean(y_train) if statistic == "mean" else np.median(y_train))
        raw = np.full(len(test_indices), value, dtype=float)
    elif kind == "read_across":
        fallback = float(np.median(y_train))
        raw, max_similarity = _knn_predict(
            fingerprints,
            train_indices,
            test_indices,
            y_train,
            neighbors=int(model_config["neighbors"]),
            similarity_power=float(model_config["similarity_power"]),
            fallback=fallback,
        )
    elif kind == "random_forest":
        common = {
            "n_estimators": int(model_config["n_estimators"]),
            "max_features": model_config["max_features"],
            "min_samples_leaf": int(model_config["min_samples_leaf"]),
            "n_jobs": int(model_config["n_jobs"]),
            "random_state": seed,
        }
        if task == "regression":
            model = RandomForestRegressor(**common)
            model.fit(x_train, y_train)
            raw = model.predict(x_test)
        else:
            model = RandomForestClassifier(
                **common,
                class_weight=model_config["classifier_class_weight"],
            )
            model.fit(x_train, y_train.astype(int))
            raw = model.predict_proba(x_test)[:, 1]
    elif kind == "lightgbm":
        common = {
            "n_estimators": int(model_config["n_estimators"]),
            "learning_rate": float(model_config["learning_rate"]),
            "num_leaves": int(model_config["num_leaves"]),
            "min_child_samples": int(model_config["min_child_samples"]),
            "subsample": float(model_config["subsample"]),
            "subsample_freq": int(model_config["subsample_freq"]),
            "colsample_bytree": float(model_config["colsample_bytree"]),
            "reg_lambda": float(model_config["reg_lambda"]),
            "random_state": seed,
            "n_jobs": int(model_config["n_jobs"]),
            "verbosity": -1,
            "deterministic": bool(model_config["deterministic"]),
            "force_col_wise": True,
        }
        if task == "regression":
            model = LGBMRegressor(
                **common,
                objective=model_config["objective_regression"],
            )
            model.fit(x_train, y_train)
            raw = model.predict(x_test)
        else:
            model = LGBMClassifier(
                **common,
                objective="binary",
                class_weight=model_config["classifier_class_weight"],
            )
            model.fit(x_train, y_train.astype(int))
            raw = model.predict_proba(x_test)[:, 1]
    else:
        raise ValueError(f"Unknown model kind for {model_name}: {kind}")

    raw = np.asarray(raw, dtype=float)
    if task == "classification":
        prediction = (raw >= threshold).astype(float)
        probability = raw
    else:
        prediction = raw
        probability = np.full(len(raw), np.nan, dtype=float)
    return prediction, probability, max_similarity


def _metric_row(
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


def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {
        "model",
        "fold",
        "task",
        "endpoint",
        "y_true",
        "y_pred",
        "y_true_lower",
        "y_true_upper",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions missing columns: {sorted(missing)}")

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
            y_true = group["y_true"].to_numpy(dtype=float)
            y_pred = group["y_pred"].to_numpy(dtype=float)
            if task == "regression":
                value = soft_threshold_rae(
                    y_true,
                    y_pred,
                    y_true_lower=group["y_true_lower"].to_numpy(dtype=float),
                    y_true_upper=group["y_true_upper"].to_numpy(dtype=float),
                )
                rows.append(
                    _metric_row(
                        model=model,
                        scope=scope,
                        fold=fold,
                        task=task,
                        endpoint=endpoint,
                        metric="ST-RAE",
                        value=value,
                        n=len(group),
                    )
                )
            else:
                rows.append(
                    _metric_row(
                        model=model,
                        scope=scope,
                        fold=fold,
                        task=task,
                        endpoint=endpoint,
                        metric="MCC",
                        value=mcc(y_true.astype(int), y_pred.astype(int)),
                        n=len(group),
                        positives_true=int(y_true.sum()),
                        positives_predicted=int(y_pred.sum()),
                    )
                )

    metrics = pd.DataFrame(rows)
    direct = metrics[(metrics["task"] == "regression") & (metrics["metric"] == "ST-RAE")]
    macro_rows: list[dict] = []
    for (model, scope, fold), group in direct.groupby(["model", "scope", "fold"], dropna=False):
        endpoint_scores = dict(zip(group["endpoint"], group["value"], strict=True))
        if len(endpoint_scores) != len(DIRECT_ISOFORMS):
            raise ValueError(
                f"Expected {len(DIRECT_ISOFORMS)} direct endpoints for {model}/{scope}/{fold}"
            )
        macro_rows.append(
            _metric_row(
                model=str(model),
                scope=str(scope),
                fold=None if pd.isna(fold) else int(fold),
                task="regression",
                endpoint="MA",
                metric="MA-ST-RAE",
                value=macro_st_rae(endpoint_scores),
                n=int(group["n"].sum()),
            )
        )
    return pd.DataFrame(rows + macro_rows)


def _render_report(metrics: pd.DataFrame, runtimes: pd.DataFrame, manifest: dict) -> str:
    pooled = metrics[metrics["scope"] == "pooled"]
    models = manifest["models_run"]
    lines = [
        "# Day 3–5 Classical Baseline Report",
        "",
        f"Generated: {manifest['generated_utc']}  ",
        f"Protocol commit: `{manifest['git_head_at_start']}`",
        "",
        "These are challenge-mimetic family-fold results. Lower MA-ST-RAE is better; higher MCC is better. No primary-model claim is authorized by this report.",
        "",
        "## Pooled primary endpoints",
        "",
        "| Model | MA-ST-RAE ↓ | CYP1A2 ST-RAE | CYP2C9 ST-RAE | CYP2D6 ST-RAE | CYP3A4 ST-RAE | CYP3A4 MCC ↑ | CYP2D6 MCC ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def value(model: str, endpoint: str, metric: str) -> float:
        match = pooled[
            (pooled["model"] == model)
            & (pooled["endpoint"] == endpoint)
            & (pooled["metric"] == metric)
        ]
        if len(match) != 1:
            raise ValueError(f"Missing pooled metric for {model}/{endpoint}/{metric}")
        return float(match.iloc[0]["value"])

    for model in models:
        cells = [
            value(model, "MA", "MA-ST-RAE"),
            *[value(model, direct_value_column(isoform), "ST-RAE") for isoform in DIRECT_ISOFORMS],
            value(model, tdi_label_column("CYP3A4"), "MCC"),
            value(model, tdi_label_column("CYP2D6"), "MCC"),
        ]
        lines.append(f"| {model} | " + " | ".join(f"{cell:.4f}" for cell in cells) + " |")

    lines.extend(
        [
            "",
            "## Runtime",
            "",
            "| Model | Fits | Wall seconds |",
            "|---|---:|---:|",
        ]
    )
    runtime_summary = runtimes.groupby("model")["seconds"].agg(["count", "sum"])
    for model in models:
        row = runtime_summary.loc[model]
        lines.append(f"| {model} | {int(row['count'])} | {float(row['sum']):.1f} |")

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This completes only the frozen classical subset. Chemprop single-task/masked-multitask baselines and random-versus-scaffold split-gap diagnostics remain pending. The Day 13–14 GO/STOP gates compare the later latent model against the strongest completed fair baseline, not against a convenient weak model.",
            "",
        ]
    )
    return "\n".join(lines)


def run_baselines(
    config_path: Path,
    *,
    requested_models: list[str] | None = None,
    allow_dirty: bool = False,
) -> dict:
    config_path = config_path.resolve()
    root = config_path.parents[1]
    config = load_yaml(config_path)
    status_at_start = _git(root, "status", "--porcelain")
    require_clean = bool(config["evaluation"]["require_clean_git_at_start"])
    if require_clean and not allow_dirty and status_at_start:
        raise RuntimeError(
            "Refusing to run a frozen baseline protocol from a dirty Git worktree; "
            "commit protocol changes or pass --allow-dirty for a labeled diagnostic run"
        )

    all_models = config["models"]
    models = list(all_models) if requested_models is None else requested_models
    unknown = set(models) - set(all_models)
    if unknown:
        raise ValueError(f"Unknown models requested: {sorted(unknown)}")

    training_path = root / "data/raw/cyp-challenge-TRAIN_TDI.csv"
    split_path = root / config["evaluation"]["split"]
    training = pd.read_csv(training_path)
    split = pd.read_csv(split_path)
    if training["Molecule_Name"].duplicated().any():
        raise ValueError("Training Molecule_Name must be unique")
    if split["Molecule_Name"].duplicated().any():
        raise ValueError("Split Molecule_Name must be unique")
    if not set(split["Molecule_Name"]).issubset(set(training["Molecule_Name"])):
        raise ValueError("Split contains molecules outside the training universe")

    records = [standardize_smiles(value) for value in training["SMILES"]]
    fp_config = config["fingerprint"]
    fingerprints = morgan_fingerprints(
        [record.canonical_smiles for record in records],
        radius=int(fp_config["radius"]),
        n_bits=int(fp_config["n_bits"]),
        include_chirality=bool(fp_config["include_chirality"]),
    )
    x = _fingerprint_matrix(fingerprints)
    name_to_index = pd.Series(training.index, index=training["Molecule_Name"]).to_dict()
    family_lookup = split.set_index("Molecule_Name")["family_id"].to_dict()

    endpoints = [
        (direct_value_column(isoform), "regression") for isoform in DIRECT_ISOFORMS
    ] + [(tdi_label_column(isoform), "classification") for isoform in TDI_ISOFORMS]
    threshold = float(config["evaluation"]["classification_threshold"])
    base_seed = int(config["random_seed"])
    prediction_rows: list[dict] = []
    runtime_rows: list[dict] = []

    for fold in range(int(config["evaluation"]["outer_folds"])):
        fold_names = split.loc[split["outer_fold"].eq(fold), "Molecule_Name"].tolist()
        test_base = np.zeros(len(training), dtype=bool)
        test_base[[name_to_index[name] for name in fold_names]] = True
        train_base = ~test_base
        if np.any(train_base & test_base):
            raise AssertionError("Train/test masks overlap")

        for endpoint_index, (endpoint, task) in enumerate(endpoints):
            if task == "classification":
                values = training[endpoint].astype("boolean")
                numeric_values = values.astype("Float64").to_numpy(dtype=float, na_value=np.nan)
            else:
                numeric_values = pd.to_numeric(training[endpoint], errors="coerce").to_numpy(float)
            labeled = np.isfinite(numeric_values)
            train_indices = np.flatnonzero(train_base & labeled)
            test_indices = np.flatnonzero(test_base & labeled)
            y_train = numeric_values[train_indices]
            if not len(test_indices):
                raise RuntimeError(f"Fold {fold} has no labeled test rows for {endpoint}")
            if task == "classification" and len(np.unique(y_train)) != 2:
                raise RuntimeError(f"Fold {fold} training labels are degenerate for {endpoint}")

            for model_name in models:
                started = time.perf_counter()
                prediction, probability, max_similarity = _fit_predict(
                    model_name,
                    all_models[model_name],
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
                            "model": model_name,
                            "fold": fold,
                            "Molecule_Name": molecule_name,
                            "family_id": family_lookup[molecule_name],
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

    predictions = pd.DataFrame(prediction_rows)
    expected_test_keys = set(
        zip(
            np.repeat(split["Molecule_Name"].to_numpy(), len(endpoints)),
            np.tile([endpoint for endpoint, _ in endpoints], len(split)),
        )
    )
    for model in models:
        actual = set(
            predictions.loc[predictions["model"].eq(model), ["Molecule_Name", "endpoint"]]
            .itertuples(index=False, name=None)
        )
        labeled_expected = {
            (name, endpoint)
            for name, endpoint in expected_test_keys
            if pd.notna(training.at[name_to_index[name], endpoint])
        }
        if actual != labeled_expected:
            raise AssertionError(f"Prediction coverage mismatch for {model}")

    metrics = summarize_predictions(predictions)
    runtimes = pd.DataFrame(runtime_rows)
    output_dir = root / "reports/baselines"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "classical_family_predictions.csv"
    metrics_path = output_dir / "classical_family_metrics.csv"
    runtimes_path = output_dir / "classical_family_runtimes.csv"
    predictions.to_csv(predictions_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    runtimes.to_csv(runtimes_path, index=False)

    manifest = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "day3_day5_classical_baselines",
        "complete_classical_set": models
        == list(config["stage_completion"]["classical_models_required"]),
        "models_run": models,
        "git_head_at_start": _git(root, "rev-parse", "HEAD"),
        "git_status_at_start": status_at_start,
        "dirty_override": allow_dirty,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "rdkit": rdkit.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
        },
        "inputs": {
            str(config_path.relative_to(root)): file_sha256(config_path),
            str(training_path.relative_to(root)): file_sha256(training_path),
            str(split_path.relative_to(root)): file_sha256(split_path),
        },
        "outputs": {
            str(predictions_path.relative_to(root)): file_sha256(predictions_path),
            str(metrics_path.relative_to(root)): file_sha256(metrics_path),
            str(runtimes_path.relative_to(root)): file_sha256(runtimes_path),
        },
        "pending": {
            "neural_models": config["stage_completion"]["neural_models_pending"],
            "split_gap_diagnostics": config["stage_completion"][
                "split_gap_diagnostics_pending"
            ],
        },
        "flagship_claim_authorized": False,
    }
    report_path = output_dir / "CLASSICAL_FAMILY_BASELINES.md"
    report_path.write_text(_render_report(metrics, runtimes, manifest), encoding="utf-8")
    manifest["outputs"][str(report_path.relative_to(root))] = file_sha256(report_path)
    manifest_path = output_dir / "classical_family_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/baselines.yaml"))
    parser.add_argument(
        "--models",
        help="Comma-separated diagnostic subset; omit to run the frozen complete classical set",
    )
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    requested = args.models.split(",") if args.models else None
    manifest = run_baselines(
        args.config,
        requested_models=requested,
        allow_dirty=args.allow_dirty,
    )
    print(
        json.dumps(
            {
                "complete_classical_set": manifest["complete_classical_set"],
                "models_run": manifest["models_run"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
