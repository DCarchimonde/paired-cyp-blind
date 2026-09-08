"""Independent arithmetic and raw-row audit for the v4 pilot.

This module does not import the training loss, model probability implementation,
or prediction-table builder. Numerical integration uses SciPy adaptive quadrature.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import ndtr
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

ISOS = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
CLASS_ISOS = ("CYP3A4", "CYP2D6")


def independent_validation_score(latents, raw, train_indices, val_indices, statistics):
    """Rebuild normalization and selection from original rows, outside the trainer."""
    training, validation = raw.iloc[train_indices], raw.iloc[val_indices]
    errors, bces = [], []
    reference, counts = [], []
    for k, iso in enumerate(ISOS):
        endpoint = iso + "_pIC50_direct_inhibition"
        scale = max(float(training[endpoint].dropna().std(ddof=0)), 0.1)
        np.testing.assert_allclose(scale, statistics["direct_scale"][k], atol=2e-7, rtol=2e-7)
        mask = validation[endpoint].notna().to_numpy()
        errors.append(float(np.abs(latents[iso + "_d"].to_numpy()[mask]
                                    - validation[endpoint].to_numpy()[mask]).mean() / scale))
        names = [iso + "_pIC50_direct_inhibition", iso + "_pIC50_TDI_condition"]
        reference.append([float(training[name + "_std"].dropna().median()) for name in names])
        counts.append([int(training[name].notna().sum()) for name in names])
    np.testing.assert_allclose(reference, statistics["reference_std"], atol=1e-7, rtol=1e-6)
    if counts != statistics["continuous_counts"] or len(training) != statistics["n_train"]:
        raise AssertionError("Training counts differ from original observations.")
    for k, iso in enumerate(CLASS_ISOS):
        label = iso + "_is_TDI"
        if int(training[label].notna().sum()) != statistics["label_counts"][k]:
            raise AssertionError("Training label mask count mismatch.")
        mask = validation[label].notna().to_numpy()
        y = validation[label].to_numpy()[mask].astype(float)
        p = latents[iso + "_probability"].to_numpy()[mask].clip(1e-6, 1 - 1e-6)
        bces.append(float(-(y * np.log(p) + (1 - y) * np.log1p(-p)).mean()))
    return float(np.mean(errors) + np.mean(bces))


def normal_rule_probability(md: float, mt: float, sd: float, st: float) -> float:
    a = (4.0 - md) / sd
    shift = math.log10(2.0)
    below = ndtr(a) * ndtr((mt - 4.0 - shift) / st)
    if a >= 12:
        return float(below)
    integral = quad(lambda z: math.exp(-z * z / 2) / math.sqrt(2 * math.pi)
                    * ndtr((mt - max(md + sd * z, 4.0) - shift) / st),
                    max(a, -12.0), 12.0, epsabs=2e-10, epsrel=2e-9, limit=150)[0]
    return float(below + integral)


def audit_latents(latents: pd.DataFrame, variant: str, tolerance: float) -> dict:
    numeric = latents.drop(columns=["Molecule_Name", "SMILES"]).to_numpy(float)
    if not np.isfinite(numeric).all():
        raise AssertionError("Non-finite latent export.")
    paired = variant.startswith("paired_")
    maximum_error = 0.0
    for iso in ISOS:
        w = latents[[f"{iso}_w{k}" for k in range(3)]].to_numpy(float)
        t = latents[[f"{iso}_t{k}" for k in range(3)]].to_numpy(float)
        d = latents[f"{iso}_d"].to_numpy(float)
        np.testing.assert_allclose(w.sum(1), 1, atol=2e-7, rtol=0)
        if (w < 0).any() or (w > 1).any():
            raise AssertionError("Invalid mixture weights.")
        np.testing.assert_allclose((w * t).sum(1), latents[f"{iso}_t_mean"], atol=2e-6, rtol=1e-6)
        if paired:
            np.testing.assert_allclose(t[:, 0], d, atol=1e-7, rtol=0)
            if (t[:, 1] < d).any() or (t[:, 2] > d).any():
                raise AssertionError("Signed hurdle ordering violated.")
        if iso not in CLASS_ISOS:
            continue
        report = latents[f"{iso}_reportability"].to_numpy(float)
        conditional = latents[f"{iso}_conditional"].to_numpy(float)
        probability = latents[f"{iso}_probability"].to_numpy(float)
        if (report < 0).any() or (report > 1).any() or (conditional < 0).any() or (conditional > 1).any():
            raise AssertionError("Invalid reporting/conditional probabilities.")
        np.testing.assert_allclose(np.clip(report * conditional, 1e-6, 1 - 1e-6), probability, atol=1e-7, rtol=1e-6)
        if paired:
            sd = latents[f"{iso}_total_sd_d"].to_numpy(float)
            st = latents[f"{iso}_total_sd_t"].to_numpy(float)
            expected = np.array([sum(w[i, k] * normal_rule_probability(d[i], t[i, k], sd[i], st[i])
                                     for k in range(3)) for i in range(len(latents))])
            maximum_error = max(maximum_error, float(np.max(np.abs(expected - conditional))))
            np.testing.assert_allclose(expected, conditional, atol=tolerance, rtol=0)
    return {"rows": len(latents), "paired_probability_adaptive_quadrature_max_error": maximum_error}


def audit_raw_predictions(predictions: pd.DataFrame, latents: pd.DataFrame, raw: pd.DataFrame,
                          family: pd.DataFrame, indices, variant: str, fold: int, seed: int) -> None:
    subset = raw.iloc[indices]
    if latents.Molecule_Name.tolist() != subset.Molecule_Name.tolist():
        raise AssertionError("Latent row identity/order mismatch.")
    if predictions.duplicated(["Molecule_Name", "endpoint"]).any():
        raise AssertionError("Duplicate prediction key.")
    expected_keys = set()
    for _, row in subset.iterrows():
        for endpoint in [*(f"{x}_pIC50_direct_inhibition" for x in ISOS), *(f"{x}_is_TDI" for x in CLASS_ISOS)]:
            if pd.notna(row[endpoint]):
                expected_keys.add((row.Molecule_Name, endpoint))
    if set(zip(predictions.Molecule_Name, predictions.endpoint, strict=True)) != expected_keys:
        raise AssertionError("Missing/extra observed prediction rows.")
    if not (predictions.variant.eq(variant).all() and predictions.fold.eq(fold).all()
            and predictions.seed.eq(seed).all() and predictions.model.eq("v4_" + variant).all()):
        raise AssertionError("Prediction job metadata mismatch.")
    lookup = raw.set_index("Molecule_Name")
    split = family.set_index("Molecule_Name")
    latent = latents.set_index("Molecule_Name")
    for row in predictions.itertuples(index=False):
        source = lookup.loc[row.Molecule_Name]
        if row.family_id != split.loc[row.Molecule_Name, "family_id"]:
            raise AssertionError("Family identity mismatch.")
        expected = float(source[row.endpoint])
        np.testing.assert_allclose(row.y_true, expected, atol=1e-10, rtol=1e-10)
        iso = row.endpoint.split("_")[0]
        if row.task == "regression":
            np.testing.assert_allclose([row.y_true_lower, row.y_true_upper],
                                      [source[row.endpoint + "_conf_low"], source[row.endpoint + "_conf_high"]],
                                      atol=1e-10, rtol=1e-10)
            np.testing.assert_allclose(row.y_pred, latent.loc[row.Molecule_Name, iso + "_d"], atol=1e-7, rtol=1e-7)
        elif row.task == "classification":
            if not np.isfinite(row.probability) or not 0 <= row.probability <= 1:
                raise AssertionError("Invalid class probability.")
            np.testing.assert_allclose(row.probability, latent.loc[row.Molecule_Name, iso + "_probability"], atol=1e-7, rtol=1e-7)
            if row.y_pred != int(row.probability >= 0.5):
                raise AssertionError("Fixed classification threshold violated.")
        else:
            raise AssertionError("Unknown prediction task.")


def summarize(predictions: pd.DataFrame, official_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = importlib.util.spec_from_file_location("v4_frozen_official_scoring", official_path)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    rows, diagnostics = [], []
    scopes = [("pooled", -1, predictions)] + [("fold", int(k), v) for k, v in predictions.groupby("fold", sort=True)]
    for scope, fold, scoped in scopes:
        for (model, seed), group in scoped.groupby(["model", "seed"], sort=True):
            direct_values = []
            direct_n = 0
            for endpoint, block in group.groupby("endpoint", sort=True):
                y, p = block.y_true.to_numpy(float), block.y_pred.to_numpy(float)
                common = {"model": model, "seed": int(seed), "scope": scope, "fold": fold, "endpoint": endpoint}
                if block.task.iloc[0] == "regression":
                    lo, hi = block.y_true_lower.to_numpy(float), block.y_true_upper.to_numpy(float)
                    numerator = np.maximum(p - hi, 0).sum() + np.maximum(lo - p, 0).sum()
                    denominator = np.maximum(y.mean() - hi, 0).sum() + np.maximum(lo - y.mean(), 0).sum()
                    if denominator <= 0:
                        raise AssertionError("Undefined official regression denominator.")
                    value = float(numerator / denominator)
                    check = official.rae_soft_threshold_absolute_error(y, p, y_true_upper=hi, y_true_lower=lo)
                    np.testing.assert_allclose(value, check, atol=1e-12, rtol=1e-12)
                    direct_values.append(value)
                    direct_n += len(y)
                    metric = "ST-RAE"
                else:
                    y, p = y.astype(int), p.astype(int)
                    tp = int(((y == 1) & (p == 1)).sum())
                    tn = int(((y == 0) & (p == 0)).sum())
                    fp = int(((y == 0) & (p == 1)).sum())
                    fn = int(((y == 1) & (p == 0)).sum())
                    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
                    value = (tp * tn - fp * fn) / denom if denom else 0.0
                    np.testing.assert_allclose(value, matthews_corrcoef(y, p), atol=1e-12, rtol=1e-12)
                    probability = block.probability.to_numpy(float).clip(1e-6, 1 - 1e-6)
                    diagnostics.append({**common, "n": len(y), "positives": int(y.sum()), "TP": tp, "TN": tn,
                                        "FP": fp, "FN": fn, "MCC": value,
                                        "BCE": float(-(y * np.log(probability) + (1 - y) * np.log1p(-probability)).mean()),
                                        "Brier": float(np.mean((probability - y) ** 2)),
                                        "AUROC": float(roc_auc_score(y, probability)) if len(set(y)) == 2 else np.nan,
                                        "AP": float(average_precision_score(y, probability)) if y.sum() else np.nan,
                                        "labeled_families": int(block.family_id.nunique()),
                                        "positive_families": int(block.loc[block.y_true.eq(1), "family_id"].nunique())})
                    metric = "MCC"
                rows.append({**common, "metric": metric, "value": float(value), "n": len(y)})
            if len(direct_values) != 4:
                raise AssertionError("Expected four direct endpoints per model/scope.")
            rows.append({"model": model, "seed": int(seed), "scope": scope, "fold": fold,
                         "endpoint": "MA", "metric": "MA-ST-RAE", "value": float(np.mean(direct_values)), "n": direct_n})
    return pd.DataFrame(rows), pd.DataFrame(diagnostics)
