import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from cyp_blind.baselines import _fingerprint_matrix, _knn_predict, summarize_predictions


def test_fingerprint_matrix_is_binary_and_stable() -> None:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=128)
    fingerprints = [generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in ["CCO", "CCN"]]
    matrix = _fingerprint_matrix(fingerprints)
    assert matrix.shape == (2, 128)
    assert matrix.dtype == np.uint8
    assert set(np.unique(matrix)).issubset({0, 1})


def test_knn_uses_only_explicit_training_indices() -> None:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=128)
    fingerprints = [
        generator.GetFingerprint(Chem.MolFromSmiles(s))
        for s in ["CCO", "CCN", "c1ccccc1"]
    ]
    prediction, maxima = _knn_predict(
        fingerprints,
        np.array([0, 2]),
        np.array([1]),
        np.array([1.0, 9.0]),
        neighbors=1,
        similarity_power=2.0,
        fallback=5.0,
    )
    assert prediction[0] == pytest.approx(1.0)
    assert 0 <= maxima[0] <= 1


def test_summary_uses_soft_bounds_and_macro_average() -> None:
    rows = []
    endpoints = [
        "CYP1A2_pIC50_direct_inhibition",
        "CYP2C9_pIC50_direct_inhibition",
        "CYP2D6_pIC50_direct_inhibition",
        "CYP3A4_pIC50_direct_inhibition",
    ]
    for endpoint in endpoints:
        for y_true, y_pred in [(4.0, 4.1), (6.0, 5.9), (8.0, 8.0)]:
            rows.append(
                {
                    "model": "inside_bounds",
                    "fold": 0,
                    "task": "regression",
                    "endpoint": endpoint,
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "y_true_lower": y_true - 0.2,
                    "y_true_upper": y_true + 0.2,
                }
            )
    metrics = summarize_predictions(pd.DataFrame(rows))
    pooled_macro = metrics[
        (metrics["scope"] == "pooled") & (metrics["metric"] == "MA-ST-RAE")
    ]
    assert len(pooled_macro) == 1
    assert pooled_macro.iloc[0]["value"] == pytest.approx(0.0)

