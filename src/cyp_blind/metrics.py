from __future__ import annotations

import numpy as np
from sklearn.metrics import matthews_corrcoef


def relative_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(
        np.sum(np.abs(y_true - y_pred))
        / np.sum(np.abs(y_true - np.mean(y_true)))
    )


def soft_threshold_rae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    y_true_upper: np.ndarray | None = None,
    y_true_lower: np.ndarray | None = None,
) -> float:
    """Exact mathematical equivalent of the official OpenADMET ST-RAE."""

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    upper = y_true if y_true_upper is None else np.asarray(y_true_upper, dtype=float)
    lower = y_true if y_true_lower is None else np.asarray(y_true_lower, dtype=float)

    above = np.clip(y_pred - upper, a_min=0, a_max=None)
    below = np.clip(lower - y_pred, a_min=0, a_max=None)
    numerator = np.sum(above + below)

    mean_true = np.mean(y_true)
    baseline_above = np.clip(mean_true - upper, a_min=0, a_max=None)
    baseline_below = np.clip(lower - mean_true, a_min=0, a_max=None)
    denominator = np.sum(baseline_above + baseline_below)
    return float(numerator / denominator)


def macro_st_rae(endpoint_scores: dict[str, float]) -> float:
    if not endpoint_scores:
        raise ValueError("endpoint_scores must not be empty")
    values = np.asarray(list(endpoint_scores.values()), dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("All endpoint ST-RAE values must be finite")
    return float(values.mean())


def mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(matthews_corrcoef(y_true, y_pred))

