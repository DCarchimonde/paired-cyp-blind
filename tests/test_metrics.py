import numpy as np
import pytest

from cyp_blind.metrics import macro_st_rae, mcc, relative_absolute_error, soft_threshold_rae


def test_predictions_inside_credible_interval_have_zero_error() -> None:
    y = np.array([4.0, 5.0, 6.0])
    pred = np.array([3.9, 5.1, 6.0])
    score = soft_threshold_rae(
        y,
        pred,
        y_true_lower=y - 0.2,
        y_true_upper=y + 0.2,
    )
    assert score == pytest.approx(0.0)


def test_no_bounds_recovers_plain_rae() -> None:
    y = np.array([3.0, 4.0, 7.0, 8.0])
    pred = np.array([2.0, 4.5, 6.5, 9.0])
    assert soft_threshold_rae(y, pred) == pytest.approx(relative_absolute_error(y, pred))


def test_constant_mean_prediction_scores_one_with_collapsed_bounds() -> None:
    y = np.array([3.0, 4.0, 7.0, 8.0])
    pred = np.full_like(y, y.mean())
    assert soft_threshold_rae(y, pred) == pytest.approx(1.0)


def test_macro_is_unweighted_endpoint_mean() -> None:
    assert macro_st_rae({"a": 0.2, "b": 0.4, "c": 0.6, "d": 0.8}) == pytest.approx(0.5)


def test_mcc_perfect_and_inverted() -> None:
    y = np.array([0, 0, 1, 1])
    assert mcc(y, y) == pytest.approx(1.0)
    assert mcc(y, 1 - y) == pytest.approx(-1.0)

