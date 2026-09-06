import numpy as np
import pandas as pd

from cyp_blind.labels import derive_tdi_labels


def test_measured_shift_uses_strict_two_fold_threshold() -> None:
    threshold = np.log10(2.0)
    # Use a zero floor so the middle subtraction is exactly the same stored
    # floating-point number as ``threshold``; adding it to pIC50=5 first would
    # introduce an unrelated rounding artefact in the test itself.
    direct = pd.Series([0.0, 0.0, 0.0])
    tdi = pd.Series([threshold - 1e-9, threshold, threshold + 1e-9])
    labels = derive_tdi_labels(
        direct,
        tdi,
        eligible=pd.Series([True] * 3),
        pic50_floor=0.0,
    )
    assert labels.tolist() == [False, False, True]


def test_inferred_positive_and_assigned_negative_at_floor() -> None:
    direct = pd.Series([3.5, 3.5, np.nan, np.nan])
    tdi = pd.Series([4.31, 4.20, 4.31, np.nan])
    labels = derive_tdi_labels(direct, tdi, eligible=pd.Series([True] * 4))
    assert labels.tolist() == [True, False, False, False]


def test_missing_direct_value_is_not_imputed_below_floor() -> None:
    direct = pd.Series([np.nan])
    tdi = pd.Series([7.0])
    labels = derive_tdi_labels(direct, tdi, eligible=pd.Series([True]))
    assert labels.tolist() == [False]


def test_ineligible_rows_remain_missing() -> None:
    direct = pd.Series([5.0, np.nan])
    tdi = pd.Series([6.0, np.nan])
    labels = derive_tdi_labels(direct, tdi, eligible=pd.Series([False, False]))
    assert labels.isna().all()


def test_missing_labels_are_never_silently_assigned_negative() -> None:
    direct = pd.Series([np.nan])
    tdi = pd.Series([np.nan])
    labels = derive_tdi_labels(direct, tdi)
    assert labels.isna().all()
