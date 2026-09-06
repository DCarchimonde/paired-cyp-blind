from __future__ import annotations

import numpy as np
import pandas as pd


def derive_tdi_labels(
    direct_pic50: pd.Series,
    tdi_pic50: pd.Series,
    *,
    eligible: pd.Series | None = None,
    pic50_floor: float = 4.0,
    fold_shift: float = 2.0,
) -> pd.Series:
    """Reproduce the challenge's assay-defined TDI label.

    A positive label requires a potency increase greater than ``fold_shift``.
    When a *reported numeric* direct potency is below the reliable range, a
    TDI-arm value above ``pic50_floor + log10(fold_shift)`` is an inferred
    positive. A missing direct value is not silently imputed below the floor;
    in the released labels, eligible rows with missing direct potency are
    assigned negative. Rows not assayed/eligible remain missing.

    ``eligible`` must come from assay provenance (for the released training set,
    the non-missing official label mask). It must never be inferred by treating
    every missing label as a negative.
    """

    if not direct_pic50.index.equals(tdi_pic50.index):
        raise ValueError("direct_pic50 and tdi_pic50 indices must match")

    if eligible is None:
        eligible = direct_pic50.notna() | tdi_pic50.notna()
    eligible = eligible.fillna(False).astype(bool)

    threshold = float(np.log10(fold_shift))
    direct_below = direct_pic50.notna() & direct_pic50.lt(pic50_floor)
    direct_quantified = direct_pic50.notna() & direct_pic50.ge(pic50_floor)
    positive_measured = (
        direct_quantified
        & tdi_pic50.notna()
        & (tdi_pic50 - direct_pic50).gt(threshold)
    )
    positive_inferred = (
        direct_below
        & tdi_pic50.notna()
        & tdi_pic50.gt(pic50_floor + threshold)
    )

    result = pd.Series(pd.NA, index=direct_pic50.index, dtype="boolean")
    result.loc[eligible] = False
    result.loc[eligible & (positive_measured | positive_inferred)] = True
    return result
