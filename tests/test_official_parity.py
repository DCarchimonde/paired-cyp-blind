from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cyp_blind.constants import TDI_ISOFORMS, direct_value_column, tdi_label_column, tdi_value_column
from cyp_blind.labels import derive_tdi_labels
from cyp_blind.metrics import soft_threshold_rae


ROOT = Path(__file__).resolve().parents[1]


def _official_metric_module():
    path = ROOT / "vendor" / "openadmet" / "custom_scoring_functions.py"
    spec = spec_from_file_location("official_openadmet_metrics", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("seed", [0, 1, 20260828])
def test_st_rae_matches_vendored_official_implementation(seed: int) -> None:
    rng = np.random.default_rng(seed)
    y = rng.normal(5.0, 1.0, size=100)
    low = y - rng.uniform(0.01, 0.8, size=100)
    high = y + rng.uniform(0.01, 0.8, size=100)
    pred = y + rng.normal(0, 0.7, size=100)
    official = _official_metric_module().rae_soft_threshold_absolute_error(
        y,
        pred,
        y_true_upper=high,
        y_true_lower=low,
    )
    ours = soft_threshold_rae(y, pred, y_true_upper=high, y_true_lower=low)
    assert ours == pytest.approx(official, rel=0, abs=1e-15)


def test_released_tdi_labels_reproduce_exactly() -> None:
    path = ROOT / "data" / "raw" / "cyp-challenge-TRAIN_TDI.csv"
    if not path.is_file():
        pytest.skip("Official data not fetched")
    table = pd.read_csv(path)
    for isoform in TDI_ISOFORMS:
        official = table[tdi_label_column(isoform)].astype("boolean")
        eligible = official.notna()
        derived = derive_tdi_labels(
            table[direct_value_column(isoform)],
            table[tdi_value_column(isoform)],
            eligible=eligible,
        )
        assert derived[eligible].equals(official[eligible])

