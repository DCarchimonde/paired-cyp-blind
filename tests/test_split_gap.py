from pathlib import Path

import pandas as pd

from cyp_blind.io import load_yaml
from cyp_blind.split_gap import build_diagnostic_splits


ROOT = Path(__file__).resolve().parents[1]


def test_diagnostic_splits_preserve_panel_and_group_integrity() -> None:
    training_path = ROOT / "data/raw/cyp-challenge-TRAIN_TDI.csv"
    if not training_path.is_file():
        return
    config = load_yaml(ROOT / "configs/split_gap.yaml")
    splits = build_diagnostic_splits(ROOT, config)
    frozen = pd.read_csv(ROOT / config["inputs"]["family_panel"])

    for frame in splits.values():
        assert set(frame["Molecule_Name"]) == set(frozen["Molecule_Name"])
        assert frame["Molecule_Name"].nunique() == 750
        assert frame.groupby("connectivity_key")["outer_fold"].nunique().eq(1).all()

    random = splits["random"]
    assert random.groupby("outer_fold").size().eq(150).all()
    assert random.groupby("family_id")["outer_fold"].nunique().gt(1).any()

    scaffold = splits["murcko_scaffold"]
    assert scaffold.groupby("diagnostic_group_id")["outer_fold"].nunique().eq(1).all()
    assert scaffold.groupby("outer_fold").size().max() - scaffold.groupby(
        "outer_fold"
    ).size().min() <= 14

