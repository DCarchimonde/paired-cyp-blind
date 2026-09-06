from pathlib import Path

import pandas as pd

from cyp_blind.io import load_yaml
from cyp_blind.splits import build_challenge_mimetic_folds


ROOT = Path(__file__).resolve().parents[1]


def test_challenge_mimetic_split_invariants() -> None:
    training_path = ROOT / "data" / "raw" / "cyp-challenge-TRAIN_TDI.csv"
    if not training_path.is_file():
        return
    training = pd.read_csv(training_path)
    contract = load_yaml(ROOT / "configs" / "family_split_contract.yaml")
    split = build_challenge_mimetic_folds(training, contract)

    assert split["connectivity_key"].nunique() == 750
    assert split["Molecule_Name"].nunique() == len(split)
    assert split["family_id"].nunique() == 75
    assert split.groupby("family_id")["connectivity_key"].nunique().eq(10).all()
    assert split.groupby("outer_fold")["connectivity_key"].nunique().eq(150).all()
    assert split.groupby("connectivity_key")["family_id"].nunique().eq(1).all()
    assert split.groupby("connectivity_key")["outer_fold"].nunique().eq(1).all()
    assert split["unit_min_tanimoto_to_anchor"].ge(
        contract["families"]["minimum_tanimoto_to_anchor"]
    ).all()
    assert not set(split["Molecule_Name"]) & set(split["anchor_Molecule_Name"])
    assert not set(split["connectivity_key"]) & set(split["anchor_connectivity_key"])
