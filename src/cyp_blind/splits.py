from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .chem import morgan_fingerprints, standardize_smiles, tanimoto_vector
from .constants import DIRECT_ISOFORMS, direct_value_column
from .io import load_yaml


@dataclass(frozen=True)
class Anchor:
    molecule_name: str
    isoform: str
    pic50: float
    rank: int
    universe_index: int
    candidate_rank: int

    @property
    def family_id(self) -> str:
        return f"{self.isoform}_R{self.rank:02d}_{self.molecule_name}"


def _standardized_universe(training: pd.DataFrame) -> pd.DataFrame:
    required = {"Molecule_Name", "SMILES"}
    missing = required - set(training.columns)
    if missing:
        raise ValueError(f"Training table missing columns: {sorted(missing)}")
    if training["Molecule_Name"].duplicated().any():
        raise ValueError("Training universe must have unique Molecule_Name values")

    records = [standardize_smiles(value) for value in training["SMILES"]]
    universe = training.copy().reset_index(drop=True)
    universe["canonical_smiles"] = [record.canonical_smiles for record in records]
    universe["inchikey"] = [record.inchikey for record in records]
    universe["connectivity_key"] = [record.connectivity_key for record in records]
    return universe


def build_challenge_mimetic_folds(
    training: pd.DataFrame,
    contract: dict,
) -> pd.DataFrame:
    if contract.get("test_structures_allowed") is not False:
        raise ValueError("Primary split contract must forbid blinded test structures")
    universe = _standardized_universe(training)
    fp_config = contract["fingerprint"]
    fingerprints = morgan_fingerprints(
        universe["canonical_smiles"].tolist(),
        radius=int(fp_config["radius"]),
        n_bits=int(fp_config["n_bits"]),
        include_chirality=bool(fp_config["include_chirality"]),
    )

    anchor_config = contract["anchors"]
    isoforms = list(anchor_config["isoforms"])
    per_isoform = int(anchor_config["per_isoform"])
    analogs_per_anchor = int(contract["families"]["analogs_per_anchor"])
    minimum_similarity = float(contract["families"]["minimum_tanimoto_to_anchor"])
    fold_count = int(contract["folds"]["count"])

    candidate_indices: dict[str, list[int]] = {}
    candidate_pointer: dict[str, int] = {}
    for isoform in isoforms:
        value_col = direct_value_column(isoform)
        candidates = universe.index[universe[value_col].notna()].tolist()
        candidate_indices[isoform] = sorted(
            candidates,
            key=lambda index: (
                -float(universe.at[index, value_col]),
                str(universe.at[index, "Molecule_Name"]),
            ),
        )
        candidate_pointer[isoform] = 0

    connectivity_members = {
        str(key): list(indices)
        for key, indices in universe.groupby("connectivity_key", sort=True).groups.items()
    }
    used_analog_units: set[str] = set()
    anchor_units: set[str] = set()
    anchors: list[Anchor] = []
    rows: list[dict] = []
    isoform_offset = {isoform: index for index, isoform in enumerate(isoforms)}

    # Interleave isoforms by selected rank. If a potent candidate lacks ten
    # unused analog connectivity units above the frozen similarity floor, skip
    # it and continue down the potency ranking. No blinded test structure is
    # consulted.
    for selected_rank in range(1, per_isoform + 1):
        for isoform in isoforms:
            selected_anchor: Anchor | None = None
            selected_units: list[tuple[str, float]] = []
            similarities: np.ndarray | None = None
            candidates = candidate_indices[isoform]

            while candidate_pointer[isoform] < len(candidates):
                candidate_rank = candidate_pointer[isoform] + 1
                index = candidates[candidate_pointer[isoform]]
                candidate_pointer[isoform] += 1
                unit = str(universe.at[index, "connectivity_key"])
                if unit in used_analog_units or unit in anchor_units:
                    continue

                candidate_similarities = tanimoto_vector(fingerprints[index], fingerprints)
                available_units: list[tuple[str, float, str]] = []
                for candidate_unit, member_indices in connectivity_members.items():
                    if (
                        candidate_unit == unit
                        or candidate_unit in used_analog_units
                        or candidate_unit in anchor_units
                    ):
                        continue
                    member_sims = candidate_similarities[member_indices]
                    unit_min_similarity = float(np.min(member_sims))
                    if unit_min_similarity < minimum_similarity:
                        continue
                    tie_name = min(
                        str(universe.at[member_index, "Molecule_Name"])
                        for member_index in member_indices
                    )
                    available_units.append((candidate_unit, unit_min_similarity, tie_name))

                available_units.sort(key=lambda item: (-item[1], item[2]))
                if len(available_units) < analogs_per_anchor:
                    continue

                selected_anchor = Anchor(
                    molecule_name=str(universe.at[index, "Molecule_Name"]),
                    isoform=isoform,
                    pic50=float(universe.at[index, direct_value_column(isoform)]),
                    rank=selected_rank,
                    universe_index=index,
                    candidate_rank=candidate_rank,
                )
                selected_units = [
                    (candidate_unit, score)
                    for candidate_unit, score, _ in available_units[:analogs_per_anchor]
                ]
                similarities = candidate_similarities
                break

            if selected_anchor is None or similarities is None:
                raise RuntimeError(
                    f"Could not select feasible anchor {selected_rank}/{per_isoform} for {isoform}"
                )

            anchors.append(selected_anchor)
            anchor_units.add(str(universe.at[selected_anchor.universe_index, "connectivity_key"]))
            fold = (
                (selected_anchor.rank - 1) + isoform_offset[selected_anchor.isoform]
            ) % fold_count
            for analog_unit_rank, (analog_unit, unit_min_similarity) in enumerate(
                selected_units,
                start=1,
            ):
                used_analog_units.add(analog_unit)
                for member_index in connectivity_members[analog_unit]:
                    rows.append(
                        {
                            "Molecule_Name": universe.at[member_index, "Molecule_Name"],
                            "connectivity_key": analog_unit,
                            "family_id": selected_anchor.family_id,
                            "outer_fold": fold,
                            "analog_unit_rank": analog_unit_rank,
                            "tanimoto_to_anchor": float(similarities[member_index]),
                            "unit_min_tanimoto_to_anchor": unit_min_similarity,
                            "anchor_Molecule_Name": selected_anchor.molecule_name,
                            "anchor_connectivity_key": universe.at[
                                selected_anchor.universe_index,
                                "connectivity_key",
                            ],
                            "anchor_isoform": selected_anchor.isoform,
                            "anchor_rank": selected_anchor.rank,
                            "anchor_candidate_rank": selected_anchor.candidate_rank,
                            "anchor_direct_pIC50": selected_anchor.pic50,
                        }
                    )

    result = pd.DataFrame(rows).sort_values(
        ["outer_fold", "anchor_isoform", "anchor_rank", "analog_unit_rank", "Molecule_Name"],
        kind="mergesort",
    )
    if result["connectivity_key"].nunique() != len(anchors) * analogs_per_anchor:
        raise AssertionError("Unexpected held-out analog connectivity-unit count")
    if result["Molecule_Name"].duplicated().any():
        raise AssertionError("A held-out analog was assigned to more than one family")
    if result["unit_min_tanimoto_to_anchor"].lt(minimum_similarity).any():
        raise AssertionError("A held-out analog unit fell below the frozen similarity floor")
    return result.reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=Path("configs/family_split_contract.yaml"))
    parser.add_argument("--training", type=Path, default=Path("data/raw/cyp-challenge-TRAIN_TDI.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/splits/challenge_mimetic_folds.csv"))
    args = parser.parse_args()
    contract = load_yaml(args.contract)
    training = pd.read_csv(args.training)
    result = build_challenge_mimetic_folds(training, contract)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(
        f"WROTE {args.output} ({len(result)} molecule rows, "
        f"{result.connectivity_key.nunique()} analog units, "
        f"{result.family_id.nunique()} families)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
