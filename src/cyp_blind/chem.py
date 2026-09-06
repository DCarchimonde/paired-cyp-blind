from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold


@dataclass(frozen=True)
class MoleculeRecord:
    input_smiles: str
    canonical_smiles: str
    connectivity_smiles: str
    inchikey: str
    connectivity_key: str
    murcko_scaffold: str


def standardize_smiles(smiles: str) -> MoleculeRecord:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")

    mol = rdMolStandardize.Cleanup(mol)
    mol = rdMolStandardize.FragmentParent(mol)
    mol = rdMolStandardize.Uncharger().uncharge(mol)
    Chem.SanitizeMol(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    connectivity = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    inchikey = Chem.MolToInchiKey(mol)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol,
        includeChirality=True,
    )
    return MoleculeRecord(
        input_smiles=smiles,
        canonical_smiles=canonical,
        connectivity_smiles=connectivity,
        inchikey=inchikey,
        connectivity_key=inchikey.split("-", maxsplit=1)[0],
        murcko_scaffold=scaffold,
    )


def morgan_fingerprints(
    smiles: list[str],
    *,
    radius: int = 2,
    n_bits: int = 2048,
    include_chirality: bool = True,
) -> list[DataStructs.ExplicitBitVect]:
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
        includeChirality=include_chirality,
    )
    fingerprints = []
    for value in smiles:
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            raise ValueError(f"Cannot fingerprint invalid SMILES: {value}")
        fingerprints.append(generator.GetFingerprint(mol))
    return fingerprints


def tanimoto_vector(
    query: DataStructs.ExplicitBitVect,
    candidates: list[DataStructs.ExplicitBitVect],
) -> np.ndarray:
    return np.asarray(DataStructs.BulkTanimotoSimilarity(query, candidates), dtype=float)

