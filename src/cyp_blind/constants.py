from __future__ import annotations

DIRECT_ISOFORMS = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
TDI_ISOFORMS = ("CYP3A4", "CYP2D6")

DIRECT_FILENAME = "cyp-challenge-TRAIN_inhibition.csv"
TEST_FILENAME = "cyp-challenge-TEST-BLINDED.csv"
TDI_FILENAME = "cyp-challenge-TRAIN_TDI.csv"
SINGLE_FILENAME = "cyp-challenge-single-concentration-TRAIN.csv"
EMAX_FILENAME = "cyp-challenge-TRAIN_Emax.csv"

FILES_BY_ROLE = {
    "direct": DIRECT_FILENAME,
    "blinded_test": TEST_FILENAME,
    "tdi": TDI_FILENAME,
    "single_concentration": SINGLE_FILENAME,
    "emax": EMAX_FILENAME,
}

IDENTIFIER_COLUMNS = ("Molecule_Name", "SMILES")


def direct_value_column(isoform: str) -> str:
    return f"{isoform}_pIC50_direct_inhibition"


def tdi_value_column(isoform: str) -> str:
    return f"{isoform}_pIC50_TDI_condition"


def tdi_label_column(isoform: str) -> str:
    return f"{isoform}_is_TDI"

