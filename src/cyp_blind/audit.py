from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rdkit

from .chem import standardize_smiles
from .constants import (
    DIRECT_ISOFORMS,
    FILES_BY_ROLE,
    TDI_ISOFORMS,
    direct_value_column,
    tdi_label_column,
    tdi_value_column,
)
from .io import file_sha256, load_tables, load_yaml
from .labels import derive_tdi_labels
from .splits import build_challenge_mimetic_folds


@dataclass(frozen=True)
class Check:
    check_id: str
    status: str
    critical: bool
    detail: str


class Auditor:
    def __init__(self) -> None:
        self.checks: list[Check] = []

    def add(self, check_id: str, passed: bool, detail: str, *, critical: bool = True) -> None:
        self.checks.append(
            Check(
                check_id=check_id,
                status="PASS" if passed else ("FAIL" if critical else "WARN"),
                critical=critical,
                detail=detail,
            )
        )

    @property
    def passed(self) -> bool:
        return not any(check.status == "FAIL" for check in self.checks)


def _required_columns() -> dict[str, set[str]]:
    direct = {"Molecule_Name", "SMILES"}
    tdi = {"Molecule_Name", "SMILES"}
    emax = {"Molecule_Name", "SMILES"}
    for isoform in DIRECT_ISOFORMS:
        direct.update(
            {
                direct_value_column(isoform),
                f"{direct_value_column(isoform)}_conf_high",
                f"{direct_value_column(isoform)}_conf_low",
                f"{direct_value_column(isoform)}_std",
            }
        )
        tdi.update(
            {
                direct_value_column(isoform),
                f"{direct_value_column(isoform)}_conf_high",
                f"{direct_value_column(isoform)}_conf_low",
                f"{direct_value_column(isoform)}_std",
                tdi_value_column(isoform),
                f"{tdi_value_column(isoform)}_conf_high",
                f"{tdi_value_column(isoform)}_conf_low",
                f"{tdi_value_column(isoform)}_std",
            }
        )
        emax.update(
            {
                f"{isoform}_EmaxVsPosCtrl_TDI_condition",
                f"{isoform}_EmaxVsPosCtrl_direct_inhibition",
                f"{isoform}_is_TDI",
            }
        )
    tdi.update(tdi_label_column(isoform) for isoform in TDI_ISOFORMS)
    return {
        "direct": direct,
        "blinded_test": {"Molecule_Name", "SMILES"},
        "tdi": tdi,
        "single_concentration": {
            "Molecule_Name",
            "SMILES",
            "enzyme",
            "concentration_M",
            "log2fc_estimate",
            "log2fc_std_error",
            "p_value",
            "log2fc_fdr",
        },
        "emax": emax,
    }


def _check_interval_columns(
    auditor: Auditor,
    table: pd.DataFrame,
    *,
    table_name: str,
    base_columns: list[str],
) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for base in base_columns:
        value = table[base]
        low = table[f"{base}_conf_low"]
        high = table[f"{base}_conf_high"]
        std = table[f"{base}_std"]
        observed = value.notna()
        missing_companion = observed & (low.isna() | high.isna() | std.isna())
        inverted = observed & (low.gt(high))
        outside = observed & (value.lt(low - 1e-10) | value.gt(high + 1e-10))
        nonpositive_std = observed & std.le(0)
        passed = not (
            missing_companion.any()
            or inverted.any()
            or outside.any()
            or nonpositive_std.any()
        )
        auditor.add(
            f"intervals.{table_name}.{base}",
            passed,
            (
                f"n={int(observed.sum())}, missing companion={int(missing_companion.sum())}, "
                f"inverted={int(inverted.sum())}, point outside={int(outside.sum())}, "
                f"nonpositive std={int(nonpositive_std.sum())}"
            ),
        )
        summary[base] = {
            "observed": int(observed.sum()),
            "missing_companion": int(missing_companion.sum()),
            "inverted": int(inverted.sum()),
            "point_outside": int(outside.sum()),
            "nonpositive_std": int(nonpositive_std.sum()),
        }
    return summary


def _render_markdown(payload: dict) -> str:
    status = "PASS" if payload["overall_pass"] else "FAIL"
    lines = [
        "# Day 1–2 Data and Metric Audit",
        "",
        f"**Overall status: {status}**  ",
        f"Generated: {payload['generated_utc']}",
        "",
        "No model performance is reported at this stage. This audit only validates the research substrate.",
        "",
        "## Official data inventory",
        "",
        "| Role | Rows | Columns | Unique molecules |",
        "|---|---:|---:|---:|",
    ]
    for role, stats in payload["tables"].items():
        lines.append(
            f"| {role} | {stats['rows']} | {stats['columns']} | {stats['unique_molecules']} |"
        )

    lines.extend(
        [
            "",
            "## TDI label balance and reproduction",
            "",
            "| Isoform | Labeled | Positive | Negative | Positive rate | Reproduction |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for isoform, stats in payload["tdi"].items():
        lines.append(
            f"| {isoform} | {stats['labeled']} | {stats['positive']} | "
            f"{stats['negative']} | {stats['positive_rate']:.3f} | "
            f"{stats['reproduction_rate']:.3f} |"
        )

    split = payload["split"]
    lines.extend(
        [
            "",
            "## Frozen challenge-mimetic split",
            "",
            f"- Held-out molecule rows: {split['molecule_rows']}",
            f"- Held-out analog connectivity units: {split['analog_units']}",
            f"- Analog families: {split['families']}",
            f"- Outer folds: {split['folds']}",
            f"- Per-fold connectivity units: {split['fold_unit_sizes']}",
            f"- Per-fold molecule rows: {split['fold_molecule_sizes']}",
            f"- Similarity range (unit minima): {split['similarity_min']:.3f}–{split['similarity_max']:.3f}; median {split['similarity_median']:.3f}",
            f"- Minimum selected-anchor pIC50 by isoform: {split['minimum_anchor_pIC50_by_isoform']}",
            "",
            "## Checks",
            "",
            "| Status | Critical | Check | Detail |",
            "|---|---|---|---|",
        ]
    )
    for check in payload["checks"]:
        detail = check["detail"].replace("|", "\\|")
        lines.append(
            f"| {check['status']} | {'yes' if check['critical'] else 'no'} | "
            f"`{check['check_id']}` | {detail} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "A PASS here permits baseline modeling to start. It does not establish model novelty, superiority, or flagship status. Those require the predeclared family-fold and blinded-test gates.",
            "",
        ]
    )
    return "\n".join(lines)


def run_audit(config_path: Path) -> dict:
    root = config_path.resolve().parents[1]
    config = load_yaml(config_path)
    paths = config["paths"]
    raw_dir = root / paths["raw_data"]
    manifest = load_yaml(root / paths["manifest"])
    split_contract = load_yaml(root / paths["split_contract"])
    reports_dir = root / paths["reports"]
    split_path = root / paths["split_output"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    split_path.parent.mkdir(parents=True, exist_ok=True)

    auditor = Auditor()
    tables = load_tables(raw_dir)
    expected_rows = config["data_contract"]["expected_rows"]

    table_stats: dict[str, dict] = {}
    required = _required_columns()
    for role, table in tables.items():
        table_stats[role] = {
            "rows": int(len(table)),
            "columns": int(len(table.columns)),
            "unique_molecules": int(table["Molecule_Name"].nunique()),
            "duplicate_molecule_rows": int(table["Molecule_Name"].duplicated().sum()),
            "duplicate_smiles_rows": int(table["SMILES"].duplicated().sum()),
        }
        auditor.add(
            f"rows.{role}",
            len(table) == int(expected_rows[role]),
            f"observed={len(table)}, expected={expected_rows[role]}",
        )
        missing_columns = required[role] - set(table.columns)
        auditor.add(
            f"schema.{role}",
            not missing_columns,
            f"missing={sorted(missing_columns)}",
        )
        if role != "single_concentration":
            auditor.add(
                f"unique_ids.{role}",
                not table["Molecule_Name"].duplicated().any(),
                f"duplicate rows={int(table['Molecule_Name'].duplicated().sum())}",
            )

    for filename, metadata in manifest["files"].items():
        path = raw_dir / filename
        actual_hash = file_sha256(path)
        actual_bytes = path.stat().st_size
        auditor.add(
            f"manifest.{filename}",
            actual_hash == metadata["sha256"] and actual_bytes == int(metadata["bytes"]),
            (
                f"sha256={actual_hash}, bytes={actual_bytes}; "
                f"expected sha256={metadata['sha256']}, bytes={metadata['bytes']}"
            ),
        )

    single = tables["single_concentration"]
    pair_duplicates = single.duplicated(["Molecule_Name", "enzyme"]).sum()
    auditor.add(
        "single.unique_molecule_enzyme",
        pair_duplicates == 0,
        f"duplicate molecule/enzyme pairs={int(pair_duplicates)}",
    )
    enzyme_counts = single.groupby("Molecule_Name")["enzyme"].nunique()
    auditor.add(
        "single.dense_four_isoforms",
        enzyme_counts.eq(4).all(),
        f"molecules with !=4 isoforms={int((~enzyme_counts.eq(4)).sum())}",
    )

    mapping = pd.concat(
        [
            table[["Molecule_Name", "SMILES"]].drop_duplicates().assign(table=role)
            for role, table in tables.items()
        ],
        ignore_index=True,
    )
    mapping_conflicts = mapping.groupby("Molecule_Name")["SMILES"].nunique().gt(1)
    auditor.add(
        "identifiers.cross_table_smiles",
        not mapping_conflicts.any(),
        f"Molecule_Name values mapped to >1 SMILES={int(mapping_conflicts.sum())}",
    )

    tdi_ids = set(tables["tdi"]["Molecule_Name"])
    direct_ids = set(tables["direct"]["Molecule_Name"])
    auditor.add(
        "direct.strict_subset_of_tdi",
        direct_ids < tdi_ids,
        f"direct={len(direct_ids)}, tdi={len(tdi_ids)}, extra_tdi={len(tdi_ids-direct_ids)}",
    )

    merged = tables["direct"].merge(
        tables["tdi"],
        on=["Molecule_Name", "SMILES"],
        suffixes=("_direct_table", "_tdi_table"),
        validate="one_to_one",
    )
    direct_disagreements = 0
    for isoform in DIRECT_ISOFORMS:
        for suffix in ("", "_conf_high", "_conf_low", "_std"):
            base = f"{direct_value_column(isoform)}{suffix}"
            left = merged[f"{base}_direct_table"]
            right = merged[f"{base}_tdi_table"]
            mismatch = (left.isna() != right.isna()) | (
                left.notna() & right.notna() & ~np.isclose(left, right, rtol=0, atol=1e-12)
            )
            direct_disagreements += int(mismatch.sum())
    auditor.add(
        "direct.cross_table_exactness",
        direct_disagreements == 0,
        f"cell disagreements={direct_disagreements}",
    )

    direct_bases = [direct_value_column(isoform) for isoform in DIRECT_ISOFORMS]
    tdi_direct_interval_summary = _check_interval_columns(
        auditor,
        tables["tdi"],
        table_name="tdi_direct_arm",
        base_columns=direct_bases,
    )
    tdi_bases = [tdi_value_column(isoform) for isoform in DIRECT_ISOFORMS]
    tdi_active_interval_summary = _check_interval_columns(
        auditor,
        tables["tdi"],
        table_name="tdi_active_arm",
        base_columns=tdi_bases,
    )

    tdi_summary: dict[str, dict] = {}
    tdi_table = tables["tdi"]
    for isoform in TDI_ISOFORMS:
        label_col = tdi_label_column(isoform)
        official = tdi_table[label_col].astype("boolean")
        eligible = official.notna()
        derived = derive_tdi_labels(
            tdi_table[direct_value_column(isoform)],
            tdi_table[tdi_value_column(isoform)],
            eligible=eligible,
            pic50_floor=float(config["data_contract"]["pic50_floor"]),
            fold_shift=float(config["data_contract"]["tdi_fold_shift"]),
        )
        matches = derived[eligible].eq(official[eligible])
        rate = float(matches.mean())
        auditor.add(
            f"labels.{isoform}.reproduction",
            rate == float(config["data_contract"]["require_label_reproduction_rate"]),
            f"matched={int(matches.sum())}/{int(eligible.sum())}, rate={rate:.6f}",
        )
        positive = int((official == True).sum())
        negative = int((official == False).sum())
        auditor.add(
            f"labels.{isoform}.minimum_positive_count",
            positive >= int(config["kill_gates"]["minimum_positive_compounds_per_tdi_isoform"]),
            f"positive={positive}, minimum={config['kill_gates']['minimum_positive_compounds_per_tdi_isoform']}",
        )
        tdi_summary[isoform] = {
            "labeled": int(eligible.sum()),
            "positive": positive,
            "negative": negative,
            "positive_rate": positive / int(eligible.sum()),
            "reproduction_rate": rate,
            "labeled_with_missing_direct_or_tdi_value": int(
                (
                    eligible
                    & (
                        tdi_table[direct_value_column(isoform)].isna()
                        | tdi_table[tdi_value_column(isoform)].isna()
                    )
                ).sum()
            ),
        }

    emax = tables["emax"].set_index("Molecule_Name")
    tdi_indexed = tdi_table.set_index("Molecule_Name")
    emax_disagreements = 0
    for isoform in TDI_ISOFORMS:
        column = tdi_label_column(isoform)
        left = emax[column].astype("boolean")
        right = tdi_indexed[column].astype("boolean")
        emax_disagreements += int((left.isna() != right.isna()).sum())
        shared = left.notna() & right.notna()
        emax_disagreements += int((left[shared] != right[shared]).sum())
    auditor.add(
        "labels.tdi_emax_consistency",
        emax_disagreements == 0,
        f"label disagreements={emax_disagreements}",
    )

    training_records = [standardize_smiles(value) for value in tdi_table["SMILES"]]
    test_records = [standardize_smiles(value) for value in tables["blinded_test"]["SMILES"]]
    training_keys = [record.connectivity_key for record in training_records]
    training_inchikeys = [record.inchikey for record in training_records]
    test_keys = [record.connectivity_key for record in test_records]
    exact_overlap = set(training_keys) & set(test_keys)
    auditor.add(
        "leakage.train_test_connectivity_overlap",
        not exact_overlap,
        f"shared connectivity keys={len(exact_overlap)}",
    )
    train_duplicate_keys = len(training_keys) - len(set(training_keys))
    train_duplicate_full_inchikeys = len(training_inchikeys) - len(set(training_inchikeys))
    structure_index = pd.DataFrame(
        {
            "Molecule_Name": tdi_table["Molecule_Name"],
            "connectivity_key": training_keys,
            "inchikey": training_inchikeys,
            "canonical_smiles": [record.canonical_smiles for record in training_records],
        }
    )
    variant_rows = structure_index[
        structure_index.duplicated("connectivity_key", keep=False)
    ]
    connectivity_variant_groups = [
        {
            "connectivity_key": str(connectivity_key),
            "molecules": group[
                ["Molecule_Name", "inchikey", "canonical_smiles"]
            ].to_dict(orient="records"),
        }
        for connectivity_key, group in variant_rows.groupby("connectivity_key", sort=True)
    ]
    auditor.add(
        "structures.training_full_inchikey_duplicates",
        train_duplicate_full_inchikeys == 0,
        f"duplicate full-InChIKey rows={train_duplicate_full_inchikeys}",
    )
    auditor.add(
        "structures.training_standardized_duplicates",
        train_duplicate_keys == 0,
        (
            f"extra connectivity-collapsed rows={train_duplicate_keys} across "
            f"{len(connectivity_variant_groups)} variant groups; full-InChIKey duplicates="
            f"{train_duplicate_full_inchikeys}; connectivity units are indivisible in splits"
        ),
        critical=False,
    )

    split = build_challenge_mimetic_folds(tdi_table, split_contract)
    split.to_csv(split_path, index=False)
    fold_molecule_sizes = split.groupby("outer_fold").size().to_dict()
    fold_unit_sizes = split.groupby("outer_fold")["connectivity_key"].nunique().to_dict()
    family_unit_sizes = split.groupby("family_id")["connectivity_key"].nunique()
    anchor_counts = split[["family_id", "anchor_Molecule_Name"]].drop_duplicates()
    analog_units = int(split["connectivity_key"].nunique())
    minimum_similarity = float(split_contract["families"]["minimum_tanimoto_to_anchor"])
    connectivity_one_family = split.groupby("connectivity_key")["family_id"].nunique().eq(1).all()
    connectivity_one_fold = split.groupby("connectivity_key")["outer_fold"].nunique().eq(1).all()
    training_unit_sizes = pd.Series(training_keys).value_counts()
    split_unit_sizes = split["connectivity_key"].value_counts()
    connectivity_complete_units = all(
        int(split_count) == int(training_unit_sizes.loc[connectivity_key])
        for connectivity_key, split_count in split_unit_sizes.items()
    )
    split_ok = (
        analog_units == 750
        and split["family_id"].nunique() == 75
        and split["outer_fold"].nunique() == 5
        and family_unit_sizes.eq(10).all()
        and set(fold_unit_sizes.values()) == {150}
        and not split["Molecule_Name"].duplicated().any()
        and not set(split["Molecule_Name"]) & set(anchor_counts["anchor_Molecule_Name"])
        and not set(split["connectivity_key"]) & set(split["anchor_connectivity_key"])
        and bool(connectivity_one_family)
        and bool(connectivity_one_fold)
        and connectivity_complete_units
        and split["unit_min_tanimoto_to_anchor"].ge(minimum_similarity).all()
    )
    auditor.add(
        "splits.challenge_mimetic_invariants",
        split_ok,
        (
            f"molecule_rows={len(split)}, analog_units={analog_units}, "
            f"families={split.family_id.nunique()}, fold_unit_sizes={fold_unit_sizes}, "
            f"family_unit_size_range={int(family_unit_sizes.min())}-"
            f"{int(family_unit_sizes.max())}, connectivity_one_family="
            f"{bool(connectivity_one_family)}, connectivity_one_fold="
            f"{bool(connectivity_one_fold)}, connectivity_complete_units="
            f"{connectivity_complete_units}, minimum_unit_similarity="
            f"{split['unit_min_tanimoto_to_anchor'].min():.6f}"
        ),
    )
    auditor.add(
        "splits.training_only_construction",
        split_contract.get("test_structures_allowed") is False,
        "split generator accepts only the training table; blinded test structures are not an input",
    )

    label_lookup = tdi_table.set_index("Molecule_Name")
    positive_family_counts: dict[str, int] = {}
    for isoform in TDI_ISOFORMS:
        labels = split["Molecule_Name"].map(label_lookup[tdi_label_column(isoform)])
        positive_families = split.loc[labels.eq(True), "family_id"].nunique()
        positive_family_counts[isoform] = int(positive_families)
        minimum = int(config["kill_gates"]["minimum_positive_families_per_tdi_isoform"])
        auditor.add(
            f"splits.{isoform}.positive_families",
            positive_families >= minimum,
            f"positive held-out families={positive_families}, minimum={minimum}",
        )

    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "overall_pass": auditor.passed,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "rdkit": rdkit.__version__,
        },
        "upstream": {
            "dataset_commit": manifest["source"]["commit"],
            "tutorial_commit": manifest["official_tutorial"]["commit"],
            "license": manifest["source"]["license"],
        },
        "tables": table_stats,
        "tdi": tdi_summary,
        "intervals": {
            "direct_arm": tdi_direct_interval_summary,
            "tdi_arm": tdi_active_interval_summary,
        },
        "structures": {
            "train_test_connectivity_overlap": len(exact_overlap),
            "training_duplicate_connectivity_rows": train_duplicate_keys,
            "training_duplicate_full_inchikey_rows": train_duplicate_full_inchikeys,
            "training_connectivity_variant_groups": connectivity_variant_groups,
        },
        "split": {
            "path": str(split_path.relative_to(root)),
            "molecule_rows": int(len(split)),
            "analog_units": analog_units,
            "families": int(split["family_id"].nunique()),
            "folds": int(split["outer_fold"].nunique()),
            "fold_unit_sizes": {
                str(key): int(value) for key, value in fold_unit_sizes.items()
            },
            "fold_molecule_sizes": {
                str(key): int(value) for key, value in fold_molecule_sizes.items()
            },
            "similarity_min": float(split["unit_min_tanimoto_to_anchor"].min()),
            "similarity_median": float(split["unit_min_tanimoto_to_anchor"].median()),
            "similarity_max": float(split["unit_min_tanimoto_to_anchor"].max()),
            "minimum_anchor_pIC50_by_isoform": {
                str(key): float(value)
                for key, value in split.groupby("anchor_isoform")["anchor_direct_pIC50"]
                .min()
                .items()
            },
            "maximum_anchor_candidate_rank_by_isoform": {
                str(key): int(value)
                for key, value in split.groupby("anchor_isoform")["anchor_candidate_rank"]
                .max()
                .items()
            },
            "positive_family_counts": positive_family_counts,
        },
        "checks": [asdict(check) for check in auditor.checks],
    }
    (reports_dir / "day1_day2_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (reports_dir / "DAY1_DAY2_AUDIT.md").write_text(
        _render_markdown(payload),
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/research_contract.yaml"))
    args = parser.parse_args()
    payload = run_audit(args.config)
    print(json.dumps({"overall_pass": payload["overall_pass"], "checks": len(payload["checks"])}, indent=2))
    return 0 if payload["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
