from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .chem import standardize_smiles
from .io import file_sha256, load_yaml
from .neural_baselines import data_specifications, neural_jobs


def _render_report(payload: dict) -> str:
    status = "PASS" if payload["overall_pass"] else "FAIL"
    lines = [
        "# Neural Baseline Protocol Audit",
        "",
        f"**Overall status: {status}**  ",
        f"Generated: {payload['generated_utc']}",
        "",
        "This audit independently reconstructs every expected train/validation/test "
        "membership from the frozen family panel and raw training table. It does not "
        "train a scientific model or authorize a flagship claim.",
        "",
        "## Frozen workload",
        "",
        f"- Data specifications: {payload['summary']['data_specs']}",
        f"- Chemprop jobs: {payload['summary']['jobs']}",
        f"- Seeds: {payload['summary']['seeds']}",
        f"- Partition-size ranges: {payload['summary']['partition_size_ranges']}",
        "",
        "## Checks",
        "",
        "| Status | Critical | Check | Detail |",
        "|---|---|---|---|",
    ]
    for check in payload["checks"]:
        critical = "yes" if check["critical"] else "no"
        detail = str(check["detail"]).replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {check['status']} | {critical} | `{check['check_id']}` | {detail} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "A PASS means the neural baseline inputs and execution matrix are internally "
            "consistent. The 200 production jobs still require the tagged clean protocol "
            "on an RTX 4090; until collection and result audit pass, neural Day 3–5 remains pending.",
            "",
        ]
    )
    return "\n".join(lines)


def audit_neural_protocol(config_path: Path) -> dict:
    config_path = config_path.resolve()
    root = config_path.parents[1]
    config = load_yaml(config_path)
    manifest_path = (
        root / config["paths"]["prepared_data"] / "preparation_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    training_path = root / config["inputs"]["training"]
    family_path = root / config["inputs"]["family_split"]
    training = pd.read_csv(training_path)
    family = pd.read_csv(family_path)
    family_fold = family.set_index("Molecule_Name")["outer_fold"].to_dict()
    molecular_records = {
        str(name): standardize_smiles(smiles)
        for name, smiles in training[["Molecule_Name", "SMILES"]].itertuples(
            index=False, name=None
        )
    }
    canonical = {
        name: record.canonical_smiles for name, record in molecular_records.items()
    }
    checks: list[dict] = []

    def add(check_id: str, passed: bool, detail: str, critical: bool = True) -> None:
        checks.append(
            {
                "check_id": check_id,
                "status": "PASS" if passed else ("FAIL" if critical else "WARN"),
                "critical": critical,
                "detail": detail,
            }
        )

    add(
        "manifest.config_hash",
        manifest["config_sha256"] == file_sha256(config_path),
        f"observed={manifest['config_sha256']}, expected={file_sha256(config_path)}",
    )
    add(
        "manifest.training_hash",
        manifest["training_sha256"] == file_sha256(training_path),
        f"observed={manifest['training_sha256']}, expected={file_sha256(training_path)}",
    )
    add(
        "manifest.family_split_hash",
        manifest["family_split_sha256"] == file_sha256(family_path),
        f"observed={manifest['family_split_sha256']}, expected={file_sha256(family_path)}",
    )

    specs = data_specifications(config)
    jobs = neural_jobs(config)
    expected_spec_keys = {spec.key for spec in specs}
    observed_spec_keys = set(manifest["data_specs"])
    add(
        "matrix.data_specs_exact",
        observed_spec_keys == expected_spec_keys,
        f"observed={len(observed_spec_keys)}, expected={len(expected_spec_keys)}, "
        f"missing={len(expected_spec_keys - observed_spec_keys)}, "
        f"extra={len(observed_spec_keys - expected_spec_keys)}",
    )
    add(
        "matrix.jobs_exact_unique",
        len(jobs) == int(config["expected"]["jobs"])
        and len({job.job_id for job in jobs}) == len(jobs),
        f"jobs={len(jobs)}, unique={len({job.job_id for job in jobs})}",
    )
    endpoints = [
        *config["models"]["single_task"]["direct"],
        *config["models"]["single_task"]["tdi"],
    ]
    panel = family.set_index("Molecule_Name")
    raw = training.set_index("Molecule_Name")
    expected_predictions = (
        2
        * len(config["seeds"])
        * sum(
            pd.notna(raw.at[name, endpoint])
            for name in panel.index
            for endpoint in endpoints
        )
    )
    add(
        "matrix.prediction_count_frozen",
        expected_predictions == int(config["expected"]["predictions"]),
        f"derived={expected_predictions}, expected={config['expected']['predictions']}",
    )

    hash_mismatches: list[str] = []
    column_mismatches: list[str] = []
    duplicate_partitions: list[str] = []
    membership_mismatches: list[str] = []
    overlap_mismatches: list[str] = []
    connectivity_overlap: list[str] = []
    smiles_mismatches: list[str] = []
    label_mismatches: list[str] = []
    count_mismatches: list[str] = []
    degenerate_tdi: list[str] = []
    partition_sizes = {membership: [] for membership in ("train", "val", "test")}
    validation_offset = int(config["cross_validation"]["inner_validation_offset"])
    fold_count = int(config["cross_validation"]["outer_folds"])

    for spec in specs:
        entry = manifest["data_specs"][spec.key]
        validation_fold = (spec.fold + validation_offset) % fold_count
        eligible = training[list(spec.targets)].notna().any(axis=1)
        folds = training["Molecule_Name"].map(family_fold)
        expected_membership = pd.Series("train", index=training.index, dtype="string")
        expected_membership.loc[folds.eq(validation_fold)] = "val"
        expected_membership.loc[folds.eq(spec.fold)] = "test"
        observed_frames: dict[str, pd.DataFrame] = {}

        for membership in ("train", "val", "test"):
            file_entry = entry["files"][membership]
            path = root / file_entry["path"]
            observed_hash = file_sha256(path)
            if observed_hash != file_entry["sha256"]:
                hash_mismatches.append(f"{spec.key}/{membership}")
            frame = pd.read_csv(path)
            observed_frames[membership] = frame
            partition_sizes[membership].append(len(frame))
            expected_columns = ["Molecule_Name", "SMILES", *spec.targets]
            if frame.columns.tolist() != expected_columns:
                column_mismatches.append(f"{spec.key}/{membership}")
            if frame["Molecule_Name"].duplicated().any():
                duplicate_partitions.append(f"{spec.key}/{membership}")
            expected_names = (
                training.loc[
                    eligible & expected_membership.eq(membership), "Molecule_Name"
                ]
                .astype(str)
                .tolist()
            )
            actual_names = frame["Molecule_Name"].astype(str).tolist()
            if actual_names != expected_names:
                membership_mismatches.append(f"{spec.key}/{membership}")
            expected_smiles = [canonical[name] for name in actual_names]
            if frame["SMILES"].astype(str).tolist() != expected_smiles:
                smiles_mismatches.append(f"{spec.key}/{membership}")
            raw_by_name = training.set_index("Molecule_Name")
            for target in spec.targets:
                expected_values = raw_by_name.loc[actual_names, target]
                if spec.task_group == "tdi":
                    expected_numeric = (
                        expected_values.astype("boolean")
                        .astype("Float64")
                        .to_numpy(dtype=float, na_value=np.nan)
                    )
                else:
                    expected_numeric = pd.to_numeric(
                        expected_values, errors="coerce"
                    ).to_numpy(dtype=float)
                actual_numeric = pd.to_numeric(frame[target], errors="coerce").to_numpy(
                    dtype=float
                )
                if not np.allclose(
                    actual_numeric, expected_numeric, rtol=0, atol=1e-12, equal_nan=True
                ):
                    label_mismatches.append(f"{spec.key}/{membership}/{target}")
                observed_target_count = int(frame[target].notna().sum())
                expected_target_count = int(entry["target_counts"][membership][target])
                if observed_target_count != expected_target_count:
                    count_mismatches.append(f"{spec.key}/{membership}/{target}")
                if spec.task_group == "tdi" and membership in ("train", "val"):
                    if set(frame[target].dropna().astype(int)) != {0, 1}:
                        degenerate_tdi.append(f"{spec.key}/{membership}/{target}")
            if len(frame) != int(entry["counts"][membership]):
                count_mismatches.append(f"{spec.key}/{membership}/rows")

        membership_sets = {
            membership: set(frame["Molecule_Name"].astype(str))
            for membership, frame in observed_frames.items()
        }
        if any(
            membership_sets[left] & membership_sets[right]
            for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
        ):
            overlap_mismatches.append(spec.key)
        connectivity_sets = {
            membership: {molecular_records[name].connectivity_key for name in names}
            for membership, names in membership_sets.items()
        }
        if any(
            connectivity_sets[left] & connectivity_sets[right]
            for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
        ):
            connectivity_overlap.append(spec.key)

    aggregates = [
        ("files.hashes", hash_mismatches),
        ("files.columns", column_mismatches),
        ("files.no_duplicate_names", duplicate_partitions),
        ("partitions.exact_ordered_membership", membership_mismatches),
        ("partitions.disjoint", overlap_mismatches),
        ("partitions.connectivity_disjoint", connectivity_overlap),
        ("chemistry.standardized_smiles", smiles_mismatches),
        ("labels.raw_parity", label_mismatches),
        ("manifest.counts", count_mismatches),
        ("tdi.train_validation_two_classes", degenerate_tdi),
    ]
    for check_id, mismatches in aggregates:
        add(
            check_id,
            not mismatches,
            f"mismatches={len(mismatches)}"
            + (f", first={mismatches[:3]}" if mismatches else ""),
        )

    add(
        "protocol.training_only",
        config["construction_data"] == "training_only"
        and config["test_structures_allowed"] is False,
        f"construction_data={config['construction_data']}, "
        f"test_structures_allowed={config['test_structures_allowed']}",
    )
    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "overall_pass": not any(check["status"] == "FAIL" for check in checks),
        "checks": checks,
        "summary": {
            "data_specs": len(specs),
            "jobs": len(jobs),
            "seeds": list(config["seeds"]),
            "partition_size_ranges": {
                membership: {
                    "min": int(min(sizes)),
                    "max": int(max(sizes)),
                }
                for membership, sizes in partition_sizes.items()
            },
        },
        "scientific_results_generated": False,
        "flagship_claim_authorized": False,
    }
    output_dir = root / config["paths"]["summary_root"]
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "neural_protocol_audit.json"
    report_path = output_dir / "NEURAL_PROTOCOL_AUDIT.md"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    report_path.write_text(_render_report(payload), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/neural_baselines.yaml")
    )
    args = parser.parse_args()
    payload = audit_neural_protocol(args.config)
    print(json.dumps(payload, indent=2))
    return 0 if payload["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
