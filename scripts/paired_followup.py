"""v5: 40 exact v4 replications + 10 fixed CYP2D6 task-weight pilot fits."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import pandas as pd
import torch

import paired_prototype as v4
from paired_audit import audit_latents, audit_raw_predictions, independent_validation_score, summarize
from paired_model import AssayModel, LABELS, fit, load_selected, predict, seed_everything, training_statistics
from paired_followup_model import fit_weighted

RUNTIME = ".runtime/paired-followup-v5"
CODE_FILES = (*v4.CODE_FILES, "configs/paired_followup.json", "scripts/paired_followup.py",
              "scripts/paired_followup_model.py", "scripts/paired_followup_diagnostics.py",
              "scripts/paired_followup_summary.py", "scripts/paired_followup_self_test.py",
              "scripts/start_paired_followup_4090.sh", "docs/PAIRED_FOLLOWUP.md",
              *("reports/paired_v4_training_diagnostics/" + name for name in (
                  "DIAGNOSTIC_AUDIT.json", "classification_train_val.csv", "observation_losses_train_val.csv",
                  "fixed_weight_std_sensitivity.csv", "training_gradient_diagnostics.csv", "selection_diagnostics.csv")))
sha, write_json, fingerprint = v4.sha, v4.write_json, v4.fingerprint


def job_matrix(config):
    result = []
    for seed in config["seeds"]:
        for fold in config["folds"]:
            for variant in config["variants"]:
                result.append({"condition": variant, "variant": variant, "fold": fold, "seed": seed,
                               "origin": "v4" if seed == config["reused_seed"] else "new", "multiplier": 1.0})
    for fold in config["folds"]:
        for variant in config["variants"]:
            result.append({"condition": variant + "_2d6x2", "variant": variant, "fold": fold,
                           "seed": config["intervention_seed"], "origin": "new",
                           "multiplier": config["tdi_2d6_loss_multiplier"]})
    for item in result:
        item["job_id"] = v4.job_id(item["condition"], item["fold"], item["seed"])
    return result


def validate_config(config, base):
    matrix = job_matrix(config)
    if (config["variants"] != ["independent_no_std", "paired_no_std"] or config["folds"] != list(range(5))
            or len(config["seeds"]) != 5 or len(set(config["seeds"])) != 5
            or config["seeds"][0] != config["reused_seed"] or config["intervention_seed"] != config["reused_seed"]
            or base["seeds"] != [config["reused_seed"]] or config["tdi_2d6_loss_multiplier"] != 2.0
            or config["threshold"] != .5 or config["threshold_tuning"] is not False
            or config["construction_data"] != "training_only" or config["test_structures_allowed"] is not False):
        raise RuntimeError("The registered v5 matrix/assay rules changed.")
    counts = {"expected_total_jobs": len(matrix), "expected_new_jobs": sum(x["origin"] == "new" for x in matrix),
              "expected_reused_jobs": sum(x["origin"] == "v4" for x in matrix)}
    if counts != {key: config[key] for key in counts} or len({x["job_id"] for x in matrix}) != len(matrix):
        raise RuntimeError("Registered v5 job counts mismatch.")


def checked_files(directory, files):
    for relative, expected in files.items():
        path = directory / relative
        if not path.resolve().is_relative_to(directory.resolve()) or not path.is_file() or sha(path) != expected:
            raise RuntimeError("Registered artifact changed or missing: " + str(path))


def verify_v4(directory, delivery, expected, *, engineering_only=False):
    """Verify all old outputs, all 20 jobs and original source; no edits/relabeling."""
    registration = json.loads((directory / "RUN_REGISTRATION.json").read_text())
    audit = json.loads((directory / "RESULT_AUDIT.json").read_text())
    if (fingerprint(registration["identity"]) != registration["fingerprint"] or registration["fingerprint"] != expected
            or audit.get("overall_pass") is not True or audit["registration"] != expected
            or audit.get("engineering_only") is not engineering_only
            or audit.get("gpu_training_qualified") is not (not engineering_only)
            or audit["summary"]["completed_jobs"] != 20):
        raise RuntimeError("The reviewed complete v4 run is required.")
    checked_files(delivery, registration["identity"]["code"])
    checked_files(directory, audit["outputs"])
    files = {"RUN_REGISTRATION.json": sha(directory / "RUN_REGISTRATION.json"),
             "RESULT_AUDIT.json": sha(directory / "RESULT_AUDIT.json"), **audit["outputs"]}
    expected_jobs = {v4.job_id(*job) for job in v4.jobs(registration["identity"]["config"])}
    if set(audit["jobs"]) != expected_jobs:
        raise RuntimeError("V4 completed job inventory mismatch.")
    for name in sorted(expected_jobs):
        job_dir = directory / "jobs" / name
        if sha(job_dir / "COMPLETE.json") != audit["jobs"][name]["complete_sha256"]:
            raise RuntimeError("V4 completion record changed: " + name)
        record = json.loads((job_dir / "COMPLETE.json").read_text())
        if record.get("complete") is not True or record["device"] != ("cpu" if engineering_only else "cuda:0"):
            raise RuntimeError("Unqualified v4 job: " + name)
        checked_files(job_dir, record["files"])
        prefix = "jobs/" + name + "/"
        files[prefix + "COMPLETE.json"] = sha(job_dir / "COMPLETE.json")
        files.update({prefix + name: digest for name, digest in record["files"].items()})
    if (directory / "graphs.pt").exists():
        graph = json.loads((directory / "GRAPH_CACHE.json").read_text())
        if sha(directory / "graphs.pt") != graph["graphs_sha256"]:
            raise RuntimeError("V4 graph cache changed.")
        files["graphs.pt"] = graph["graphs_sha256"]
    return registration, audit, files


def register(root, delivery, config, base):
    # Reuse existing version/GPU/frozen-v3 checks, with the unchanged v4 code.
    identity = v4.preflight(root, delivery, base)
    old_dir = root / v4.RUNTIME
    old_registration, _, old_files = verify_v4(old_dir, delivery, config["expected_v4_registration"])
    if old_registration["identity"]["config"] != base or old_registration["identity"]["environment"] != identity["environment"]:
        raise RuntimeError("Exact replication requires the reviewed v4 configuration and environment.")
    if v4.git(delivery, "status", "--porcelain", "--", *CODE_FILES):
        raise RuntimeError("Follow-up source/config has uncommitted changes.")
    identity.update(config=config, base_training_config=base, code={name: sha(delivery / name) for name in CODE_FILES},
                    v4_registration=old_registration["fingerprint"], v4_files=old_files)
    registration = {"identity": identity, "fingerprint": fingerprint(identity)}
    marker = root / RUNTIME / "RUN_REGISTRATION.json"
    if marker.exists() and json.loads(marker.read_text()) != registration:
        raise RuntimeError("V5 code/data/environment fingerprint changed; existing results are preserved.")
    if not marker.exists():
        write_json(marker, registration)
    return registration


def training_config(base, spec):
    # Replication jobs use the original configuration byte-for-byte as a mapping;
    # seed is already an explicit fit argument. Only the intervention adds a key.
    return dict(base) if spec["multiplier"] == 1 else {**base, "tdi_2d6_loss_multiplier": spec["multiplier"]}


def source_directory(root, spec):
    return root / (v4.RUNTIME if spec["origin"] == "v4" else RUNTIME) / "jobs" / spec["job_id"]


def run_job(root, delivery, config, base, registration, spec, graphs, table, raw, family, arrays, *, device="cuda:0"):
    directory = source_directory(root, spec)
    parts = v4.partitions(raw, family, spec["fold"])
    statistics = training_statistics(arrays, parts["train"])
    cfg = training_config(base, spec)
    complete = directory / "COMPLETE.json"
    ident = fingerprint({"registration": registration["fingerprint"], "job": spec, "statistics": statistics, "fit_config": cfg})
    if complete.exists():
        saved = json.loads(complete.read_text())
        checked_files(directory, saved["files"])
        if saved.get("complete") is not True or saved.get("device") != device:
            raise RuntimeError("Existing fit is incomplete or uses a different device.")
        if spec["origin"] == "new" and (saved["identity"] != ident or saved.get("spec") != spec):
            raise RuntimeError("Existing follow-up job identity changed.")
        if spec["origin"] == "v4":
            key = "jobs/" + spec["job_id"] + "/COMPLETE.json"
            if sha(complete) != registration["identity"]["v4_files"][key]:
                raise RuntimeError("Original v4 completion changed.")
        print("REUSE verified " + spec["job_id"], flush=True)
        return {"spec": spec, "record": saved, "complete_sha256": sha(complete)}
    if spec["origin"] == "v4":
        raise RuntimeError("An original completed job is missing; v5 never retrains/replaces it.")
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "TRAINING.json", {"identity": ident, "spec": spec, "statistics": statistics,
                                            "fit_config": cfg, "registration": registration["fingerprint"],
                                            "partitions": {key: len(value) for key, value in parts.items()}, "device": device})
    fitter = fit if spec["multiplier"] == 1 else fit_weighted
    model, state = fitter(cfg, spec["variant"], statistics, graphs, arrays, parts["train"], parts["val"], spec["seed"],
                          directory, ident, device)
    write_json(directory / "TRAINED.json", {"identity": ident, "best_sha256": sha(directory / "best.pt"),
                                           "resume_sha256": sha(directory / "resume.pt"), "best_epoch": state["best_epoch"]})
    audits = {}
    print("  AUDIT selected model, held-out row coverage, probability integral and fixed threshold...", flush=True)
    for partition in ("val", "test"):
        prediction = predict(model, graphs, parts[partition], cfg["batch_size"], device)
        v4.latent_frame(prediction, table, parts[partition]).to_csv(directory / (partition + "_latents.csv"), index=False)
        v4.prediction_frame(prediction, raw, table, parts[partition], spec["variant"], spec["fold"], spec["seed"]).to_csv(
            directory / (partition + "_predictions.csv"), index=False)
        latent = pd.read_csv(directory / (partition + "_latents.csv"))
        scored = pd.read_csv(directory / (partition + "_predictions.csv"))
        audit_raw_predictions(scored, latent, raw, family, parts[partition], spec["variant"], spec["fold"], spec["seed"])
        audits[partition] = audit_latents(latent, spec["variant"], base["quadrature_audit_atol"])
    selection = v4.audit_checkpoint(directory, ident, statistics, cfg, model, graphs, arrays, parts, table, device)
    independent = independent_validation_score(pd.read_csv(directory / "val_latents.csv"), raw,
                                               parts["train"], parts["val"], statistics)
    np.testing.assert_allclose(independent, selection["best_validation_score"], atol=2e-6, rtol=2e-6)
    v4.verify_sources(root, delivery, registration["identity"])
    filenames = ["TRAINING.json", "TRAINED.json", "best.pt", "resume.pt", "history.csv",
                 "val_latents.csv", "test_latents.csv", "val_predictions.csv", "test_predictions.csv"]
    saved = {"identity": ident, "spec": spec, "complete": True, "selection": selection, "prediction_audit": audits,
             "parameters": sum(p.numel() for p in model.parameters()), "device": device,
             "files": {name: sha(directory / name) for name in filenames}}
    write_json(complete, saved)
    del model, state
    torch.cuda.empty_cache()
    return {"spec": spec, "record": saved, "complete_sha256": sha(complete)}


def smoke_test(root, config, base, registration, raw, family, arrays, graphs, table, *, device="cuda:0"):
    output = root / RUNTIME / "smoke"
    marker = output / "PASS.json"
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved["registration"] != registration["fingerprint"]:
            raise RuntimeError("Smoke registration changed.")
        checked_files(output, saved["files"])
        if saved["overall_pass"] is not True or any(row["device"] != device for row in saved["results"]):
            raise RuntimeError("Smoke device/qualification changed.")
        return saved
    parts = v4.partitions(raw, family, 0)
    # Deterministic TRAIN-only fixture covering classes, endpoints and censoring.
    selected = set()
    for k in range(4):
        for a in range(2):
            for below in (False, True):
                mask = arrays["mask"][parts["train"], k, a] & ((arrays["y"][parts["train"], k, a] < 4) == below)
                selected.update(parts["train"][mask][:12].tolist())
    for k in range(2):
        for label in (0, 1):
            mask = arrays["label_mask"][parts["train"], k] & (arrays["labels"][parts["train"], k] == label)
            selected.update(parts["train"][mask][:16].tolist())
    train = np.array(sorted(selected), dtype=int)
    stats = training_statistics(arrays, train)
    results, files = [], {}
    for multiplier in (1, 2):
        for variant in config["variants"]:
            condition = variant + ("_2d6x2" if multiplier == 2 else "")
            small = {**training_config(base, {"multiplier": multiplier}), "max_epochs": 3, "patience": 3}
            fitter = fit if multiplier == 1 else fit_weighted
            ident = fingerprint({"registration": registration["fingerprint"], "smoke": condition})
            directory = output / condition
            print(f"REAL {'GPU' if device.startswith('cuda') else 'CPU FIXTURE'} SMOKE {condition}: three epochs, train/val only.", flush=True)
            model, state = fitter(small, variant, stats, graphs, arrays, train, parts["val"], config["reused_seed"], directory, ident, device)
            fitter(small, variant, stats, graphs, arrays, train, parts["val"], config["reused_seed"], directory / "restart", ident, device, stop_after_epoch=0)
            resumed, rs = fitter(small, variant, stats, graphs, arrays, train, parts["val"], config["reused_seed"], directory / "restart", ident, device)
            if state["best_epoch"] != rs["best_epoch"] or state["best_score"] != rs["best_score"]:
                raise AssertionError("GPU restart changed selection.")
            for key in state["model"]:
                if not torch.equal(state["model"][key], rs["model"][key]):
                    raise AssertionError("GPU restart changed final weights.")
            loaded, _ = load_selected(directory / "best.pt", ident, device)
            before = predict(model, graphs, parts["val"], small["batch_size"], device)
            after = predict(loaded, graphs, parts["val"], small["batch_size"], device)
            for key in before:
                np.testing.assert_allclose(before[key], after[key], atol=1e-7, rtol=1e-6)
            frame = v4.latent_frame(after, table, parts["val"])
            frame.to_csv(directory / "validation_latents.csv", index=False)
            audit = audit_latents(pd.read_csv(directory / "validation_latents.csv"), variant, small["quadrature_audit_atol"])
            results.append({"condition": condition, "device": device, "epochs": len(state["history"]),
                            "restart_final_weights_bitwise_equal": True, **audit})
            for name in ("best.pt", "resume.pt", "history.csv", "validation_latents.csv", "restart/best.pt", "restart/resume.pt"):
                files[condition + "/" + name] = sha(directory / name)
            del model, state, resumed, rs, loaded
            torch.cuda.empty_cache()
    result = {"overall_pass": True, "registration": registration["fingerprint"], "results": results, "files": files}
    write_json(marker, result)
    return result


def finish(root, delivery, config, base, registration, raw, family, arrays, table, records, *, engineering_only=False):
    from paired_followup_summary import summarize_followup
    output, old = root / RUNTIME, root / v4.RUNTIME
    required_device = "cpu" if engineering_only else "cuda:0"
    cpu = json.loads((output / "CPU_SELF_TEST.json").read_text())
    smoke = json.loads((output / "smoke/PASS.json").read_text())
    expected_conditions = {x["condition"] for x in job_matrix(config)}
    if (cpu.get("overall_pass") is not True or smoke.get("overall_pass") is not True
            or smoke["registration"] != registration["fingerprint"]
            or len(smoke["results"]) != 4
            or {r["condition"] for r in smoke["results"]} != expected_conditions
            or any(r["device"] != required_device for r in smoke["results"])
            or any(r["record"]["device"] != required_device for r in records)):
        raise RuntimeError("Self-test/job device or registration mismatch.")
    checked_files(output / "smoke", smoke["files"])
    if [r["spec"] for r in records] != job_matrix(config):
        raise RuntimeError("Missing, duplicate or reordered final job inventory.")
    checked_files(old, registration["identity"]["v4_files"])
    frames, ledger = [], []
    for result in records:
        spec, record = result["spec"], result["record"]
        directory = source_directory(root, spec)
        if sha(directory / "COMPLETE.json") != result["complete_sha256"]:
            raise RuntimeError("Completion record changed during final audit.")
        checked_files(directory, record["files"])
        prediction = pd.read_csv(directory / "test_predictions.csv")
        latent = pd.read_csv(directory / "test_latents.csv")
        audit_raw_predictions(prediction, latent, raw, family, v4.partitions(raw, family, spec["fold"])["test"],
                              spec["variant"], spec["fold"], spec["seed"])
        # Explicit aggregation aliases; original v4 files retain their exact bytes.
        prediction["condition"], prediction["origin"] = spec["condition"], spec["origin"]
        prediction["model"] = "v5_" + spec["condition"]
        frames.append(prediction)
        ledger.append({**spec, "source_runtime": v4.RUNTIME if spec["origin"] == "v4" else RUNTIME,
                       "complete_sha256": result["complete_sha256"], "parameters": record["parameters"],
                       "identity": record["identity"], **record["selection"]})
    if max(row["parameters"] for row in ledger) / min(row["parameters"] for row in ledger) > 1.01:
        raise RuntimeError("Matched active parameter budgets differ by more than 1%.")
    pred = pd.concat(frames, ignore_index=True)
    if len(pred) != config["expected_prediction_rows"] or pred.duplicated(["model", "seed", "Molecule_Name", "endpoint"]).any():
        raise RuntimeError("Follow-up prediction coverage mismatch.")
    metadata = ["Molecule_Name", "endpoint", "fold", "family_id", "y_true", "y_true_lower", "y_true_upper"]
    reference = pd.read_csv(old / "predictions.csv.gz")
    reference = reference[reference.variant.eq("independent_no_std")][metadata].sort_values(metadata[:2]).reset_index(drop=True)
    for _, block in pred.groupby(["condition", "seed"]):
        pd.testing.assert_frame_equal(block[metadata].sort_values(metadata[:2]).reset_index(drop=True), reference,
                                      check_dtype=False, atol=1e-10, rtol=1e-10)
    pred.to_csv(output / "predictions.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    roundtrip = pd.read_csv(output / "predictions.csv.gz")
    pd.testing.assert_frame_equal(pred, roundtrip, check_dtype=False, atol=1e-10, rtol=1e-10)
    metrics, diagnostics = summarize(roundtrip, root / "vendor/openadmet/custom_scoring_functions.py")
    if len(metrics) != config["expected_metric_rows"] or len(diagnostics) != config["expected_diagnostic_rows"]:
        raise RuntimeError("Follow-up metric/diagnostic counts mismatch.")
    metrics.to_csv(output / "metrics.csv", index=False)
    diagnostics.to_csv(output / "tdi_diagnostics.csv", index=False)
    pd.DataFrame(ledger).to_csv(output / "job_ledger.csv", index=False)
    summary_files = summarize_followup(roundtrip, metrics, family, config, output)
    diagnosis = json.loads((output / "diagnostics/DIAGNOSTIC_AUDIT.json").read_text())
    if (diagnosis.get("complete") is not True or diagnosis["v4_registration"] != registration["identity"]["v4_registration"]
            or diagnosis["partitions_evaluated"] != ["train", "val"] or diagnosis["outer_predictions_used"] is not False
            or diagnosis["device"] != required_device or diagnosis["threshold_tuned"] is not False):
        raise RuntimeError("Training/validation diagnostic provenance mismatch.")
    checked_files(output / "diagnostics", diagnosis["outputs"])
    diagnostic_files = ["diagnostics/DIAGNOSTIC_AUDIT.json", *("diagnostics/" + name for name in diagnosis["outputs"])]
    files = ["predictions.csv.gz", "metrics.csv", "tdi_diagnostics.csv", "job_ledger.csv", "CPU_SELF_TEST.json",
             "smoke/PASS.json", "molecules.csv", *summary_files, *diagnostic_files]
    v4.verify_sources(root, delivery, registration["identity"])
    checked_files(old, registration["identity"]["v4_files"])
    audit = {"overall_pass": True, "protocol_id": config["protocol_id"], "registration": registration["fingerprint"],
             "summary": {"new_jobs": config["expected_new_jobs"], "reused_v4_jobs": config["expected_reused_jobs"],
                         "total_jobs": len(records), "preserved_original_jobs": 20, "prediction_rows": len(pred),
                         "metric_rows": len(metrics), "diagnostic_rows": len(diagnostics), "seeds": config["seeds"]},
             "checks": ["all_v4_artifacts_unchanged", "registered_matrix_and_no_duplicate_seeds", "training_only_diagnostics",
                        "real_CPU_training_and_weighting_self_test", "real_training_restart_and_saved_model_smoke",
                        "validation_only_original_selection", "saved_model_replay_and_probability_quadrature",
                        "raw_row_and_old_v4_coverage_parity", "official_metrics_independently_recomputed",
                        "fixed_half_threshold", "per_seed_results_and_all_declared_contrasts", "within_fold_family_bootstrap"],
             "outputs": {name: sha(output / name) for name in files},
             "jobs": {r["spec"]["job_id"]: {"origin": r["spec"]["origin"], "complete_sha256": r["complete_sha256"]} for r in records},
             "engineering_only": engineering_only, "scientific_results_generated": not engineering_only,
             "gpu_training_qualified": not engineering_only, "outer_v4_results_seen": True,
             "intervention_pilot_seeds": [config["intervention_seed"]], "confirmatory_inference": False,
             "independent_blind_validation_completed": False, "flagship_claim_authorized": False}
    write_json(output / "RESULT_AUDIT.json", audit)
    target_dir = output if engineering_only else delivery
    target_name = "CYP_paired_v5_engineering_test" if engineering_only else "CYP_paired_v5_review"
    target = target_dir / (target_name + ".zip")
    if target.exists():
        target = target_dir / (target_name + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".zip")
    # Zip is atomic: an interrupted archive cannot look like a completed review.
    temporary = target.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in files + ["RESULT_AUDIT.json", "RUN_REGISTRATION.json"]:
            archive.write(output / name, name)
        for name in CODE_FILES:
            archive.write(delivery / name, "source/" + name)
        for result in records:
            if result["spec"]["origin"] == "new":
                directory = source_directory(root, result["spec"])
                for name in ["COMPLETE.json", *(name for name in result["record"]["files"] if name != "resume.pt")]:
                    archive.write(directory / name, "jobs/" + result["spec"]["job_id"] + "/" + name)
        # Include ALL original conditions, including the unsuccessful full model.
        for name in registration["identity"]["v4_files"]:
            if name == "graphs.pt" or name.endswith("/resume.pt"):
                continue
            archive.write(old / name, "v4_preserved/" + name)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None or len(archive.namelist()) != len(set(archive.namelist())):
            raise RuntimeError("Review archive CRC/path uniqueness failed.")
    temporary.replace(target)
    write_json(output / "EXPORT.json", {"path": str(target), "sha256": sha(target), "registration": registration["fingerprint"]})
    if engineering_only:
        print("ENGINEERING_PIPELINE_PASS: synthetic CPU fixture; no scientific results.", flush=True)
    else:
        print("PASS: all 50 new jobs, 10 verified v4 jobs, diagnostics and independent result audit completed.", flush=True)
    print("Review ZIP: " + str(target), flush=True)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--delivery", type=Path, required=True)
    args = parser.parse_args()
    root, delivery = args.root.resolve(), args.delivery.resolve()
    config = json.loads((delivery / "configs/paired_followup.json").read_text())
    base = json.loads((delivery / config["base_config_file"]).read_text())
    validate_config(config, base)
    registration = register(root, delivery, config, base)
    output, old = root / RUNTIME, root / v4.RUNTIME
    print("PREFLIGHT_PASS: unchanged complete v4, frozen v3/data, exact pinned environment and RTX 4090.", flush=True)
    from paired_followup_self_test import self_test
    self_test(output / "cpu_self_test", delivery / "configs/paired_prototype.json", output / "CPU_SELF_TEST.json")
    seed_everything(config["reused_seed"], base["cpu_threads"])
    raw, family, arrays = v4.read_inputs(root, base)
    cache_dir = old if all((old / name).is_file() for name in ("GRAPH_CACHE.json", "graphs.pt", "molecules.csv")) else output / "cache"
    graphs, table = v4.graph_cache(root, base, raw, family, cache_dir)
    table.to_csv(output / "molecules.csv", index=False)
    smoke_test(root, config, base, registration, raw, family, arrays, graphs, table)
    from paired_followup_diagnostics import run_diagnostics
    marker = output / "diagnostics/DIAGNOSTIC_AUDIT.json"
    if marker.exists():
        prior = json.loads(marker.read_text())
        checked_files(output / "diagnostics", prior["outputs"])
        if prior["v4_registration"] != config["expected_v4_registration"] or prior["complete"] is not True:
            raise RuntimeError("Existing diagnostic provenance changed.")
        print("REUSE verified training/validation diagnostics.", flush=True)
    else:
        run_diagnostics(old, output / "diagnostics", raw, family, arrays, graphs, device="cuda:0",
                        gradient_sample=config["diagnostic_gradient_sample"])
    records = []
    for number, spec in enumerate(job_matrix(config), 1):
        print(f"[{number}/{config['expected_total_jobs']}] {spec['origin'].upper()} {spec['job_id']}", flush=True)
        v4.verify_sources(root, delivery, registration["identity"])
        records.append(run_job(root, delivery, config, base, registration, spec, graphs, table, raw, family, arrays))
    print("All fits complete; auditing all seeds, fixed contrasts and preserved v4 results...", flush=True)
    finish(root, delivery, config, base, registration, raw, family, arrays, table, records)


if __name__ == "__main__":
    main()
