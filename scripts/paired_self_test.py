"""Actual CPU numerical/gradient/training/restart tests, before any GPU pilot job."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import torch

from paired_audit import normal_rule_probability, audit_latents, audit_raw_predictions, independent_validation_score
from paired_model import (AssayModel, DIRECT, ISOS, LABELS, TREATED, arrays_from_raw,
                          assay_probability, batch_graphs, fit, load_selected, molecular_graph,
                          observation_nll, predict, seed_everything, tensor_targets,
                          training_loss, training_statistics)


def toy_data():
    """Synthetic engineering fixture only; never enters scientific training."""
    rng = np.random.default_rng(219)
    smiles = ["CCO", "CCN", "CC(=O)O", "c1ccccc1", "CCOC", "CC(C)N", "C1CCCCC1", "CCCl",
              "CC(=O)N", "CCS", "C[C@H](O)C(=O)O", "[Na+]"]
    n = 80
    raw = pd.DataFrame({"Molecule_Name": [f"toy-{i}" for i in range(n)], "SMILES": [smiles[i % len(smiles)] for i in range(n)]})
    for k, (direct, treated) in enumerate(zip(DIRECT, TREATED, strict=True)):
        d = 4.0 + 0.2 * (np.arange(n) % 12) + rng.normal(0, 0.1, n)
        d[::7] = 3.3
        t = d + np.where(np.arange(n) % 3 == 0, 0.9, -0.2)
        d[np.arange(n) % 13 == k] = np.nan
        t[np.arange(n) % 17 == k] = np.nan
        for name, value in ((direct, d), (treated, t)):
            raw[name] = value
            raw[name + "_std"] = np.where(np.isfinite(value), 0.1 + 0.02 * (np.arange(n) % 5), np.nan)
            raw[name + "_conf_low"] = value - 0.2
            raw[name + "_conf_high"] = value + 0.2
    for iso, label in zip(("CYP3A4", "CYP2D6"), LABELS, strict=True):
        d, t = raw[iso + "_pIC50_direct_inhibition"], raw[iso + "_pIC50_TDI_condition"]
        raw[label] = np.where(np.arange(n) % 19 == 0, np.nan,
                              (d.notna() & t.notna() & (t > np.maximum(d, 4) + np.log10(2))).astype(float))
    family = pd.DataFrame({"Molecule_Name": raw.Molecule_Name, "family_id": [f"toy-family-{i // 4}" for i in range(n)],
                           "outer_fold": [i // 16 for i in range(n)]})
    table = raw[["Molecule_Name", "SMILES"]].merge(family, on="Molecule_Name")
    graphs = [molecular_graph(s) for s in raw.SMILES]
    return raw, family, table, graphs, arrays_from_raw(raw)


def self_test(output: Path, config_path: Path, result_path: Path | None = None) -> dict:
    from paired_prototype import latent_frame, prediction_frame, audit_checkpoint, write_json
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text())
    small = {**config, "hidden_dim": 24, "max_epochs": 3, "patience": 3, "batch_size": 16, "cpu_threads": 1}
    seed_everything(7, 1)
    nodes, weights = np.polynomial.legendre.leggauss(config["quadrature_nodes"])
    nodes, weights = torch.tensor((nodes + 1) / 2), torch.tensor(weights / 2)
    rng = np.random.default_rng(79)
    md = rng.uniform(2, 8, 160)
    mt = rng.uniform(1, 9, (160, 3))
    sd = rng.uniform(0.05, 2, 160)
    st = rng.uniform(0.05, 2, 160)
    # Include the sharpest permitted scale ratio at the floor boundary.
    md[:4], sd[:4], st[:4] = [4, 4, 7, 2], [2, .05, 2, 2], [.05, 2, .05, .05]
    mix = rng.dirichlet([1, 1, 1], 160)
    actual = assay_probability(*[torch.tensor(x) for x in (md, mt, sd, st, mix)], nodes, weights).numpy()
    expected = np.array([sum(mix[i, k] * normal_rule_probability(md[i], mt[i, k], sd[i], st[i]) for k in range(3)) for i in range(160)])
    error = float(np.max(np.abs(actual - expected)))
    np.testing.assert_allclose(actual, expected, atol=config["quadrature_audit_atol"], rtol=0)
    inputs = (torch.tensor([4.1, 5.2], dtype=torch.double, requires_grad=True),
              torch.tensor([[4.1, 4.7, 3.6], [5.2, 5.8, 4.7]], dtype=torch.double, requires_grad=True),
              torch.tensor([.4, .2], dtype=torch.double, requires_grad=True),
              torch.tensor([.3, .6], dtype=torch.double, requires_grad=True))
    fixed_mix = torch.tensor([[.6, .2, .2], [.6, .2, .2]], dtype=torch.double)
    if not torch.autograd.gradcheck(lambda *x: assay_probability(*x, fixed_mix, nodes, weights), inputs, eps=1e-6, atol=2e-5, rtol=2e-4):
        raise AssertionError("Probability gradients fail finite differences.")
    raw, family, table, graphs, arrays = toy_data()
    parts = {"train": np.arange(48), "val": np.arange(48, 64), "test": np.arange(64, 80)}
    stats = training_statistics(arrays, parts["train"])
    # Training statistics must not depend on validation/test values or uncertainty.
    changed = {k: v.copy() for k, v in arrays.items()}
    changed["y"][48:] = 999
    changed["std"][48:] = 999
    changed["labels"][48:] = 1
    if training_statistics(changed, parts["train"]) != stats:
        raise AssertionError("Held-out data changed training statistics.")
    trained = []
    with tempfile.TemporaryDirectory(prefix="actual-", dir=output) as temporary:
        temporary = Path(temporary)
        for variant in config["variants"]:
            seed_everything(11, 1)
            model = AssayModel(small, variant, stats)
            target = tensor_targets(arrays, np.arange(16), "cpu")
            model.eval()
            result = model(batch_graphs(graphs[:16], "cpu"))
            loss = training_loss(result, target, model, stats)
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise AssertionError("Missing or non-finite parameter gradient: " + variant)
            if not any(torch.count_nonzero(p.grad) for p in model.encoder.parameters()):
                raise AssertionError("Molecular encoder receives no gradient.")
            zeroed = {k: v.clone() for k, v in target.items()}
            zeroed["y"][~zeroed["mask"]] = 1e20
            zeroed["std"][~zeroed["mask"]] = 1e10
            zeroed["labels"][~zeroed["label_mask"]] = 0
            torch.testing.assert_close(training_loss(result, target, model, stats), training_loss(result, zeroed, model, stats))
            censored = {k: v.clone() for k, v in target.items()}
            censored["y"][(censored["y"] < 4) & censored["mask"]] = -100
            torch.testing.assert_close(observation_nll(result, target, model.use_std, 4),
                                       observation_nll(result, censored, model.use_std, 4))
            if not model.use_std:
                altered = {k: v.clone() for k, v in target.items()}
                altered["std"] *= 100
                torch.testing.assert_close(observation_nll(result, target, False, 4), observation_nll(result, altered, False, 4))
            del model
            whole = temporary / (variant + "-whole")
            resume = temporary / (variant + "-resume")
            model, state = fit(small, variant, stats, graphs, arrays, parts["train"], parts["val"], 11, whole, "self-test", "cpu")
            fit(small, variant, stats, graphs, arrays, parts["train"], parts["val"], 11, resume, "self-test", "cpu", stop_after_epoch=0)
            resumed, resumed_state = fit(small, variant, stats, graphs, arrays, parts["train"], parts["val"], 11, resume, "self-test", "cpu")
            final_file = torch.load(whole / "resume.pt", map_location="cpu", weights_only=False)
            for key in state["model"]:
                torch.testing.assert_close(state["model"][key], final_file["model"][key], atol=0, rtol=0)
                torch.testing.assert_close(state["model"][key], resumed_state["model"][key], atol=0, rtol=0)
            for key in ("best_epoch", "best_score"):
                if state[key] != resumed_state[key]:
                    raise AssertionError("Resume changed selected checkpoint.")
            fresh, _ = load_selected(whole / "best.pt", "self-test", "cpu")
            for partition in ("val", "test"):
                a = predict(model, graphs, parts[partition], small["batch_size"], "cpu")
                b = predict(fresh, graphs, parts[partition], small["batch_size"], "cpu")
                for key in a:
                    np.testing.assert_array_equal(a[key], b[key])
                latent = latent_frame(b, table, parts[partition])
                prediction = prediction_frame(b, raw, table, parts[partition], variant, 0, 11)
                latent.to_csv(whole / (partition + "_latents.csv"), index=False)
                prediction.to_csv(whole / (partition + "_predictions.csv"), index=False)
                reread = pd.read_csv(whole / (partition + "_latents.csv"))
                audit_latents(reread, variant, config["quadrature_audit_atol"])
                audit_raw_predictions(pd.read_csv(whole / (partition + "_predictions.csv")), reread, raw, family, parts[partition], variant, 0, 11)
            audit_checkpoint(whole, "self-test", stats, small, model, graphs, arrays, parts, table, "cpu")
            independent = independent_validation_score(pd.read_csv(whole / "val_latents.csv"), raw,
                                                        parts["train"], parts["val"], stats)
            np.testing.assert_allclose(independent, state["best_score"], atol=2e-6, rtol=2e-6)
            final_differs = any(not torch.equal(state["model"][k], state["best_state"][k]) for k in state["model"])
            if variant.startswith("paired_") and (state["best_epoch"] != 0 or not final_differs):
                raise AssertionError("Fixture must exercise selected weights different from final weights.")
            if variant.startswith("paired_"):
                early = {**small, "max_epochs": 6, "patience": 1}
                _, stopped = fit(early, variant, stats, graphs, arrays, parts["train"], parts["val"], 11,
                                 temporary / (variant + "-early-stop"), "early-stop-test", "cpu")
                if len(stopped["history"]) != 2 or stopped["best_epoch"] != 0:
                    raise AssertionError("Real training did not stop when patience was exhausted.")
            # Batch composition must not change a molecule's deterministic output.
            p1 = predict(fresh, graphs, parts["test"], 1, "cpu")
            p16 = predict(fresh, graphs, parts["test"], 16, "cpu")
            np.testing.assert_allclose(p1["d"], p16["d"], atol=2e-6, rtol=1e-6)
            np.testing.assert_allclose(p1["probability"], p16["probability"], atol=2e-6, rtol=1e-6)
            reordered = predict(fresh, [molecular_graph("CCO"), molecular_graph("OCC")], np.arange(2), 2, "cpu")
            np.testing.assert_allclose(reordered["d"][0], reordered["d"][1], atol=2e-6, rtol=1e-6)
            trained.append({"variant": variant, "epochs": len(state["history"]), "best_epoch": state["best_epoch"],
                            "restart_final_weights_bitwise_equal": True, "export_reload_replay_pass": True,
                            "selected_weights_differ_from_final": final_differs})
    result = {"overall_pass": True, "torch": torch.__version__, "device": "cpu", "synthetic_data_only": True,
              "adaptive_quadrature_max_error": error, "finite_difference_gradient_pass": True,
              "missing_target_and_censoring_pass": True, "held_out_statistics_isolation_pass": True,
              "actual_early_stopping_pass": True,
              "variants": trained}
    if result_path:
        write_json(result_path, result)
    print("PAIRED_CPU_SELF_TEST_PASS " + json.dumps(result, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/paired_prototype.json")
    args = parser.parse_args()
    self_test(args.output, args.config, args.output / "SELF_TEST.json")
