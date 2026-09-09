"""Every registered v5 contrast, seed variability and conditional family bootstrap."""
from __future__ import annotations

import numpy as np
import pandas as pd

ENDPOINTS = ("MA", "CYP3A4_is_TDI", "CYP2D6_is_TDI")


def contrasts(config):
    return [("pairing_no_std", "paired_no_std", "independent_no_std", config["seeds"]),
            ("2d6_weight_in_independent", "independent_no_std_2d6x2", "independent_no_std", [config["intervention_seed"]]),
            ("2d6_weight_in_paired", "paired_no_std_2d6x2", "paired_no_std", [config["intervention_seed"]]),
            ("pairing_with_2d6_weight", "paired_no_std_2d6x2", "independent_no_std_2d6x2", [config["intervention_seed"]])]


def family_samples(predictions, family, config):
    """Resample whole families within each fold; same draws for every fitted seed.

    This is conditional on the fitted weights. Seeds are not extra molecules,
    families are not independent of adaptive model choice, and no fits are rerun.
    """
    fam = family[["family_id", "outer_fold"]].drop_duplicates().sort_values(["outer_fold", "family_id"]).reset_index(drop=True)
    if fam.family_id.duplicated().any():
        raise AssertionError("A bootstrap family spans outer folds.")
    count = config["family_bootstrap_draws"]
    rng = np.random.default_rng(config["family_bootstrap_seed"])
    weights = np.zeros((count, len(fam)), dtype=float)
    for _, group in fam.groupby("outer_fold", sort=True):
        ix = group.index.to_numpy()
        chosen = rng.choice(ix, size=(count, len(ix)), replace=True)
        np.add.at(weights, (np.arange(count)[:, None], chosen), 1)
    lookup = {name: i for i, name in enumerate(fam.family_id)}
    samples = {}
    regression_cache = {}
    for (model, seed), group in predictions.groupby(["model", "seed"], sort=True):
        macro = np.zeros(count)
        for endpoint, block in group.groupby("endpoint", sort=True):
            block = block.sort_values("Molecule_Name").reset_index(drop=True)
            if block.task.iloc[0] == "regression":
                if endpoint not in regression_cache:
                    weight = weights[:, block.family_id.map(lookup).to_numpy()]
                    y, lo, hi = [block[key].to_numpy(float) for key in ("y_true", "y_true_lower", "y_true_upper")]
                    size = weight.sum(1)
                    mean = np.divide(weight @ y, size, out=np.full(count, np.nan), where=size > 0)
                    denominator = (weight * (np.maximum(mean[:, None] - hi, 0) + np.maximum(lo - mean[:, None], 0))).sum(1)
                    regression_cache[endpoint] = (block.Molecule_Name.tolist(), weight, lo, hi, denominator)
                names, weight, lo, hi, denominator = regression_cache[endpoint]
                if block.Molecule_Name.tolist() != names:
                    raise AssertionError("Bootstrap regression row alignment changed.")
                p = block.y_pred.to_numpy(float)
                numerator = weight @ (np.maximum(p - hi, 0) + np.maximum(lo - p, 0))
                macro += np.divide(numerator, denominator, out=np.full(count, np.nan), where=denominator > 0) / 4
            else:
                for name, y, p in (("TP", 1, 1), ("TN", 0, 0), ("FP", 0, 1), ("FN", 1, 0)):
                    block[name] = ((block.y_true == y) & (block.y_pred == p)).astype(int)
                counts = block.groupby("family_id")[["TP", "TN", "FP", "FN"]].sum().reindex(fam.family_id, fill_value=0).to_numpy()
                tp, tn, fp, fn = (weights @ counts).T
                denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
                samples[(model, int(seed), endpoint)] = np.divide(tp * tn - fp * fn, denominator,
                                                                                out=np.zeros(count), where=denominator > 0)
        samples[(model, int(seed), "MA")] = macro
    return samples, len(fam)


def summarize_followup(predictions, metrics, family, config, output):
    primary = metrics[metrics.scope.eq("pooled") & metrics.endpoint.isin(ENDPOINTS)].copy()
    primary.to_csv(output / "primary_per_seed.csv", index=False)
    if primary.duplicated(["model", "seed", "endpoint"]).any():
        raise AssertionError("Duplicate pooled seed result.")
    seed_rows = []
    for (model, endpoint), group in primary.groupby(["model", "endpoint"], sort=True):
        scopes = [("all_registered_seeds", group)]
        if model in ("v5_independent_no_std", "v5_paired_no_std"):
            scopes.append(("additional_four_seeds", group[~group.seed.eq(config["reused_seed"])]))
        for scope, subset in scopes:
            seed_rows.append({"model": model, "endpoint": endpoint, "seed_scope": scope, "n_seeds": len(subset),
                              "mean": subset.value.mean(), "seed_sd": subset.value.std(ddof=1),
                              "minimum": subset.value.min(), "maximum": subset.value.max(),
                              "interpretation": "same frozen molecules across seeds; SD is not a confidence interval"})
    pd.DataFrame(seed_rows).to_csv(output / "seed_summary.csv", index=False)
    lookup = primary.set_index(["model", "seed", "endpoint"]).value
    rows = []
    for question, treatment, control, seeds in contrasts(config):
        for seed in seeds:
            for endpoint in ENDPOINTS:
                left, right = lookup.loc[("v5_" + treatment, seed, endpoint)], lookup.loc[("v5_" + control, seed, endpoint)]
                rows.append({"question": question, "treatment": treatment, "control": control, "seed": seed,
                             "endpoint": endpoint, "treatment_value": left, "control_value": right,
                             "raw_difference": left - right, "benefit_direction_difference": right - left if endpoint == "MA" else left - right})
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(output / "contrasts_per_seed.csv", index=False)
    samples, families = family_samples(predictions, family, config)
    interval_rows = []
    for question, treatment, control, seeds in contrasts(config):
        scopes = [("all_registered_seeds", seeds)]
        if question == "pairing_no_std":
            scopes.append(("additional_four_seeds", [s for s in seeds if s != config["reused_seed"]]))
        for scope, selected in scopes:
            for endpoint in ENDPOINTS:
                empirical = per_seed[per_seed.question.eq(question) & per_seed.endpoint.eq(endpoint) & per_seed.seed.isin(selected)]
                differences = [samples[("v5_" + treatment, seed, endpoint)] - samples[("v5_" + control, seed, endpoint)] for seed in selected]
                benefit = np.mean(differences, axis=0) * (-1 if endpoint == "MA" else 1)
                valid = np.isfinite(benefit)
                if valid.sum() < .95 * config["family_bootstrap_draws"]:
                    raise RuntimeError("Too many undefined family bootstrap regression denominators.")
                low, high = np.quantile(benefit[valid], [.025, .975])
                interval_rows.append({"question": question, "endpoint": endpoint, "treatment": treatment, "control": control,
                                      "seed_scope": scope, "n_seeds": len(selected), "seeds_with_positive_effect": int((empirical.benefit_direction_difference > 0).sum()),
                                      "mean_benefit": float(empirical.benefit_direction_difference.mean()),
                                      "seed_sd_of_benefit": float(empirical.benefit_direction_difference.std(ddof=1)),
                                      "family_bootstrap_low": float(low), "family_bootstrap_high": float(high),
                                      "draws": config["family_bootstrap_draws"], "defined_draws": int(valid.sum()), "families": families,
                                      "conditional_on_fitted_weights": True, "confirmatory_inference": False})
    pd.DataFrame(interval_rows).to_csv(output / "contrast_summary.csv", index=False)
    return ["primary_per_seed.csv", "seed_summary.csv", "contrasts_per_seed.csv", "contrast_summary.csv"]
