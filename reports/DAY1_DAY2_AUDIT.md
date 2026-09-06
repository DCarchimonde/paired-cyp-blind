# Day 1–2 Data and Metric Audit

**Overall status: PASS**  
Generated: 2026-08-28T17:27:43.938763+00:00

No model performance is reported at this stage. This audit only validates the research substrate.

## Official data inventory

| Role | Rows | Columns | Unique molecules |
|---|---:|---:|---:|
| direct | 4905 | 18 | 4905 |
| blinded_test | 750 | 2 | 750 |
| tdi | 6145 | 36 | 6145 |
| single_concentration | 17504 | 12 | 4376 |
| emax | 6145 | 30 | 6145 |

## TDI label balance and reproduction

| Isoform | Labeled | Positive | Negative | Positive rate | Reproduction |
|---|---:|---:|---:|---:|---:|
| CYP3A4 | 3584 | 764 | 2820 | 0.213 | 1.000 |
| CYP2D6 | 1497 | 324 | 1173 | 0.216 | 1.000 |

## Frozen challenge-mimetic split

- Held-out molecule rows: 750
- Held-out analog connectivity units: 750
- Analog families: 75
- Outer folds: 5
- Per-fold connectivity units: {'0': 150, '1': 150, '2': 150, '3': 150, '4': 150}
- Per-fold molecule rows: {'0': 150, '1': 150, '2': 150, '3': 150, '4': 150}
- Similarity range (unit minima): 0.333–0.793; median 0.401
- Minimum selected-anchor pIC50 by isoform: {'CYP1A2': 5.595434417, 'CYP2C9': 5.2200345, 'CYP3A4': 5.66007574}

## Checks

| Status | Critical | Check | Detail |
|---|---|---|---|
| PASS | yes | `rows.direct` | observed=4905, expected=4905 |
| PASS | yes | `schema.direct` | missing=[] |
| PASS | yes | `unique_ids.direct` | duplicate rows=0 |
| PASS | yes | `rows.blinded_test` | observed=750, expected=750 |
| PASS | yes | `schema.blinded_test` | missing=[] |
| PASS | yes | `unique_ids.blinded_test` | duplicate rows=0 |
| PASS | yes | `rows.tdi` | observed=6145, expected=6145 |
| PASS | yes | `schema.tdi` | missing=[] |
| PASS | yes | `unique_ids.tdi` | duplicate rows=0 |
| PASS | yes | `rows.single_concentration` | observed=17504, expected=17504 |
| PASS | yes | `schema.single_concentration` | missing=[] |
| PASS | yes | `rows.emax` | observed=6145, expected=6145 |
| PASS | yes | `schema.emax` | missing=[] |
| PASS | yes | `unique_ids.emax` | duplicate rows=0 |
| PASS | yes | `manifest.cyp-challenge-TEST-BLINDED.csv` | sha256=a342f8444a8dcb531ca12f3685293f0bd6c36ae9073f491e44a9bc1cc4b741f9, bytes=44935; expected sha256=a342f8444a8dcb531ca12f3685293f0bd6c36ae9073f491e44a9bc1cc4b741f9, bytes=44935 |
| PASS | yes | `manifest.cyp-challenge-TRAIN_Emax.csv` | sha256=482f686a9a9f9166f290e6f5ea463a99de1da478b179a88f4baef48bc66501f1, bytes=1130831; expected sha256=482f686a9a9f9166f290e6f5ea463a99de1da478b179a88f4baef48bc66501f1, bytes=1130831 |
| PASS | yes | `manifest.cyp-challenge-TRAIN_TDI.csv` | sha256=b458f599a792412292664386e8f18adc5d4a4129d6bd212ae80a60fb9b96bb60, bytes=1218524; expected sha256=b458f599a792412292664386e8f18adc5d4a4129d6bd212ae80a60fb9b96bb60, bytes=1218524 |
| PASS | yes | `manifest.cyp-challenge-TRAIN_inhibition.csv` | sha256=b8f79addd266fb6f9f4c222c5e4e73d926362328b6a8d2841871a54e46bd2278, bytes=653407; expected sha256=b8f79addd266fb6f9f4c222c5e4e73d926362328b6a8d2841871a54e46bd2278, bytes=653407 |
| PASS | yes | `manifest.cyp-challenge-single-concentration-TRAIN.csv` | sha256=cf275440aa33d10b16e7a3a19f1b196d59dc379052175ab79b47ca43db663c7a, bytes=3619009; expected sha256=cf275440aa33d10b16e7a3a19f1b196d59dc379052175ab79b47ca43db663c7a, bytes=3619009 |
| PASS | yes | `single.unique_molecule_enzyme` | duplicate molecule/enzyme pairs=0 |
| PASS | yes | `single.dense_four_isoforms` | molecules with !=4 isoforms=0 |
| PASS | yes | `identifiers.cross_table_smiles` | Molecule_Name values mapped to >1 SMILES=0 |
| PASS | yes | `direct.strict_subset_of_tdi` | direct=4905, tdi=6145, extra_tdi=1240 |
| PASS | yes | `direct.cross_table_exactness` | cell disagreements=0 |
| PASS | yes | `intervals.tdi_direct_arm.CYP1A2_pIC50_direct_inhibition` | n=1412, missing companion=0, inverted=0, point outside=0, nonpositive std=0 |
| PASS | yes | `intervals.tdi_direct_arm.CYP2C9_pIC50_direct_inhibition` | n=1285, missing companion=0, inverted=0, point outside=0, nonpositive std=0 |
| PASS | yes | `intervals.tdi_direct_arm.CYP2D6_pIC50_direct_inhibition` | n=1493, missing companion=0, inverted=0, point outside=0, nonpositive std=0 |
| PASS | yes | `intervals.tdi_direct_arm.CYP3A4_pIC50_direct_inhibition` | n=2335, missing companion=0, inverted=0, point outside=0, nonpositive std=0 |
| PASS | yes | `intervals.tdi_active_arm.CYP1A2_pIC50_TDI_condition` | n=1413, missing companion=0, inverted=0, point outside=0, nonpositive std=0 |
| PASS | yes | `intervals.tdi_active_arm.CYP2C9_pIC50_TDI_condition` | n=1285, missing companion=0, inverted=0, point outside=0, nonpositive std=0 |
| PASS | yes | `intervals.tdi_active_arm.CYP2D6_pIC50_TDI_condition` | n=1497, missing companion=0, inverted=0, point outside=0, nonpositive std=0 |
| PASS | yes | `intervals.tdi_active_arm.CYP3A4_pIC50_TDI_condition` | n=3583, missing companion=0, inverted=0, point outside=0, nonpositive std=0 |
| PASS | yes | `labels.CYP3A4.reproduction` | matched=3584/3584, rate=1.000000 |
| PASS | yes | `labels.CYP3A4.minimum_positive_count` | positive=764, minimum=100 |
| PASS | yes | `labels.CYP2D6.reproduction` | matched=1497/1497, rate=1.000000 |
| PASS | yes | `labels.CYP2D6.minimum_positive_count` | positive=324, minimum=100 |
| PASS | yes | `labels.tdi_emax_consistency` | label disagreements=0 |
| PASS | yes | `leakage.train_test_connectivity_overlap` | shared connectivity keys=0 |
| PASS | yes | `structures.training_full_inchikey_duplicates` | duplicate full-InChIKey rows=0 |
| WARN | no | `structures.training_standardized_duplicates` | extra connectivity-collapsed rows=4 across 4 variant groups; full-InChIKey duplicates=0; connectivity units are indivisible in splits |
| PASS | yes | `splits.challenge_mimetic_invariants` | molecule_rows=750, analog_units=750, families=75, fold_unit_sizes={0: 150, 1: 150, 2: 150, 3: 150, 4: 150}, family_unit_size_range=10-10, connectivity_one_family=True, connectivity_one_fold=True, connectivity_complete_units=True, minimum_unit_similarity=0.333333 |
| PASS | yes | `splits.training_only_construction` | split generator accepts only the training table; blinded test structures are not an input |
| PASS | yes | `splits.CYP3A4.positive_families` | positive held-out families=52, minimum=20 |
| PASS | yes | `splits.CYP2D6.positive_families` | positive held-out families=22, minimum=20 |

## Interpretation boundary

A PASS here permits baseline modeling to start. It does not establish model novelty, superiority, or flagship status. Those require the predeclared family-fold and blinded-test gates.
