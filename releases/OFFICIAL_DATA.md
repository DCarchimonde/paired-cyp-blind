# Frozen OpenADMET public data copy

`official-data-3ac9c5d.zip` contains unchanged public CSV files from
[OpenADMET's CYP challenge dataset](https://huggingface.co/datasets/openadmet/cyp-challenge-train-test/tree/3ac9c5dbb83eec5780ec7fa511908698cfe1396d),
at commit `3ac9c5dbb83eec5780ec7fa511908698cfe1396d`.
The dataset is produced by OpenADMET and released under Apache-2.0.
The archive includes the full Apache-2.0 license, the preserved upstream dataset
card, and this project's original provenance/hash manifest.

Archive SHA256: `ed3b68085181651410ed8bc4f31a64b3b29f2b818e5d236c73738c5997a1e8a5`.

All five CSV byte counts and SHA256 values match
`data/manifests/official_dataset.yaml`. The CSVs have not been reformatted,
filtered, relabeled, or otherwise changed. The archive only changes how the same
public files are delivered to the user's machine.

The blinded test file contains the released test inputs; it does not add withheld
test labels. Dataset provenance, model selection rules and the frozen scientific
protocol remain unchanged.

`scripts/build_official_data_archive.py` builds the archive from already verified
raw files. `scripts/restore_official_data.py` checks the archive and every CSV,
preserves conflicting existing files, respects the workflow lock, and only
publishes missing verified files. No network access is needed during restoration.
