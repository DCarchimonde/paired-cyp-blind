# GitHub delivery and immutable experiment execution

This repository was uploaded using the authenticated GitHub connector. GitHub
creates new delivery commit IDs. Those IDs are not presented as the original
experiment history. The complete, original nine-commit history and five tags
are retained in `releases/paired-cyp-blind-reviewed-20260906.bundle`.

- Experiment commit: `bc00641390c559ec696506d6fd07edb77c514c13`
- Required experiment tag: `neural-protocol-hardened-20260906`
- Bundle SHA-256: `1aa7a8d7ac8f734a4e758fef9232ddd98d77a711485ee6b87bc5c363e1f6e79f`
- Bundle size: 627,436 bytes

The new `start_reviewed_neural_baselines_4090.sh` entrypoint verifies that
archive, restores it into `.runtime/frozen-experiment`, verifies its exact
commit, tag, and clean worktree, and calls the original reviewed launcher.
The original GPU, dependency, input, resume, and result-audit guards are
unchanged. No token or write access is needed to run from a public clone.
No credentials are stored in this repository.

The outer `src/`, `configs/`, `data/splits/`, `vendor/`, `pyproject.toml`,
`uv.lock`, and original experiment/test scripts mirror the reviewed source
for inspection. Formal training executes the immutable archived checkout.
Editing this outer source mirror does not change the frozen experiment.
Any scientific revision requires a separately reviewed freeze and bundle.

The preparation helper serializes imports, validates before publishing the
checkout directory, and preserves unexpected existing files. Corruption,
source modifications, wrong tags, or wrong commits stop startup. Verified
checkouts are reused on resume. Short artifact aliases point to the actual
files inside the frozen checkout:

- `.runtime/neural-run.log`
- `.runtime/run-status.txt`
- `.runtime/runner.pid`
- `reports/neural_v2/`

Preserve the complete `.runtime/frozen-experiment/reports/neural_v2` directory
(the real directory behind the alias), including models and logs, before
releasing a rented instance. Copying only the symlink is not a results backup.
The output directory and source archive are separate: no production results
are uploaded to GitHub automatically.

The frozen implementation has 46 passing regression tests and a 17-check
real-data input audit; CPU smoke and short real-data integration checks passed.
This delivery adds four passing archive-import checks. The source-only
checkout suite returned 49 PASS and one expected skip because raw official
data had not yet been fetched; that data-parity test passed in the frozen
46-test review. The launcher fetches the official data before its full test
suite. All mirrored frozen files, apart from the documented delivery README,
runbook, Makefile entrypoint, and ignore rules, were compared byte-for-byte.
A physical RTX 4090 full
run remains pending. The command covers 200 neural baseline jobs and audits,
not the unimplemented primary model or an entire finished paper.
