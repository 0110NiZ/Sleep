# E2 naming maintenance (round two)

Round one was merged into main at `61f4d48`; this round remains on `maintenance/e2-naming-round2`.

| Current entry | Compatible old entry |
|---|---|
| `scripts/noise_conditions.py`, `apply_noise` | `scripts/noise_conditions_v2.py`, `apply_noise_v2` |
| `scripts/run_e2_evaluation.py` | `scripts/run_e2_evaluation_v2.py` |
| `scripts/run_e2_evaluation_resumable.py` | `scripts/run_e2_evaluation_v2_resumable.py` |
| `scripts/visualize_noise.py` | `scripts/visualize_noise_v2.py` |

Old Python modules alias the canonical module object, including private helpers and mutable globals. Old CLI commands execute the canonical implementation. There is only one noise algorithm. The original generator is preserved verbatim in the canonical file, followed only by an `apply_noise` alias. E6 verifies the original content against the archived E5 hash before permitting any new run; it does not rewrite or bypass archived manifests.

## Paths and preserved records

- `configs/noise_conditions_v2.json` remains the sole noise configuration, byte-identical, including `version: noise_v2`. No competing configuration is introduced.
- Published `result/e2_v2/` and all archived results, figures, reports, selection records and hashes stay untouched. This is a historical archive, not a second algorithm to delete.
- New local E2 runs default to `SLEEP_RESULT_DIR/e2` (default root: `result/local`). If only an existing `e2_v2` directory is present under that root, it remains selected. If both exist, fail explicitly rather than guessing. Use a fresh output root for maintenance-code runs.
- The resumable E2 entry accepts explicit `--output-dir`, including old paths; its new default is the local result directory, not the committed archive. No output was generated there by this maintenance.
- The exact historical reference `c2547fb:result/e2_v2/predictions_with_ids.csv` remains unchanged.
- Source hashes change when entry points/imports move. Resume historical experiments with the original commit/environment. Do not edit old manifests to make a new checkout resume them.

## Verification and limits

This maintenance checks old/new imports, CLI help, ten-condition noise equality against the pre-maintenance generator, batching and input layouts, protocol checks, archived file hashes and the descriptive summary. A small CPU inference comparison may verify rename equivalence; it is not a rerun of the five-fold experiment. No training, tuning or official output regeneration is authorized.

Demo source is not delivered. Its compatibility and prediction agreement remain pending; the old imports are retained to ease later integration.

Checks passed; see [verification record](../reports/e2_naming_round2/README.md). Runtime checks used an Ubuntu LF checkout, matching the original experiment environment. Windows CRLF conversion can trigger the existing byte-hash protections; do not bypass them.
