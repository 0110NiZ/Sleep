# Handover

Share the Git branch/commit, formal [results](../result/) and [figures](../visualization/). Share EEG, complete checkpoints and original prediction NPZs separately; keep large files out of Git.

E5/E6 checkpoint packages should retain `fold0`–`fold4`, `best.pth` and original `selection.json`. E5 needs its full checkpoint including normalization scale, not just a partial state dictionary. Test prediction packages contain ten NPZs per fold (50 total), including sample IDs, labels and predictions. These packages were not regenerated during this cleanup.

## Updating existing work

- Experiment entry points are now in `code/`, named `E1_baseline.py` through `E6_augmentation.py`.
- Formal results use matching `result/E*/` names. Config aliases use `E*_*.json`, `subject_splits.json` and `noise.json`.
- Plotting code and exports are grouped together in `visualization/`. Latest main's clean E1/E6 figure was retained; older branch snapshots did not overwrite it.
- Old `scripts/` imports, old result paths and config names remain compatibility links. Noise v2 protocol identity and historical Git object paths are unchanged.
- Audit records are in `docs/checks/`; original instructions and environment records are in `docs/history/`.
- Demo source is still pending. No web application was moved, tested or claimed ready by this cleanup. Its imports should be reviewed when delivered.

Use Ubuntu/WSL with symbolic links. Historical commands still resolve, but an archived run may reject current code hashes: use its recorded commit for historical resume. Never update an old manifest to hide a code change.

See [original maintenance notes](HANDOVER_MAINTENANCE.md) for experiment-specific limits and [cleanup checks](checks/folder_cleanup/README.md) for this change's scope.
