# Setup

Use Ubuntu or WSL for this layout. Activate the existing Sleep environment when available. `requirements.txt` retains the previous environment requirements unchanged; the historical lock and E4 dependency lists are in `docs/history/`.

Run commands from the repository root. No training is required to inspect results or use saved CSVs.

| Local content | Default location | Optional environment variable |
|---|---|---|
| Processed EEG | `local/data/processed/sleep-edf-20/` | `SLEEP_DATA_DIR` |
| Raw EDF | `local/data/raw/sleep-edf-20/` | `SLEEP_RAW_DIR` |
| E1 weights | `local/weights/E1_baseline/` | `SLEEP_BASELINE_DIR` |
| New run output | `local/runs/` | `SLEEP_RESULT_DIR` |

E1 handover supports `fold0/` or `fold0_seed123/` (likewise folds 1–4), containing `best_validation_macro_f1.pth`. Do not provide both layouts for the same fold. E5/E6 original complete checkpoints can be stored under `local/weights/E5_dae/fold*/best.pth` and `local/weights/E6_augmentation/fold*/best.pth`; callers must pass the appropriate path rather than assume automatic discovery.

The shared path helper supports these variables; older entry points or archived configs can still contain original absolute paths or explicit CLI arguments. Consult the experiment guide before starting any future run. Existing datasets elsewhere are not moved by this cleanup. Set paths to their real location.

`data`, `runs` and `result/local` are relative compatibility links into `local/`. `data` and `runs` are still referenced by the unchanged original configs; `result/local` is still used by experiment output paths. Keep these links intact. Use `local/logs/` and `local/weights/E1_baseline/` directly; the old root logs and weight aliases have been removed. Windows-native Git must materialize symbolic links; a checkout that turns them into text files is not supported by this verification.

Saved results stay in `result/E*/`. Use a fresh output directory for any future authorized run. Do not modify original manifests or selection records to avoid a hash mismatch.
