> 路径与当前交付状态以 [HANDOVER_MAINTENANCE.md](HANDOVER_MAINTENANCE.md) 为准。下方保留历史运行说明，当前无需重跑。

# E5 – Denoising autoencoder

Branch: `E5-dae`, based on `E4-wavelet` commit `9d27c8b`.
Use the existing `/home/nzd/Sleep/.venv`; no dependency upgrade is needed.

## Run

```bash
sleep
cd /home/nzd/Sleep/e5_project
export SLEEP_DATA_DIR=/home/nzd/Sleep/data/processed/sleep-edf-20
python scripts/check_e5.py
python scripts/run_e5_dae.py --smoke
python scripts/run_e5_dae.py --stage verify
python scripts/run_e5_dae.py --stage train
python scripts/run_e5_dae.py --stage test
```

`verify` reproduces all 423,080 original E2 CPU predictions with matching IDs and labels. It does not select DAE settings. Exact prediction agreement is required before E5 testing. The local NPZ archive hashes differ from E4's regenerated archives; no claim of byte-identical files is made. Original baseline weight hashes, split contents, per-record epoch counts and label counts are checked.

The model/configuration is fixed in `configs/e5_dae.json`. A small residual encoder–decoder with 72,689 parameters receives 3000 points and outputs 3000 points. It starts at the identity transformation and learns an additive correction. Internal scaling uses a single training-only RMS scalar and is reversed before AttnSleep; there is no per-test normalization or calibration.

Each fold starts independently. All nine noisy training conditions have equal sampling probability, generated on the fly with fold/epoch-specific training seeds. Original clean epochs are MSE targets. Validation/test retain the original seed 123 and sample IDs, so E2–E5 see the same noise. Baseline is in eval mode with frozen weights. Training minimizes normalized MSE, but best checkpoints maximize ten-condition validation mean Macro-F1. Maximum 30 epochs, patience 6, minimum 5. No hyperparameter decisions use test results.

Run inputs/code are hashed. Restarting the same unchanged run resumes `latest.pth`; completed folds are skipped. A changed configuration/code requires a separate output directory/run. All five selections must exist before official testing. The test runner uses CPU and each baseline's original batch size for comparability. Timing includes DAE plus classifier, excludes noise generation and disk access, and uses warm-up; comparisons on other hardware require remeasurement.

Outputs are in `result/local/e5_dae/`: manifest/data audit, E2 reproduction audit, per-epoch validation and loss history, five DAE checkpoints and selections, 50 prediction NPZs with IDs, test metrics, per-stage results, pooled confusion matrices, RMSE/SNR improvement, timings, and summary. The nine-noise summary first averages nine conditions within each fold, then reports mean and sample SD across five folds. Clean performance is separate. No statistical significance is claimed from these descriptive results.

Training/GPU verification may differ slightly from CPU floating-point inference; final test uses CPU. Synthetic-noise and single-seed limitations remain. This is one prespecified DAE, not an exhaustive comparison of autoencoders. Do not tune it after reading test performance.
