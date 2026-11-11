> 已完成实验。下文命令和代码/数据路径均以仓库根目录为基准，不以 docs/ 为基准。历史运行请使用记录的原提交，不用维护代码继续写入原输出目录。目录迁移见 [DELIVERY_LAYOUT.md](DELIVERY_LAYOUT.md)。

> 路径与当前交付状态以 [HANDOVER_MAINTENANCE.md](HANDOVER_MAINTENANCE.md) 为准。下方保留历史运行说明，当前无需重跑。

# E5 – Denoising autoencoder

Historical development branch: `E5-dae`, based on `E4-wavelet` commit `9d27c8b`; E5 is now integrated into main. The original author used `/home/nzd/Sleep/.venv`. Other members should activate their own compatible environment; the `sleep` shortcut is personal, not installed by this repository.

## Run

These are optional reproduction commands, not a request to rerun completed experiments. Replace `/path/to/...` with your local paths; activate your environment first. Keep the same output root across stages. Baseline folders may be `fold0`–`fold4` or `fold0_seed123`–`fold4_seed123`, with `best_validation_macro_f1.pth` and the original configuration.

```bash
cd /path/to/Sleep
export SLEEP_DATA_DIR=/path/to/clean_npz
export SLEEP_BASELINE_DIR=/path/to/fivefold_baseline
export SLEEP_RESULT_DIR=/path/to/new_output_root
python scripts/check_e5.py
python scripts/run_e5_dae.py --smoke
python scripts/run_e5_dae.py --stage verify
python scripts/run_e5_dae.py --stage train
python scripts/run_e5_dae.py --stage test
```

`verify` reproduces all 423,080 original E2 CPU predictions with matching IDs and labels. It does not select DAE settings. Exact prediction agreement is required before E5 testing. E4 and E5 NPZ container hashes differ, but the x/y/fs/ch_label arrays from the actual experimental inputs were independently verified as identical. Object headers were not compared and the container-hash difference was not attributed to a confirmed cause. Original baseline weight hashes, split contents, per-record epoch counts and label counts are checked.

The model/configuration is fixed in `configs/e5_dae.json`. A small residual encoder–decoder with 72,689 parameters receives 3000 points and outputs 3000 points. It starts at the identity transformation and learns an additive correction. Internal scaling uses a single training-only RMS scalar and is reversed before AttnSleep; there is no per-test normalization or calibration.

Each fold starts independently. All nine noisy training conditions have equal sampling probability, generated on the fly with fold/epoch-specific training seeds. Original clean epochs are MSE targets. Validation/test retain the original seed 123 and sample IDs, so E2–E5 see the same noise. Baseline is in eval mode with frozen weights. Training minimizes normalized MSE, but best checkpoints maximize ten-condition validation mean Macro-F1. Maximum 30 epochs, patience 6, minimum 5. No hyperparameter decisions use test results.

Run inputs/code are hashed. Restarting the same unchanged run resumes `latest.pth`; completed folds are skipped. A changed configuration/code requires a separate output directory/run. All five selections must exist before official testing. The test runner uses CPU and each baseline's original batch size for comparability. Timing includes DAE plus classifier, excludes noise generation and disk access, and uses warm-up; comparisons on other hardware require remeasurement.

Full local outputs are in `SLEEP_RESULT_DIR/e5_dae/` (default `result/local/e5_dae/`); committed handover results are in `result/e5_dae/`. Local outputs include: manifest/data audit, E2 reproduction audit, per-epoch validation and loss history, five DAE checkpoints and selections, 50 prediction NPZs with IDs, test metrics, per-stage results, pooled confusion matrices, RMSE/SNR improvement, timings, and summary. The nine-noise summary first averages nine conditions within each fold, then reports mean and sample SD across five folds. Clean performance is separate. No statistical significance is claimed from these descriptive results.

Training/GPU verification may differ slightly from CPU floating-point inference; final test uses CPU. Synthetic-noise and single-seed limitations remain. This is one prespecified DAE, not an exhaustive comparison of autoencoders. Do not tune it after reading test performance.
