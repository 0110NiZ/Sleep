# E4 · Wavelet denoising

Shrink wavelet detail coefficients before classifying each EEG segment with the frozen E1 model.

## Inputs

Use the preprocessed EEG, five E1 checkpoints and the local reference CSV from [E2](E2_noisy_baseline.md) at `results/local/e2/e2_per_fold.csv`. The original E4 evaluation ran on Windows with Python 3.12, PyTorch 2.6.0 CPU and PyWavelets 1.8.0. Its [evaluation dependencies](../requirements-e4.txt) are supplied separately from the Ubuntu GPU environment.

## Run from the repository root

```bash
python experiments/E4_wavelet/run_e4_wavelet.py --stage all
python experiments/E4_wavelet/plot_e4_results.py
```

The [runner](../experiments/E4_wavelet/run_e4_wavelet.py) writes to `results/local/e4_wavelet/` by default. The second command plots that run. To use a separate root, set `SLEEP_RESULT_DIR` for both the E2 reference and the E4 run.

The [denoiser](../experiments/E4_wavelet/wavelet_denoising.py) tests db4 and sym8, each at threshold scales 0.5 and 1.0. It uses four decomposition levels, symmetric extension and soft thresholds on detail coefficients. It keeps the approximation coefficients, so slow drift can remain. The runner scores all four candidates across the ten validation conditions and freezes one winner per fold before testing. [Candidate settings](../configs/e4_wavelet.json)

## Results

[Selected test scores](../results/E4_wavelet/test_per_fold.csv) · [Frozen selections](../results/E4_wavelet/selection_frozen.json) · [Validation candidate scores](../results/E4_wavelet/validation_candidate_scores.csv)

The archive contains 200 validation evaluations and 50 test evaluations. The [E2 reference checksum](checks/reproduction/E2_reference.sha256) identifies the exact baseline table used for comparison.
