# Test results

These are the scores from the original five-fold experiment runs. Start with the CSV for the experiment you want to inspect.

| Experiment | Fold-level test scores |
| --- | --- |
| E1 · Clean baseline | [Five clean folds](e1_clean_baseline/clean_baseline_5fold_results.csv) |
| E2 · Noisy baseline | [Five folds × ten conditions](e2_noisy_baseline/e2_per_fold.csv) |
| E3 · Band-pass | [Test scores for the filter selected on each validation fold](e3_bandpass/comparison/selected_per_fold.csv) |
| E4 · Wavelet | [Test scores](e4_wavelet/test_per_fold.csv) |
| E5 · Denoising autoencoder | [Test scores](e5_dae/test_per_fold.csv) |
| E6 · Noise augmentation | [Test scores](e6_noise_augmentation/test_per_fold.csv) |

The [presentation tables](../figures/tables/) contain the values plotted in the slides. The [figure guide](../figures/README.md) explains how the conditions and folds were averaged. Older manifests describe paths at the time of the run. See the [path history](../docs/history/layout.md).
