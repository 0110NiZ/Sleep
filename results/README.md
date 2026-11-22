# Test results

These are the scores from the original five-fold experiment runs. Start with the CSV for the experiment you want to inspect.

| Experiment | Fold-level test scores |
| --- | --- |
| E1 · Clean baseline | [Five clean folds](E1_clean_baseline/clean_baseline_5fold_results.csv) |
| E2 · Noisy baseline | [Five folds × ten conditions](E2_noisy_baseline/e2_per_fold.csv) |
| E3 · Band-pass | [Test scores for the filter selected on each validation fold](E3_bandpass/comparison/selected_per_fold.csv) |
| E4 · Wavelet | [Test scores](E4_wavelet/test_per_fold.csv) |
| E5 · Denoising autoencoder | [Test scores](E5_dae/test_per_fold.csv) |
| E6 · Noise augmentation | [Test scores](E6_noise_augmentation/test_per_fold.csv) |

The [presentation tables](../figures/tables/) contain the values plotted in the slides. The [figure guide](../figures/README.md) explains how the conditions and folds were averaged. Original manifests record paths at the time of each run. The [run guides](../docs/README.md) link to the current code and inputs.
