# Experiment code

Each experiment has a matching folder in [results](../results/). Open the code to see how a method works, or follow its guide to run it from the repository root.

| Experiment | Main code | Run guide | Test results |
| --- | --- | --- | --- |
| E1 · Clean baseline | [Train](E1_clean_baseline/train_clean_fivefold.py) · [Verify](E1_clean_baseline/verify_e1_baseline.py) | [Run E1](../docs/E1_clean_baseline.md) | [Five clean folds](../results/E1_clean_baseline/clean_baseline_5fold_results.csv) |
| E2 · Noisy baseline | [Run the noise tests](E2_noisy_baseline/run_e2_evaluation_resumable.py) · [Base evaluator](E2_noisy_baseline/run_e2_evaluation.py) | [Run E2](../docs/E2_noisy_baseline.md) | [Scores by fold and condition](../results/E2_noisy_baseline/e2_per_fold.csv) |
| E3 · Band-pass | [Filter and evaluate](E3_bandpass/run_e3_bandpass.py) · [Select filters and summarize](E3_bandpass/finalize_e3_results.py) | [Run E3](../docs/E3_bandpass.md) | [Validation-selected filters](../results/E3_bandpass/comparison/selected_per_fold.csv) |
| E4 · Wavelet | [Run E4](E4_wavelet/run_e4_wavelet.py) · [Wavelet denoising](E4_wavelet/wavelet_denoising.py) | [Run E4](../docs/E4_wavelet.md) | [Test scores](../results/E4_wavelet/test_per_fold.csv) |
| E5 · Denoising autoencoder | [Train and evaluate](E5_dae/run_e5_dae.py) · [DAE architecture](E5_dae/dae_model.py) · [Run the pipeline](E5_dae/run_e5_pipeline.py) | [Run E5](../docs/E5_dae.md) | [Test scores](../results/E5_dae/test_per_fold.csv) |
| E6 · Noise augmentation | [Train and evaluate](E6_noise_augmentation/run_e6_noise_augmentation.py) | [Run E6](../docs/E6_noise_augmentation.md) | [Test scores](../results/E6_noise_augmentation/test_per_fold.csv) |

## Shared code

- [Data preparation](support/data_prep/) downloads and preprocesses Sleep-EDF, checks alignment and summarizes the subject splits.
- [Noise and paths](support/shared/) contains the shared noise generator, waveform preview and local data paths. E3–E6 also reuse the evaluation functions in the [E2 evaluator](E2_noisy_baseline/run_e2_evaluation.py).
- [Method comparisons](support/comparison/) summarizes the experiment scores and produces the additional comparison plots.

The [PPT figure guide](../figures/README.md) links each slide to its image, plotting script and input data. The [browser demo](../demo/README.md) compares all six methods on the same test segment.
