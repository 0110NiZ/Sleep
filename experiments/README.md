# Experiment code

We trained a clean baseline, measured how noise affected it, then tested four ways to handle the noise. Commands for each step are in the [run guides](../docs/README.md).

| Experiment | Main Python file |
| --- | --- |
| E1 · Train the clean baseline | [train_clean_fivefold.py](train_clean_fivefold.py) |
| E2 · Test the baseline under noise | [run_e2_evaluation_resumable.py](run_e2_evaluation_resumable.py) |
| E3 · Band-pass filtering | [run_e3_bandpass.py](run_e3_bandpass.py) |
| E4 · Wavelet denoising | [run_e4_wavelet.py](run_e4_wavelet.py) and [wavelet_denoising.py](wavelet_denoising.py) |
| E5 · Denoising autoencoder | [run_e5_dae.py](run_e5_dae.py) and [dae_model.py](dae_model.py) |
| E6 · Noise-augmented training | [run_e6_noise_augmentation.py](run_e6_noise_augmentation.py) |

The experiments share a [noise generator](noise_conditions.py), [subject splits and settings](../configs/) and the [AttnSleep model](../AttnSleep/model/model.py). The [PPT plotting scripts](../figures/plots/) read the [test results](../results/) to draw the metric charts.
