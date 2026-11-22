# Run the six experiments

| Experiment | Run guide | What it produces |
| --- | --- | --- |
| E1 | [Clean baseline](E1_clean_baseline.md) | One AttnSleep checkpoint per fold and its clean test score |
| E2 | [Noisy baseline](E2_noisy_baseline.md) | Scores and predictions from the frozen E1 model under ten conditions |
| E3 | [Band-pass filtering](E3_bandpass.md) | Four filter candidates and the test results for each fold's selected filter |
| E4 | [Wavelet denoising](E4_wavelet.md) | Validation-selected wavelet settings and their test scores |
| E5 | [Denoising autoencoder](E5_dae.md) | Reconstructed-signal errors and frozen-classifier scores |
| E6 | [Noise-augmented training](E6_noise_augmentation.md) | A newly trained classifier per fold and its clean and noisy scores |

For the submitted work, open the [experiment results](../results/) or the [PPT figures, code and plotting instructions](../figures/README.md). The [demo](../demo/README.md) includes its own test samples and models.

## Set up a training environment

Run the commands below from the repository root in Ubuntu. We used Python 3.10.12 on Ubuntu 22.04.5 (WSL), PyTorch 2.6.0 with CUDA 12.4, and an RTX 4060 Laptop GPU with driver 560.94. The [dependency list](../requirements.txt) records that working environment. A fresh installation of the entire list has not been independently verified. The E4 guide also documents its original Windows CPU environment.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## Prepare the shared inputs

- **EEG.** Put the 39 preprocessed Sleep-EDF-20 NPZ recordings in `data/processed/sleep-edf-20/`. They contain 42,308 labeled 30-second segments from 20 subjects. The [download and preprocessing scripts](../experiments/support/data_prep/) and [recording checks](checks/) document the inputs.
- **Folds and noise.** The [subject splits](../configs/fivefold_subject_splits.json) assign 12 training, 4 validation and 4 test subjects per fold. E2–E6 evaluate the same clean reference and nine [noise conditions](../configs/noise_conditions_v2.json).
- **Baseline weights.** For an exact E2–E5 replay, obtain the five original E1 checkpoints and associated configs from the group. Place them under `Sleep_Baseline_5fold/`, with one folder per fold containing `best_validation_macro_f1.pth`. Both `fold0` and `fold0_seed123` naming conventions are supported. To train new checkpoints, start with E1. E6 starts from its own initialization.
- **New runs.** E1 training uses the output paths in its configs. The other examples write to `results/local/`. The guides show when to use explicit command-line paths for data, weights or output. Full EEG recordings, original training checkpoints and prediction bundles are distributed separately.
