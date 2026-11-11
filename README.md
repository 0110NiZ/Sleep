# AttnSleep Robustness under Simulated EEG Noise

A COMP 4330SEF group project evaluating how simulated EEG noise affects single-channel sleep staging and comparing two strategies for improving robustness:

- **Input denoising:** process EEG with band-pass filtering, wavelet denoising or a denoising autoencoder (DAE), then classify it with a frozen AttnSleep baseline.
- **Noise-augmented training:** train AttnSleep with clean and corrupted inputs so that the classifier learns to handle interference directly.

**E1–E6 are complete.** This repository contains the experiment code, configurations, archived results and analysis tools. The web demonstration is awaiting source-code delivery.

## Experimental design

We use **Sleep-EDF-20**: 20 subjects, 39 nights and 42,308 single-channel Fpz-Cz EEG segments. Each segment is 30 seconds long, sampled at 100 Hz.

All methods use the same subject-wise five-fold splits: 12 training, 4 validation and 4 test subjects per fold. All nights from a subject stay in the same partition within a fold. Noise is applied to the preprocessed 30-second segments, using three types—Gaussian white noise, baseline drift and oscillatory burst artifacts—at three severity levels each. Together with the clean condition, this gives ten evaluation conditions. Here, **clean means no additional synthetic noise**, not an artifact-free recording.

| Experiment | Method | Purpose |
|---|---|---|
| E1 | Clean AttnSleep baseline | Establish performance without additional synthetic noise |
| E2 | Noisy baseline | Measure degradation using the same frozen baseline under noise |
| E3 | Band-pass filtering + frozen baseline | Evaluate traditional frequency-based filtering |
| E4 | Wavelet denoising + frozen baseline | Evaluate multiscale input denoising |
| E5 | DAE + frozen baseline | Evaluate learned signal reconstruction |
| E6 | Noise-augmented AttnSleep | Evaluate robustness learned during classifier training |

Parameters and checkpoints are selected using validation data. E2–E6 share the same test segments and noise instances. E3–E5 retain the corresponding fold's frozen E1 classifier; E6 trains a separate AttnSleep from random initialization. This compares complete robustness strategies, rather than isolating a single training component. Exact noise settings and subject splits are in [configs/](configs/).

## Main results

**Test Macro-F1 (%), five-fold mean ± sample standard deviation.** For the noisy score, we first average the nine noisy conditions within each fold, then calculate the mean and standard deviation across the five folds.

| Method | Clean Macro-F1 | Mean noisy Macro-F1 |
|---|---:|---:|
| E2 — Baseline | 75.08 ± 4.32 | 58.14 ± 3.46 |
| E3 — Band-pass | 63.65 ± 7.77 | 55.98 ± 6.88 |
| E4 — Wavelet | 71.42 ± 5.78 | 62.93 ± 5.81 |
| E5 — DAE | 73.81 ± 3.61 | 69.89 ± 7.74 |
| E6 — Noise augmentation | 74.24 ± 4.73 | **73.37 ± 4.46** |

E2's clean score is the E1 reference, not a separately trained model. E3 uses the formal per-fold validation-selected filters, not the candidate with the highest test score.

- **Noise substantially reduces baseline performance:** mean noisy Macro-F1 is 58.14%, compared with 75.08% on clean inputs.
- **Input denoising has mixed effects:** band-pass filtering falls below the noisy baseline on average, while wavelet denoising and DAE improve it. All three reduce clean-input performance.
- **Noise augmentation achieves the highest mean noisy score in these experiments:** E6 improves over E2 by **15.22 percentage points**, while clean Macro-F1 decreases by **0.84 percentage points**. These differences use the unrounded means.

These are descriptive results for the specified synthetic-noise protocol. They do not establish superiority under every noise condition or real-world device setting. The five training sets overlap, so the folds are not five independent repeat experiments. Reconstruction metrics apply to E3–E5; timing should only be compared when hardware and measurement protocols match. See [interpretation limits](docs/HANDOVER_MAINTENANCE.md).

## Explore and analyze the results

| Experiment | Archived results | Run and handover instructions |
|---|---|---|
| E1 | [Clean baseline](result/e1/) | [Environment and baseline notes](docs/ENVIRONMENT.txt) |
| E2 | [Noisy baseline](result/e2_v2/) | [E2 guide](docs/E2_HANDOVER.md) |
| E3 | [Band-pass results](result/e3_bandpass/) | [E3 guide](docs/E3_HANDOVER.md) |
| E4 | [Wavelet results](result/e4_wavelet/) | [E4 guide](docs/E4_RUN.txt) |
| E5 | [DAE results](result/e5_dae/) | [E5 guide](docs/E5_RUN.md) |
| E6 | [Noise-augmentation results](result/e6_noise_augmentation/) | [E6 guide](docs/E6_START_HERE.md) |

To summarize all five methods without rerunning any model, run this from the repository root:

```bash
python scripts/summarize_all_experiments.py --e3 result/e3_bandpass/comparison/selected_per_fold.csv --out result/local/analysis_summary
```

The output directory must not already exist. The script reads archived CSVs and creates `overall.csv` and `paired_vs_e2.csv`; it does not train models, perform inference or generate noise. Its default E2 source is the verified reference CSV in the E4 handover.

## Environment and running experiments

The recorded project environment uses **Ubuntu/WSL, Python 3.10 and PyTorch 2.6.0**; GPU training uses CUDA. Read the [environment instructions](docs/ENVIRONMENT.txt) and the relevant experiment guide before installing dependencies or running an experiment. The dependency snapshot describes the working environment; a fresh installation has not been independently verified.

EEG data, model checkpoints and full prediction NPZs are **shared separately** and are not included in a normal clone. Local locations can be configured with `SLEEP_DATA_DIR`, `SLEEP_BASELINE_DIR` and `SLEEP_RESULT_DIR`; individual guides document additional options. Existing results can be analyzed without downloading the model weights.

The current E2 evaluation entry is `scripts/run_e2_evaluation_resumable.py`; the shared noise API is `from noise_conditions import apply_noise` when `scripts/` is on the Python path. Old `*_v2` commands and imports remain compatible. The sole noise configuration is still `configs/noise_conditions_v2.json`, identifying the formal Noise v2 protocol. See [compatibility and path guidance](docs/E2_NAMING.md).

For historical reproduction or resuming an old run, use its recorded code commit and environment. Preserve the archived results and manifests, and use a fresh output directory for any new experiment.

## Repository guide

| Directory | Contents |
|---|---|
| [docs/](docs/README.md) | Environment, run instructions and handover guidance |
| [configs/](configs/) | Subject splits, noise protocol and experiment settings |
| [scripts/](scripts/) | Training, evaluation, verification and analysis tools |
| [result/](result/) | Formal experiment archives |
| [reports/](reports/README.md) | Data checks, reproduction evidence and integration audits |
| [AttnSleep/](AttnSleep/) | Original author implementation and license |
| [demo/](demo/README.md) | Reserved location for the forthcoming local web demo |

## Local web demonstration

The planned demo will display EEG corruption, denoising and sleep-stage predictions. **Only a placeholder is currently included in `demo/`; there is no runnable webpage in this repository yet.** Once the source is delivered, its paths and sample predictions will be checked against the formal experiment archives.

## Attribution and license

This project builds on [the original AttnSleep implementation](https://github.com/emadeldeen24/AttnSleep), vendored at commit `6b4d2665884628c8a7bb09f36589a8ec0992f8e2`. Please cite the original paper, *An attention-based deep learning approach for sleep stage classification with single-channel EEG* (Eldele et al., 2021). The author's [MIT license](AttnSleep/LICENSE) is retained, and compatibility changes are recorded in [author_compatibility.patch](patches/author_compatibility.patch).

Sleep-EDF-20 is drawn from PhysioNet's Sleep-EDF Expanded database. Dataset use remains subject to its source terms; this repository does not grant a separate dataset license.
