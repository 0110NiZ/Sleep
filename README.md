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

These are descriptive results for the specified synthetic-noise protocol. They do not establish superiority under every noise condition or real-world device setting. The five training sets overlap, so the folds are not five independent repeat experiments. Reconstruction metrics apply to E3–E5; timing should only be compared when hardware and measurement protocols match. See [interpretation limits](docs/handover.md).


## Where to find things

| Folder | Contents |
|---|---|
| [code/](code/) | Our E1–E6 experiment entry points and shared functions |
| [AttnSleep/](AttnSleep/) | Original AttnSleep model and supporting code |
| [configs/](configs/) | Experiment settings, subject splits and Noise v2 protocol |
| [result/](result/) | Original formal experiment results, grouped by E1–E6 |
| [visualization/](visualization/) | Plotting code, source tables and exported figures, grouped by topic |
| [docs/](docs/) | Setup, experiment guides, handover notes and checks |
| [demo/](demo/) | Placeholder for the web demo; source still pending |
| [local/](local/) | Local-only EEG, weights, logs and new outputs; not uploaded |

```text
Sleep/
├── README.md
├── requirements.txt
├── AttnSleep/
├── code/
│   ├── E1_baseline.py
│   ├── E2_noisy_baseline.py
│   ├── E2_evaluation.py
│   ├── E3_bandpass.py
│   ├── E4_wavelet.py
│   ├── E5_dae.py
│   ├── E6_augmentation.py
│   ├── noise.py, dae_model.py, wavelet.py, paths.py
│   ├── summarize_results.py
│   ├── data/
│   └── checks/
├── configs/
│   ├── E1_baseline_fold0.json … E1_baseline_fold4.json
│   ├── E4_wavelet.json, E5_dae.json, E6_augmentation.json
│   ├── subject_splits.json
│   └── noise.json
├── result/
│   ├── E1_baseline/
│   ├── E2_noisy_baseline/
│   ├── E3_bandpass/
│   ├── E4_wavelet/
│   ├── E5_dae/
│   └── E6_augmentation/
├── visualization/
│   ├── E2_noise_effect/
│   ├── noise_examples/
│   ├── method_comparison/
│   ├── clean_performance/
│   ├── reconstruction/
│   ├── E4_results/
│   └── comparison_plots.py
├── docs/
│   ├── setup.md
│   ├── E1_baseline.md … E6_augmentation.md
│   ├── results.md
│   ├── handover.md
│   ├── checks/
│   └── history/
├── demo/
└── local/
    ├── data/
    ├── weights/
    ├── logs/
    └── runs/
```

E2 shares the noise configuration; E3 keeps its existing parameter choices in code and archived selection records. No new E2/E3 parameter files have been invented.

## Getting started

For a report or presentation, start with [results](docs/results.md) and [figures](visualization/README.md). For code, read [setup](docs/setup.md) and the [experiment guides](docs/README.md). The completed experiments do not need to be rerun.

Large EEG files, five-fold weights and full prediction NPZ packages are shared separately. GitHub contains the formal tables and audit records; it is not a complete copy of local experiment storage.

## Compatibility and history

Old `scripts/`, result names and configuration aliases remain where active imports or hash checks need them. Root `data/` and `runs/` remain because archived configuration values refer to them. They point to the same files, not duplicate implementations. Use the folders above for new work. The directory cleanup preserves experiment configurations, archived results and Noise v2 identity; it does not rename fields inside historical manifests.

This layout uses relative symbolic links and is checked in **Ubuntu/WSL**. Windows-native clones must support Git symbolic links; that environment has not been validated. Historical reproduction/resume should use the recorded commit: current code-path changes can correctly trigger archived code-hash guards. Do not edit manifests to bypass them.

See [the cleanup record](docs/checks/folder_cleanup/README.md) for source branches, path mappings and validation. The demo integration is deferred until its source is delivered.

Removed root aliases: `reports`, `patches`, `logs`, `Sleep_Baseline_5fold`, `requirement.txt`, `requirements-e4.txt` and `requirements-lock.txt`. Current code uses their real locations directly. Use `requirements.txt`; original E4/lock dependency records remain in `docs/history/`. Legacy top-level documentation links and the obsolete reports figure links were also removed. See [retained-link reasons](docs/checks/folder_cleanup/compatibility.md).
