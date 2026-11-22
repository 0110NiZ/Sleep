# AttnSleep Robustness under Simulated EEG Noise

**[Experiment code](experiments/) · [Test results](results/) · [PPT figures, data and plotting code](figures/README.md)**

We tested how three types of noise affect sleep staging from single-channel EEG. We compared cleaning the signal before classification with training AttnSleep on noisy examples. This is our COMP 4330SEF Group 7 project.

**[Open the live demo](https://sleep-lab-7qq.pages.dev/)** or [run the same demo locally](demo/README.md) with `python demo/serve.py`. It shows how E1–E6 handle the same test segment. Use the full test results below to compare methods across subjects.

## Our six experiments

| Experiment | What we did | Code | Test results |
| --- | --- | --- | --- |
| E1 · Clean baseline | Train AttnSleep and test with no added synthetic noise. | [Train](experiments/train_clean_fivefold.py) | [Five folds](results/e1_clean_baseline/clean_baseline_5fold_results.csv) |
| E2 · Noisy baseline | Add noise to the EEG and test the same frozen E1 classifier. | [Evaluate](experiments/run_e2_evaluation_resumable.py) | [By fold and condition](results/e2_noisy_baseline/e2_per_fold.csv) |
| E3 · Band-pass | Filter the input before the frozen classifier. Select the passband on validation data. | [Filter and evaluate](experiments/run_e3_bandpass.py) | [Selected filters on test data](results/e3_bandpass/comparison/selected_per_fold.csv) |
| E4 · Wavelet | Apply wavelet denoising before the frozen classifier. | [Run E4](experiments/run_e4_wavelet.py) · [Denoiser](experiments/wavelet_denoising.py) | [By fold and condition](results/e4_wavelet/test_per_fold.csv) |
| E5 · Denoising autoencoder | Train a DAE on noisy–clean pairs, then classify the reconstructed EEG with the frozen classifier. | [Train and evaluate](experiments/run_e5_dae.py) · [DAE model](experiments/dae_model.py) | [By fold and condition](results/e5_dae/test_per_fold.csv) |
| E6 · Noise augmentation | Train a separate AttnSleep from random initialization on a mix of clean and noisy inputs. | [Train and evaluate](experiments/run_e6_noise_augmentation.py) | [By fold and condition](results/e6_noise_augmentation/test_per_fold.csv) |

## What we found

| Method | Clean Macro-F1 (%) | Mean noisy Macro-F1 (%) |
| --- | ---: | ---: |
| E1 / E2 baseline | 75.08 ± 4.32 | 58.14 ± 3.46 |
| E3 · Band-pass | 63.65 ± 7.77 | 55.98 ± 6.88 |
| E4 · Wavelet | 71.42 ± 5.78 | 62.93 ± 5.81 |
| E5 · DAE | 73.81 ± 3.61 | 69.89 ± 7.74 |
| E6 · Noise augmentation | 74.24 ± 4.73 | **73.37 ± 4.46** |

E6 gained **15.22 percentage points** over E2 on noisy EEG and lost **0.84 points** against E1 on clean EEG, using unrounded scores. With the classifier frozen, E5 had the highest mean noisy score. Band-pass filtering lowered both scores. The [condition-by-condition figure](figures/images/08_condition_heatmap.png) shows how these averages vary with the type and strength of noise.

We used Sleep-EDF-20 (20 subjects, 39 nights, 42,308 EEG segments). Each fold had 12 training, 4 validation and 4 test subjects. We applied three noise types at three strengths to the same test segments for E2–E6. We averaged the nine noisy scores within each fold, then reported the mean and sample standard deviation across five folds. “Clean” means no added synthetic noise. E2's clean reference comes from E1. [Subject splits and noise settings](configs/)

The comparison covers complete strategies. We used one training seed and synthetic noise. The experiments do not isolate the effects of retraining, initialization or model selection, or establish performance on clinical artifacts. [Experiment notes](docs/experiment_notes.md)

## Run the code

- **Recreate the slides.** Follow the [figure guide](figures/README.md) to rebuild five charts from the archived results, or all six including the waveform with the original test EEG. The guide links each slide to its PNG, code and input data.
- **Run an experiment.** The [run guides](docs/README.md) cover setup, data, weights and commands for each method. Original recordings, training checkpoints and full prediction bundles are distributed separately.
- **Compare all methods.** From the repository root, run the command below with a fresh output directory to create `overall.csv` and `paired_vs_e2.csv` from the archived scores.

```bash
python experiments/summarize_all_experiments.py --e3 results/e3_bandpass/comparison/selected_per_fold.csv --out results/local/analysis_summary
```

## Credits

We use [AttnSleep by Eldele et al. (2021)](https://github.com/emadeldeen24/AttnSleep), originally at commit `6b4d2665884628c8a7bb09f36589a8ec0992f8e2`. The [original implementation and MIT license](AttnSleep/) and our [compatibility changes](patches/author_compatibility.patch) are included. Sleep-EDF-20 is drawn from PhysioNet's Sleep-EDF Expanded database and is subject to the dataset's terms.
