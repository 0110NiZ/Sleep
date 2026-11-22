# Presentation figures and source data

This directory accompanies the Group 7 presentation. Slide numbers below follow the 11-slide version with the noise-impact analysis on slide 3. Filenames retain the team's original plot identifiers.

| Slide | Figure | Script | Inputs supplied with the figure |
| --- | --- | --- | --- |
| 2 | [Real test EEG with three high-noise examples](images/02_eeg_waveforms.png) | [Held-out EEG](plots/02_eeg_waveforms.py) | [Recording, epoch, fold, seed and noise settings](images/02_eeg_waveforms.provenance.json) |
| 3 | [Baseline response to noise](images/06_noise_impact.png) | [Noise impact](plots/06_noise_impact.py) | [Condition scores](tables/baseline_noise.csv) and [clean reference](tables/overall_performance.csv) |
| 6 | [Clean and noisy Macro-F1](images/07_clean_noisy_overview.png) | [Overall comparison](plots/07_clean_noisy_overview.py) | [Means and sample SD](tables/overall_performance.csv) |
| 7 | [Change from E2 across nine conditions](images/08_condition_heatmap.png) | [Condition heatmap](plots/08_condition_heatmap.py) | [Paired differences in percentage points](tables/condition_deltas.csv) |
| 8 | [Waveform error and sleep-stage classification](images/09_rmse_vs_f1.png) | [RMSE and F1](plots/09_rmse_vs_f1.py) | [E4 and E5 under medium Gaussian noise](tables/reconstruction_case.csv) |
| 9 | [Stage-wise F1 under noise](images/10_stagewise_f1.png) | [Stage comparison](plots/10_stagewise_f1.py) | [Five-stage scores](tables/stage_f1.csv) |

## Rebuild from the repository root

With Python 3.10 and the packages in this directory's requirements.txt:

```bash
python -m pip install -r figures/requirements.txt
python figures/make_figures.py --out results/local/presentation-new
```

This writes five PNGs in `images/` and five CSVs in `tables/` inside the new output directory. To draw **all six figures**, point the following command at the directory containing the preprocessed recording `SC4021E0.npz`. Choose a fresh output directory each time.

```bash
python figures/make_figures.py --out results/local/presentation-with-eeg --data-dir /path/to/processed/sleep-edf-20
```

The waveform uses epoch 100 (zero-based) from test fold 0, stage N1, at 100 Hz with noise seed 123. The original EEG is distributed separately. See the [recording manifest](../docs/checks/sleep_edf20_manifest.json) and [preprocessing code](../experiments/preprocess_full.py).

## How the numbers are calculated

- E1 has five clean-fold results. Its original result CSV numbers folds 1–5. E2–E6 use 0–4. The clean reference for strategy comparisons is the team's E1 CPU recheck, which matches E2 Clean. One original fold differs by about 0.0072 percentage points from the CPU recheck. See [the reproduction record](../docs/checks/reproduction/e1_comparison.json).
- E2–E6 each have five folds with one clean and nine noisy conditions. E3 uses [the validation-selected filters](../results/e3_bandpass/comparison/selected_per_fold.csv). Noise-average Macro-F1 first averages nine conditions within a fold and then averages the five folds. Standard deviation uses the five fold averages.
- Heatmap cells use each method's score minus E2 for the same condition. The reconstruction comparison uses the five-fold mean RMSE and classification Macro-F1. Stage F1 uses nine equally weighted condition means per stage. E2's stage table uses percent, while E5 and E6 store fractions; the builder converts those to percent.
- Full-precision inputs are kept in the plotting tables. Rounding is applied to the figure labels.

## Other exports and execution assets

- [Existing clean-baseline comparison and reconstruction exports](../docs/additional_figures/) include SVG and PDF.
- Alternative layouts remain available in [visualization-MAI](https://github.com/0110NiZ/Sleep/tree/dd28f22e95a1405b5aee8755e513302e366996ff) and [tang-ppt-figures](https://github.com/0110NiZ/Sleep/tree/d767e440b83483e9bc0b68f49062ac4ad8a394fc).
- For model inference or exact prediction replay, use the processed EEG and the corresponding trained weights. [Handover notes](../docs/experiment_notes.md) describe the separate prediction and checkpoint bundles. E2 and selected E3 also include per-sample prediction CSVs in the repository. In an extracted ZIP, set `E2_REFERENCE_CSV=results/e2_noisy_baseline/predictions_with_ids.csv` when running E5's E2 verification.
