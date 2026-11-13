# COMP4330 Group 7 — Slides 6 and 7 plots

These figures compare the **formal E2–E6 test results** on simulated EEG noise. They do not reproduce EEG inference or training. The script **only generates PNG and SVG figures**; it does not open or edit PowerPoint files.

## Color palette

The colors come exactly from the presentation reference image:

| Color | Hex |
|---|---|
| Dusty blue | `#9EAAD1` |
| Coral pink | `#F59790` |
| Lavender | `#DACFE5` |
| Soft apricot | `#F5DBB6` |
| Ice blue | `#CDE2E8` |
| Pale blue | `#C8D4E9` |

## Generated files

- `slide6_overall_morandi.png` / `.svg`: E2–E6 nine-noise mean Macro-F1, with five-fold SD error bars.
- `slide7_noise_conditions_morandi.png` / `.svg`: Macro-F1 changes relative to E2 in the nine matched noise settings.
- `data/condition_fivefold_means.csv`: method-by-condition five-fold means.
- `data/overall_nine_noisy.csv`: method-level nine-noise mean + across-fold sample SD.

Both figures use **fivefold means**, not pooled predictions. The displayed labels are rounded after calculations. Differences on slide 7 are in **percentage points**, not relative percent changes. Positive means better than E2; negative means worse.

## Requirements

```bash
python -m pip install numpy matplotlib
```

## Regenerate from the CSVs shipped with these figures

From the `Sleep` repository root:

```bash
python scripts/plot_presentation_6_7.py --out reports/figures/presentation_6_7
```

The script automatically reads `reports/figures/presentation_6_7/data/` when installed at `scripts/`.

## Stronger verification: recompute directly from official fold files

```bash
python scripts/plot_presentation_6_7.py --repo . --out reports/figures/presentation_6_7
```

This mode reads these **current main branch** paths:

```
result/e4_wavelet/e2_reference_per_fold.csv
result/e3_bandpass/comparison/selected_per_fold.csv
result/e4_wavelet/test_per_fold.csv
result/e5_dae/test_per_fold.csv
result/e6_noise_augmentation/test_per_fold.csv
```

It expects 5 folds × (Clean + 9 noise conditions) = 50 unique rows for every method. It recomputes the condition-level means and nine-noise means/SD from the full fold tables. For E3, only the validation-selected official test configuration is used. Never replace it with an arbitrary band-pass candidate.

### Provenance and limitations

The offline CSVs were constructed from the group's formal archived results, cross-checked with the Group 7 presentation and the project README. The offline summary has **full-precision means**. E5/E6 sample SD values were carried from the Group 7 presentation and are shown to two decimal places (`7.74`, `4.46` pp); running `--repo .` recomputes these SDs at full precision from individual folds. The two resulting error bars may therefore differ by less than a chart pixel.

The GitHub project's README reports E2–E6 noisy means of 58.14, 55.98, 62.93, 69.89 and 73.37 percent. These are descriptive results; the five folds are not statistically independent repeated trials. No statistical significance is claimed. Synthetic noise severities are relative within each noise type, not equivalent across types.

Do not commit or modify archived experiment inputs, checkpoints or source prediction files when only updating presentation visuals.
