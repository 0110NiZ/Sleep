# Robustness comparison figures

Generated from formal archived test CSVs, not rounded PPT labels. No training or inference.

- Style: white background, pastel bars, two-decimal value labels. Clean/noisy comparison uses horizontal grouped bars; reconstruction/classification comparison uses two basic bar charts.
- Clean/noisy comparison: clean Macro-F1 and equally weighted nine-noise Macro-F1. Average nine conditions within each fold, then average five folds. CSV SD is sample SD over five fold values. E3 uses validation-selected per-fold filters.
- Reconstruction/classification comparison: Gaussian_Medium only. Average per-fold reported RMSE and Macro-F1 for E4 and E5. Both methods use corresponding frozen E1 classifiers. The reported RMSE is averaged from archived fold metrics, not recomputed from pooled waveforms.
- `clean_noisy_per_fold.csv` supports later paired descriptive analysis. No statistical significance is implied and folds are not independent repetitions.
- PNG: 300 dpi for slides. SVG/PDF: vector outputs. SVG retains text. These are figure files; they do not contain PowerPoint chart workbooks. Source CSV and plotting script support regeneration.
- Differences use unrounded source values; subtraction of displayed two-decimal labels can differ by 0.01 pp.
- `provenance.json` records input hashes and source commit. Original result files remain unchanged.

## Files and interpretation

| File | Contents |
|---|---|
| `clean_vs_noisy_macro_f1.png` | E2–E6 clean and nine-noise mean Macro-F1; pastel horizontal grouped bars |
| `reconstruction_vs_classification_gaussian_medium.png` | E4/E5 RMSE and Macro-F1 under Gaussian Medium; two bar charts |
| `clean_noisy_summary.csv` | Five-fold means, sample SDs and clean/noisy gaps |
| `clean_noisy_per_fold.csv` | Paired per-fold clean and noisy scores |
| `reconstruction_classification_gaussian_medium.csv` | Values plotted in the E4/E5 comparison |
| `provenance.json` | Formal input hashes, source commit and full-precision differences |

Each chart also has SVG and PDF vector versions. Labels show two decimals; source CSVs retain full precision. In the first figure blue means Clean and green means noisy mean; in the second blue means E4 and green means E5. RMSE is lower-is-better, Macro-F1 higher-is-better. Both charts show descriptive means without uncertainty bars; sample SDs for the overall comparison are retained in the summary CSV. Gaussian Medium is one illustrative condition, not a conclusion about every condition.

From repository root in the Sleep environment:

```bash
source /home/nzd/Sleep/.venv/bin/activate
python scripts/plot_robustness_comparison.py --out /tmp/sleep-robustness-figures-new
```

Use a new output directory. Requirements: numpy and matplotlib. The default output is `reports/figures/robustness_comparison`; the script refuses to overwrite an existing directory.
