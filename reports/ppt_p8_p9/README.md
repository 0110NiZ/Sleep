# PPT pages 8–9 figures

Generated from formal archived test CSVs, not rounded PPT labels. No training or inference.

- Style: white background, pastel bars, two-decimal value labels. Page 8 uses horizontal grouped bars; page 9 uses two basic bar charts.
- Page 8: clean Macro-F1 and equally weighted nine-noise Macro-F1. Average nine conditions within each fold, then average five folds. CSV SD is sample SD over five fold values. E3 uses validation-selected per-fold filters.
- Page 9: Gaussian_Medium only. Average per-fold reported RMSE and Macro-F1 for E4 and E5. Both methods use corresponding frozen E1 classifiers. The reported RMSE is averaged from archived fold metrics, not recomputed from pooled waveforms.
- `p8_per_fold.csv` supports later paired descriptive analysis. No statistical significance is implied and folds are not independent repetitions.
- PNG: 300 dpi for slides. SVG/PDF: vector outputs. SVG retains text. These are figure files; they do not contain PowerPoint chart workbooks. Source CSV and plotting script support regeneration.
- Differences use unrounded source values; subtraction of displayed two-decimal labels can differ by 0.01 pp.
- `provenance.json` records input hashes and source commit. Original result files remain unchanged.

From repository root in the Sleep environment:

```bash
source /home/nzd/Sleep/.venv/bin/activate
python scripts/plot_ppt_p8_p9.py --out /tmp/sleep-p8-p9-new
```

Use a new output directory. Requirements: numpy and matplotlib. The default output is `reports/ppt_p8_p9`; the script refuses to overwrite an existing directory.
