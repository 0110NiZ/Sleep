# reconstruction

E4/E5 RMSE and classification under Gaussian Medium only (PNG/SVG/PDF). Lower RMSE does not guarantee higher Macro-F1. Both use the same noisy segments and fold-specific frozen classifier.

From repo root: `python visualization/reconstruction/plot.py --out local/runs/reconstruction_new`. Use a fresh output directory. This reads saved CSVs, without training or inference. It also writes supporting tables and provenance. The shared generator calculates both sets of summaries but exports only the selected figure.

Source values are unrounded five-fold means; labels are rounded. Stored provenance describes the original generation commit, not this relocation. See [original figure notes](../../docs/history/comparison_figures_README.md). The shared original provenance.json is in ../clean_performance/.
