# clean_performance

Latest main clean E1 vs E6 Macro-F1 chart (PNG/SVG/PDF). E1 is the clean reference of the same baseline evaluated under noise in E2. Full E2–E6 clean/noisy tables remain for analysis; they are not all plotted in this two-bar figure.

From repo root: `python visualization/clean_performance/plot.py --out local/runs/clean_performance_new`. Use a fresh output directory. This reads saved CSVs, without training or inference. It also writes supporting tables and provenance. The shared generator calculates both sets of summaries but exports only the selected figure.

Source values are unrounded five-fold means; labels are rounded. Stored provenance describes the original generation commit, not this relocation. See [original figure notes](../../docs/history/comparison_figures_README.md). Reconstruction exports are in ../reconstruction/.
