# Visualization

Each topic contains its plotting entry and available saved figures/tables. All existing figure bytes were preserved; no plots or experiments were rerun for this cleanup.

- [E2_noise_effect](E2_noise_effect/): E2 baseline performance across noise types and intensities. `plot.py` reads formal E2 tables and exports `e2_baseline_sensitivity.png`. Source: visualization-MAI.
- [noise_examples](noise_examples/): Low/medium/high EEG noise examples imported unchanged from visualization-MAI. `plot.py` is the existing waveform illustration program and requires local EEG. It uses the shared data path; it is not a classifier or training program. Imported example PNGs retain their original names; this cleanup did not regenerate them.
- [method_comparison](method_comparison/): Overall E2–E6 noisy performance and condition-specific improvements. Source: tang-ppt-figures. `overall_noisy_macro_f1` and `noise_condition_improvement` have PNG/SVG versions, renamed from slide-number filenames without changing image bytes.
- [clean_performance](clean_performance/): Latest main clean E1 vs E6 Macro-F1 chart (PNG/SVG/PDF). E1 is the clean reference of the same baseline evaluated under noise in E2. Full E2–E6 clean/noisy tables remain for analysis; they are not all plotted in this two-bar figure.
- [reconstruction](reconstruction/): E4/E5 RMSE and classification under Gaussian Medium only (PNG/SVG/PDF). Lower RMSE does not guarantee higher Macro-F1. Both use the same noisy segments and fold-specific frozen classifier.
- [E4_results](E4_results/): Existing E4 result plotting code. Original exported E4 experiment figures remain in result/E4_wavelet; they were not duplicated or regenerated.


`comparison_plots.py` is the single shared implementation used by clean_performance/plot.py and reconstruction/plot.py. Those entry points select which figure to export and default to fresh output folders under `local/runs/`. The old scripts/plot_robustness_comparison.py command remains available.

Topic scripts do not have identical command-line interfaces. Read the topic notes before running; the E2 and waveform scripts have fixed output paths and can replace their existing figure exports. No separate stage-level plotting program was found in these source branches, so none is invented here. Original stage results remain in result/E*/.

Tables describe five-fold means, not pooled scores. Preserve the exact aggregation and condition definitions when making new figures. Do not use slide numbers as new filenames.
