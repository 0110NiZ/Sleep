# method_comparison

Overall E2–E6 noisy performance and condition-specific improvements. Source: tang-ppt-figures. `overall_noisy_macro_f1` and `noise_condition_improvement` have PNG/SVG versions, renamed from slide-number filenames without changing image bytes.

`plot.py --data-dir visualization/method_comparison --out local/runs/method_figures` uses the included tables. `plot.py --repo . --out local/runs/method_figures` reads formal fold results instead. Prefix either command with `python visualization/method_comparison/` from the repo root.

The imported CSVs keep their original full-precision means. E5/E6 SDs in the offline table were rounded to 7.74/4.46; --repo recomputes full precision from archived folds. Error bars are five-fold sample SD, not confidence intervals. Differences are percentage points. CSVs and original figure bytes are unchanged; see the [original source notes](../../docs/history/method_comparison_README.md).
