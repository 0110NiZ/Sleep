# Results and analysis

Formal outputs live in [result/](../result/), with one folder per experiment. All existing result files were retained, including E3 candidates and formal validation-selected summaries, E4 parameter selections, E5/E6 fold selections and E6's 500 validation records.

Use the root [README](../README.md) for the overall table. Use [visualization](../visualization/README.md) for presentation figures and their source CSVs.

For noisy performance, average nine conditions inside each fold, then calculate the mean and sample SD across five folds. Do not substitute pooled prediction metrics. E3 comparisons must use `E3_bandpass/comparison/selected_per_fold.csv`, not the best test-scoring candidate. E2's clean row is the shared E1 baseline reference.

`code/summarize_results.py` provides the existing summary program under its new name. Check its CLI options before use and write any new output outside the formal archive.

Keep interpretation descriptive: folds overlap in training data; improvements are not necessarily significant or universal. Compare classification, clean performance and noise-specific effects together. Lower RMSE need not imply higher Macro-F1. E6 does not reconstruct waveforms, so RMSE/SNR improvement does not apply. Timing comparisons require matching hardware and measurement scope.

For paired analyses, obtain the original full prediction packages separately. Do not infer that a package is present locally merely because its archived manifest is on GitHub.
