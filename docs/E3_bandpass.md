# E3 · Band-pass filtering

Test whether removing frequencies outside a fixed passband helps the frozen E1 classifier.

## Inputs

Use the same preprocessed EEG, E1 checkpoints and noise as [E2](E2_noisy_baseline.md). First complete the E2 run including finalization. The commands below use its local `results/local/e2/` output, including the prediction NPZ files.

## Run from the repository root

```bash
python experiments/E3_bandpass/run_e3_bandpass.py --stage validation --fold all
python experiments/E3_bandpass/run_e3_bandpass.py --stage test --fold all
python experiments/E3_bandpass/run_e3_bandpass.py --stage summary
python experiments/E3_bandpass/finalize_e3_results.py --e3-dir "$(cat results/local/.e3_active_run.txt)" --e2-dir results/local/e2
```

Validation creates a timestamped working directory. The following stages reuse it. If you set `SLEEP_RESULT_DIR`, use the same root in the last command.

The [filtering code](../experiments/E3_bandpass/run_e3_bandpass.py) applies a fourth-order Butterworth filter forward and backward to each 30-second segment. The candidate passbands were A (0.3–35 Hz), B (0.5–30 Hz), C (0.5–20 Hz) and D (1–30 Hz). The [finalizer](../experiments/E3_bandpass/finalize_e3_results.py) selects one passband for each fold using the highest mean Macro-F1 over the ten validation conditions. The original team recorded the selections after evaluating all candidates on test data. The selection calculation uses validation scores only.

## Results

[Selected test scores](../results/E3_bandpass/comparison/selected_per_fold.csv) · [Filter chosen per fold](../results/E3_bandpass/comparison/parameter_selection.csv) · [All validation candidates](../results/E3_bandpass/validation/) · [All test candidates](../results/E3_bandpass/test/)

Folds 0–3 selected A and fold 4 selected B. Use the 50 selected scores for the main method comparison and the full candidate tables to study sensitivity to passband choice.
