# E2 · Noisy baseline

Test how much input noise changes the prediction of the same frozen E1 classifier.

## Inputs

Use the 39 preprocessed recordings and the five E1 checkpoints from the [shared setup](README.md). The [noise generator](../experiments/support/shared/noise_conditions.py) uses seed 123 and the recording name and epoch index to reproduce each sample's noise.

| Noise | Low / medium / high | Shape |
| --- | --- | --- |
| Gaussian white noise | 20 / 10 / 5 dB | Scaled to the specified signal-to-noise ratio |
| Baseline drift | 10 / 20 / 40 µV | 0.5 Hz sinusoid |
| Oscillatory burst | 1.75 / 3.5 / 7 × the clean EEG standard deviation | One 0.5–2 s event with a 15–35 Hz carrier |

Three noise types at three strengths give nine noisy conditions. The clean reference makes ten in total. [Noise settings](../configs/noise_conditions_v2.json)

## Run from the repository root

```bash
python experiments/E2_noisy_baseline/run_e2_evaluation_resumable.py --fold all --conditions all --data-dir data/processed/sleep-edf-20 --baseline-dir Sleep_Baseline_5fold --output-dir results/local/e2
python experiments/E2_noisy_baseline/finalize_e2_results.py --e2-dir results/local/e2 --data-dir data/processed/sleep-edf-20
```

The [evaluator](../experiments/E2_noisy_baseline/run_e2_evaluation_resumable.py) resumes completed fold–condition pairs. The [finalizer](../experiments/E2_noisy_baseline/finalize_e2_results.py) matches predictions to EEG segment IDs and computes pooled and stage scores. To check a full local run, use the evaluator with `--check-only --output-dir results/local/e2`. This requires its 50 prediction NPZ files.

## Results

[50 fold–condition scores](../results/E2_noisy_baseline/e2_per_fold.csv) · [423,080 predictions with sample IDs](../results/E2_noisy_baseline/predictions_with_ids.csv) · [Reproduction checks](checks/reproduction/)

The nine-noise score averages the nine conditions within each fold and then averages the five folds. Pooled metrics combine individual predictions across folds.
