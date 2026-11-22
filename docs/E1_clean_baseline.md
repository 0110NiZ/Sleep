# E1 · Clean baseline

Train AttnSleep on clean EEG to establish the reference classifier reused in E2–E5.

## Inputs

Use the 39 preprocessed recordings in `data/processed/sleep-edf-20/` and the [five subject splits](../configs/fivefold_subject_splits.json). Follow the [shared setup](README.md) for the GPU training environment.

## Run from the repository root

```bash
for fold in 0 1 2 3 4; do
  python experiments/E1_clean_baseline/train_clean_fivefold.py "configs/clean_baseline_fold${fold}.json" || break
done
```

Start with fresh output folders for this run. Each [fold config](../configs/clean_baseline_fold0.json) points to `runs/clean_baseline_5fold/`. The [training code](../experiments/E1_clean_baseline/train_clean_fivefold.py) calculates class weights from training labels, selects the checkpoint with the highest validation Macro-F1, then evaluates the held-out test subjects. To use these new weights in E2, pass `runs/clean_baseline_5fold` as the baseline directory.

## Results

[Five clean test scores](../results/E1_clean_baseline/clean_baseline_5fold_results.csv) · [CPU reproduction check](checks/reproduction/e1_comparison.json)

The original CSV labels the folds 1–5. The evaluation scripts label those same folds 0–4. The PPT uses the five-fold CPU clean reference matched to E2.
