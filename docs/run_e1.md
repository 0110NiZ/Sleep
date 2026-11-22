# E1 · Clean baseline

We trained one AttnSleep model for each of five [subject splits](../configs/fivefold_subject_splits.json). The [five-fold results](../results/e1_clean_baseline/clean_baseline_5fold_results.csv) can be read directly.

For a new run, use the [Ubuntu GPU environment](environment_setup.txt). Place the 39 preprocessed NPZ recordings in `data/processed/sleep-edf-20/`. The [download](../experiments/download_edf20.py) and [preprocessing](../experiments/preprocess_full.py) scripts prepare those files from Sleep-EDF. From the repository root, train the five folds in sequence:

```bash
for fold in 0 1 2 3 4; do
  python experiments/train_clean_fivefold.py "configs/clean_baseline_fold${fold}.json" || break
done
```

Each config names its output under `runs/clean_baseline_5fold/`. Start with fresh fold directories. The runner selects the checkpoint with the highest validation Macro-F1, then evaluates it on the held-out test subjects. To use these trained classifiers in E2, pass `runs/clean_baseline_5fold/` as the baseline directory.
