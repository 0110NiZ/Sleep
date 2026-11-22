# E6 · Noise-augmented training

Train AttnSleep to classify noisy EEG directly. Each fold starts with a new classifier and its original sleep-stage labels.

## Inputs

Use the same 39 preprocessed NPZ files as [E5](E5_dae.md). The runner checks their hashes against the E5 data record during loading. For a repacked copy, compare the actual arrays with the originals before using `--allow-repacked-data`. The [E5 guide](E5_dae.md) describes which arrays the group verified for its repacked inputs.

## Run from the repository root

```bash
python experiments/E6_noise_augmentation/run_e6_noise_augmentation.py --stage check --data-dir data/processed/sleep-edf-20
python experiments/E6_noise_augmentation/run_e6_noise_augmentation.py --stage train --device cuda --data-dir data/processed/sleep-edf-20 --out results/local/e6_noise_augmentation
python experiments/E6_noise_augmentation/run_e6_noise_augmentation.py --stage test --device cuda --data-dir data/processed/sleep-edf-20 --out results/local/e6_noise_augmentation
```

Use the same output directory and training-device argument for train and test. The [runner](../experiments/E6_noise_augmentation/run_e6_noise_augmentation.py) checks the training environment and uses CPU inference for the official test. For a one-epoch trial, use `--stage smoke --device cuda --out results/local/e6_smoke` instead of the train and test stages.

Each training segment has a 50% chance of staying clean. The other half sample one of the nine noisy conditions with equal probability. Noise is sampled again each epoch. Each fold uses initialization seed 123 and trains for 100 epochs with batch size 128. Mean Macro-F1 over the ten validation conditions selects the checkpoint, with the earlier epoch winning ties. All five selections are frozen before testing.

## Results

[Fold and condition scores](../results/E6_noise_augmentation/test_per_fold.csv) · [Frozen selections](../results/E6_noise_augmentation/all_selections_frozen.json) · [Five-fold summary](../results/E6_noise_augmentation/overall.json) · [Prediction and weight checks](checks/integration_20261110/)

The archive keeps 100 validation records per fold. The group shares the 50 prediction NPZs and five model weights separately. E6 and the frozen-classifier methods use matched test conditions. Retraining, initialization and checkpoint selection can all contribute to the observed difference. The 50% clean probability was fixed in advance.
