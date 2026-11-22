# E5 · Denoising autoencoder

Train a network to reconstruct clean EEG and evaluate its output through each fold's frozen E1 classifier.

## Inputs

Use the 39 original preprocessed EEG recordings, the corresponding E1 checkpoints and the submitted E2 predictions with sample IDs. The original E4 and E5 NPZ files were packed differently. The group separately verified that the signal, label, sampling-rate and channel arrays match. The [preprocessing checks](checks/full_alignment_validation.json) document the original EEG/annotation alignment.

## Run from the repository root

```bash
export SLEEP_DATA_DIR=data/processed/sleep-edf-20
export SLEEP_BASELINE_DIR=Sleep_Baseline_5fold
export E2_REFERENCE_CSV=results/E2_noisy_baseline/predictions_with_ids.csv
python experiments/E5_dae/check_e5.py
python experiments/E5_dae/run_e5_pipeline.py
python experiments/E5_dae/finalize_e5.py
```

The [pipeline](../experiments/E5_dae/run_e5_pipeline.py) first verifies the 423,080 E2 CPU predictions and their IDs, then trains the five DAEs and tests the frozen selections. The last command checks the completed run and prepares the result package. Working files go to `results/local/e5_dae/`. For a small fold-0 training check, use `python experiments/E5_dae/run_e5_dae.py --smoke`.

The [residual Conv1D model](../experiments/E5_dae/dae_model.py) has 72,689 parameters. It learns from noisy–clean pairs by minimizing MSE scaled by the training fold's EEG mean-square amplitude. The reconstructed signal returns to µV before classification. Choose the checkpoint with the highest mean validation Macro-F1 across the ten conditions. The [training configuration](../configs/e5_dae.json) allows up to 30 epochs, a minimum of 5 and early stopping with patience 6.

Training samples the nine noisy conditions uniformly, with noise varying by fold and epoch. Each fold has one initialization (seeds 123–127). Validation and test both use seed 123 with disjoint subjects and sample IDs. The proposal described separate validation and test seeds. The submitted implementation follows the shared E2 evaluation protocol.

## Results

[50 test scores and reconstruction errors](../results/E5_dae/test_per_fold.csv) · [Model selections](../results/E5_dae/all_selections_frozen.json) · [Results and limitations](../results/E5_dae/E5_RESULT_REPORT.md) · [Verification](../results/E5_dae/final_verification.json)

RMSE and SNR pool signal samples within each fold and condition. Clean SNR improvement is undefined. CPU timing includes both DAE and classifier, and training time includes validation. Strong Gaussian noise produced variable scores across folds. The group shares the five trained DAEs and the 50 prediction NPZs (E5_predictions.zip) separately. Individual waveform reconstructions require another inference run with the EEG and weights.
