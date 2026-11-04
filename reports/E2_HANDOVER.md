# E2 — Noise Robustness Evaluation: Experiment Handover

## 1. Overview

E2 evaluates the robustness of pretrained AttnSleep models under simulated EEG noise using the Sleep-EDF-20 dataset.

The five AttnSleep models obtained from E1 remain frozen throughout E2. No model retraining or denoising is performed.

**Evaluation pipeline:**

Clean EEG → Noise v2 Generation → Frozen AttnSleep → Sleep Stage Prediction → Evaluation

**Status: Completed**

- Five-fold Test evaluation completed.
- All ten evaluation conditions completed.
- Per-fold and five-fold summary metrics saved.
- Stage-wise metrics and confusion matrices saved.
- Sample identities reconstructed and verified.
- Pooled five-fold metrics computed.

## 2. Dataset and Experimental Setup

| Item | Configuration |
|---|---|
| Dataset | Sleep-EDF-20 |
| Subjects | 20 |
| Recordings | 39 |
| EEG epochs | 42,308 |
| EEG channel | Fpz-Cz |
| Sampling rate | 100 Hz |
| Epoch duration | 30 seconds |
| Samples per epoch | 3,000 |
| Signal unit | μV |
| Sleep stages | W, N1, N2, N3, REM |
| Random seed | 123 |

The experiment uses subject-wise five-fold cross-validation.

Each fold contains 12 training subjects, 4 validation subjects and 4 test subjects. Every subject appears in the Test set exactly once across the five folds.

**Official split configuration:**

`configs/fivefold_subject_splits.json`

The E1 checkpoints are reused for inference. E2 does not retrain AttnSleep.

## 3. Official Noise v2 Implementation

**Noise generator:**

`scripts/noise_conditions_v2.py`

**Noise configuration:**

`configs/noise_conditions_v2.json`

**Main function:**

`apply_noise_v2()`

The previous `noise_generator.py` implementation belongs to Noise v1 and is no longer used for official E2–E6 comparisons.

### 3.1 Evaluation Conditions

E2 includes ten conditions: one Clean condition and nine noisy conditions.

| Condition | Parameter |
|---|---|
| Clean | Original EEG |
| Gaussian_Low | SNR = 20 dB |
| Gaussian_Medium | SNR = 10 dB |
| Gaussian_High | SNR = 5 dB |
| Drift_Low | 0.5 Hz, 10 μV |
| Drift_Medium | 0.5 Hz, 20 μV |
| Drift_High | 0.5 Hz, 40 μV |
| Burst_Low | Scaling coefficient = 1.75σ |
| Burst_Medium | Scaling coefficient = 3.5σ |
| Burst_High | Scaling coefficient = 7σ |

### 3.2 Noise Definitions

**Gaussian Noise**

Gaussian noise is scaled to the specified signal-to-noise ratio (SNR). Signal power is calculated using `mean(x²)`, including the DC component.

**Baseline Drift**

Baseline drift is simulated using low-frequency baseline wander at 0.5 Hz, with amplitude settings of 10, 20 and 40 μV.

**Transient Burst**

Burst noise is implemented as a Hann-windowed, nonsinusoidal oscillatory artifact.

Its waveform combines a sinusoidal carrier and a square-wave component:

`burst_wave = (carrier + 0.5 * np.sign(carrier)) * window`

The carrier frequency is sampled from 15–35 Hz. This range does not represent a strict spectral limit for the complete waveform.

The σ-based values are scaling coefficients relative to the clean EEG standard deviation, not exact final peak amplitudes.

## 4. Reproducibility

Noise v2 is deterministic when the following remain unchanged:

- Clean EEG signal and preprocessing
- Sample ID
- Noise condition and parameters
- Random seed
- Noise generator implementation

The official random seed is **123**.

Sample ID format:

`recording_filename.npz:epoch_index`

Example:

`SC4001E0.npz:100`

The same Sample ID must be used for the same EEG epoch across experiments.

Evaluation uses fixed, reproducible noise instances.

## 5. Evaluation Procedure

For each fold:

1. Load the corresponding E1 AttnSleep checkpoint.
2. Load the fold-specific Test EEG.
3. Apply the ten Noise v2 conditions.
4. Run inference using the frozen AttnSleep model.
5. Save predictions and evaluation metrics.

E2 evaluates five folds × ten conditions = **50 fold-condition combinations**.

The original E2 evaluation saved classification predictions but did not permanently store complete noisy EEG waveforms.

The noisy Test EEG can be regenerated using the official noise generator, configuration and sample identities.

## 6. Results

**Official results directory:**

`result/e2_v2/`

| File | Description |
|---|---|
| `e2_per_fold.csv` | Per-fold metrics for 50 combinations |
| `e2_fivefold_summary.csv` | Five-fold Mean ± SD |
| `e2_details.json` | Stage-wise metrics and confusion matrices |
| `predictions/` | 50 fold-condition prediction NPZ files |
| `predictions_with_ids.csv` | Predictions with subject and sample identities |
| `pooled_summary.csv` | Pooled five-fold Accuracy, Macro-F1 and Kappa |
| `pooled_stage_metrics.csv` | Pooled stage-wise metrics |
| `pooled_confusion_matrix.json` | Pooled confusion matrices |
| `stage_summary.csv` | Five-fold stage-wise Mean ± SD |
| `e2_analysis_manifest.json` | Analysis metadata and verification information |
| `figures/` | Noise waveform visualizations |

### 6.1 Main Results

| Evaluation | Macro-F1 |
|---|---|
| Clean Test | 75.08% |
| Nine noisy Test conditions, equally weighted five-fold mean | 58.14% ± 3.46 percentage points |

Noise substantially reduces the classification performance of the frozen AttnSleep models, with the magnitude of degradation depending on noise type and intensity.

### 6.2 Five-Fold and Pooled Metrics

Two types of aggregated results are reported:

**Five-fold Mean ± SD:** Calculate each fold's metric first, followed by the arithmetic mean and sample standard deviation across five folds (`ddof=1`).

**Pooled metrics:** Concatenate Test predictions from all five folds for a particular noise condition, then calculate the metric from the combined predictions.

Pooled Macro-F1 is not necessarily equal to the average of fold-level Macro-F1 values.

### 6.3 Stage-Wise Metrics

Stage-wise Precision, Recall and F1 are available for:

- W
- N1
- N2
- N3
- REM

Both five-fold summary and pooled stage-wise metrics are provided.

Confusion matrices are also available for individual folds and pooled predictions.

### 6.4 Sample-Level Prediction Records

`predictions_with_ids.csv` contains:

`fold, subject_id, recording_id, epoch_index, sample_id, condition, y_true, y_pred`

The analysis script verified the sample ordering against the preprocessed EEG labels and the archived E2 evaluation results.

These identities allow subsequent analysis by subject, recording, epoch, noise condition and sleep stage.

## 7. Scripts and Commands

### 7.1 Historical E2 Evaluation

`scripts/run_e2_evaluation_v2.py`

This is the original evaluation implementation used for the completed E2 results.

It is preserved for experimental provenance, including the code hash recorded by E3.

It may contain machine-specific absolute paths and should not be treated as the portable default execution entry point.

### 7.2 Resumable E2 Evaluation

`scripts/run_e2_evaluation_v2_resumable.py`

This additional script supports configurable paths, checking existing results and fold-condition-level resumable execution.

Its default result directory is `result/e2_v2/`.

**Verify completed E2 results:**

`python scripts/run_e2_evaluation_v2_resumable.py --check-only`

Verified result:

`E2: 50 complete, 0 missing, 0 partial/inconsistent conditions`

`CHECK ONLY PASS`

This operation does not perform model inference.

### 7.3 E2 Postprocessing

`scripts/finalize_e2_results.py`

**Command:**

`python scripts/finalize_e2_results.py`

This script generates sample-identified predictions, pooled metrics, pooled confusion matrices and stage-wise summaries using saved predictions and preprocessed EEG labels.

It does not retrain models, generate noise or perform inference.

## 8. Requirements for E4–E6

Subsequent experiments should use the same:

- Preprocessed Sleep-EDF-20 dataset
- Five-fold subject splits
- Noise v2 generator and configuration
- Sample identity convention
- Fixed Validation and Test noise generation protocol

**Training:** E5 and E6 may dynamically generate varied noise instances. E6 specifically investigates noise-augmented training and should not be restricted to nine fixed noise realizations.

**Validation and Test:** Use fixed, reproducible noise instances for fair comparison.

Any parameter or model selection must use Validation data, not Test performance.

Report five-fold Mean ± sample SD and pooled metrics separately.

## 9. Data Sharing and Limitations

Preprocessed EEG data and E1 model checkpoints are stored outside the Git repository.

Prediction NPZ files may also be excluded by `.gitignore`.

Therefore, cloning the repository alone may not provide all files necessary for original model inference or full E3 reproduction.

The missing datasets, checkpoints and prediction NPZ files must be obtained separately or regenerated using the documented protocol.

The sample-identified prediction CSV contains labels and identities, not complete EEG waveforms.

## 10. Important Notes

- Noise v2 is the official implementation.
- Historical Noise v1 results must not be mixed with Noise v2 results.
- E2 uses frozen E1 AttnSleep checkpoints.
- All five Test folds and ten conditions have been completed.
- E2 sample identities and pooled metrics have been generated and verified.
- The original E2 evaluation code is retained for historical reproducibility.
- No model retraining is required for the completed E2 postprocessing.

**E2 experiment and statistical analysis: Completed.**