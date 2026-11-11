> 已完成实验。下文命令和代码/数据路径均以仓库根目录为基准，不以 docs/ 为基准。历史运行请使用记录的原提交，不用维护代码继续写入原输出目录。目录迁移见 [DELIVERY_LAYOUT.md](DELIVERY_LAYOUT.md)。

# E3 — Butterworth Band-Pass Filtering: Experiment Handover

## 1. Overview

E3 investigates whether Butterworth band-pass filtering can improve AttnSleep sleep-stage classification under simulated EEG noise.

E3 uses the same Sleep-EDF-20 dataset, five-fold subject splits, Noise v2 protocol and frozen E1 AttnSleep checkpoints as E2.

The classification models are not retrained.

**Evaluation pipeline:**

Clean EEG → Noise v2 Generation → Band-Pass Filtering → Frozen AttnSleep → Classification Evaluation

**Status: Completed**

- Four filter candidates evaluated.
- All five Validation folds completed.
- All five Test folds completed.
- Validation-based parameter selection completed.
- Official fold-wise selected Test results generated.
- Sample-level predictions and pooled metrics generated.
- Stage-wise metrics and confusion matrices generated.
- Noise-quality and runtime measurements recorded.

## 2. Shared Experimental Setup

E3 uses the same configuration as E2.

| Item | Configuration |
|---|---|
| Dataset | Sleep-EDF-20 |
| Subjects | 20 |
| Recordings | 39 |
| Total EEG epochs | 42,308 |
| EEG channel | Fpz-Cz |
| Sampling rate | 100 Hz |
| Epoch duration | 30 seconds |
| Sleep stages | W, N1, N2, N3, REM |
| Number of folds | 5 |
| Noise conditions | Clean + 9 noisy conditions |
| Noise seed | 123 |
| Classifier | Frozen E1 AttnSleep |

**Shared files:**

- `configs/fivefold_subject_splits.json`
- `configs/noise_conditions_v2.json`
- `scripts/noise_conditions_v2.py`

See [E2_HANDOVER.md](E2_HANDOVER.md) for the complete Noise v2 definitions and reproducibility requirements.

## 3. Band-Pass Filtering

E3 evaluates four fourth-order Butterworth band-pass candidates.

| Filter ID | Passband |
|---|---|
| A | 0.3–35 Hz |
| B | 0.5–30 Hz |
| C | 0.5–20 Hz |
| D | 1.0–30 Hz |

Filtering is implemented using second-order sections (SOS) and forward-backward filtering (`sosfiltfilt`).

Filtering is applied independently to each 30-second EEG epoch.

This implementation provides zero-phase forward-backward filtering under the chosen boundary-handling method.

The resulting filtered EEG is classified using the corresponding frozen AttnSleep checkpoint.

No AttnSleep parameters are modified during E3.

## 4. Evaluation Protocol

Four candidate filters are evaluated for every fold under all ten noise conditions.

### 4.1 Validation

For each fold:

1. Load the Validation EEG.
2. Generate fixed Noise v2 conditions.
3. Apply each of the four filter candidates.
4. Evaluate using the frozen E1 AttnSleep checkpoint.
5. Save classification and signal-quality metrics.

Total Validation combinations:

5 folds × 4 filters × 10 conditions = **200 combinations**.

### 4.2 Parameter Selection

Following the project proposal, E3 selects one filter per fold.

For each fold, calculate the equally weighted mean Macro-F1 across all ten Validation conditions, including Clean, for each candidate filter.

Select the candidate with the highest Validation average Macro-F1.

**Final Validation-selected parameters:**

| Fold | Selected filter |
|---|---|
| 0 | A |
| 1 | A |
| 2 | A |
| 3 | A |
| 4 | B |

Each fold uses one selected filter across all ten Test conditions.

Different filters are not selected separately for Gaussian, Drift or Burst conditions.

**Important methodological note:** The selection record was reconstructed from the saved Validation results after all candidate Test evaluations had already been performed. It must not be described as an automatically locked, timestamped selection made before the original Test runs.

The official selection rule uses Validation metrics only. Test performance is not used to choose the selected filters.

### 4.3 Test

All four filter candidates were also evaluated on the Test sets.

Total candidate Test combinations:

5 folds × 4 filters × 10 conditions = **200 combinations**.

The complete A–D candidate results are preserved for parameter-sensitivity analysis.

For the official E3 comparison, only the Validation-selected candidate for each fold is used:

- Fold 0: A
- Fold 1: A
- Fold 2: A
- Fold 3: A
- Fold 4: B

Official selected Test combinations:

5 folds × 10 conditions = **50 combinations**.

No additional model inference was necessary to construct this final selected result because all candidate predictions had already been saved.

## 5. Main Results

### 5.1 Official Nine-Noise Performance

The overall nine-noise score is calculated by averaging each fold's Macro-F1 equally across the nine noisy conditions, followed by a five-fold mean and sample standard deviation.

| Method | Test Macro-F1 Mean ± SD |
|---|---|
| E2 — No Filtering | 58.14% ± 3.46 pp |
| E3 — Validation-selected Filtering | 55.98% ± 6.88 pp |

**E3 − E2 = −2.16 percentage points**

Under the evaluated conditions, Validation-selected band-pass filtering does not improve the overall average robustness of frozen AttnSleep.

### 5.2 Interpretation

Filtering can help under certain severe noise conditions, including some strong Gaussian and baseline-drift settings.

However, the effects depend on noise type, intensity and filter passband.

Filtering also changes the distribution of EEG signals provided to the pretrained classifier.

The observed performance reductions on Clean and several noisy conditions are consistent with possible removal or alteration of useful EEG information or input-distribution changes. These explanations are hypotheses, not independently established causes.

The negative overall result is a valid experimental finding and should be reported without modifying the already established Test noise protocol.

### 5.3 Pooled Five-Fold Results

Pooled metrics are calculated by concatenating the five Test folds' predictions under the selected filter for each condition.

For example, the selected E3 Clean condition has:

- 42,308 pooled Test epochs.
- Pooled Macro-F1: **64.41%**.

This pooled Macro-F1 must not be confused with the arithmetic mean of five fold-wise Macro-F1 scores.

The same distinction applies to all noisy conditions.

## 6. Original Experiment Results

**Experiment directory:**

`result/e3_bandpass/`

The directory name identifies the original E3 experiment run.

### 6.1 Validation

`validation/`

Contains:

- `per_fold.csv`
- `summary.csv`
- `details.json`
- `e2_reference_per_fold.csv`

These files preserve the four candidate filters' Validation results.

### 6.2 Test

`test/`

Contains:

- `per_fold.csv`
- `summary.csv`
- `details.json`
- `e2_reference_per_fold.csv`

These files preserve the complete A–D Test results.

### 6.3 Figures

`figures/`

Contains experiment visualizations including:

- Validation versus Test performance.
- E2 versus E3 condition-level comparisons.
- EEG signal and filter comparisons.

### 6.4 Experiment Metadata

`experiment_info.json`

This file belongs to E3 and stores information about the experiment configuration, implementation and source-file hashes.

It should remain in the original E3 experiment directory.

Historical metadata should not be overwritten merely to make a modified source file appear identical to the original experiment.

## 7. Official Finalized E3 Results

The official Validation-selected results are saved under:

`result/e3_bandpass/comparison/`

| File | Description |
|---|---|
| `validation_candidate_scores.csv` | Validation scores of A–D for all folds |
| `parameter_selection.csv` | Final fold-wise filter selection |
| `selected_per_fold.csv` | Official 50-row selected Test result |
| `selected_summary.csv` | Condition-wise five-fold Mean ± SD and E2 comparison |
| `selected_stage_summary.csv` | Stage-wise five-fold Mean ± SD |
| `selected_pooled_summary.csv` | Pooled five-fold classification metrics |
| `selected_pooled_stage_metrics.csv` | Pooled stage-wise Precision, Recall and F1 |
| `selected_pooled_confusion_matrix.json` | Pooled confusion matrices |
| `selected_fold_details.json` | Per-fold stage metrics and confusion matrices |
| `selected_noisy_per_fold.csv` | Fold-wise averages across nine noisy conditions |
| `selected_overall.json` | Overall selected E3 and E2 comparison |
| `selected_analysis_manifest.json` | Finalization metadata |
| `selected_quality_time_summary.csv` | Signal-quality and timing summaries, generated when supported columns are available |

The original A–D comparison tables remain available and are not replaced by the selected-results tables.

### 7.1 Sample-Level Predictions

Official selected predictions are saved in:

`predictions/selected_predictions_with_ids.csv`

Fields:

`fold, subject_id, recording_id, epoch_index, sample_id, condition, filter_id, y_true, y_pred`

Sample identity mappings are obtained from the verified E2 sample-identification records.

The E3 finalization script checks the corresponding `y_true` values and sample ordering before attaching identities.

This enables:

- E2 versus E3 error comparison at the EEG-epoch level.
- Analysis by subject and sleep stage.
- Pooled five-fold evaluation.
- Confusion-matrix calculation.

The complete original A–D prediction NPZ files are retained for additional analysis.

## 8. Denoising Quality and Computational Cost

The original E3 Test results record signal-quality and computational measurements, including:

- RMSE-related measurements.
- SNR improvement.
- Filtering time.
- Classification inference time.

These measurements are preserved in:

`test/per_fold.csv`

The finalization script summarizes compatible measurements across the five selected folds in:

`comparison/selected_quality_time_summary.csv`

The exact available columns should be checked before reporting numerical values.

Filtering and classification time should be clearly distinguished from noise-generation time.

Signal-quality improvements do not necessarily translate into improved classification Macro-F1.

## 9. Scripts and Execution

### 9.1 Original E3 Experiment

`scripts/run_e3_bandpass.py`

The original experiment implements:

- Noise v2 generation.
- Butterworth filtering.
- Frozen AttnSleep inference.
- Validation and Test evaluation.
- Candidate-filter comparison and visualization.

Historical source code and experiment metadata are retained for reproducibility.

### 9.2 E3 Finalization

`scripts/finalize_e3_results.py`

**Command:**

`python scripts/finalize_e3_results.py`

This script:

1. Reads the saved Validation results.
2. Selects one filter per fold using the predefined rule.
3. Extracts the corresponding Test metrics and saved predictions.
4. Verifies sample identity alignment with E2.
5. Generates the official selected results.
6. Computes five-fold and pooled metrics.

**Verified local output:**

`Validation-selected filters: ['A', 'A', 'A', 'A', 'B']`

`E3 FINALIZATION PASS`

`E3 nine-noise Test Macro-F1: 55.98% +/- 6.88 pp`

`E2 nine-noise Test Macro-F1: 58.14% +/- 3.46 pp`

`Difference: -2.16 pp`

Finalization does not retrain AttnSleep, regenerate noise, rerun filtering or perform inference.

## 10. Reproducibility Requirements

To reproduce E3, use:

- The same preprocessed EEG recordings.
- The same subject-wise five-fold splits.
- The same E1 checkpoints.
- The same Noise v2 configuration and source implementation.
- The same sample IDs and random seed.
- The same candidate filter definitions.
- The predefined Validation-based parameter selection rule.

The original E3 experiment also refers to E2 results and saved predictions.

The E2 Test prediction NPZ files may not be included in the Git repository because of `.gitignore`.

A full original E3 rerun may therefore require separately shared E2 predictions, clean EEG data and model checkpoints.

The E3 finalization script does not need the original EEG waveforms, but it requires the saved E3 predictions, E2 sample-identity CSV and the corresponding metric tables.

## 11. Requirements for Cross-Experiment Comparison

For formal comparison with E2, E4, E5 and E6:

- Use the Validation-selected E3 result, not the highest Test-performing candidate.
- Preserve the same evaluation conditions and fixed noise instances.
- Report Clean performance separately from the nine-noise average.
- Report five-fold Mean ± sample SD.
- Report pooled metrics separately.
- Include stage-wise Recall/F1 and confusion matrices.
- Use the full A–D results only for supplementary parameter-sensitivity analysis.

The official fold selections are **A, A, A, A, B**.

## 12. Important Notes

- E3 does not retrain AttnSleep.
- The existing E1 checkpoints remain frozen.
- All A–D Validation and Test results are preserved.
- Final E3 parameters are selected using Validation only.
- The saved selection record was reconstructed after the candidate evaluations, without inventing a pre-Test selection timestamp.
- Sample IDs and pooled classification metrics have been generated.
- E3 does not improve the overall nine-noise average Macro-F1 under the evaluated settings.
- No additional training, filtering or inference is required for the completed final analysis.

**E3 experiment and statistical analysis: Completed.**