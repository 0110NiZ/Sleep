# E5 handoff and proposal alignment

## Outcome

E5 is complete: five independently trained DAEs, selected only using ten-condition validation Macro-F1 through each fold's frozen AttnSleep, followed by 50 official CPU test evaluations. The nine-noise score is calculated within each fold first, then summarized across five folds (sample SD). See `E5_RESULT_REPORT.md` for results. No test-driven retraining or parameter changes were made.

## Proposal review

Checked against the user's final `SessionP01_Group7_ Proposal.pdf`, Sections 3.1–3.5 and 4.2.

- Consistent: 39 records / 42,308 preprocessed 30-second epochs, subject-disjoint 12/4/4 splits, common Noise v2, one mixed-noise DAE per fold, paired clean targets, MSE training, frozen fold-specific classifier, validation classification selection, all ten test conditions, fold summaries separate from pooled metrics, reconstruction and computational measures.
- Documented difference: the proposal says independent training/validation/test noise seeds. The implemented shared E2–E5 evaluation protocol uses base seed 123 plus noise type and unique sample ID for both validation and test. Subjects and sample IDs are disjoint within each fold; this is not identical validation/test data or evidence of leakage. Training uses separate fold/epoch seeds. Keep the existing evaluation instances for comparison, and describe this actual protocol in the final report instead of claiming three distinct phase seeds.
- Clarification: the proposal's EEG input remains in microvolts. DAE uses a reversible internal scalar calculated exclusively from that fold's training EEG; output returns to microvolts before AttnSleep. MSE is divided by the same constant squared. No test-statistic normalization is used.
- Implementation details filled in after proposal: residual Conv1D DAE, 72,689 parameters, Adam, maximum 30 epochs, patience 6, minimum 5. These are one experimental configuration, not an exhaustive DAE search. Fold-specific initialization seeds are 123–127, with one run per fold; this does not evaluate repeated-seed variability.
- Noise v2 actually uses Gaussian 20/10/5 dB; drift 0.5 Hz with 10/20/40 microvolt peak amplitude; one 15–35 Hz burst of 0.5–2 seconds with amplitude ratios 1.75/3.5/7. Do not describe drift as 0.1–0.4 Hz.

## Uploaded small deliverables

| File or directory | Use |
|---|---|
| `test_per_fold.csv` | 50 fold-condition rows: accuracy, Macro-F1, kappa, difference from E2, RMSE, SNR improvement, combined inference time |
| `test_summary.csv`, `overall.json` | Ten-condition five-fold mean/sample SD, nine-noise aggregate and separate Clean score |
| `stage_summary.csv` | Stage-wise precision, recall and F1 mean/sample SD |
| `fold*/test_metrics.json` | Full per-fold confusion matrices, stage metrics and reconstruction/timing metrics |
| `pooled_details.json` | Separately labelled pooled-epoch metrics and confusion matrices |
| `fold*/history.json`, `fold*/validation/` | Training loss, all epoch validation scores, condition sampling counts and epoch durations |
| `fold*/selection.json`, `all_selections_frozen.json` | Selected epochs, validation scores, checkpoint hashes and training-plus-validation elapsed time |
| `manifest.json`, `data_audit.json`, `e2_reproduction.json`, `final_verification.json` | Configuration/environment/source hashes, alignment and independent result checks |
| `e2_e4_e5_comparison.csv`, `figures/` | Matched-condition comparisons, aggregate chart, validation curves and pooled confusion matrices |
| `E5_RESULT_REPORT.md`, `E5_HANDOFF.md` | Interpretation, limitations and delivery inventory |

Implementation is in `scripts/dae_model.py`, `scripts/run_e5_dae.py`, `scripts/check_e5.py`, `scripts/background_e5.py`, `scripts/finalize_e5.py`; configuration is `configs/e5_dae.json`; setup/run instructions are in `E5_RUN.md`.

## Kept local: share separately when needed

`/home/nzd/Sleep/e5_project/result/local/e5_dae/` holds five `best.pth`, five `latest.pth`, and 50 `predictions_*.npz` containing sample IDs, ground truth and predicted classes (423,080 prediction rows in total), plus pipeline logs. No raw EEG, processed EEG, model weights or prediction NPZs are uploaded by the E5 commits. Baseline weights and Clean NPZ inputs must also be obtained through shared storage. Set `SLEEP_DATA_DIR` as documented before running checks.

The branch is based on `E4-wavelet`, so it includes inherited E4 files/history. It is not a main merge.

## Analysis coverage and limits

The delivered small files support the planned fold/condition/stage comparisons, confusion matrices, Clean trade-offs, reconstruction versus classification and descriptive timing analysis. Subject-level analysis and paired resampling additionally need the local prediction NPZs. Reconstructed waveforms and class probabilities were not saved: waveform examples require inference again using frozen weights; calibration/AUROC analysis is not covered by the current prediction files.

RMSE/SNR are pooled over sample points within each fold/condition, not an average of per-epoch SNRs. Clean SNR improvement is undefined and left blank. The independent check verifies saved reconstruction metrics' algebraic consistency, not a fresh reconstruction run. Reconstruction quality uses preprocessed Clean as reference, not artifact-free physiological ground truth.

Reported training duration includes validation. CPU inference includes DAE plus classifier with warm-up, excludes noise generation/disk I/O, and is not a dedicated repeated benchmark. Cross-method speed comparisons require matched hardware and timing procedures. No statistical significance or multi-seed stability claim is made. Strong-Gaussian variance and early stopping in folds 1/3 are material limitations, not reasons for post-test tuning.
