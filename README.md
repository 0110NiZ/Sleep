# Sleep: AttnSleep EEG robustness project

COMP 4330SEF group project. E1–E6 experiments have completed. This branch contains E4–E6 handovers; E2/E3 archives remain on E3-bandpass. See HANDOVER_MAINTENANCE.md for paths, historical result provenance, outstanding E6 files and merge notes. No experiment is rerun by these maintenance changes.

## Setup

Use Linux / WSL Ubuntu 22.04, Python 3.10 and an NVIDIA GPU. This training entry point requires CUDA and Linux file locking; Windows-native/CPU execution is not supported. The dependency snapshot was exported from the working environment, not tested by a fresh installation.

Run all commands from the repository root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirement.txt
mkdir -p reports logs runs
python scripts/download_edf20.py
python scripts/preprocess_full.py
python scripts/validate_full_alignment.py
python scripts/train_clean_fivefold.py configs/clean_baseline_fold0.json
```

Download and preprocessing can take time. Existing preprocessed files can instead be placed in `data/processed/sleep-edf-20/` (39 NPZ files). Do not commit EEG datasets or model weights. Each training command trains exactly one fold for 100 epochs and refuses to overwrite an existing output. To train another fold, use its corresponding configuration (fold1 through fold4). No training is triggered when this repository is cloned.

## Evaluation protocol

20 subjects, 39 nights, 42,308 epochs; EEG Fpz-Cz at 100 Hz, 30-second epochs and five sleep stages. Each fold contains 12 training, 4 validation and 4 test subjects. Both nights of a subject stay together. Splits and seed 123 are fixed in `configs/`. Class weights use training labels only. Validation Macro-F1 selects the epoch before test data are loaded. Do not tune using test results.

All five folds completed: Accuracy 80.99% ± 4.41 percentage points; Macro-F1 75.08% ± 4.32 percentage points; Kappa 0.7409 ± 0.0561 (sample standard deviation across folds). Full results and confusion matrix are in `reports/`. These are project fivefold results, not the original paper's 20-fold reproduction.

## Team workflow

Create a branch for each change, submit a pull request and have another member review it. Keep the committed splits unchanged across methods. Select noise and denoising parameters using validation data. Share large data/checkpoints separately. Everyone should use their own virtual environment.

## Attribution and packaging

`AttnSleep/` vendors the author's code from https://github.com/emadeldeen24/AttnSleep at commit `6b4d2665884628c8a7bb09f36589a8ec0992f8e2`, with local compatibility changes recorded in `patches/author_compatibility.patch`. The original MIT license is retained at `AttnSleep/LICENSE`. Model architecture is unchanged. Cite the original AttnSleep paper when reporting results.

This shareable copy uses repository-relative paths. The split manifest checksum changed because its data path changed; subject/file assignments and all training hyperparameters are unchanged. Historical results retain their original split checksum. The source experiment folder and its checkpoints were not modified. No license is granted here for the dataset; obtain it from PhysioNet under its applicable terms.


## E4: wavelet denoising

E4 is complete on this branch. See [E4_RUN.txt](E4_RUN.txt) for setup and commands, [the result report](result/e4_wavelet/E4_RESULT_REPORT.txt) for methods and limitations, and [the comparison table](result/e4_wavelet/e2_e4_comparison.csv).

Five-fold nine-noise mean Macro-F1: E2 58.14%, E4 62.93% (+4.79 percentage points). Clean: E2 75.08%, E4 71.42%. Gains are concentrated in medium/high Gaussian noise; the other seven noisy conditions decline. Validation-selected candidates: W3, W3, W3, W1, W1.

This branch includes the shared Noise v2 generator and CPU E2 evaluator required by E4. It does not depend on E3 filtering. Shared EEG and baseline weights must be obtained separately from the group; they are not stored in Git.
