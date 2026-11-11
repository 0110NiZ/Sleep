# Sleep: AttnSleep EEG robustness project

COMP 4330SEF group project. **E1–E6 are complete.** This repository contains their code and archived results. Round-one delivery organization is merged at `61f4d48`. Round-two E2 naming maintenance (`4a1ec93`) is also integrated into main, retaining legacy compatibility without changing experimental settings, noise logic, models or recorded results.

## Find what you need

| Location | Purpose |
|---|---|
| [docs/](docs/README.md) | Environment, experiment run instructions and handover guidance |
| [configs/](configs/) | Fixed subject splits, Noise v2 and experiment settings |
| [scripts/](scripts/) | Training, evaluation, checks and analysis entry points |
| [result/](result/) | Completed E1–E6 experiment archives |
| [reports/](reports/README.md) | Data checks, reproduction evidence and integration audits |
| [demo/](demo/README.md) | Placeholder only: web-demo source has not been delivered |
| [AttnSleep/](AttnSleep/) | Vendored author implementation and license |

EEG data, prediction NPZs and model weights are shared separately. `data/`, `runs/` and `logs/` are local storage locations, not evidence that these large files are in Git. Cloning the repository does not run any experiment.

## Completed experiments

| Experiment | Archived results | Instructions |
|---|---|---|
| E1 Clean baseline | [e1](result/e1/) | [Environment and baseline entry point](docs/ENVIRONMENT.txt) |
| E2 Noisy baseline | [e2_v2](result/e2_v2/) | [E2 handover](docs/E2_HANDOVER.md) |
| E3 Band-pass | [e3_bandpass](result/e3_bandpass/) | [E3 handover](docs/E3_HANDOVER.md) |
| E4 Wavelet | [e4_wavelet](result/e4_wavelet/) | [E4 run notes](docs/E4_RUN.txt) |
| E5 DAE | [e5_dae](result/e5_dae/) | [E5 run notes](docs/E5_RUN.md) |
| E6 Noise augmentation | [e6_noise_augmentation](result/e6_noise_augmentation/) | [E6 guide](docs/E6_START_HERE.md) |

E3's formal comparison uses [selected_per_fold.csv](result/e3_bandpass/comparison/selected_per_fold.csv), selected using validation data; the other candidate test results are retained for sensitivity analysis. Do not replace formal scores with the best candidate Test scores.

All five E1 folds completed: Accuracy 80.99% ± 4.41 percentage points; Macro-F1 75.08% ± 4.32 percentage points; Kappa 0.7409 ± 0.0561. These are project five-fold results, not the original paper's 20-fold reproduction.

## Analyze existing results without rerunning models

Run commands from the repository root:

```bash
python scripts/summarize_all_experiments.py --e3 result/e3_bandpass/comparison/selected_per_fold.csv --out result/local/analysis_round1
```

The output directory must not exist. This command only reads archived CSVs and writes a new summary; it does not train, infer or regenerate noise. Its E2 reference comes from the verified E4 handover CSV. `summarize_e6.py` remains a historical E2/E4/E5/E6 comparison entry point; use the command above for all five methods.

## Protocol and environment

Sleep-EDF-20: 20 subjects, 39 nights and 42,308 30-second Fpz-Cz EEG epochs at 100 Hz. Within each fold, 12 subjects train, 4 validate and 4 test; all nights of one subject stay together. Splits and seed 123 are fixed in `configs/`. Model/parameter selection uses validation data, never Test results. Noise v2 protocol parameters and original generator logic are unchanged; current entry names have compatibility aliases.

The E1 training entry point requires Linux/WSL, Python 3.10 and CUDA; other experiment environments are recorded in their run notes and manifests. See [ENVIRONMENT.txt](docs/ENVIRONMENT.txt) before installing dependencies. The full dependency snapshot has not been verified by a fresh installation and is not a universal Windows/Mac installer.

## Handover and provenance

See [current handover status](docs/HANDOVER_MAINTENANCE.md) and [old-to-new path map](docs/DELIVERY_LAYOUT.md). E6's 500 validation files and the separately shared prediction/checkpoint package passed independent checks recorded in [integration_20261110](reports/integration_20261110/). Demo prediction agreement is a separate check and is not claimed by this branch.

Archive contents, manifest hashes and historical commit references are preserved. Historical runs must be reproduced/resumed using the recorded original code and environment; do not rewrite old manifests to match maintenance code or continue writing into old output directories.

`AttnSleep/` vendors [the author's implementation](https://github.com/emadeldeen24/AttnSleep) at commit `6b4d2665884628c8a7bb09f36589a8ec0992f8e2`. Compatibility changes are recorded in [author_compatibility.patch](patches/author_compatibility.patch); the original [MIT license](AttnSleep/LICENSE) is retained. Cite the original AttnSleep paper. Dataset terms are provided by PhysioNet; this repository does not grant a dataset license.

Changes should use a separate branch and peer review. Do not change committed splits or tune from Test results. Keep data and weights outside ordinary Git.

## E2 naming maintenance

Round one is merged at `61f4d48`. Round two (`4a1ec93`) is integrated into main. Use `scripts/run_e2_evaluation_resumable.py` and `from noise_conditions import apply_noise`. Old `*_v2` imports/commands remain compatibility entries. See [compatibility and archive rules](docs/E2_NAMING.md). The official configuration remains `configs/noise_conditions_v2.json`; the version identifies the experimental protocol.
