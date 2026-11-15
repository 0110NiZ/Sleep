# Experiment code

Start with `E1_baseline.py`, `E2_noisy_baseline.py`, `E3_bandpass.py`, `E4_wavelet.py`, `E5_dae.py` or `E6_augmentation.py`. `E2_evaluation.py` is the shared evaluation implementation used by other experiments.

Shared functions: `noise.py`, `dae_model.py`, `wavelet.py`, `paths.py`. Data preparation is in `data/`; audit utilities are in `checks/`. The `finalize_*`, `summarize_*` and `background_E5.py` files retain their original roles.

Run from repository root. These are experimental programs, not all import-safe libraries: E1 and some preparation/finalization utilities start work directly. Read [the guides](../docs/README.md) before running them. Old names in `scripts/` remain compatibility entries to these same implementations.
