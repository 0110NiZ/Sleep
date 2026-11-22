# Running the project

Run commands from the repository root. The [environment notes](environment_setup.txt) describe our Ubuntu/WSL setup. Use [requirements.txt](../requirements.txt) for the full environment or [the figure guide](../figures/README.md) for plotting only.

| Experiment | What the guide covers |
| --- | --- |
| [E1 · Clean baseline](run_e1.md) | Data, five subject splits and baseline training |
| [E2 · Noisy baseline](run_e2.md) | Add noise and evaluate the frozen classifier |
| [E3 · Band-pass](run_e3.md) | Filter candidates and validation-based selection |
| [E4 · Wavelet](run_e4.md) | Wavelet settings, dependencies and evaluation |
| [E5 · Denoising autoencoder](run_e5.md) | Training pairs, model selection and verification |
| [E6 · Noise augmentation](run_e6.md) | Check the data, train and test the augmented classifier |

For the browser demo, see [local setup and validation](../demo/README.md). For additional analysis, see the [experiment notes](experiment_notes.md), [data and reproduction checks](checks/README.md) and [additional figure exports](additional_figures/README.md). The [archive path history](history/layout.md) and [E2 compatibility notes](e2_compatibility.md) explain older filenames in the original run records.
