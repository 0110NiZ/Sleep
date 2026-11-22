# Data and experiment checks

| Question | Evidence |
| --- | --- |
| Which recordings and subjects were used? | [39-record manifest](sleep_edf20_manifest.json), [preprocessing summary](preprocess_full_summary.json) and [subject and class balance](data_audit/) |
| Did the EEG segments align with the sleep-stage annotations? | [Data alignment check](full_alignment_validation.json) |
| Did the E1 and E2 evaluations reproduce the baseline? | [CPU and sample-level comparisons](reproduction/) |
| Which E2 scores were used in the E4 comparison? | The original copy and the current [50-row E2 table](../../results/E2_noisy_baseline/e2_per_fold.csv) had identical bytes. The [recorded SHA-256](reproduction/E2_reference.sha256) lets the comparison scripts check that table directly. |
| Were E6's epoch records, weights and predictions checked? | [Validation and package checks](integration_20261110/) |

Original audit files use the paths from their recorded revision. The [run guides](../README.md) give the current commands. The [figure guide](../../figures/README.md) links each PPT figure to its code and data.
