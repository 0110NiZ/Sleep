# Round-one delivery layout

Base commit: `091185df1ff993e7b65109e8c1db63a31f009d06`.
Branch: `maintenance/delivery-layout-round1`. Main is not modified or merged by this work.

## Moved paths

| Old repository path | Current repository path |
|---|---|
| `E4_RUN.txt` | `docs/E4_RUN.txt` |
| `E5_RUN.md` | `docs/E5_RUN.md` |
| `E6_START_HERE.md` | `docs/E6_START_HERE.md` |
| `ENVIRONMENT.txt` | `docs/ENVIRONMENT.txt` |
| `HANDOVER_MAINTENANCE.md` | `docs/HANDOVER_MAINTENANCE.md` (current); original narrative in `docs/history/HANDOVER_MAINTENANCE_20261110.md` |
| `reports/E2_HANDOVER.md` | `docs/E2_HANDOVER.md` |
| `reports/E3_HANDOVER.md` | `docs/E3_HANDOVER.md` |
| `result/20261104-050058-E3/` | `result/e3_bandpass/` |
| `result/data_audit/` | `reports/data_audit/` |
| `result/reproduction/` | `reports/reproduction/` |

E3 retains its entire candidate Validation/Test outputs, formal selected results and predictions. No archive is deleted as an alleged old E2. The two E2 evaluator entry points remain: the resumable entry depends on the base evaluator.

## Paths versus historical evidence

Current instructions/default paths are updated, but archived results and audits are byte-identical. Consequently old paths remain inside archived reports/manifests. For example, `result/e4_wavelet/E4_RESULT_REPORT.txt` mentions `../reproduction/`; its current location is `reports/reproduction/` from the repository root. The E5 archived handover mentions `E5_RUN.md`; its current location is `docs/E5_RUN.md`.

`c2547fb:result/e2_v2/predictions_with_ids.csv` is a path inside a specific historical commit. It is unchanged and must not be remapped. Noise v2 module/function/config names, sampling rules and all experiment configurations are unchanged.

The E3 postprocessing default points to the moved archive. Running a finalizer can write results: it was not executed during this maintenance. Future users must not use maintenance code to regenerate archived records in place. The unified summary reads archives and writes only to a fresh directory.

## Demo scope

The user confirmed demo source has not been uploaded. `demo/README.md` reserves its location only. No webpage files, model adapters or startup commands are invented. Six-method webpage execution and agreement with official predictions remain pending delivery of the actual code and shared assets. The E5 prediction ZIP has been prepared separately.

## Validation scope

Verify all original result/report payload hashes after resolving the mappings above, all configs and training/noise/model code unchanged, Python parsing and current documentation links, and a read-only E2–E6 summary. No training, model inference, noise generation or rewriting of old manifest records is part of this round. The detailed check result is in `reports/delivery_layout_round1/validation.json`.
