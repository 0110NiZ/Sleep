# Compatibility entries

Only active code/configuration paths and supported legacy Python entry points remain. There are no duplicate algorithm implementations.

| Kept | Reason |
|---|---|
| `data`, `runs` | E1 configs and the shared subject-split manifest still contain these paths; their bytes and hashes must stay unchanged. |
| `result/local` | Existing output defaults use it; it points to local/runs. |
| Old result names | Evaluation and summary programs still use them; original result identities and historical commit references are preserved. |
| Old config names | Current CLI defaults and E6 protocol hash checks depend on them. |
| scripts/ Python entries | Existing imports, code hash inventories and old group commands use them. They forward to the single code/ or visualization/ implementation. |

Removed: root reports, patches, logs, Sleep_Baseline_5fold, old dependency filenames; old docs entry aliases; obsolete chart-location links. The targets themselves were retained. Current code/documentation now points to the real paths.

`link_cleanup.json` lists every removed and retained link, its target and why it is kept. `link_validation.json` records this follow-up check. `validation.json` is the previous cleanup's unchanged historical validation. `check.py` performs the current basic check.

No experiment parameters, original configuration values, archived result bytes or historical records were edited. No training, inference or plotting jobs were run. Ubuntu/WSL link behavior is checked; Windows-native cloning and demo integration remain outside this check.
