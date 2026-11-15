# Folder cleanup

Branch: `organize/project-folders`. Base main: `0b5dd26f79d89b0d5ce984b105bc71a5ba3e8cec`.

Source additions: `visualization-MAI` and `tang-ppt-figures`; only their new visualization files were imported. Latest main code/results/figures took precedence over older branch snapshots.

- `file_map.json`: complete pre-cleanup tracked-file SHA256 map, moves and imported-file provenance.
- `validation.json`: completed static, compatibility, saved-table and byte-preservation checks.
- `check.py`: repeat the basic read-only check using the active Sleep Python environment. It prints a report and does not rewrite the stored validation record.

No training, experiment inference or plot regeneration. Formal results, parameter files, selection records and existing figure exports are preserved byte-for-byte. Python changes are path/bootstrap/output routing only. Relative symlinks keep old paths usable on Ubuntu/WSL. New filenames do not erase Noise v2 protocol identity or historical commit paths.

Original documentation copies are retained in docs/history. Current guides use the new layout. No demo verification was attempted because its source is not present. Windows-native checkout and real training/resume are outside this check.

Validation: 1,051 original archived/configuration/model files preserved; 10 imported figure/table files preserved; new and legacy CLI help checks passed, including v2 aliases. E6 protocol validation passed without reading EEG or executing models. Imported summary means match formal fold tables; their original E5/E6 SD rounding is documented. Generated SVG whitespace and CSV line endings are deliberately preserved.

## Follow-up: remove unnecessary links

The first validation above is retained as history. The current layout has 47 compatibility links, down from 75. [Compatibility reasons](compatibility.md), `link_cleanup.json` and `link_validation.json` describe the 28 removed links and current checks. Targets were not deleted. Current code uses docs/checks and the real dependency/weight locations directly; original result/configuration/history files are unchanged.
