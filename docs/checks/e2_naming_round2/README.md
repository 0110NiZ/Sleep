# Round-two verification

See [validation.json](validation.json) for counts and scope. No training or official result regeneration was performed.

- Compared all ten conditions against the pre-maintenance generator: 30 NumPy layout/condition combinations, ten Torch conditions, and split-batch equality.
- Checked canonical/legacy module identity (including mutable attributes), six CLI help entries, E3–E6 imports, and E6's original protocol hash verification.
- Compared 80 predictions (eight existing EEG segments, ten conditions, one frozen E1 fold0 checkpoint) using old/new noise output. This is rename-equivalence evidence, not a fresh five-fold experiment or webpage validation.
- Checked new E2 local directory selection, legacy fallback and explicit rejection of ambiguous dual directories.
- Checked 1,039 protected pre-existing files by SHA256; archived results, original reports/manifests, configs, model code and historical docs were not altered.
- Read-only E2–E6 summary outputs match round one's output files exactly.

The Ubuntu runtime check used Git's LF contents (`core.autocrlf=false` export). Windows CRLF checkouts can differ from the original Linux code/config hashes; hashes were not relaxed or rewritten to hide this difference. Historical runs must use the original commit and environment. New source files/import paths produce new provenance hashes and should use fresh output directories.
