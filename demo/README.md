# Sleep Lab — E1–E6 browser demo

The final five-second zoom version used in the presentation.
Public site: **https://sleep-lab-7qq.pages.dev/**

## Run locally

Download or clone the **whole repository**, then run from its root:

```bash
python demo/serve.py
```

Windows users can double-click **`demo/start_demo.bat`** instead. Requires Python
3.9+ and a modern Chrome, Edge or Firefox browser. The launcher opens
**http://127.0.0.1:8766/**. Keep the terminal open; Ctrl+C stops the server.
If the port is busy, use `python demo/serve.py --port 8767`.

**No pip installation, PyTorch, full NPZ dataset, or external checkpoint download
is needed to run this version.** All runtime assets are included. Models run on
the visitor's computer in the browser. After downloading, local operation does
not depend on the public site. Do not double-click `index.html` or try to run it
in GitHub's file preview: model loading requires HTTP.

Paths resolve relative to the launcher, so starting it from another directory
or a folder with spaces/Chinese characters works. The server binds only to
localhost. For public hosting, deploy the contents of `dist/` to an HTTPS static
host, with `index.html` at the site's root.

## Controls

- **Sample / Noise / Compare:** select a segment and condition, then run all six paths.
- **View:** six five-second windows or Full 30 s; changing View only redraws plots.
- **Clean overlay:** grey dashed clean reference; all six plots share a scale within the window.
- **Random:** choose and run another sample, keeping the noise condition.
- **Export:** download completed results as JSON.
- **Details & full-test results:** configurations, RMSE, class scores and formal aggregate metrics.

Startup selects `Fold 2 · SC4162E0.npz:868 · N2`, `Gaussian · High`, view `0–5 s`.
For the two presentation examples, use **5–10 s**, then compare Gaussian High
and Drift High on the same sample. [One-page Chinese guide](Quick-start-ZH.docx).
Predictions always use **all 3,000 samples (30 seconds at 100 Hz)**, not the crop.

## E1–E6 mapping

| Path | Classifier input | Classifier |
|---|---|---|
| E1 | Clean EEG | Corresponding fold's frozen E1 model |
| E2 | Noisy EEG | Same frozen E1 model |
| E3 | Band-pass output | Same frozen E1 model |
| E4 | Wavelet output | Same frozen E1 model |
| E5 | Live DAE reconstruction of noisy EEG | Same frozen E1 model |
| E6 | Exactly the same noisy EEG as E2 | Corresponding fold's noise-trained model |

All classifications and E5 DAE inference run live with ONNX Runtime Web 1.20.1.
Noise and E3/E4 waveforms were prepared with the experiment Python code. Stored
predictions do not supply the normal Compare results. E2/E6 waveforms are
intentionally identical: E6 changes the classifier, not the signal.

The 50 test epochs contain two per stage per fold, sampled with seed 4330 before
inspecting predictions, under Clean and all nine Noise v2 conditions.
`dist/manifest.json` records models, hashes, sample IDs, labels and folds.
Each binary input contains four consecutive little-endian float32 arrays of
3,000 values: **clean, noisy, band-pass, wavelet**, in microvolts. E5's checkpoint
training scale is included inside its exported model.

E3's validation selections are A, A, A, A, B (0.3–35 Hz or 0.5–30 Hz, fourth-order
Butterworth). E4's are W3, W3, W3, W1, W1. Internal folds 0–4 display as Fold 1–5.
Full-test metrics are fixed formal experiment results, not demo-subset estimates.
“Clean” means no added synthetic noise, not artifact-free EEG.

## Validate

```bash
python demo/validate.py
```

The launcher also runs this check: hashes for 527 runtime files, 15 model files,
signal lengths, labels and per-fold sample coverage. Assets are ordinary Git
files, not LFS pointers. Missing/truncated files fail explicitly before serving.

Optional full model check (requires NumPy and ONNX Runtime in your Python environment):

```bash
python demo/validate.py --inference
```

This reruns 500 sample/condition combinations and checks 3,000 class predictions
against `verification.json`, the original PyTorch-to-ONNX conversion record.
It is a native ONNX test, not a check of every browser/device. The additional
`audit-20261119.json` records exact comparison of 2,000 input arrays with original
Python preprocessing and 3,000 predictions. `dist/browser-checks.json` contains
browser test fixtures, not the source of live predictions.

## Source and optional rebuilding

- `page_template.html`: page source; run `python demo/prepare_page.py` after edits.
- `dist/inference.js`: live inference and model hash checking.
- `dist/`: complete ready-to-serve site, including 15 models, 500 signal files and runtime libraries.
- `build_assets.py`: optional exporter, requiring original experiment inputs and checkpoints.
- `refresh_asset_hashes.py`: refresh the inventory after intentional reviewed asset changes, then validate again.

Rebuilding is **not needed to run the demo**. Use the original experiment Python
environment plus `onnx==1.17.0` and `onnxruntime==1.20.1`, with your own paths:

```bash
python demo/build_assets.py --data-dir "/path/to/clean-npz" --baseline-dir "/path/to/Sleep_Baseline_5fold" --e5-dir "/path/to/e5_weights" --e6-dir "/path/to/e6_weights"
python demo/prepare_page.py
python demo/refresh_asset_hashes.py
python demo/validate.py --inference
```

Baseline: `fold0`–`fold4/best_validation_macro_f1.pth`.
E5 and E6: `fold0`–`fold4`, each containing `best.pth` and matching `selection.json`.
The clean NPZ files must be the formal experiment inputs. E5/E6 checkpoint hashes
are checked. The exporter reads project code, splits and validation selections
relative to this repository; all private data/weight paths are explicit arguments.
It does not depend on the author's desktop paths or old local server adapters.
Keep `dist/inference.js` and `dist/vendor/` when rebuilding.

## Troubleshooting and attribution

Use the local HTTP URL, run validation and wait for the first model load. On the
public site, retry Compare after checking the connection. The generic run error
alone does not identify the failing operation. Check the sample/condition above
the cards: an unsuccessful new run can leave previous results visible.

Retain [data attribution](dist/ATTRIBUTION.txt), [AttnSleep's MIT license](dist/ATTNSLEEP-LICENSE.txt)
and [ONNX Runtime's MIT license](dist/vendor/LICENSE.txt). This folder contains
the public demo subset and exported models, not the complete training dataset or
original PyTorch checkpoints. Source dataset terms still apply.
