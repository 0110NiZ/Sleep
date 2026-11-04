"""Shared local paths; no changes to experiment or noise settings."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def configured(name, default):
    value = Path(os.environ.get(name, str(default))).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()
DATA_DIR = configured("SLEEP_DATA_DIR", ROOT / "data/processed/sleep-edf-20")
BASELINE_DIR = configured("SLEEP_BASELINE_DIR", ROOT / "Sleep_Baseline_5fold")
RESULT_ROOT = configured("SLEEP_RESULT_DIR", ROOT / "result/local")
E2_DIR = RESULT_ROOT / "e2_v2"
