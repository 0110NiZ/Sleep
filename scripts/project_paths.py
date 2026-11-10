"""Local paths only; experiment settings remain in configs/."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def configured(name, default):
    value = Path(os.environ.get(name, str(default))).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()
DATA_DIR = configured("SLEEP_DATA_DIR", ROOT / "data/processed/sleep-edf-20")
RAW_DIR = configured("SLEEP_RAW_DIR", ROOT / "data/raw/sleep-edf-20")
BASELINE_DIR = configured("SLEEP_BASELINE_DIR", ROOT / "Sleep_Baseline_5fold")
RESULT_ROOT = configured("SLEEP_RESULT_DIR", ROOT / "result/local")
E2_DIR = RESULT_ROOT / "e2_v2"
def baseline_fold(fold, root=None):
    """Accept the two existing handover layouts; never silently pick between both."""
    root = BASELINE_DIR if root is None else Path(root)
    candidates = [root / f"fold{fold}", root / f"fold{fold}_seed123"]
    found = [p for p in candidates if (p / "best_validation_macro_f1.pth").is_file()]
    if len(found) != 1:
        raise ValueError(f"Expected exactly one checkpoint folder for fold {fold}: {candidates}; found {found}")
    return found[0]
