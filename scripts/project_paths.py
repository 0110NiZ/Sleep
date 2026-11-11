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
def e2_result_dir(root):
    """Use an existing legacy local directory; never silently choose between two."""
    root = Path(root)
    current, legacy = root / "e2", root / "e2_v2"
    if current.exists() and legacy.exists():
        raise ValueError(f"Both E2 directories exist; resolve ambiguity: {current}, {legacy}")
    return legacy if legacy.exists() else current

E2_DIR = e2_result_dir(RESULT_ROOT)
def baseline_fold(fold, root=None):
    """Accept the two existing handover layouts; never silently pick between both."""
    root = BASELINE_DIR if root is None else Path(root)
    candidates = [root / f"fold{fold}", root / f"fold{fold}_seed123"]
    found = [p for p in candidates if (p / "best_validation_macro_f1.pth").is_file()]
    if len(found) != 1:
        raise ValueError(f"Expected exactly one checkpoint folder for fold {fold}: {candidates}; found {found}")
    return found[0]
