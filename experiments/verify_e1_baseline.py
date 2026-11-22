
"""E1 Baseline verification using original evaluation logic.

Loads the original five-fold checkpoints and preprocessed EEG data.
CPU inference only. No training.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
)

# ============================================================
# 1. Paths and environment
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "AttnSleep"))

from model.model import AttnSleep
from utils.util import calc_class_weight

DATA_DIR = Path(
    "/mnt/d/SleepProject/data/processed/sleep-edf-20"
)

BASELINE_DIR = Path(
    "/mnt/d/SleepProject/baseline"
)

DEVICE = torch.device("cpu")


# ============================================================
# 2. Load preprocessed EEG data
# Original E1 data-loading logic
# ============================================================

def load_split(group):

    xs, ys = [], []

    for filename in group["files"]:

        with np.load(
            DATA_DIR / filename,
            allow_pickle=False
        ) as z:

            x = z["x"]
            y = z["y"]

            assert x.shape == (len(y), 3000, 1)
            assert np.isfinite(x).all()
            assert np.isin(y, range(5)).all()

            xs.append(x.transpose(0, 2, 1))
            ys.append(y)

    return TensorDataset(
        torch.from_numpy(
            np.concatenate(xs).copy()
        ).float(),

        torch.from_numpy(
            np.concatenate(ys)
        ).long(),
    )


# ============================================================
# 3. Original E1 evaluation logic
# Uses sklearn for Accuracy and Macro-F1
# ============================================================

def original_evaluate(model, loader, weights):

    model.eval()

    ys = []
    preds = []

    loss_sum = 0.0
    weight_sum = 0.0

    with torch.no_grad():

        for x, y in loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            out = model(x)

            assert torch.isfinite(out).all()

            loss_sum += torch.nn.functional.cross_entropy(
                out,
                y,
                weight=weights,
                reduction="sum",
            ).item()

            weight_sum += weights[y].sum().item()

            ys.extend(
                y.cpu().tolist()
            )

            preds.extend(
                out.argmax(1).cpu().tolist()
            )

    scores = {
        "loss": loss_sum / weight_sum,

        "accuracy": accuracy_score(
            ys, preds
        ),

        "macro_f1": f1_score(
            ys,
            preds,
            labels=list(range(5)),
            average="macro",
            zero_division=0,
        ),
    }

    return (
        scores,
        np.asarray(ys),
        np.asarray(preds),
    )


# ============================================================
# 4. Verify one fold
# ============================================================

def verify_fold(fold, manifest):

    split = manifest["folds"][fold]

    fold_dir = (
        BASELINE_DIR / f"fold{fold}"
    )

    # Read original E1 configuration
    cfg = json.loads(
        (fold_dir / "config.json").read_text(
            encoding="utf-8"
        )
    )


    # Verify the actual five-fold subject split

    expected_groups = [
        ["02", "07", "11", "17"],  # A
        ["04", "10", "15", "16"],  # B
        ["00", "08", "18", "19"],  # C
        ["01", "06", "09", "12"],  # D
        ["03", "05", "13", "14"],  # E
    ]

    assert manifest["seed"] == 123
    assert cfg["fold"] == fold

    # Check all five folds against the original baseline split
    for k, entry in enumerate(manifest["folds"]):
        assert entry["fold"] == k

        assert set(entry["test"]["subjects"]) == set(
            expected_groups[k]
        )

        assert set(entry["validation"]["subjects"]) == set(
            expected_groups[(k + 1) % 5]
        )

        expected_train = set().union(
            *[
                set(expected_groups[j])
                for j in range(5)
                if j not in (k, (k + 1) % 5)
            ]
        )

        assert set(entry["train"]["subjects"]) == expected_train

        # Check that recording files belong to their assigned subjects
        for part in ("train", "validation", "test"):
            subjects = set(entry[part]["subjects"])
            files = entry[part]["files"]

            assert len(files) == len(set(files))
            assert all(
                filename[3:5] in subjects
                for filename in files
            )

    print("Five-fold subject split: VERIFIED")

    # --------------------------------------------------------
    # Compute class weights from training set
    # --------------------------------------------------------

    train_counts = np.zeros(
        5,
        dtype=np.int64,
    )

    for filename in split["train"]["files"]:

        with np.load(
            DATA_DIR / filename,
            allow_pickle=False
        ) as z:

            train_counts += np.bincount(
                z["y"],
                minlength=5,
            )

    weights = torch.tensor(
        calc_class_weight(
            train_counts.tolist()
        ),
        dtype=torch.float32,
        device=DEVICE,
    )

    # --------------------------------------------------------
    # Load original AttnSleep model and checkpoint
    # --------------------------------------------------------

    model = AttnSleep().to(DEVICE)

    checkpoint_path = (
        fold_dir / "best_validation_macro_f1.pth"
    )

    checkpoint = torch.load(
        checkpoint_path,
        weights_only=True,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["state_dict"],
        strict=True,
    )

    # --------------------------------------------------------
    # Load original E1 test set
    # --------------------------------------------------------

    test = load_split(
        split["test"]
    )

    loader = DataLoader(
        test,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    # --------------------------------------------------------
    # Original evaluation
    # --------------------------------------------------------

    metrics, y, pred = original_evaluate(
        model,
        loader,
        weights,
    )

    cm = confusion_matrix(
        y,
        pred,
        labels=list(range(5)),
    )

    kappa = cohen_kappa_score(
        y,
        pred,
    )

    # --------------------------------------------------------
    # Compare against original E1 results
    # --------------------------------------------------------

    original = json.loads(
        (
            fold_dir / "test_metrics.json"
        ).read_text(encoding="utf-8")
    )

    assert (
        len(test) == original["test_samples"]
    )

    acc_delta = 100 * (
        metrics["accuracy"]
        - original["accuracy"]
    )

    mf1_delta = 100 * (
        metrics["macro_f1"]
        - original["macro_f1"]
    )

    cm_delta = (
        cm - np.asarray(
            original["confusion_matrix"]
        )
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        f"\n========== Fold {fold} ==========",
        flush=True,
    )

    print(f"Test samples: {len(test)}")
    print(f"Batch size: {cfg['batch_size']}")

    print(
        f"Accuracy: "
        f"{metrics['accuracy'] * 100:.4f}%"
    )

    print(
        f"Original Accuracy: "
        f"{original['accuracy'] * 100:.4f}%"
    )

    print(
        f"Macro-F1: "
        f"{metrics['macro_f1'] * 100:.4f}%"
    )

    print(
        f"Original Macro-F1: "
        f"{original['macro_f1'] * 100:.4f}%"
    )

    print(
        f"Kappa: {kappa:.6f}"
    )

    print(
        f"Original Kappa: "
        f"{original['kappa']:.6f}"
    )

    print(
        f"Accuracy difference: "
        f"{acc_delta:+.4f} percentage points"
    )

    print(
        f"Macro-F1 difference: "
        f"{mf1_delta:+.4f} percentage points"
    )

    matched = (
        abs(acc_delta) < 0.01
        and abs(mf1_delta) < 0.01
    )

    print(
        f"Status: "
        f"{'MATCH' if matched else 'DIFFERENT'}"
    )

    print(
        "Confusion matrix exact match:",
        not np.any(cm_delta),
    )

    if np.any(cm_delta):

        print(
            "Confusion matrix differences "
            "(local - original):"
        )

        print(cm_delta)

    # --------------------------------------------------------
    # Save per-sample predictions
    # Outside GitHub repository
    # --------------------------------------------------------

    output = (
        fold_dir / "e1_cpu_recheck_predictions.npz"
    )

    np.savez_compressed(
        output,
        y_true=y,
        y_pred=pred,
    )

    print(
        f"Predictions saved to: {output}",
        flush=True,
    )

    return metrics


# ============================================================
# 5. Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fold",
        choices=[
            "0", "1", "2", "3", "4", "all"
        ],
        default="2",
    )

    args = parser.parse_args()

    torch.set_num_threads(4)

    manifest = json.loads(
        (
            ROOT / "configs" / "fivefold_subject_splits.json"
        ).read_text(encoding="utf-8")
    )

    folds = (
        range(5)
        if args.fold == "all"
        else [int(args.fold)]
    )

    results = []

    for fold in folds:

        result = verify_fold(
            fold,
            manifest,
        )

        results.append(result)

    if len(results) == 5:

        mean_acc = np.mean([
            r["accuracy"]
            for r in results
        ])

        mean_f1 = np.mean([
            r["macro_f1"]
            for r in results
        ])

        print(
            "\n========== Five-Fold Summary =========="
        )

        print(
            f"Mean Accuracy: "
            f"{mean_acc * 100:.2f}%"
        )

        print(
            f"Mean Macro-F1: "
            f"{mean_f1 * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
