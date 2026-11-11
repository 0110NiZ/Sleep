
"""E2: Evaluate frozen AttnSleep under Noise v2.

Uses the same five-fold splits, checkpoints, batch sizes,
and sklearn evaluation metrics as the E1 baseline.

No training or fine-tuning.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

# ============================================================
# 1. Paths and imports
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "AttnSleep"))
sys.path.insert(0, str(ROOT / "scripts"))

from model.model import AttnSleep
from noise_conditions import apply_noise, apply_noise_v2

from project_paths import DATA_DIR, BASELINE_DIR, E2_DIR, baseline_fold
OUTPUT_DIR = E2_DIR

SPLIT_FILE = (
    ROOT / "configs/fivefold_subject_splits.json"
)

NOISE_FILE = (
    ROOT / "configs/noise_conditions_v2.json"
)

LABELS = ["W", "N1", "N2", "N3", "REM"]

# Match the local E1 verification environment
DEVICE = torch.device("cpu")


# ============================================================
# 2. Load the original E1 test data
# ============================================================

def load_test(group):

    xs = []
    ys = []
    sample_ids = []

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

        # E1 input: (N, 3000, 1) -> (N, 1, 3000)
        xs.append(x.transpose(0, 2, 1))
        ys.append(y)

        # Fixed IDs: one unique ID per recording and epoch
        for i in range(len(y)):
            sample_ids.append(f"{filename}:{i}")

    assert len(sample_ids) == len(set(sample_ids))

    clean_x = (
        np.concatenate(xs)
        .copy()
        .astype(np.float32, copy=False)
    )

    true_y = np.concatenate(ys).astype(
        np.int64, copy=False
    )

    return clean_x, true_y, sample_ids


# ============================================================
# 3. Original E1 inference logic
# ============================================================

def predict(model, x, batch_size):

    model.eval()
    predictions = []

    with torch.no_grad():

        for start in range(0, len(x), batch_size):

            batch = x[start:start + batch_size]

            tensor = torch.from_numpy(
                np.ascontiguousarray(batch)
            ).float().to(DEVICE)

            output = model(tensor)

            assert torch.isfinite(output).all()

            predictions.append(
                output.argmax(1).cpu().numpy()
            )

    return np.concatenate(predictions)


# ============================================================
# 4. Original sklearn metrics + Proposal metrics
# ============================================================

def calculate_metrics(y_true, y_pred):

    precision, recall, stage_f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=list(range(5)),
            zero_division=0,
        )
    )

    result = {
        "samples": int(len(y_true)),

        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),

        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=list(range(5)),
                average="macro",
                zero_division=0,
            )
        ),

        "kappa": float(
            cohen_kappa_score(y_true, y_pred)
        ),

        "stage_metrics": {},

        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(5)),
        ).tolist(),
    }

    for i, stage in enumerate(LABELS):

        result["stage_metrics"][stage] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(stage_f1[i]),
            "support": int(support[i]),
        }

    return result


# ============================================================
# 5. CSV helper
# ============================================================

def write_csv(path, rows):

    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 6. E2 experiment
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fold",
        choices=["0", "1", "2", "3", "4", "all"],
        default="0",
    )

    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["Clean", "Gaussian_Low"],
        help="Condition names, or 'all'",
    )

    args = parser.parse_args()

    torch.set_num_threads(4)

    # Original E1 subject-wise split
    split_info = json.loads(
        SPLIT_FILE.read_text(encoding="utf-8")
    )

    # Our finalized Noise v2 settings
    noise_info = json.loads(
        NOISE_FILE.read_text(encoding="utf-8")
    )

    assert split_info["seed"] == 123
    assert noise_info["seed"] == 123
    assert noise_info["sampling_rate"] == 100
    assert noise_info["signal_unit"] == "uV"

    condition_map = {
        c["name"]: c
        for c in noise_info["conditions"]
    }

    assert len(condition_map) == 10, (
        "Expected Clean + 9 noisy conditions"
    )

    assert len(split_info["folds"]) == 5

    # Select noise conditions
    if args.conditions == ["all"]:

        selected = list(condition_map)

    else:

        invalid = set(args.conditions) - set(condition_map)

        if invalid:
            parser.error(
                f"Unknown conditions: {sorted(invalid)}"
            )

        selected = list(
            dict.fromkeys(args.conditions)
        )

    # Select folds
    folds = (
        range(5)
        if args.fold == "all"
        else [int(args.fold)]
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_dir = OUTPUT_DIR / "predictions"

    prediction_dir.mkdir(exist_ok=True)

    rows = []
    full_details = {}

    # --------------------------------------------------------
    # Evaluate each fold
    # --------------------------------------------------------

    for fold in folds:

        print(
            f"\n========== E2 Fold {fold} ==========",
            flush=True,
        )

        split = split_info["folds"][fold]

        assert split["fold"] == fold
        assert len(split["test"]["subjects"]) == 4

        assert all(
            f[3:5] in split["test"]["subjects"]
            for f in split["test"]["files"]
        )

        # Same clean test data as E1
        clean_x, true_y, sample_ids = load_test(
            split["test"]
        )

        fold_dir = baseline_fold(fold)

        cfg = json.loads(
            (fold_dir / "config.json").read_text(
                encoding="utf-8"
            )
        )

        assert cfg["fold"] == fold

        batch_size = int(cfg["batch_size"])

        # Same frozen AttnSleep checkpoint
        checkpoint = torch.load(
            fold_dir / "best_validation_macro_f1.pth",
            weights_only=True,
            map_location=DEVICE,
        )

        model = AttnSleep().to(DEVICE)

        model.load_state_dict(
            checkpoint["state_dict"],
            strict=True,
        )

        model.eval()

        print(f"Samples: {len(true_y)}")
        print(f"Batch size: {batch_size}")
        print(f"Device: {DEVICE}")

        # ----------------------------------------------------
        # Evaluate selected conditions
        # ----------------------------------------------------

        for name in selected:

            condition = condition_map[name]

            # Clean or Noise v2
            if condition["noise_type"] == "clean":

                input_x = clean_x

            else:

                input_x = apply_noise(
                    clean_x,
                    condition,
                    sample_ids=sample_ids,
                    seed=int(noise_info["seed"]),
                    fs=float(noise_info["sampling_rate"]),
                )

            input_x = np.asarray(input_x)

            assert input_x.shape == clean_x.shape
            assert np.isfinite(input_x).all()

            # Frozen model inference
            pred = predict(
                model,
                input_x,
                batch_size,
            )

            score = calculate_metrics(
                true_y,
                pred,
            )

            # ------------------------------------------------
            # Verify E2 Clean = E1 CPU recheck
            # ------------------------------------------------

            if name == "Clean":

                e1_file = (
                    fold_dir
                    / "e1_cpu_recheck_predictions.npz"
                )

                if e1_file.exists():

                    with np.load(
                        e1_file,
                        allow_pickle=False,
                    ) as e1:

                        assert np.array_equal(
                            true_y,
                            e1["y_true"],
                        )

                        assert np.array_equal(
                            pred,
                            e1["y_pred"],
                        ), (
                            f"Fold {fold}: E2 Clean "
                            "predictions differ from E1"
                        )

                    print(
                        "  Clean predictions match "
                        "local E1 exactly"
                    )

            # ------------------------------------------------
            # Save individual predictions
            # ------------------------------------------------

            prediction_path = (
                prediction_dir
                / f"fold{fold}_{name}.npz"
            )

            np.savez_compressed(
                prediction_path,
                y_true=true_y,
                y_pred=pred,
            )

            # Detailed results
            full_details[f"fold{fold}_{name}"] = score

            rows.append({
                "fold": fold,
                "condition": name,
                "samples": score["samples"],
                "accuracy_pct": 100 * score["accuracy"],
                "macro_f1_pct": 100 * score["macro_f1"],
                "kappa": score["kappa"],
            })

            print(
                f"  {name:<18} "
                f"Acc={score['accuracy'] * 100:.4f}%  "
                f"Macro-F1={score['macro_f1'] * 100:.4f}%  "
                f"Kappa={score['kappa']:.6f}",
                flush=True,
            )

            del input_x

    # --------------------------------------------------------
    # 7. Save per-fold results
    # --------------------------------------------------------

    write_csv(
        OUTPUT_DIR / "e2_per_fold.csv",
        rows,
    )

    (
        OUTPUT_DIR / "e2_details.json"
    ).write_text(
        json.dumps(
            full_details,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # 8. Five-fold summary
    # --------------------------------------------------------

    if args.fold == "all":

        summary = []

        for name in selected:

            subset = [
                r for r in rows
                if r["condition"] == name
            ]

            assert len(subset) == 5

            accs = np.array([
                r["accuracy_pct"]
                for r in subset
            ])

            f1s = np.array([
                r["macro_f1_pct"]
                for r in subset
            ])

            kappas = np.array([
                r["kappa"]
                for r in subset
            ])

            summary.append({
                "condition": name,

                "mean_accuracy_pct": float(
                    accs.mean()
                ),

                "sd_accuracy_pct": float(
                    accs.std(ddof=1)
                ),

                "mean_macro_f1_pct": float(
                    f1s.mean()
                ),

                "sd_macro_f1_pct": float(
                    f1s.std(ddof=1)
                ),

                "mean_kappa": float(
                    kappas.mean()
                ),

                "sd_kappa": float(
                    kappas.std(ddof=1)
                ),
            })

        write_csv(
            OUTPUT_DIR / "e2_fivefold_summary.csv",
            summary,
        )

        print(
            "\n========== E2 Five-Fold Summary =========="
        )

        for item in summary:

            print(
                f"{item['condition']:<18} "
                f"Acc={item['mean_accuracy_pct']:.2f} "
                f"± {item['sd_accuracy_pct']:.2f}%  "
                f"Macro-F1={item['mean_macro_f1_pct']:.2f} "
                f"± {item['sd_macro_f1_pct']:.2f}%"
            )

    print(
        f"\nResults saved in: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
