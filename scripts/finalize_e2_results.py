
"""Finalize already-completed E2 Noise-v2 test results WITHOUT running AttnSleep.

Reads saved 50 prediction NPZs and original preprocessed label arrays,
validates alignment and existing E2 metrics, then adds:
  - predictions_with_ids.csv
  - pooled_summary.csv
  - pooled_stage_metrics.csv
  - pooled_confusion_matrix.json
  - stage_summary.csv
  - e2_analysis_manifest.json

Usage (from Sleep repo root):
  python scripts/finalize_e2_results.py

Optional different locations:
  python scripts/finalize_e2_results.py --e2-dir result/e2_v2 \
      --data-dir /mnt/d/SleepProject/data/processed/sleep-edf-20
"""

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parents[1]
STAGES = ["W", "N1", "N2", "N3", "REM"]
LABELS = list(range(5))
CONDITIONS = [
    "Clean",
    "Gaussian_Low", "Gaussian_Medium", "Gaussian_High",
    "Drift_Low", "Drift_Medium", "Drift_High",
    "Burst_Low", "Burst_Medium", "Burst_High",
]


def load_json(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def save_json(path, value):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def save_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows to write: {path}")

    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def sample_mean_sd(values):
    a = np.asarray(values, dtype=np.float64)
    if len(a) != 5:
        raise ValueError(f"Five folds expected, found {len(a)}")
    return float(a.mean()), float(a.std(ddof=1))


def calculate_metrics(y_true, y_pred):
    """Same label set and sklearn settings as run_e2_evaluation_v2.py."""

    precision, recall, stage_f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=LABELS,
            zero_division=0,
        )
    )

    stage_metrics = {
        stage: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(stage_f1[i]),
            "support": int(support[i]),
        }
        for i, stage in enumerate(STAGES)
    }

    return {
        "samples": int(len(y_true)),
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "kappa": float(
            cohen_kappa_score(y_true, y_pred)
        ),
        "stage_metrics": stage_metrics,
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=LABELS,
        ).tolist(),
    }


def assert_close(actual, expected, description, atol=1e-7):
    if not np.isfinite(actual) or not np.isclose(
        actual, expected, rtol=0, atol=atol
    ):
        raise ValueError(
            f"Existing result mismatch for {description}: "
            f"recomputed={actual}, saved={expected}"
        )


def fold_test_identity(split, data_dir):
    """Rebuild exactly E2 load_test() order: files, then epoch indices."""

    y_chunks = []
    sample_ids = []
    subject_ids = []
    recording_ids = []
    epoch_indices = []

    allowed_subjects = set(split["test"]["subjects"])

    for filename in split["test"]["files"]:
        path = data_dir / filename

        if not path.is_file():
            raise FileNotFoundError(
                f"Preprocessed EEG file missing: {path}"
            )

        recording_id = Path(filename).stem

        if not recording_id.startswith("SC") or len(recording_id) < 5:
            raise ValueError(
                f"Unexpected Sleep-EDF recording ID: {filename}"
            )

        # Match original E2 extraction
        subject_id = recording_id[3:5]

        if subject_id not in allowed_subjects:
            raise ValueError(
                f"Subject {subject_id} not in fold Test set: {filename}"
            )

        with np.load(path, allow_pickle=False) as z:
            labels = (
                np.asarray(z["y"])
                .reshape(-1)
                .astype(np.int64)
            )

        if not np.isin(labels, LABELS).all():
            raise ValueError(
                f"Unexpected labels in {path}"
            )

        count = len(labels)

        y_chunks.append(labels)

        sample_ids.extend(
            f"{filename}:{i}" for i in range(count)
        )

        subject_ids.extend([subject_id] * count)
        recording_ids.extend([recording_id] * count)
        epoch_indices.extend(range(count))

    if not y_chunks:
        raise ValueError(
            "No Test EEG files in split"
        )

    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(
            "Duplicate sample IDs inside Test fold"
        )

    return {
        "y_true": np.concatenate(y_chunks),
        "sample_ids": sample_ids,
        "subject_ids": subject_ids,
        "recording_ids": recording_ids,
        "epoch_indices": epoch_indices,
    }


def verify_one_result(
    fold,
    condition,
    score,
    saved_row,
    saved_details,
):
    title = f"Fold {fold}, {condition}"

    if int(saved_row["samples"]) != score["samples"]:
        raise ValueError(
            f"Sample count differs in e2_per_fold.csv: {title}"
        )

    if int(saved_details["samples"]) != score["samples"]:
        raise ValueError(
            f"Sample count differs in e2_details.json: {title}"
        )

    for key, actual in [
        ("accuracy_pct", score["accuracy"] * 100),
        ("macro_f1_pct", score["macro_f1"] * 100),
        ("kappa", score["kappa"]),
    ]:
        assert_close(
            actual,
            float(saved_row[key]),
            f"{title} / {key}",
        )

    for key in [
        "accuracy",
        "macro_f1",
        "kappa",
    ]:
        assert_close(
            score[key],
            float(saved_details[key]),
            f"{title} / details.{key}",
        )

    if score["confusion_matrix"] != saved_details["confusion_matrix"]:
        raise ValueError(
            f"Saved confusion matrix differs: {title}"
        )

    for stage in STAGES:
        for key in [
            "precision",
            "recall",
            "f1",
        ]:
            assert_close(
                score["stage_metrics"][stage][key],
                float(saved_details["stage_metrics"][stage][key]),
                f"{title} / {stage}.{key}",
            )

        if score["stage_metrics"][stage]["support"] != int(
            saved_details["stage_metrics"][stage]["support"]
        ):
            raise ValueError(
                f"Stage support differs: {title} / {stage}"
            )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--e2-dir",
        type=Path,
        default=ROOT / "result/e2_v2",
        help="Directory containing E2 CSV, JSON and predictions/",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(
            "/mnt/d/SleepProject/data/processed/sleep-edf-20"
        ),
        help="Same preprocessed NPZ directory used in E2 evaluation",
    )

    parser.add_argument(
        "--split-file",
        type=Path,
        default=ROOT / "configs/fivefold_subject_splits.json",
    )

    parser.add_argument(
        "--noise-file",
        type=Path,
        default=ROOT / "configs/noise_conditions_v2.json",
    )

    args = parser.parse_args()

    e2_dir = args.e2_dir.resolve()
    data_dir = args.data_dir.resolve()
    split_file = args.split_file.resolve()
    noise_file = args.noise_file.resolve()

    splits = load_json(split_file)
    noise = load_json(noise_file)

    condition_names = [
        item["name"]
        for item in noise["conditions"]
    ]

    if condition_names != CONDITIONS or len(splits["folds"]) != 5:
        raise ValueError(
            "Expected the original 5 folds and 10 Noise-v2 conditions"
        )

    existing_rows = load_csv(
        e2_dir / "e2_per_fold.csv"
    )

    existing_details = load_json(
        e2_dir / "e2_details.json"
    )

    by_key = {}

    for row in existing_rows:
        key = (
            int(row["fold"]),
            row["condition"],
        )

        if key in by_key:
            raise ValueError(
                f"Duplicate result: {key}"
            )

        by_key[key] = row

    required = {
        (fold, name)
        for fold in range(5)
        for name in CONDITIONS
    }

    if set(by_key) != required:
        missing = sorted(required - set(by_key))
        extras = sorted(set(by_key) - required)

        raise ValueError(
            f"E2 per-fold CSV incomplete: "
            f"missing={missing}, extra={extras}"
        )

    expected_details = {
        f"fold{f}_{c}"
        for f, c in required
    }

    if set(existing_details) != expected_details:
        raise ValueError(
            "e2_details.json must contain the same "
            "50 fold-condition entries"
        )

    # Read label arrays, not full EEG waveforms.
    identities = {}
    all_ids = set()

    for fold in range(5):
        group = splits["folds"][fold]

        if int(group["fold"]) != fold:
            raise ValueError(
                f"Invalid fold ordering: {fold}"
            )

        identities[fold] = fold_test_identity(
            group,
            data_dir,
        )

        ids = set(
            identities[fold]["sample_ids"]
        )

        if ids & all_ids:
            raise ValueError(
                f"Test epochs appear in multiple folds: {fold}"
            )

        all_ids.update(ids)

        print(
            f"Fold {fold}: {len(ids)} Test epochs identified",
            flush=True,
        )

    prediction_output = (
        e2_dir / "predictions_with_ids.csv"
    )

    prediction_tmp = prediction_output.with_name(
        prediction_output.name + ".tmp"
    )

    per_fold_scores = {}
    pooled_rows = []
    pooled_stages = []

    pooled_cms = {
        "labels": STAGES,
        "axis0": "y_true",
        "axis1": "y_pred",
        "conditions": {},
    }

    # Stream predictions to CSV.
    # Do not hold all output rows in memory.
    try:
        with prediction_tmp.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                "fold",
                "subject_id",
                "recording_id",
                "epoch_index",
                "sample_id",
                "condition",
                "y_true",
                "y_pred",
            ])

            for condition in CONDITIONS:
                pooled_true = []
                pooled_pred = []

                for fold in range(5):
                    path = (
                        e2_dir
                        / "predictions"
                        / f"fold{fold}_{condition}.npz"
                    )

                    if not path.is_file():
                        raise FileNotFoundError(
                            f"Missing E2 predictions: {path}"
                        )

                    with np.load(
                        path,
                        allow_pickle=False,
                    ) as z:
                        y_true = (
                            np.asarray(z["y_true"])
                            .reshape(-1)
                            .astype(np.int64)
                        )

                        y_pred = (
                            np.asarray(z["y_pred"])
                            .reshape(-1)
                            .astype(np.int64)
                        )

                    identity = identities[fold]
                    expected_y = identity["y_true"]

                    # Critical: verify sample order before attaching IDs
                    if not np.array_equal(y_true, expected_y):
                        raise ValueError(
                            "Saved y_true does NOT match EEG "
                            "labels/order: "
                            f"Fold {fold}, {condition}. "
                            "Cannot assign Sample IDs safely."
                        )

                    if (
                        len(y_pred) != len(y_true)
                        or not np.isin(y_pred, LABELS).all()
                    ):
                        raise ValueError(
                            f"Invalid y_pred: Fold {fold}, {condition}"
                        )

                    score = calculate_metrics(
                        y_true,
                        y_pred,
                    )

                    # Verify against original E2 result files
                    verify_one_result(
                        fold,
                        condition,
                        score,
                        by_key[(fold, condition)],
                        existing_details[
                            f"fold{fold}_{condition}"
                        ],
                    )

                    per_fold_scores[(fold, condition)] = score

                    pooled_true.append(y_true)
                    pooled_pred.append(y_pred)

                    writer.writerows(
                        (
                            fold,
                            identity["subject_ids"][i],
                            identity["recording_ids"][i],
                            identity["epoch_indices"][i],
                            identity["sample_ids"][i],
                            condition,
                            int(y_true[i]),
                            int(y_pred[i]),
                        )
                        for i in range(len(y_true))
                    )

                # Pool five folds for this condition
                pooled_score = calculate_metrics(
                    np.concatenate(pooled_true),
                    np.concatenate(pooled_pred),
                )

                pooled_rows.append({
                    "condition": condition,
                    "samples": pooled_score["samples"],
                    "accuracy_pct": pooled_score["accuracy"] * 100,
                    "macro_f1_pct": pooled_score["macro_f1"] * 100,
                    "kappa": pooled_score["kappa"],
                })

                pooled_cms["conditions"][condition] = (
                    pooled_score["confusion_matrix"]
                )

                for stage in STAGES:
                    item = pooled_score["stage_metrics"][stage]

                    pooled_stages.append({
                        "condition": condition,
                        "stage": stage,
                        "precision_pct": item["precision"] * 100,
                        "recall_pct": item["recall"] * 100,
                        "f1_pct": item["f1"] * 100,
                        "support": item["support"],
                    })

                print(
                    f"Validated {condition:<17} "
                    f"| pooled epochs={pooled_score['samples']:,} "
                    f"| Macro-F1="
                    f"{100 * pooled_score['macro_f1']:.2f}%",
                    flush=True,
                )

        # Publish predictions only after all checks succeed
        prediction_tmp.replace(prediction_output)

    finally:
        if prediction_tmp.exists():
            prediction_tmp.unlink()

    # Stage-wise Mean ± SD across five folds
    stage_summary = []

    for condition in CONDITIONS:
        for stage in STAGES:

            fold_stage = [
                per_fold_scores[(fold, condition)]
                ["stage_metrics"][stage]
                for fold in range(5)
            ]

            row = {
                "condition": condition,
                "stage": stage,
                "total_support": sum(
                    item["support"]
                    for item in fold_stage
                ),
            }

            for key in [
                "precision",
                "recall",
                "f1",
            ]:
                mean, sd = sample_mean_sd([
                    item[key] * 100
                    for item in fold_stage
                ])

                row[f"mean_{key}_pct"] = mean
                row[f"sd_{key}_pct"] = sd

            stage_summary.append(row)

    # Save derived result files
    save_csv(
        e2_dir / "pooled_summary.csv",
        pooled_rows,
    )

    save_csv(
        e2_dir / "pooled_stage_metrics.csv",
        pooled_stages,
    )

    save_json(
        e2_dir / "pooled_confusion_matrix.json",
        pooled_cms,
    )

    save_csv(
        e2_dir / "stage_summary.csv",
        stage_summary,
    )

    metadata = {
        "experiment": "E2 Noise v2 - Test postprocessing",
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "input_prediction_files": 50,
        "unique_test_epochs": len(all_ids),
        "prediction_rows": (
            len(all_ids) * len(CONDITIONS)
        ),

        "data_dir": str(data_dir),
        "e2_dir": str(e2_dir),

        "split_file": str(split_file),
        "split_sha256": hashlib.sha256(
            split_file.read_bytes()
        ).hexdigest(),

        "noise_file": str(noise_file),
        "noise_config_sha256": hashlib.sha256(
            noise_file.read_bytes()
        ).hexdigest(),

        "sample_id_rule": (
            "recording_filename_with_npz_extension:"
            "zero_based_epoch_index"
        ),

        "subject_id_rule": (
            "SleepEDF recording stem characters [3:5]"
        ),

        "verification": (
            "Predictions checked against preprocessed labels, "
            "saved per-fold CSV and details JSON; "
            "no AttnSleep inference was run."
        ),

        "mean_sd_note": (
            "stage_summary uses sample SD across 5 folds "
            "(ddof=1)"
        ),

        "pooled_note": (
            "pooled metrics concatenate 5 Test folds "
            "and are NOT five-fold means"
        ),
    }

    save_json(
        e2_dir / "e2_analysis_manifest.json",
        metadata,
    )

    print(
        "\nE2 postprocessing complete. "
        "All saved predictions verified."
    )

    for name in [
        "predictions_with_ids.csv",
        "pooled_summary.csv",
        "pooled_stage_metrics.csv",
        "pooled_confusion_matrix.json",
        "stage_summary.csv",
        "e2_analysis_manifest.json",
    ]:
        print("  ", e2_dir / name)

    print(
        "No training, no noise generation, "
        "no model inference."
    )


if __name__ == "__main__":
    main()
