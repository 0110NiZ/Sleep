
"""Resumable E2 Noise-v2 evaluation.

Keeps the historical run_e2_evaluation_v2.py unchanged.

Usage:
    python scripts/run_e2_evaluation_v2_resumable.py --check-only
    python scripts/run_e2_evaluation_v2_resumable.py --fold all --conditions all
    python scripts/run_e2_evaluation_v2_resumable.py --fold 3 --conditions Gaussian_High

Existing complete results are validated and skipped.
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "AttnSleep"))

import run_e2_evaluation_v2 as legacy_e2
from model.model import AttnSleep
from noise_conditions_v2 import apply_noise_v2


def load_json(path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def load_rows(path):
    if not path.exists():
        return []

    with path.open(
        newline="",
        encoding="utf-8-sig",
    ) as f:
        return list(csv.DictReader(f))


def save_json(path, data):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = path.with_name(
        path.name + ".tmp"
    )

    temp.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    temp.replace(path)


def save_rows(path, rows):
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = path.with_name(
        path.name + ".tmp"
    )

    with temp.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)

    temp.replace(path)


def sha256(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def verify_noise_modules():
    path = (
        ROOT / "scripts/noise_conditions_v2.py"
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Official Noise v2 generator missing: {path}"
        )

    print(
        f"Noise v2 generator: {path.name} "
        f"(SHA256 {sha256(path)[:12]}...)"
    )


def read_prediction_score(path):
    with np.load(
        path,
        allow_pickle=False,
    ) as z:
        y_true = np.asarray(
            z["y_true"]
        ).reshape(-1)

        y_pred = np.asarray(
            z["y_pred"]
        ).reshape(-1)

    if (
        len(y_true) != len(y_pred)
        or not np.isin(y_pred, range(5)).all()
    ):
        raise ValueError(
            f"Invalid predictions: {path}"
        )

    return legacy_e2.calculate_metrics(
        y_true,
        y_pred,
    )


def check_pair(pair, rows_by_key, details, pred_dir):
    fold, name = pair

    pred_path = (
        pred_dir / f"fold{fold}_{name}.npz"
    )

    row = rows_by_key.get(pair)

    detail = details.get(
        f"fold{fold}_{name}"
    )

    present = [
        row is not None,
        detail is not None,
        pred_path.exists(),
    ]

    if not any(present):
        return "missing"

    if not all(present):
        return "partial"

    score = read_prediction_score(
        pred_path
    )

    comparisons = [
        (
            float(row["accuracy_pct"]),
            100 * score["accuracy"],
        ),
        (
            float(row["macro_f1_pct"]),
            100 * score["macro_f1"],
        ),
        (
            float(row["kappa"]),
            score["kappa"],
        ),
        (
            float(detail["macro_f1"]),
            score["macro_f1"],
        ),
    ]

    if int(row["samples"]) != score["samples"]:
        return "inconsistent"

    if any(
        not np.isclose(
            a,
            b,
            rtol=0,
            atol=1e-6,
        )
        for a, b in comparisons
    ):
        return "inconsistent"

    if (
        detail["confusion_matrix"]
        != score["confusion_matrix"]
    ):
        return "inconsistent"

    return "complete"


def make_summary(rows_by_key, condition_names):
    summary = []

    for name in condition_names:
        values = [
            rows_by_key.get((fold, name))
            for fold in range(5)
        ]

        if any(row is None for row in values):
            continue

        entry = {
            "condition": name
        }

        for metric in (
            "accuracy_pct",
            "macro_f1_pct",
            "kappa",
        ):
            x = np.asarray(
                [
                    float(v[metric])
                    for v in values
                ],
                dtype=float,
            )

            entry[f"mean_{metric}"] = float(
                x.mean()
            )

            entry[f"sd_{metric}"] = float(
                x.std(ddof=1)
            )

        summary.append(entry)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--fold",
        choices=["all", "0", "1", "2", "3", "4"],
        default="all",
    )

    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["all"],
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "result/e2_v2",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(
            "/mnt/d/SleepProject/data/processed/sleep-edf-20"
        ),
    )

    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path(
            "/mnt/d/SleepProject/baseline"
        ),
    )

    parser.add_argument(
        "--split-file",
        type=Path,
        default=ROOT
        / "configs/fivefold_subject_splits.json",
    )

    parser.add_argument(
        "--noise-file",
        type=Path,
        default=ROOT
        / "configs/noise_conditions_v2.json",
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Inspect existing results without model inference",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Intentionally recompute existing results",
    )

    args = parser.parse_args()

    if args.check_only and args.force:
        parser.error(
            "--check-only and --force cannot be combined"
        )

    verify_noise_modules()

    output_dir = args.output_dir.resolve()
    pred_dir = output_dir / "predictions"

    conditions_info = load_json(
        args.noise_file
    )

    split_info = load_json(
        args.split_file
    )

    if (
        conditions_info["seed"] != 123
        or conditions_info["sampling_rate"] != 100
    ):
        raise RuntimeError(
            "Unexpected noise config"
        )

    conditions = {
        c["name"]: c
        for c in conditions_info["conditions"]
    }

    if (
        len(conditions) != 10
        or len(split_info["folds"]) != 5
    ):
        raise RuntimeError(
            "Expected 5 folds and 10 noise conditions"
        )

    if args.conditions == ["all"]:
        names = list(conditions)

    else:
        unknown = (
            set(args.conditions)
            - set(conditions)
        )

        if unknown:
            parser.error(
                f"Unknown conditions: {sorted(unknown)}"
            )

        names = list(
            dict.fromkeys(args.conditions)
        )

    folds = (
        range(5)
        if args.fold == "all"
        else [int(args.fold)]
    )

    rows_file = output_dir / "e2_per_fold.csv"
    details_file = output_dir / "e2_details.json"

    rows_by_key = {}

    for row in load_rows(rows_file):
        key = (
            int(row["fold"]),
            row["condition"],
        )

        if key in rows_by_key:
            raise ValueError(
                f"Duplicate E2 result: {key}"
            )

        rows_by_key[key] = row

    details = (
        load_json(details_file)
        if details_file.exists()
        else {}
    )

    missing = []
    incomplete = []
    complete = 0

    for fold in folds:
        for name in names:
            state = check_pair(
                (fold, name),
                rows_by_key,
                details,
                pred_dir,
            )

            if state == "complete":
                complete += 1

            elif state == "missing":
                missing.append(
                    (fold, name)
                )

            else:
                incomplete.append(
                    (fold, name, state)
                )

    print(
        f"E2: {complete} complete, "
        f"{len(missing)} missing, "
        f"{len(incomplete)} partial/inconsistent conditions"
    )

    if args.check_only:
        for item in missing[:15]:
            print("MISSING", item)

        for item in incomplete[:15]:
            print("NEEDS ATTENTION", item)

        if missing or incomplete:
            raise SystemExit(
                "Incomplete E2 results"
            )

        print(
            "CHECK ONLY PASS: "
            "saved results are complete; "
            "no inference performed"
        )
        return

    if incomplete and not args.force:
        raise RuntimeError(
            "Partial/inconsistent results exist. "
            f"Inspect first: {incomplete[:5]}"
        )

    legacy_e2.DATA_DIR = (
        args.data_dir.resolve()
    )

    legacy_e2.DEVICE = torch.device(
        "cpu"
    )

    torch.set_num_threads(4)

    pred_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    changed = 0

    for fold in folds:
        todo = [
            name
            for name in names
            if (
                args.force
                or check_pair(
                    (fold, name),
                    rows_by_key,
                    details,
                    pred_dir,
                ) == "missing"
            )
        ]

        if not todo:
            print(
                f"Fold {fold}: "
                "all conditions already saved; skipping"
            )
            continue

        split = split_info["folds"][fold]

        if int(split["fold"]) != fold:
            raise ValueError(
                f"Fold mismatch: {fold}"
            )

        clean_x, y, sample_ids = (
            legacy_e2.load_test(
                split["test"]
            )
        )

        folder = (
            args.baseline_dir.resolve()
            / f"fold{fold}"
        )

        config = load_json(
            folder / "config.json"
        )

        if int(config["fold"]) != fold:
            raise ValueError(
                f"Baseline fold mismatch: {fold}"
            )

        checkpoint = torch.load(
            folder / "best_validation_macro_f1.pth",
            map_location="cpu",
            weights_only=True,
        )

        model = AttnSleep().to(
            "cpu"
        )

        model.load_state_dict(
            checkpoint["state_dict"],
            strict=True,
        )

        model.eval()

        batch_size = int(
            config["batch_size"]
        )

        for name in todo:
            condition = conditions[name]

            if condition["noise_type"] == "clean":
                noisy_x = clean_x

            else:
                noisy_x = apply_noise_v2(
                    clean_x,
                    condition,
                    sample_ids=sample_ids,
                    seed=int(
                        conditions_info["seed"]
                    ),
                    fs=float(
                        conditions_info["sampling_rate"]
                    ),
                )

            y_pred = legacy_e2.predict(
                model,
                np.asarray(
                    noisy_x,
                    dtype=np.float32,
                ),
                batch_size,
            )

            score = legacy_e2.calculate_metrics(
                y,
                y_pred,
            )

            out = (
                pred_dir
                / f"fold{fold}_{name}.npz"
            )

            temp = (
                pred_dir
                / f"fold{fold}_{name}.tmp.npz"
            )

            np.savez_compressed(
                temp,
                y_true=y,
                y_pred=y_pred,
                sample_ids=np.asarray(
                    sample_ids,
                    dtype=str,
                ),
            )

            temp.replace(out)

            rows_by_key[(fold, name)] = {
                "fold": fold,
                "condition": name,
                "samples": score["samples"],
                "accuracy_pct":
                    100 * score["accuracy"],
                "macro_f1_pct":
                    100 * score["macro_f1"],
                "kappa": score["kappa"],
            }

            details[
                f"fold{fold}_{name}"
            ] = score

            ordered = [
                rows_by_key[key]
                for key in sorted(
                    rows_by_key
                )
            ]

            save_rows(
                rows_file,
                ordered,
            )

            save_json(
                details_file,
                details,
            )

            save_rows(
                output_dir
                / "e2_fivefold_summary.csv",
                make_summary(
                    rows_by_key,
                    list(conditions),
                ),
            )

            changed += 1

            print(
                f"SAVED Fold {fold} {name}: "
                f"Macro-F1="
                f"{100 * score['macro_f1']:.2f}%",
                flush=True,
            )

        del model, clean_x

    if changed:
        print(
            "WARNING: derived pooled/stage results "
            "may be stale. "
            "Rerun scripts/finalize_e2_results.py"
        )

    else:
        print(
            "No rerun needed; "
            "all requested results are complete"
        )

    print(
        f"E2 results directory: {output_dir}"
    )


if __name__ == "__main__":
    main()
