
"""E3: Butterworth Band-pass Filtering.

Evaluate four predefined filters on both Validation and Test.

Stages:
    validation - evaluate all filters and E2 no-filter baseline
    test       - evaluate all filters and compare with saved E2 results
    summary    - merge results and generate visualizations

Run from repository root:
    python scripts/run_e3_bandpass.py --stage validation --fold all
    python scripts/run_e3_bandpass.py --stage test --fold all
    python scripts/run_e3_bandpass.py --stage summary

Output:
    Sleep/result/YYYY/MM/DD-HHMMSS-E3/
"""

import argparse
import csv
import hashlib
import json
import sys
import time

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch

from scipy.signal import butter, sosfiltfilt


# ============================================================
# 1. Paths and imports
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "AttnSleep"))

from model.model import AttnSleep
from scripts.noise_conditions_v2 import apply_noise_v2

import run_e2_evaluation_v2 as e2


BASELINE_DIR = Path(
    "/mnt/d/SleepProject/baseline"
)

E2_DIR = ROOT / "result" / "e2_v2"

SPLIT_FILE = (
    ROOT / "configs/fivefold_subject_splits.json"
)

NOISE_FILE = (
    ROOT / "configs/noise_conditions_v2.json"
)

RESULT_ROOT = ROOT / "result"

ACTIVE_FILE = (
    RESULT_ROOT / ".e3_active_run.txt"
)

DEVICE = torch.device("cpu")


# ============================================================
# 2. Predefined filter candidates
# Do not modify these after looking at Test performance.
# ============================================================

CANDIDATES = [
    {
        "id": "A",
        "low_hz": 0.3,
        "high_hz": 35.0,
        "order": 4,
    },
    {
        "id": "B",
        "low_hz": 0.5,
        "high_hz": 30.0,
        "order": 4,
    },
    {
        "id": "C",
        "low_hz": 0.5,
        "high_hz": 20.0,
        "order": 4,
    },
    {
        "id": "D",
        "low_hz": 1.0,
        "high_hz": 30.0,
        "order": 4,
    },
]


# ============================================================
# 3. File utilities
# ============================================================

def read_json(path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def write_json(path, value):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def write_csv(path, rows):

    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
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


def read_csv(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Required result missing: {path}"
        )

    with path.open(
        newline="",
        encoding="utf-8-sig",
    ) as f:

        return list(csv.DictReader(f))


def update_fold_csv(path, fold, rows):

    previous = (
        read_csv(path)
        if path.exists()
        else []
    )

    previous = [
        r for r in previous
        if int(r["fold"]) != fold
    ]

    write_csv(
        path,
        previous + rows,
    )


def update_fold_json(path, fold, result):

    previous = (
        read_json(path)
        if path.exists()
        else {}
    )

    previous[str(fold)] = result

    write_json(path, previous)


def sha256(path):

    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


# ============================================================
# 4. Record experiment settings
# ============================================================

def settings():

    return {
        "noise_config_sha256": sha256(
            NOISE_FILE
        ),

        "split_config_sha256": sha256(
            SPLIT_FILE
        ),

        "noise_generator_sha256": sha256(
            ROOT / "scripts/noise_conditions_v2.py"
        ),

        "e2_evaluation_sha256": sha256(
            ROOT / "scripts/run_e2_evaluation_v2.py"
        ),

        "candidates": CANDIDATES,
    }


# ============================================================
# 5. Create/reuse one timestamped experiment folder
# ============================================================

def get_run_directory(stage, requested):

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if requested is not None:

        run_dir = requested.resolve()

        if not (
            run_dir / "experiment_info.json"
        ).exists():

            raise FileNotFoundError(
                f"Invalid E3 experiment directory: {run_dir}"
            )

    elif stage == "validation":

        # Hong Kong local date and time
        now = datetime.now(
            ZoneInfo("Asia/Hong_Kong")
        )

        base = (
            RESULT_ROOT
            / now.strftime("%Y%m%d-%H%M%S-E3")
        )

        run_dir = base

        index = 1

        while run_dir.exists():

            run_dir = Path(
                str(base) + f"-{index:02d}"
            )

            index += 1

        run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        info = {
            "experiment": "E3 Band-pass Comparison",

            "created_at": now.isoformat(),

            "timezone": "Asia/Hong_Kong",

            "seed": 123,

            "device": str(DEVICE),

            "data_dir": str(e2.DATA_DIR),

            "checkpoint_dir": str(BASELINE_DIR),

            "e2_reference_dir": str(E2_DIR),

            "torch_version": torch.__version__,

            "numpy_version": np.__version__,

            "scipy_version": scipy.__version__,

            "filter_method": (
                "Butterworth SOS + zero-phase sosfiltfilt"
            ),

            "selection_policy": (
                "All four candidates are evaluated and reported. "
                "No selection based on Test performance."
            ),

            **settings(),
        }

        write_json(
            run_dir / "experiment_info.json",
            info,
        )

        # Keep the same directory for Test and Summary
        ACTIVE_FILE.write_text(
            str(run_dir),
            encoding="utf-8",
        )

    else:

        if not ACTIVE_FILE.exists():

            raise FileNotFoundError(
                "No active E3 experiment. "
                "Run Validation first."
            )

        run_dir = Path(
            ACTIVE_FILE.read_text(
                encoding="utf-8"
            ).strip()
        )

    # Ensure the experiment still uses the same inputs
    info = read_json(
        run_dir / "experiment_info.json"
    )

    for key, value in settings().items():

        if info[key] != value:

            raise RuntimeError(
                f"Experiment inputs changed: {key}. "
                "Restore the original file or start a new run."
            )

    for folder in [
        "validation",
        "test",
        "comparison",
        "figures",
        "predictions",
    ]:

        (
            run_dir / folder
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    return run_dir


# ============================================================
# 6. Load frozen AttnSleep checkpoint
# ============================================================

def load_model(fold):

    folder = (
        BASELINE_DIR / f"fold{fold}"
    )

    cfg = read_json(
        folder / "config.json"
    )

    assert int(cfg["fold"]) == fold

    checkpoint = torch.load(
        folder / "best_validation_macro_f1.pth",
        map_location=DEVICE,
        weights_only=True,
    )

    model = AttnSleep().to(DEVICE)

    model.load_state_dict(
        checkpoint["state_dict"],
        strict=True,
    )

    model.eval()

    return model, int(cfg["batch_size"])


# ============================================================
# 7. Butterworth band-pass filter
# ============================================================

def make_filter(candidate, fs):

    low = candidate["low_hz"]
    high = candidate["high_hz"]

    if not 0 < low < high < fs / 2:

        raise ValueError(
            "Invalid band-pass frequencies"
        )

    return butter(
        candidate["order"],
        [low, high],
        btype="bandpass",
        fs=fs,
        output="sos",
    )


# ============================================================
# 8. E3 inference: Noise -> Filter -> Frozen AttnSleep
# ============================================================

def filtered_inference(
    model,
    batch_size,
    clean_x,
    y,
    sample_ids,
    condition,
    noise_cfg,
    sos,
):

    fs = float(
        noise_cfg["sampling_rate"]
    )

    seed = int(
        noise_cfg["seed"]
    )

    predictions = []

    before_error = 0.0
    after_error = 0.0

    duration = 0.0

    # Chunking reduces RAM usage
    chunk_size = batch_size * 4

    for start in range(
        0,
        len(y),
        chunk_size,
    ):

        end = min(
            start + chunk_size,
            len(y),
        )

        clean = clean_x[start:end]

        # ----------------------------------------
        # Generate identical noise instances to E2
        # ----------------------------------------

        if condition["noise_type"] == "clean":

            noisy = clean

        else:

            noisy = apply_noise_v2(
                clean,
                condition,
                sample_ids=sample_ids[start:end],
                seed=seed,
                fs=fs,
            )

        noisy = np.asarray(
            noisy,
            dtype=np.float32,
        )

        assert noisy.shape == clean.shape
        assert np.isfinite(noisy).all()

        # ----------------------------------------
        # Band-pass filtering
        # ----------------------------------------

        t0 = time.perf_counter()

        filtered = sosfiltfilt(
            sos,
            noisy.astype(np.float64),
            axis=-1,
        )

        filtered = np.ascontiguousarray(
            filtered,
            dtype=np.float32,
        )

        assert np.isfinite(filtered).all()

        # ----------------------------------------
        # Same prediction function as E2
        # ----------------------------------------

        pred = e2.predict(
            model,
            filtered,
            batch_size,
        )

        duration += (
            time.perf_counter() - t0
        )

        predictions.append(pred)

        # ----------------------------------------
        # Signal recovery metrics
        # ----------------------------------------

        reference = clean.astype(
            np.float64
        )

        before_error += float(
            np.square(
                noisy.astype(np.float64) - reference
            ).sum()
        )

        after_error += float(
            np.square(
                filtered.astype(np.float64) - reference
            ).sum()
        )

    pred = np.concatenate(predictions)

    score = e2.calculate_metrics(
        y,
        pred,
    )

    n_points = clean_x.size

    recovery = {
        "rmse_noisy_uv": float(
            np.sqrt(before_error / n_points)
        ),

        "rmse_filtered_uv": float(
            np.sqrt(after_error / n_points)
        ),

        "filter_and_inference_seconds": duration,

        "filter_and_inference_ms_per_epoch": (
            duration * 1000 / len(y)
        ),
    }

    if condition["noise_type"] == "clean":

        recovery["snr_improvement_db"] = None

    else:

        recovery["snr_improvement_db"] = float(
            10 * np.log10(
                max(before_error, 1e-30)
                / max(after_error, 1e-30)
            )
        )

    return score, pred, recovery


# ============================================================
# 9. E2 reference inference on Validation
# No filtering
# ============================================================

def e2_validation_inference(
    model,
    batch_size,
    clean_x,
    y,
    sample_ids,
    condition,
    noise_cfg,
):

    prediction_parts = []

    chunk_size = batch_size * 4

    for start in range(
        0,
        len(y),
        chunk_size,
    ):

        end = min(
            start + chunk_size,
            len(y),
        )

        chunk = clean_x[start:end]

        if condition["noise_type"] != "clean":

            chunk = apply_noise_v2(
                chunk,
                condition,
                sample_ids=sample_ids[start:end],
                seed=int(noise_cfg["seed"]),
                fs=float(noise_cfg["sampling_rate"]),
            )

        pred = e2.predict(
            model,
            np.asarray(
                chunk,
                dtype=np.float32,
            ),
            batch_size,
        )

        prediction_parts.append(pred)

    predictions = np.concatenate(
        prediction_parts
    )

    return e2.calculate_metrics(
        y,
        predictions,
    )


# ============================================================
# 10. Convert scores to CSV rows
# ============================================================

def metric_row(
    fold,
    condition,
    candidate,
    score,
    extra=None,
):

    row = {
        "fold": fold,

        "condition": condition,

        "filter_id": (
            candidate["id"]
            if candidate else "E2"
        ),

        "low_hz": (
            candidate["low_hz"]
            if candidate else ""
        ),

        "high_hz": (
            candidate["high_hz"]
            if candidate else ""
        ),

        "order": (
            candidate["order"]
            if candidate else ""
        ),

        "samples": score["samples"],

        "accuracy_pct": (
            score["accuracy"] * 100
        ),

        "macro_f1_pct": (
            score["macro_f1"] * 100
        ),

        "kappa": score["kappa"],
    }

    if extra:
        row.update(extra)

    return row


# ============================================================
# 11. Evaluate one fold: Validation or Test
# ============================================================

def evaluate_fold(
    run_dir,
    stage,
    fold,
    split,
    noise_cfg,
):

    model, batch_size = load_model(fold)

    # Validation and Test use their own subjects
    group = (
        split["validation"]
        if stage == "validation"
        else split["test"]
    )

    # Reuse exact E2-v2 data loading
    clean_x, y, sample_ids = e2.load_test(
        group
    )

    conditions = noise_cfg["conditions"]

    fs = float(
        noise_cfg["sampling_rate"]
    )

    filter_map = {
        c["id"]: make_filter(c, fs)
        for c in CANDIDATES
    }

    rows = []
    e2_rows = []
    details = {}

    print(
        f"\n========== E3 {stage.upper()} "
        f"Fold {fold} ==========",
        flush=True,
    )

    print(f"Samples: {len(y)}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {DEVICE}")

    for condition in conditions:

        name = condition["name"]

        # ----------------------------------------
        # E2 reference
        # ----------------------------------------

        if stage == "validation":

            e2_score = e2_validation_inference(
                model,
                batch_size,
                clean_x,
                y,
                sample_ids,
                condition,
                noise_cfg,
            )

        else:

            # Reuse saved E2 Test predictions
            reference_file = (
                E2_DIR
                / "predictions"
                / f"fold{fold}_{name}.npz"
            )

            if not reference_file.exists():

                raise FileNotFoundError(
                    f"E2 result not found: {reference_file}"
                )

            with np.load(
                reference_file,
                allow_pickle=False,
            ) as data:

                e2_y = data["y_true"]
                e2_pred = data["y_pred"]

            if not np.array_equal(y, e2_y):

                raise RuntimeError(
                    f"E2/E3 labels differ: Fold {fold}, {name}"
                )

            e2_score = e2.calculate_metrics(
                e2_y,
                e2_pred,
            )

        e2_rows.append(
            metric_row(
                fold,
                name,
                None,
                e2_score,
            )
        )

        # ----------------------------------------
        # E3: evaluate all four filters
        # ----------------------------------------

        for candidate in CANDIDATES:

            filter_id = candidate["id"]

            score, pred, recovery = filtered_inference(
                model,
                batch_size,
                clean_x,
                y,
                sample_ids,
                condition,
                noise_cfg,
                filter_map[filter_id],
            )

            delta_f1 = 100 * (
                score["macro_f1"]
                - e2_score["macro_f1"]
            )

            delta_acc = 100 * (
                score["accuracy"]
                - e2_score["accuracy"]
            )

            rows.append(
                metric_row(
                    fold,
                    name,
                    candidate,
                    score,
                    {
                        **recovery,

                        "delta_accuracy_vs_e2_pp": delta_acc,

                        "delta_macro_f1_vs_e2_pp": delta_f1,
                    },
                )
            )

            details[
                f"{filter_id}_{name}"
            ] = score

            # Save Test predictions for later analysis
            if stage == "test":

                output = (
                    run_dir
                    / "predictions"
                    / f"fold{fold}_{filter_id}_{name}.npz"
                )

                np.savez_compressed(
                    output,
                    y_true=y,
                    y_pred=pred,
                )

            print(
                f"  {name:<18} "
                f"Filter={filter_id} "
                f"F1={score['macro_f1']*100:.2f}% "
                f"E2={e2_score['macro_f1']*100:.2f}% "
                f"Delta={delta_f1:+.2f} pp",
                flush=True,
            )

    # --------------------------------------------
    # Save this fold
    # --------------------------------------------

    update_fold_csv(
        run_dir / stage / "per_fold.csv",
        fold,
        rows,
    )

    update_fold_csv(
        run_dir / stage / "e2_reference_per_fold.csv",
        fold,
        e2_rows,
    )

    update_fold_json(
        run_dir / stage / "details.json",
        fold,
        details,
    )

    print(
        f"Fold {fold} saved.",
        flush=True,
    )


# ============================================================
# 12. Verify that all five folds are complete
# ============================================================

def ensure_complete(
    rows,
    rows_per_fold,
    description,
):

    unique = {
        (
            int(r["fold"]),
            r["condition"],
            r["filter_id"],
        )
        for r in rows
    }

    expected = 5 * rows_per_fold

    if (
        len(rows) != expected
        or len(unique) != len(rows)
    ):

        raise RuntimeError(
            f"Incomplete {description}: "
            f"found {len(rows)} rows, expected {expected}"
        )


def mean_sd(numbers):

    values = np.asarray(
        numbers,
        dtype=np.float64,
    )

    return (
        float(values.mean()),
        float(values.std(ddof=1))
        if len(values) > 1 else 0.0,
    )


# ============================================================
# 13. Summarize one stage by filter and noise condition
# ============================================================

def aggregate_stage(
    rows,
    filter_ids,
    condition_names,
):

    summary = []

    for filter_id in filter_ids:

        for condition in condition_names:

            selected = [
                r for r in rows
                if (
                    r["filter_id"] == filter_id
                    and r["condition"] == condition
                )
            ]

            if len(selected) != 5:

                raise RuntimeError(
                    f"Missing results: {filter_id}, {condition}"
                )

            item = {
                "filter_id": filter_id,
                "condition": condition,
            }

            for key in [
                "accuracy_pct",
                "macro_f1_pct",
                "kappa",
            ]:

                values = [
                    float(r[key])
                    for r in selected
                ]

                avg, sd = mean_sd(values)

                item[f"mean_{key}"] = avg
                item[f"sd_{key}"] = sd

            summary.append(item)

    return summary


# ============================================================
# 14. Build final comparison tables
# ============================================================

def build_summary(run_dir, noise_cfg):

    names = [
        c["name"]
        for c in noise_cfg["conditions"]
    ]

    noisy_names = [
        name for name in names
        if name != "Clean"
    ]

    filter_ids = (
        ["E2"]
        + [c["id"] for c in CANDIDATES]
    )

    all_data = {}

    # --------------------------------------------
    # Read Validation and Test
    # --------------------------------------------

    for stage in [
        "validation",
        "test",
    ]:

        e3_rows = read_csv(
            run_dir / stage / "per_fold.csv"
        )

        e2_rows = read_csv(
            run_dir / stage / "e2_reference_per_fold.csv"
        )

        # Four filters × ten conditions = 40 per fold
        ensure_complete(
            e3_rows,
            40,
            f"{stage} E3",
        )

        # One E2 baseline × ten conditions
        ensure_complete(
            e2_rows,
            10,
            f"{stage} E2",
        )

        combined = e2_rows + e3_rows

        all_data[stage] = combined

        stage_summary = aggregate_stage(
            combined,
            filter_ids,
            names,
        )

        write_csv(
            run_dir / stage / "summary.csv",
            stage_summary,
        )

    # --------------------------------------------
    # Overall comparison
    # --------------------------------------------

    overall = []

    for filter_id in filter_ids:

        item = {
            "filter_id": filter_id
        }

        for stage in [
            "validation",
            "test",
        ]:

            rows = all_data[stage]

            noisy_f1_by_fold = []
            noisy_acc_by_fold = []

            for fold in range(5):

                selected = [
                    r for r in rows
                    if (
                        r["filter_id"] == filter_id
                        and int(r["fold"]) == fold
                        and r["condition"] in noisy_names
                    )
                ]

                if len(selected) != 9:

                    raise RuntimeError(
                        f"Incomplete {stage}: "
                        f"{filter_id}, Fold {fold}"
                    )

                noisy_f1_by_fold.append(
                    np.mean([
                        float(r["macro_f1_pct"])
                        for r in selected
                    ])
                )

                noisy_acc_by_fold.append(
                    np.mean([
                        float(r["accuracy_pct"])
                        for r in selected
                    ])
                )

            f1_mean, f1_sd = mean_sd(
                noisy_f1_by_fold
            )

            acc_mean, acc_sd = mean_sd(
                noisy_acc_by_fold
            )

            item[
                f"{stage}_noisy_macro_f1_pct"
            ] = f1_mean

            item[
                f"{stage}_noisy_macro_f1_sd"
            ] = f1_sd

            item[
                f"{stage}_noisy_accuracy_pct"
            ] = acc_mean

            item[
                f"{stage}_noisy_accuracy_sd"
            ] = acc_sd

            # Clean is reported separately
            clean_f1 = [
                float(r["macro_f1_pct"])
                for r in rows
                if (
                    r["filter_id"] == filter_id
                    and r["condition"] == "Clean"
                )
            ]

            clean_acc = [
                float(r["accuracy_pct"])
                for r in rows
                if (
                    r["filter_id"] == filter_id
                    and r["condition"] == "Clean"
                )
            ]

            item[
                f"{stage}_clean_macro_f1_pct"
            ] = mean_sd(clean_f1)[0]

            item[
                f"{stage}_clean_accuracy_pct"
            ] = mean_sd(clean_acc)[0]

        overall.append(item)

    # --------------------------------------------
    # E3 - E2 difference
    # --------------------------------------------

    e2_row = next(
        r for r in overall
        if r["filter_id"] == "E2"
    )

    for item in overall:

        item["delta_test_noisy_f1_vs_e2_pp"] = (
            item["test_noisy_macro_f1_pct"]
            - e2_row["test_noisy_macro_f1_pct"]
        )

        item["delta_test_clean_f1_vs_e2_pp"] = (
            item["test_clean_macro_f1_pct"]
            - e2_row["test_clean_macro_f1_pct"]
        )

    write_csv(
        run_dir
        / "comparison"
        / "overall_comparison.csv",
        overall,
    )

    # --------------------------------------------
    # Detailed noise comparison
    # --------------------------------------------

    test_summary = aggregate_stage(
        all_data["test"],
        filter_ids,
        names,
    )

    lookup = {
        (r["filter_id"], r["condition"]): r
        for r in test_summary
    }

    comparison = []

    for candidate in CANDIDATES:

        filter_id = candidate["id"]

        for name in names:

            e2_result = lookup[
                ("E2", name)
            ]

            e3_result = lookup[
                (filter_id, name)
            ]

            comparison.append({
                "filter_id": filter_id,

                "condition": name,

                "e2_macro_f1_pct":
                    e2_result["mean_macro_f1_pct"],

                "e3_macro_f1_pct":
                    e3_result["mean_macro_f1_pct"],

                "delta_macro_f1_pp": (
                    e3_result["mean_macro_f1_pct"]
                    - e2_result["mean_macro_f1_pct"]
                ),

                "e2_accuracy_pct":
                    e2_result["mean_accuracy_pct"],

                "e3_accuracy_pct":
                    e3_result["mean_accuracy_pct"],

                "delta_accuracy_pp": (
                    e3_result["mean_accuracy_pct"]
                    - e2_result["mean_accuracy_pct"]
                ),
            })

    write_csv(
        run_dir
        / "comparison"
        / "noise_comparison.csv",
        comparison,
    )

    return overall, comparison


# ============================================================
# 15. Generate all visualizations
# ============================================================

def save_figures(
    run_dir,
    overall,
    comparison,
    noise_cfg,
):

    figure_dir = (
        run_dir / "figures"
    )

    filters = [
        "E2", "A", "B", "C", "D"
    ]

    lookup = {
        row["filter_id"]: row
        for row in overall
    }

    # ========================================================
    # Figure 1: Validation vs Test
    # ========================================================

    x = np.arange(
        len(filters)
    )

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(10, 5),
        dpi=150,
    )

    val_scores = [
        lookup[f]["validation_noisy_macro_f1_pct"]
        for f in filters
    ]

    test_scores = [
        lookup[f]["test_noisy_macro_f1_pct"]
        for f in filters
    ]

    ax.bar(
        x - width / 2,
        val_scores,
        width,
        label="Validation",
    )

    ax.bar(
        x + width / 2,
        test_scores,
        width,
        label="Test",
    )

    ax.set_xticks(
        x,
        filters,
    )

    ax.set_ylabel(
        "Mean Macro-F1 (%)"
    )

    ax.set_title(
        "E3 Band-pass: Validation vs Test\n"
        "Nine Noisy Conditions, Five-Fold Mean"
    )

    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        figure_dir / "validation_vs_test.png",
        dpi=200,
    )

    plt.close(fig)

    # ========================================================
    # Figure 2: E3 - E2 Heatmap
    # ========================================================

    names = [
        c["name"]
        for c in noise_cfg["conditions"]
        if c["name"] != "Clean"
    ]

    matrix = np.asarray([
        [
            next(
                r["delta_macro_f1_pp"]
                for r in comparison
                if (
                    r["filter_id"] == filter_id
                    and r["condition"] == name
                )
            )
            for name in names
        ]
        for filter_id in filters[1:]
    ])

    fig, ax = plt.subplots(
        figsize=(13, 5),
        dpi=150,
    )

    limit = max(
        float(np.max(np.abs(matrix))),
        1.0,
    )

    image = ax.imshow(
        matrix,
        cmap="RdBu",
        vmin=-limit,
        vmax=limit,
        aspect="auto",
    )

    ax.set_yticks(
        range(4),
        filters[1:],
    )

    ax.set_xticks(
        range(9),
        names,
        rotation=35,
        ha="right",
    )

    ax.set_title(
        "E3 - E2 Test Macro-F1 (Percentage Points)"
    )

    for i in range(4):

        for j in range(9):

            ax.text(
                j,
                i,
                f"{matrix[i, j]:+.1f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    fig.colorbar(
        image,
        ax=ax,
        label="Difference (pp)",
    )

    fig.tight_layout()

    fig.savefig(
        figure_dir / "e3_vs_e2_heatmap.png",
        dpi=200,
    )

    plt.close(fig)

    # ========================================================
    # Figure 3: EEG waveforms
    # Clean / Noisy / Filtered A-D
    # ========================================================

    sample_file = "SC4001E0.npz"
    epoch = 100

    with np.load(
        e2.DATA_DIR / sample_file,
        allow_pickle=False,
    ) as z:

        clean = np.asarray(
            z["x"][epoch, :, 0],
            dtype=np.float32,
        )

    sample_id = (
        f"{sample_file}:{epoch}"
    )

    condition = next(
        c for c in noise_cfg["conditions"]
        if c["name"] == "Gaussian_Medium"
    )

    fs = float(
        noise_cfg["sampling_rate"]
    )

    noisy = apply_noise_v2(
        clean.reshape(1, 1, -1),
        condition,
        sample_ids=[sample_id],
        seed=int(noise_cfg["seed"]),
        fs=fs,
    )[0, 0]

    signals = [
        ("Clean EEG", clean),
        ("Gaussian 10 dB", noisy),
    ]

    for candidate in CANDIDATES:

        sos = make_filter(
            candidate,
            fs,
        )

        filtered = sosfiltfilt(
            sos,
            np.asarray(
                noisy,
                dtype=np.float64,
            ),
            axis=-1,
        )

        label = (
            f"Filter {candidate['id']} "
            f"({candidate['low_hz']}-"
            f"{candidate['high_hz']} Hz)"
        )

        signals.append(
            (label, filtered)
        )

    t = np.arange(
        len(clean)
    ) / fs

    fig, axes = plt.subplots(
        len(signals),
        1,
        figsize=(14, 11),
        dpi=150,
        sharex=True,
    )

    ymax = max(
        float(np.max(np.abs(signal)))
        for _, signal in signals
    ) * 1.08

    for ax, (label, signal) in zip(
        axes,
        signals,
    ):

        ax.plot(
            t,
            signal,
            linewidth=0.8,
        )

        ax.set_title(
            label,
            fontsize=10,
        )

        ax.set_ylabel("uV")
        ax.set_ylim(-ymax, ymax)
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel(
        "Time (seconds)"
    )

    fig.suptitle(
        f"E3: Clean / Noisy / Filtered EEG\n"
        f"{sample_id}"
    )

    fig.tight_layout()

    fig.savefig(
        figure_dir / "eeg_filter_comparison.png",
        dpi=200,
    )

    plt.close(fig)


# ============================================================
# 16. Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stage",
        required=True,
        choices=[
            "validation",
            "test",
            "summary",
        ],
    )

    parser.add_argument(
        "--fold",
        default="all",
        choices=[
            "0", "1", "2", "3", "4", "all"
        ],
    )

    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Reuse a specific E3 experiment directory",
    )

    args = parser.parse_args()

    torch.set_num_threads(4)

    noise_cfg = read_json(
        NOISE_FILE
    )

    splits = read_json(
        SPLIT_FILE
    )

    assert (
        noise_cfg["seed"]
        == splits["seed"]
        == 123
    )

    assert noise_cfg["sampling_rate"] == 100
    assert noise_cfg["signal_unit"] == "uV"

    assert len(
        noise_cfg["conditions"]
    ) == 10

    assert len(
        splits["folds"]
    ) == 5

    assert (
        noise_cfg["conditions"][0]["name"]
        == "Clean"
    )

    run_dir = get_run_directory(
        args.stage,
        args.run_dir,
    )

    print(
        f"\nExperiment folder: {run_dir}",
        flush=True,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    if args.stage == "summary":

        overall, comparison = build_summary(
            run_dir,
            noise_cfg,
        )

        save_figures(
            run_dir,
            overall,
            comparison,
            noise_cfg,
        )

        print(
            "\n========== E3 Overall Comparison =========="
        )

        print(
            "Filter | Validation F1 | Test F1 | Delta vs E2"
        )

        for row in overall:

            print(
                f"{row['filter_id']:>5} | "
                f"{row['validation_noisy_macro_f1_pct']:>8.2f}% | "
                f"{row['test_noisy_macro_f1_pct']:>8.2f}% | "
                f"{row['delta_test_noisy_f1_vs_e2_pp']:+.2f} pp"
            )

        print(
            "\nComparison tables and figures saved."
        )

        return

    # --------------------------------------------------------
    # Validation or Test
    # --------------------------------------------------------

    folds = (
        range(5)
        if args.fold == "all"
        else [int(args.fold)]
    )

    if args.stage == "test":

        validation_rows = read_csv(
            run_dir / "validation/per_fold.csv"
        )

        completed_folds = {
            int(r["fold"])
            for r in validation_rows
        }

        missing = (
            set(folds) - completed_folds
        )

        if missing:

            raise RuntimeError(
                f"Validation missing for folds: {sorted(missing)}"
            )

    for fold in folds:

        split = splits["folds"][fold]

        assert split["fold"] == fold

        evaluate_fold(
            run_dir,
            args.stage,
            fold,
            split,
            noise_cfg,
        )

    print(
        f"\n{args.stage.upper()} completed."
    )

    print(
        f"Results saved to: {run_dir}"
    )


if __name__ == "__main__":
    main()
