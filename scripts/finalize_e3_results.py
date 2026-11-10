
"""E3 final analysis, without training, filtering or model inference.

From Sleep/:
    python scripts/finalize_e3_results.py

Uses all ten Validation conditions (including Clean) to choose one A-D
filter for EACH fold; only then extracts the corresponding saved Test outputs.
Sample IDs come from the E2 predictions_with_ids.csv already validated.
Existing E2/E3 files are never overwritten.
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                             confusion_matrix, precision_recall_fscore_support)

ROOT = Path(__file__).resolve().parents[1]
STAGES = ["W", "N1", "N2", "N3", "REM"]
FILTERS = list("ABCD")


def csv_read(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows


def csv_save(path, rows):
    if not rows:
        raise ValueError(f"No data to save: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    with temp.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def json_save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def column(row, names, meaning):
    for name in names:
        if name in row and str(row[name]).strip():
            return str(row[name]).strip()
    raise ValueError(f"Missing {meaning}; available CSV columns: {list(row)}")


def row_fold(row):
    value = column(row, ["fold", "fold_id", "Fold"], "fold")
    match = re.fullmatch(r"(?:fold)?\s*([0-4])", value, flags=re.I)
    if match is None:
        raise ValueError(f"Invalid fold: {value}")
    return int(match.group(1))


def row_filter(row):
    value = column(row, ["filter", "filter_id", "filter_name", "setting",
                         "bandpass", "parameter"], "filter")
    match = re.match(r"^([ABCD])(?:$|[\s_:(-])", value, flags=re.I)
    if match is None:
        raise ValueError(f"Invalid filter: {value}")
    return match.group(1).upper()


def row_condition(row):
    return column(row, ["condition", "noise_condition", "noise", "Condition"], "condition")


def row_f1(row):
    value = float(column(row, ["macro_f1_pct", "macro_f1", "macro_f1_score",
                               "Macro-F1"], "Macro-F1"))
    if not np.isfinite(value) or not 0 <= value <= 100:
        raise ValueError(f"Invalid Macro-F1: {value}")
    return value * 100 if value <= 1 else value


def index_rows(rows, conditions):
    indexed = {}
    for row in rows:
        key = row_fold(row), row_filter(row), row_condition(row)
        if key in indexed or key[2] not in conditions:
            raise ValueError(f"Duplicate or unknown fold/filter/condition: {key}")
        indexed[key] = row
    return indexed


def average_sd(values):
    x = np.asarray(values, dtype=float)
    if len(x) != 5 or not np.isfinite(x).all():
        raise ValueError("Five finite fold values required")
    return float(x.mean()), float(x.std(ddof=1))


def metrics(y, pred):
    if len(y) != len(pred) or not len(y):
        raise ValueError("Invalid prediction count")
    if not np.isin(y, range(5)).all() or not np.isin(pred, range(5)).all():
        raise ValueError("Stage labels must be 0..4")
    p, r, f, support = precision_recall_fscore_support(
        y, pred, labels=list(range(5)), zero_division=0
    )
    return {
        "samples": len(y),
        "accuracy_pct": float(accuracy_score(y, pred) * 100),
        "macro_f1_pct": float(f.mean() * 100),
        "kappa": float(cohen_kappa_score(y, pred)),
        "confusion_matrix": confusion_matrix(y, pred, labels=list(range(5))).tolist(),
        "stage_metrics": {
            name: {
                "precision_pct": float(p[i] * 100),
                "recall_pct": float(r[i] * 100),
                "f1_pct": float(f[i] * 100),
                "support": int(support[i]),
            }
            for i, name in enumerate(STAGES)
        },
    }


def load_e2_ids(path):
    """Only Clean records are needed; same IDs apply to every test noise."""
    identities = defaultdict(list)
    observed = set()
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"fold", "subject_id", "recording_id", "epoch_index",
                    "sample_id", "condition", "y_true"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"E2 identity CSV is missing: {required - set(reader.fieldnames or [])}")
        for row in reader:
            if row["condition"] != "Clean":
                continue
            fold, sid = int(row["fold"]), row["sample_id"]
            if fold not in range(5) or sid in observed:
                raise ValueError(f"Duplicate or invalid E2 Sample ID: {sid}")
            observed.add(sid)
            identities[fold].append(row)
    if set(identities) != set(range(5)):
        raise ValueError("E2 Sample IDs missing for some folds")
    print(f"E2 verified Sample IDs: {len(observed):,} unique Test epochs")
    return identities


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e3-dir", type=Path,
                    default=ROOT / "result/20261104-050058-E3")
    ap.add_argument("--e2-dir", type=Path, default=ROOT / "result/e2_v2")
    args = ap.parse_args()
    e3, e2 = args.e3_dir.resolve(), args.e2_dir.resolve()
    cfg = json.loads((ROOT / "configs/noise_conditions_v2.json").read_text())
    conditions = [item["name"] for item in cfg["conditions"]]
    if len(conditions) != 10 or len(set(conditions)) != 10 or "Clean" not in conditions:
        raise ValueError("Expected 10 unique conditions including Clean")
    noisy = [c for c in conditions if c != "Clean"]

    # Select using Validation ONLY (before reading Test).
    val = index_rows(csv_read(e3 / "validation/per_fold.csv"), conditions)
    expected = {(i, f, c) for i in range(5) for f in FILTERS for c in conditions}
    if set(val) != expected:
        raise ValueError(f"Incomplete Validation results: {list(expected - set(val))[:8]}")
    selected, candidate_rows, selection_rows = {}, [], []
    for fold in range(5):
        means = {f: float(np.mean([row_f1(val[fold, f, c]) for c in conditions]))
                 for f in FILTERS}
        ranking = sorted(FILTERS, key=lambda f: (-means[f], FILTERS.index(f)))
        selected[fold] = ranking[0]
        for f in FILTERS:
            candidate_rows.append({"fold": fold, "filter_id": f,
                                   "validation_mean_macro_f1_pct": means[f],
                                   "selected": int(f == selected[fold])})
        selection_rows.append({
            "fold": fold, "selected_filter": selected[fold],
            "selection_rule": "10 equally weighted Validation conditions including Clean",
            "validation_mean_macro_f1_pct": means[selected[fold]],
            "runner_up_filter": ranking[1], "margin_pp": means[ranking[0]] - means[ranking[1]],
            **{f"validation_{f}_macro_f1_pct": means[f] for f in FILTERS},
        })
    print("Validation-selected filters:", [selected[i] for i in range(5)])
    if [selected[i] for i in range(5)] != list("AAAAB"):
        print("WARNING: Not AAAAB. Verify the actual Validation table before reporting.")

    test = index_rows(csv_read(e3 / "test/per_fold.csv"), conditions)
    for fold in range(5):
        for condition in conditions:
            if (fold, selected[fold], condition) not in test:
                raise ValueError(f"Missing selected Test row: fold {fold}, {condition}")
    e2_scores = {}
    for row in csv_read(e2 / "e2_per_fold.csv"):
        key = int(row["fold"]), row["condition"]
        if key in e2_scores:
            raise ValueError(f"Duplicate E2 score: {key}")
        e2_scores[key] = row_f1(row)
    if {(i, c) for i in range(5) for c in conditions} - set(e2_scores):
        raise ValueError("E2 reference table is incomplete")
    identities = load_e2_ids(e2 / "predictions_with_ids.csv")

    comp = e3 / "comparison"
    output = e3 / "predictions/selected_predictions_with_ids.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + ".partial")
    selected_rows, fold_scores, details = [], {}, {}
    pooled_rows, pooled_stages = [], []
    pooled_cm = {"labels": STAGES, "axis0": "y_true", "axis1": "y_pred", "conditions": {}}
    try:
        with temp.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            writer.writerow(["fold", "subject_id", "recording_id", "epoch_index",
                             "sample_id", "condition", "filter_id", "y_true", "y_pred"])
            for condition in conditions:
                true_chunks, pred_chunks = [], []
                for fold in range(5):
                    f = selected[fold]
                    filename = f"fold{fold}_{f}_{condition}.npz"
                    with np.load(e3 / "predictions" / filename, allow_pickle=False) as npz:
                        true = np.asarray(npz["y_true"], dtype=int).reshape(-1)
                        pred = np.asarray(npz["y_pred"], dtype=int).reshape(-1)
                    ids = identities[fold]
                    if not np.array_equal(true, np.array([int(x["y_true"]) for x in ids])):
                        raise ValueError(f"Sample ordering differs between E2 and E3: {filename}")
                    score = metrics(true, pred)
                    original = test[fold, f, condition]
                    if abs(score["macro_f1_pct"] - row_f1(original)) > 0.05:
                        raise ValueError(f"Archived E3 Macro-F1 differs from predictions: {filename}")
                    if "samples" in original and original["samples"]:
                        if int(float(original["samples"])) != len(true):
                            raise ValueError(f"E3 sample count mismatch: {filename}")
                    fold_scores[fold, condition] = score
                    details[f"fold{fold}_{f}_{condition}"] = {
                        "fold": fold, "filter_id": f, "condition": condition, **score}
                    saved_row = dict(original)  # preserves RMSE/SNR/time values
                    saved_row.update({"selected_filter": f,
                                      "e2_macro_f1_pct": e2_scores[fold, condition],
                                      "delta_macro_f1_pp": score["macro_f1_pct"] - e2_scores[fold, condition]})
                    selected_rows.append(saved_row)
                    writer.writerows((fold, x["subject_id"], x["recording_id"],
                                      x["epoch_index"], x["sample_id"], condition, f,
                                      int(true[i]), int(pred[i])) for i, x in enumerate(ids))
                    true_chunks.append(true)
                    pred_chunks.append(pred)
                combined = metrics(np.concatenate(true_chunks), np.concatenate(pred_chunks))
                pooled_rows.append({"condition": condition, **{k: combined[k]
                                   for k in ("samples", "accuracy_pct", "macro_f1_pct", "kappa")}})
                pooled_cm["conditions"][condition] = combined["confusion_matrix"]
                pooled_stages.extend({"condition": condition, "stage": stage, **value}
                                     for stage, value in combined["stage_metrics"].items())
                print(f"Validated {condition:<17} | pooled epochs={combined['samples']:,} | Macro-F1={combined['macro_f1_pct']:.2f}%")
        temp.replace(output)
    finally:
        if temp.exists():
            temp.unlink()

    summary, stage_summary, quality_summary = [], [], []
    quality_cols = sorted({k for row in selected_rows for k in row
                           if any(tag in k.lower() for tag in
                                  ("rmse", "snr", "time", "seconds", "latency", "duration"))
                           and k.lower() not in ("timestamp", "datetime")})
    for condition in conditions:
        items = [fold_scores[i, condition] for i in range(5)]
        result = {"condition": condition,
                  "selected_filters": ",".join(selected[i] for i in range(5))}
        for key in ("accuracy_pct", "macro_f1_pct", "kappa"):
            mean, sd = average_sd([x[key] for x in items])
            result[f"mean_{key}"], result[f"sd_{key}"] = mean, sd
        result["e2_mean_macro_f1_pct"], result["e2_sd_macro_f1_pct"] = average_sd(
            [e2_scores[i, condition] for i in range(5)])
        result["delta_macro_f1_pp"] = result["mean_macro_f1_pct"] - result["e2_mean_macro_f1_pct"]
        summary.append(result)
        for stage in STAGES:
            values = [x["stage_metrics"][stage] for x in items]
            row = {"condition": condition, "stage": stage,
                   "total_support": sum(x["support"] for x in values)}
            for key in ("precision_pct", "recall_pct", "f1_pct"):
                mean, sd = average_sd([x[key] for x in values])
                row[f"mean_{key}"], row[f"sd_{key}"] = mean, sd
            stage_summary.append(row)
        for col in quality_cols:
            condition_rows = [next(r for r in selected_rows if row_fold(r) == i
                                   and row_condition(r) == condition) for i in range(5)]
            try:
                vals = [float(r[col]) for r in condition_rows]
                mean, sd = average_sd(vals)
            except (KeyError, ValueError, TypeError):
                continue
            quality_summary.append({"condition": condition, "metric": col,
                                    "mean": mean, "sd": sd})

    noisy_folds = []
    for fold in range(5):
        v3 = float(np.mean([fold_scores[fold, c]["macro_f1_pct"] for c in noisy]))
        v2 = float(np.mean([e2_scores[fold, c] for c in noisy]))
        noisy_folds.append({"fold": fold, "selected_filter": selected[fold],
                            "e3_noisy_macro_f1_pct": v3,
                            "e2_noisy_macro_f1_pct": v2, "delta_pp": v3 - v2})
    e3_mean, e3_sd = average_sd([x["e3_noisy_macro_f1_pct"] for x in noisy_folds])
    e2_mean, e2_sd = average_sd([x["e2_noisy_macro_f1_pct"] for x in noisy_folds])
    overall = {"selected_by_fold": [selected[i] for i in range(5)],
               "e3_noisy_mean_pct": e3_mean, "e3_noisy_sd_pp": e3_sd,
               "e2_noisy_mean_pct": e2_mean, "e2_noisy_sd_pp": e2_sd,
               "delta_pp": e3_mean - e2_mean,
               "selection_note": "Reconstructed from saved Validation (including Clean); no Test-based selection.",
               "model_inference": False}

    csv_save(comp / "validation_candidate_scores.csv", candidate_rows)
    csv_save(comp / "parameter_selection.csv", selection_rows)
    csv_save(comp / "selected_per_fold.csv", selected_rows)
    csv_save(comp / "selected_summary.csv", summary)
    csv_save(comp / "selected_stage_summary.csv", stage_summary)
    csv_save(comp / "selected_pooled_summary.csv", pooled_rows)
    csv_save(comp / "selected_pooled_stage_metrics.csv", pooled_stages)
    csv_save(comp / "selected_noisy_per_fold.csv", noisy_folds)
    if quality_summary:
        csv_save(comp / "selected_quality_time_summary.csv", quality_summary)
    json_save(comp / "selected_fold_details.json", details)
    json_save(comp / "selected_pooled_confusion_matrix.json", pooled_cm)
    json_save(comp / "selected_overall.json", overall)
    json_save(comp / "selected_analysis_manifest.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "e2_identity_source": str(e2 / "predictions_with_ids.csv"),
        "selection": selected, "selected_test_files": 50,
        "mean_sd": "sample SD across five folds (ddof=1)",
        "note": "Validation-based selection reconstructed after candidate Test runs; no model inference."
    })
    print("\nE3 FINALIZATION PASS")
    print("Selected:", [selected[i] for i in range(5)])
    print(f"E3 nine-noise Test Macro-F1: {e3_mean:.2f}% +/- {e3_sd:.2f} pp")
    print(f"E2 nine-noise Test Macro-F1: {e2_mean:.2f}% +/- {e2_sd:.2f} pp")
    print(f"Difference: {e3_mean-e2_mean:+.2f} pp")
    print("Saved to:", comp, "and", output)
    print("No training, no filtering, no noise generation, no inference.")


if __name__ == "__main__":
    main()
