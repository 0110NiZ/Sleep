"""Build presentation figure inputs from the committed test results."""
import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULT = REPO / "results"
FOLDS = range(5)
FAMILIES = ("Gaussian", "Drift", "Burst")
LEVELS = ("Low", "Medium", "High")
NOISY = [f"{family}_{level}" for family in FAMILIES for level in LEVELS]
METHODS = {
    "E2": ("Noisy baseline", "E2_noisy_baseline/e2_per_fold.csv"),
    "E3": ("Band-pass", "E3_bandpass/comparison/selected_per_fold.csv"),
    "E4": ("Wavelet", "E4_wavelet/test_per_fold.csv"),
    "E5": ("DAE", "E5_dae/test_per_fold.csv"),
    "E6": ("Augmentation", "E6_noise_augmentation/test_per_fold.csv"),
}
PLOTS = [
    "E2_noise_impact", "clean_vs_noisy_macro_f1", "gain_vs_E2_by_noise_condition",
    "E4_E5_rmse_vs_macro_f1", "E2_E5_E6_f1_by_sleep_stage",
]


def read_rows(relative):
    with (RESULT / relative).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_table(path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def build_tables(destination):
    destination.mkdir()
    tables = {}
    means = {}
    expected = {(fold, name) for fold in FOLDS for name in ["Clean", *NOISY]}
    for method, (_, path) in METHODS.items():
        rows = read_rows(path)
        table = {(int(row["fold"]), row["condition"]): row for row in rows}
        if set(table) != expected or len(rows) != len(expected):
            raise ValueError(f"Incomplete or duplicated test results in {path}")
        tables[method] = table
        means[method] = {
            name: statistics.mean(float(table[fold, name]["macro_f1_pct"]) for fold in FOLDS)
            for name in ["Clean", *NOISY]
        }
    baseline = read_rows("E1_clean_baseline/clean_baseline_5fold_results.csv")
    if len(baseline) != 5 or {int(row["fold"]) for row in baseline} != set(range(1, 6)):
        raise ValueError("Incomplete E1 test results")
    cpu = {r["fold"]: r for r in json.loads((REPO / "docs/checks/reproduction/e1_comparison.json").read_text())}
    for row in baseline:
        fold = int(row["fold"]) - 1
        e2 = tables["E2"][fold, "Clean"]
        if (abs(cpu[fold]["macro_f1"] * 100 - float(e2["macro_f1_pct"])) > 1e-6
                or int(row["test_samples"]) != int(e2["samples"])):
            raise ValueError("E1 CPU recheck and the E2 clean reference differ")
    overall = []
    for method, (label, _) in METHODS.items():
        t = tables[method]
        clean = [float(t[f, "Clean"]["macro_f1_pct"]) for f in FOLDS]
        noisy = [statistics.mean(float(t[f, c]["macro_f1_pct"]) for c in NOISY) for f in FOLDS]
        overall.append([method, label, statistics.mean(clean), statistics.stdev(clean),
                        statistics.mean(noisy), statistics.stdev(noisy)])
    write_table(destination / "overall_performance.csv",
                ["method", "name", "clean_f1", "clean_sample_sd", "mean_noisy_f1", "noisy_sample_sd"], overall)
    protocol = json.loads((REPO / "configs/noise_conditions_v2.json").read_text())
    conditions = {c["name"]: c for c in protocol["conditions"]}
    settings = {"Gaussian": ("snr_db", " dB"), "Drift": ("amplitude_uv", " μV"),
                "Burst": ("amplitude_ratio", "σ")}
    baseline_noise = []
    for name in NOISY:
        family, level = name.split("_")
        key, unit = settings[family]
        baseline_noise.append([family, level, f"{conditions[name][key]:g}{unit}", means["E2"][name]])
    write_table(destination / "baseline_noise.csv", ["noise", "severity", "setting", "macro_f1"], baseline_noise)
    write_table(destination / "condition_deltas.csv", ["method", *[c.replace("_", " ") for c in NOISY]],
                ([m, *[means[m][c] - means["E2"][c] for c in NOISY]] for m in METHODS if m != "E2"))
    case = "Gaussian_Medium"
    write_table(destination / "reconstruction_case.csv", ["method", "rmse_uv", "macro_f1"],
                ([m, statistics.mean(float(tables[m][f, case]["rmse_denoised_uv"]) for f in FOLDS),
                  means[m][case]] for m in ("E4", "E5")))
    stage_names = ("W", "N1", "N2", "N3", "REM")
    stage_means = {}
    for m, column, scale in [("E2", "mean_f1_pct", 1), ("E5", "mean_f1", 100), ("E6", "mean_f1", 100)]:
        stage_path = Path(METHODS[m][1]).parent / "stage_summary.csv"
        rows = read_rows(stage_path)
        indexed = {(r["condition"], r["stage"]): float(r[column]) * scale for r in rows}
        if len(indexed) != len(rows) or set(indexed) != {(c,s) for c in ["Clean", *NOISY] for s in stage_names}:
            raise ValueError(f"Invalid stage summary: {stage_path}")
        for stage in stage_names:
            stage_means[stage, m] = statistics.mean(indexed[c, stage] for c in NOISY)
    write_table(destination / "stage_f1.csv", ["stage", "E2", "E5", "E6"],
                ([s, *[stage_means[s, m] for m in ("E2", "E5", "E6")]] for s in stage_names))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="New output directory")
    parser.add_argument("--data-dir", type=Path, help="Processed Sleep-EDF directory for the optional EEG figure")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    build_tables(out / "tables")
    env = dict(os.environ, SLEEP_FIGURE_BUILD=str(out))
    for plot in PLOTS:
        subprocess.run([sys.executable, HERE / "plots" / (plot + ".py")], env=env, check=True)
    if args.data_dir:
        subprocess.run([sys.executable, HERE / "plots/test_eeg_with_noise.py", "--data-dir", args.data_dir],
                       env=env, check=True)


if __name__ == "__main__":
    main()
