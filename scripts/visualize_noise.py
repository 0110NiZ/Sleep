
"""
Visualize E2 Noise v2 on a real 30-second EEG epoch.

Four waveforms per figure:
1. Clean EEG
2. Gaussian noise
3. Baseline drift
4. Burst artifact

Uses the same noise configuration, sample IDs, and seed
as run_e2_evaluation.py.

The original visualize_noise.py is not modified.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from noise_conditions import apply_noise, apply_noise_v2
from project_paths import E2_DIR


# ============================================================
# 1. Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(
    "/mnt/d/SleepProject/data/processed/sleep-edf-20"
)

CONFIG_FILE = (
    ROOT / "configs/noise_conditions_v2.json"
)

OUTPUT_DIR = (
    E2_DIR / "figures"
)

STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]


# ============================================================
# 2. Load one real EEG epoch
# ============================================================

def load_epoch(filename, epoch_index):

    file_path = DATA_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"EEG file not found: {file_path}"
        )

    with np.load(
        file_path,
        allow_pickle=False,
    ) as data:

        x = data["x"]
        y = data["y"]

        if not 0 <= epoch_index < len(y):
            raise ValueError(
                f"Epoch index must be between "
                f"0 and {len(y) - 1}"
            )

        clean_signal = np.asarray(
            x[epoch_index, :, 0],
            dtype=np.float32,
        ).copy()

        stage = int(y[epoch_index])

    assert clean_signal.shape == (3000,)
    assert np.isfinite(clean_signal).all()
    assert stage in range(5)

    # Must match the E2 evaluation sample ID format
    sample_id = f"{filename}:{epoch_index}"

    return clean_signal, stage, sample_id


# ============================================================
# 3. Generate Noise v2
# ============================================================

def generate_noise(
    clean_signal,
    condition,
    sample_id,
    seed,
    fs,
):

    # Match E2 input shape: (N, 1, 3000)
    input_x = clean_signal.reshape(1, 1, -1)

    noisy_x = apply_noise(
        input_x,
        condition,
        sample_ids=[sample_id],
        seed=seed,
        fs=fs,
    )

    noisy_signal = np.asarray(
        noisy_x,
        dtype=np.float32,
    )[0, 0, :]

    assert noisy_signal.shape == clean_signal.shape
    assert np.isfinite(noisy_signal).all()

    return noisy_signal


# ============================================================
# 4. Plot one severity level
# ============================================================

def plot_level(
    level,
    clean_signal,
    stage,
    sample_id,
    condition_map,
    seed,
    fs,
):

    level_name = level.capitalize()

    # Get finalized E2-v2 conditions
    gaussian = condition_map[
        f"Gaussian_{level_name}"
    ]

    drift = condition_map[
        f"Drift_{level_name}"
    ]

    burst = condition_map[
        f"Burst_{level_name}"
    ]

    # Generate the same noise instances as E2
    gaussian_signal = generate_noise(
        clean_signal, gaussian, sample_id, seed, fs
    )

    drift_signal = generate_noise(
        clean_signal, drift, sample_id, seed, fs
    )

    burst_signal = generate_noise(
        clean_signal, burst, sample_id, seed, fs
    )

    # Correct sample time: 0 to 29.99 seconds
    time = np.arange(len(clean_signal)) / fs

    # --------------------------------------------------------
    # Four vertically stacked EEG waveforms
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(14, 10),
        dpi=150,
        sharex=True,
    )

    stage_name = STAGE_NAMES[stage]

    fig.suptitle(
        f"E2 Noise v2 - {level_name} Severity\n"
        f"EEG Epoch: {sample_id} | "
        f"Sleep Stage: {stage_name} | Seed: {seed}",
        fontsize=14,
        fontweight="bold",
    )

    waveforms = [
        (
            clean_signal,
            f"Clean EEG (Stage: {stage_name})",
            "#1f77b4",
        ),
        (
            gaussian_signal,
            f"Gaussian White Noise "
            f"(SNR = {gaussian['snr_db']:g} dB)",
            "#ff7f0e",
        ),
        (
            drift_signal,
            f"Baseline Drift "
            f"({drift['frequency_hz']:g} Hz, "
            f"Amplitude = {drift['amplitude_uv']:g} uV)",
            "#2ca02c",
        ),
        (
            burst_signal,
            f"Burst Artifact "
            f"({burst['amplitude_ratio']:g} sigma, "
            f"15-35 Hz, 0.5-2.0 s)",
            "#d62728",
        ),
    ]

    for ax, (signal, title, color) in zip(
        axes, waveforms
    ):

        ax.plot(
            time,
            signal,
            color=color,
            linewidth=0.9,
        )

        ax.set_title(
            title,
            fontsize=11,
            fontweight="bold",
        )

        ax.set_ylabel(
            "Amplitude (uV)",
            fontsize=10,
        )

        ax.grid(
            True,
            linestyle="--",
            alpha=0.4,
        )

        ax.set_xlim(0, 30)

    axes[-1].set_xlabel(
        "Time (seconds)",
        fontsize=11,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.94]
    )

    # --------------------------------------------------------
    # Save figure
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR /
        f"eeg_noise_comparison_{level}.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"[SAVED] {output_path}",
        flush=True,
    )


# ============================================================
# 5. Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--level",
        choices=["low", "medium", "high", "all"],
        default="all",
    )

    parser.add_argument(
        "--file",
        default="SC4001E0.npz",
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    # Load the exact E2 noise configuration
    config = json.loads(
        CONFIG_FILE.read_text(
            encoding="utf-8"
        )
    )

    seed = int(config["seed"])
    fs = float(config["sampling_rate"])

    assert config["signal_unit"] == "uV"

    condition_map = {
        condition["name"]: condition
        for condition in config["conditions"]
    }

    assert len(condition_map) == 10

    clean_signal, stage, sample_id = load_epoch(
        args.file,
        args.epoch,
    )

    print("========== E2 Noise v2 Visualization ==========")
    print(f"EEG sample: {sample_id}")
    print(f"Stage: {STAGE_NAMES[stage]}")
    print(f"Sampling rate: {fs:g} Hz")
    print(f"Seed: {seed}")

    if args.level == "all":
        levels = ["low", "medium", "high"]
    else:
        levels = [args.level]

    for level in levels:

        plot_level(
            level=level,
            clean_signal=clean_signal,
            stage=stage,
            sample_id=sample_id,
            condition_map=condition_map,
            seed=seed,
            fs=fs,
        )

    print("\nVisualization completed.")


if __name__ == "__main__":
    main()
