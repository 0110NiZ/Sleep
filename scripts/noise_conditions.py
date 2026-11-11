
"""
Noise Generator v2
Sleep-EDF-20 / AttnSleep Robustness Evaluation

Supported noise:
1. Gaussian White Noise
2. Baseline Drift
3. Oscillatory Burst Artifact

Configurations:
    configs/noise_conditions_v2.json

Requirements:
    - EEG sampling rate: 100 Hz
    - EEG input unit: microvolts (uV)
    - NumPy required
    - PyTorch optional

Features:
    - Supports NumPy and PyTorch
    - Supports (N, 3000), (N, 1, 3000), (N, 3000, 1)
    - Reproducible noise generation
    - Stable sample IDs
    - Same noise instances for E2-E6
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from collections.abc import Sequence

import numpy as np


# ============================================================
# 1. Reproducible Random Seed
# ============================================================

def _seed_for(
    seed: int,
    noise_type: str,
    sample_id: str
) -> int:

    raw = f"{seed}|{noise_type}|{sample_id}".encode("utf-8")

    return int.from_bytes(
        hashlib.blake2b(raw, digest_size=8).digest(),
        "little"
    )


# ============================================================
# 2. EEG Shape Conversion
# ============================================================

def _as_nt(x: np.ndarray):

    if x.ndim == 1:
        return x[None, :], "t"

    if x.ndim == 2:
        return x, "nt"

    if x.ndim == 3 and x.shape[1] == 1:
        return x[:, 0, :], "n1t"

    if x.ndim == 3 and x.shape[2] == 1:
        return x[:, :, 0], "nt1"

    raise ValueError(
        "Expected EEG shape: "
        "(T,), (N,T), (N,1,T), or (N,T,1)"
    )


def _back_to_shape(x_nt, layout):

    if layout == "t":
        return x_nt[0]

    if layout == "nt":
        return x_nt

    if layout == "n1t":
        return x_nt[:, None, :]

    return x_nt[:, :, None]


# ============================================================
# 3. Main Noise Generator
# ============================================================

def apply_noise_v2(
    x,
    condition: dict,
    *,
    sample_ids: Sequence[str] | None = None,
    seed: int = 123,
    fs: float = 100.0
):
    """
    Apply one noise condition to EEG data.

    Parameters
    ----------
    x:
        EEG data (NumPy array or PyTorch Tensor).
        Input values must be in microvolts.

    condition:
        A condition dictionary from noise_conditions_v2.json.

    sample_ids:
        Unique, stable identifiers for each EEG epoch.
        Required for noisy conditions.

    seed:
        Random seed, default 123.

    fs:
        Sampling frequency, default 100 Hz.

    Returns
    -------
    EEG with noise added.
    Preserves the original shape and data type.
    """

    # --------------------------------------------------------
    # Detect NumPy / PyTorch input
    # --------------------------------------------------------

    is_torch = False
    torch = None

    try:
        import torch as _torch

        torch = _torch
        is_torch = isinstance(x, torch.Tensor)

    except ImportError:
        pass

    if is_torch:

        if not x.is_floating_point():
            raise TypeError(
                "Torch EEG input must have floating dtype"
            )

        x_np = x.detach().cpu().numpy()

    else:

        x_np = np.asarray(x)

        if not np.issubdtype(x_np.dtype, np.floating):
            raise TypeError(
                "EEG input must use float32 or float64"
            )

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    x_nt, layout = _as_nt(x_np)

    n, t_len = x_nt.shape

    if fs <= 0 or t_len < 3:
        raise ValueError("Invalid sampling rate or EEG length")

    if not np.isfinite(x_nt).all():
        raise ValueError("EEG contains NaN or infinity")

    noise_type = condition.get("noise_type")

    # Clean condition: no added noise
    if noise_type == "clean":
        return x.clone() if is_torch else x_np.copy()

    if noise_type not in {
        "gaussian",
        "baseline_wander",
        "burst"
    }:
        raise ValueError(
            f"Unknown noise type: {noise_type}"
        )

    if sample_ids is None:
        raise ValueError(
            "Noisy conditions require stable sample_ids"
        )

    if len(sample_ids) != n:
        raise ValueError(
            "Number of sample_ids must match batch size"
        )

    if len(set(map(str, sample_ids))) != n:
        raise ValueError(
            "sample_ids must be unique within the batch"
        )

    out = x_nt.copy()

    time = np.arange(
        t_len, dtype=np.float64
    ) / fs

    # --------------------------------------------------------
    # Process each EEG epoch
    # --------------------------------------------------------

    for i, sid in enumerate(sample_ids):

        rng = np.random.default_rng(
            _seed_for(
                seed,
                noise_type,
                str(sid)
            )
        )

        signal = x_nt[i].astype(
            np.float64,
            copy=False
        )

        # ====================================================
        # Gaussian White Noise
        # ====================================================

        if noise_type == "gaussian":

            snr_db = float(condition["snr_db"])

            white = rng.standard_normal(t_len)

            white -= white.mean()

            power_signal = np.mean(signal ** 2)

            power_white = np.mean(white ** 2)

            if power_signal == 0:

                artifact = np.zeros(
                    t_len,
                    dtype=np.float64
                )

            else:

                target_noise_power = (
                    power_signal /
                    (10.0 ** (snr_db / 10.0))
                )

                artifact = white * np.sqrt(
                    target_noise_power / power_white
                )

        # ====================================================
        # Baseline Drift
        # ====================================================

        elif noise_type == "baseline_wander":

            frequency = float(
                condition["frequency_hz"]
            )

            amplitude = float(
                condition["amplitude_uv"]
            )

            if not (0 < frequency < fs / 2):
                raise ValueError(
                    "Invalid baseline drift frequency"
                )

            if amplitude < 0:
                raise ValueError(
                    "Invalid baseline drift amplitude"
                )

            phase = rng.uniform(
                0.0,
                2.0 * np.pi
            )

            artifact = amplitude * np.sin(
                2.0 * np.pi * frequency * time + phase
            )

        # ====================================================
        # Oscillatory Burst Artifact
        # ====================================================

        elif noise_type == "burst":

            f_low, f_high = map(
                float,
                condition["frequency_range_hz"]
            )

            d_low, d_high = map(
                float,
                condition["duration_range_s"]
            )

            ratio = float(
                condition["amplitude_ratio"]
            )

            count = int(
                condition["burst_count"]
            )

            if not (0 < f_low <= f_high < fs / 2):
                raise ValueError(
                    "Burst frequency exceeds Nyquist limit"
                )

            if not (0 < d_low <= d_high):
                raise ValueError(
                    "Invalid burst duration"
                )

            if ratio < 0 or count < 1:
                raise ValueError(
                    "Invalid burst parameters"
                )

            artifact = np.zeros(
                t_len,
                dtype=np.float64
            )

            signal_std = np.std(signal)

            amplitude = signal_std * ratio

            for _ in range(count):

                # Random duration
                duration = rng.uniform(
                    d_low,
                    d_high
                )

                points = max(
                    3,
                    int(round(duration * fs))
                )

                if points > t_len:
                    raise ValueError(
                        "Burst is longer than EEG epoch"
                    )

                # Random position
                start = int(
                    rng.integers(
                        0,
                        t_len - points + 1
                    )
                )

                # Random frequency
                frequency = rng.uniform(
                    f_low,
                    f_high
                )

                # Random phase
                phase = rng.uniform(
                    0,
                    2 * np.pi
                )

                burst_time = (
                    np.arange(points) / fs
                )

                # Oscillatory component
                carrier = np.sin(
                    2 * np.pi * frequency * burst_time
                    + phase
                )

                # Hann window
                window = np.hanning(points)

                # Preserve E2 burst waveform design
                burst_wave = (
                    carrier
                    + 0.5 * np.sign(carrier)
                ) * window

                artifact[
                    start:start + points
                ] += amplitude * burst_wave

        # ====================================================
        # Add noise to original EEG
        # ====================================================

        out[i] = signal + artifact

    # --------------------------------------------------------
    # Restore original shape and type
    # --------------------------------------------------------

    reshaped = _back_to_shape(
        out,
        layout
    )

    if is_torch:

        noise_delta = torch.as_tensor(
            reshaped - x_np,
            dtype=x.dtype,
            device=x.device
        )

        return x + noise_delta

    return reshaped


# ============================================================
# 4. Self-Test
# ============================================================

if __name__ == "__main__":

    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "noise_conditions_v2.json"
    )

    config = json.loads(
        config_path.read_text(
            encoding="utf-8"
        )
    )

    fs = config["sampling_rate"]

    # Simulated 30-second EEG
    t = np.arange(3000) / fs

    eeg = (
        25 * np.sin(
            2 * np.pi * 8 * t
        )
    ).astype(np.float32)

    eeg = eeg[None, None, :]

    sample_ids = ["selftest:0"]

    print("=" * 50)
    print("Noise Generator v2 Self-Test")
    print("=" * 50)

    for condition in config["conditions"]:

        y = apply_noise_v2(
            eeg,
            condition,
            sample_ids=sample_ids,
            seed=config["seed"],
            fs=fs
        )

        # Shape check
        assert y.shape == eeg.shape

        # Numerical check
        assert np.isfinite(y).all()

        # Determinism check
        y_repeat = apply_noise_v2(
            eeg,
            condition,
            sample_ids=sample_ids,
            seed=config["seed"],
            fs=fs
        )

        assert np.array_equal(
            y,
            y_repeat
        )

        # Gaussian SNR verification
        if condition["noise_type"] == "gaussian":

            error = (
                y.astype(np.float64)
                - eeg.astype(np.float64)
            )

            power_signal = np.mean(
                eeg.astype(np.float64) ** 2
            )

            power_noise = np.mean(
                error ** 2
            )

            measured_snr = (
                10 * np.log10(
                    power_signal / power_noise
                )
            )

            expected_snr = condition["snr_db"]

            assert abs(
                measured_snr - expected_snr
            ) < 0.02

            print(
                f"{condition['name']}: "
                f"SNR={measured_snr:.2f} dB"
            )

        else:

            print(
                f"{condition['name']}: "
                "shape OK, deterministic OK"
            )

    print("=" * 50)
    print(
        "Noise v2 self-test PASSED "
        "(10 conditions)"
    )

# Public version-neutral name; protocol remains Noise v2.
apply_noise = apply_noise_v2
