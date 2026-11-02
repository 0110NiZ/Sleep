"""
scripts/visualize_noise.py
从真实预处理脑电切片中提取一段 30 秒信号，绘制加噪前后时域与频域对比图，
并保存至 reports/figures/ 供团队汇报使用。
"""

import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from noise_generator import add_gaussian_noise, add_baseline_wander, add_burst_artifact

# 1. 确保输出目录存在
output_dir = Path("reports/figures")
output_dir.mkdir(parents=True, exist_ok=True)

# 2. 读取一段真实的干净脑电片段 (Fold 0 首晚数据)
data_dir = Path("data/processed/sleep-edf-20")
npz_files = sorted(glob.glob(str(data_dir / "SC4001E0.npz")))
if not npz_files:
    npz_files = sorted(glob.glob(str(data_dir / "*.npz")))

assert len(npz_files) > 0, "未找到预处理数据，请确认 data/processed/ 目录！"

with np.load(npz_files[0]) as data:
    # 提取第 100 个 30 秒片段 (3000 点)
    idx = 100
    clean_signal = data["x"][idx].flatten()  # (3000,)
    label = data["y"][idx]

fs = 100.0
time = np.linspace(0, 30, len(clean_signal))

# 3. 生成三种核心噪声信号 (锁定 seed=123)
noisy_gauss_10db = add_gaussian_noise(clean_signal, snr_db=10.0, seed=123)
noisy_wander = add_baseline_wander(clean_signal, fs=fs, wander_ratio=1.2, seed=123)
noisy_burst = add_burst_artifact(clean_signal, fs=fs, amplitude_ratio=3.5, seed=123)

# 4. 绘图对比
plt.figure(figsize=(14, 10), dpi=150)

# 子图 1: 干净原始信号
plt.subplot(4, 1, 1)
plt.plot(time, clean_signal, color="#1f77b4", linewidth=1.0)
plt.title(f"Clean EEG Epoch (Sleep Stage: {label})", fontsize=11, fontweight="bold")
plt.ylabel("Amp (uV)")
plt.grid(True, linestyle="--", alpha=0.5)

# 子图 2: 高斯白噪声 (SNR = 10 dB)
plt.subplot(4, 1, 2)
plt.plot(time, noisy_gauss_10db, color="#ff7f0e", linewidth=1.0)
plt.title("Gaussian White Noise (SNR = 10 dB, seed=123)", fontsize=11, fontweight="bold")
plt.ylabel("Amp (uV)")
plt.grid(True, linestyle="--", alpha=0.5)

# 子图 3: 基线漂移 (0.1 ~ 0.5 Hz)
plt.subplot(4, 1, 3)
plt.plot(time, noisy_wander, color="#2ca02c", linewidth=1.0)
plt.title("Baseline Wander (0.1~0.5 Hz Slow Drift, seed=123)", fontsize=11, fontweight="bold")
plt.ylabel("Amp (uV)")
plt.grid(True, linestyle="--", alpha=0.5)

# 子图 4: 突发运动/肌电伪迹
plt.subplot(4, 1, 4)
plt.plot(time, noisy_burst, color="#d62728", linewidth=1.0)
plt.title("Burst / Motion Artifact (Transient High-Amp Spike, seed=123)", fontsize=11, fontweight="bold")
plt.xlabel("Time (seconds)")
plt.ylabel("Amp (uV)")
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
save_path = output_dir / "eeg_noise_comparison.png"
plt.savefig(save_path)
print(f"✔ 噪声波形对比图已成功生成并保存至: {save_path}")