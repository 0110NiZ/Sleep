# E2 实验交接文档：噪声注入模块与 Baseline 鲁棒性评估

- **代码分支**：`feature/noise-generator`
- **随机种子**：全局锁定 `seed = 123`（严格与团队规范对齐）
- **核心模块**：`scripts/noise_generator.py`
- **评测报告**：`reports/e2_robustness_results.csv`
- **波形对比**：`reports/figures/eeg_noise_comparison.png`

---

## 1. 噪声模块调用说明 (供后续训练增强 E3~E6 使用)

模块原生兼容 **PyTorch Tensor** 与 **NumPy ndarray**，无论在 CPU 还是 GPU 设备上均能自动适配，无需修改设备迁移逻辑。

### 常用调用方式：

```python
from scripts.noise_generator import apply_noise, add_gaussian_noise

# 方式 1: 在 PyTorch Dataset / DataLoader 中动态调用
# 输入 x 形状支持 (Batch, 1, 3000) 或 (Batch, 3000)
x_noisy = apply_noise(x, noise_type="gaussian", snr_db=10.0, seed=123)

# 方式 2: 支持的全部噪声类型
# 1) 高斯白噪声 (局部功率自适应匹配 SNR)
x_noisy = apply_noise(x, noise_type="gaussian", snr_db=10.0)

# 2) 低频基线漂移 (0.1~0.5 Hz 呼吸与出汗模拟)
x_noisy = apply_noise(x, noise_type="baseline_wander", wander_ratio=1.0)

# 3) 突发性高幅运动伪迹 (0.5~2.0 秒冲击)
x_noisy = apply_noise(x, noise_type="burst", amplitude_ratio=3.5)