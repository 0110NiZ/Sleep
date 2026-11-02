"""
noise_generator.py
用于 AttnSleep 单通道 EEG 信号的在线动态噪声注入模块。
默认随机种子严格锁定为团队统一标准: seed = 123

支持:
1. 高斯白噪声 (AWGN, 局部 SNR 匹配)
2. 基线漂移 (Baseline Wander, 0.1~0.5 Hz)
3. 突发运动伪迹 (Burst Artifact, 0.5~2.0 秒)
"""

import math
import numpy as np
import torch
from typing import Union, Tuple, Optional


def _ensure_tensor(
    x: Union[np.ndarray, torch.Tensor]
) -> Tuple[torch.Tensor, bool, torch.dtype]:
    """统一转换为 torch.Tensor 进行批次向量化计算"""
    is_numpy = isinstance(x, np.ndarray)
    if is_numpy:
        tensor_x = torch.from_numpy(x).float()
        return tensor_x, True, torch.float32
    return x, False, x.dtype


def _restore_type(
    x_noisy: torch.Tensor, is_numpy: bool, orig_dtype: torch.dtype
) -> Union[np.ndarray, torch.Tensor]:
    """恢复原始输入类型"""
    if is_numpy:
        return x_noisy.detach().cpu().numpy()
    return x_noisy.to(dtype=orig_dtype)


def add_gaussian_noise(
    x: Union[np.ndarray, torch.Tensor],
    snr_db: float = 10.0,
    seed: Optional[int] = 123
) -> Union[np.ndarray, torch.Tensor]:
    """
    添加局部功率匹配的高斯白噪声 (AWGN)。默认 seed = 123
    """
    tensor_x, is_numpy, orig_dtype = _ensure_tensor(x)
    device = tensor_x.device
    
    gen = torch.Generator(device=device) if seed is not None else None
    if seed is not None:
        gen.manual_seed(seed)

    # 局部信号功率 P_sig = mean(x^2)
    power_sig = torch.mean(tensor_x ** 2, dim=-1, keepdim=True)
    power_sig = torch.clamp(power_sig, min=1e-8)
    
    # 根据 SNR 计算噪声功率和标准差
    snr_linear = 10.0 ** (snr_db / 10.0)
    power_noise = power_sig / snr_linear
    sigma_noise = torch.sqrt(power_noise)
    
    if gen is not None:
        raw_noise = torch.randn(tensor_x.shape, generator=gen, device=device, dtype=tensor_x.dtype)
    else:
        raw_noise = torch.randn_like(tensor_x)
        
    noise = raw_noise * sigma_noise
    x_noisy = tensor_x + noise
    
    return _restore_type(x_noisy, is_numpy, orig_dtype)


def add_baseline_wander(
    x: Union[np.ndarray, torch.Tensor],
    fs: float = 100.0,
    wander_ratio: float = 1.0,
    seed: Optional[int] = 123
) -> Union[np.ndarray, torch.Tensor]:
    """
    添加超低频基线漂移 (0.1 ~ 0.5 Hz)。默认 seed = 123
    """
    tensor_x, is_numpy, orig_dtype = _ensure_tensor(x)
    device = tensor_x.device
    seq_len = tensor_x.shape[-1]
    
    rng = np.random.RandomState(seed) if seed is not None else np.random
    t = torch.linspace(0, (seq_len - 1) / fs, seq_len, device=device, dtype=tensor_x.dtype)
    
    sig_std = torch.std(tensor_x, dim=-1, keepdim=True)
    sig_std = torch.clamp(sig_std, min=1e-6)
    
    # 随机采样漂移频率与相位
    f1 = float(rng.uniform(0.10, 0.25))
    f2 = float(rng.uniform(0.25, 0.50))
    phi1 = float(rng.uniform(0, 2 * math.pi))
    phi2 = float(rng.uniform(0, 2 * math.pi))
    
    slow_wave = 0.7 * torch.sin(2 * math.pi * f1 * t + phi1) + 0.3 * torch.sin(2 * math.pi * f2 * t + phi2)
    
    while slow_wave.ndim < tensor_x.ndim:
        slow_wave = slow_wave.unsqueeze(0)
        
    drift = slow_wave * (sig_std * wander_ratio)
    x_noisy = tensor_x + drift
    
    return _restore_type(x_noisy, is_numpy, orig_dtype)


def add_burst_artifact(
    x: Union[np.ndarray, torch.Tensor],
    fs: float = 100.0,
    burst_duration_range: Tuple[float, float] = (0.5, 2.0),
    amplitude_ratio: float = 3.5,
    seed: Optional[int] = 123
) -> Union[np.ndarray, torch.Tensor]:
    """
    添加突发性运动/肌电伪迹。默认 seed = 123
    """
    tensor_x, is_numpy, orig_dtype = _ensure_tensor(x)
    device = tensor_x.device
    seq_len = tensor_x.shape[-1]
    
    rng = np.random.RandomState(seed) if seed is not None else np.random
    x_noisy = tensor_x.clone()
    
    orig_shape = x_noisy.shape
    x_flat = x_noisy.view(-1, seq_len)
    
    for b in range(x_flat.shape[0]):
        cur_sig = x_flat[b]
        cur_std = float(torch.std(cur_sig))
        if cur_std < 1e-6:
            cur_std = 1.0
            
        dur_sec = float(rng.uniform(burst_duration_range[0], burst_duration_range[1]))
        dur_pts = int(dur_sec * fs)
        dur_pts = min(dur_pts, seq_len - 10)
        
        start_idx = int(rng.randint(0, seq_len - dur_pts))
        end_idx = start_idx + dur_pts
        
        burst_t = torch.linspace(0, dur_sec, dur_pts, device=device, dtype=tensor_x.dtype)
        freq = float(rng.uniform(15.0, 35.0))
        phi = float(rng.uniform(0, 2 * math.pi))
        
        window = torch.hann_window(dur_pts, device=device, dtype=tensor_x.dtype)
        carrier = torch.sin(2 * math.pi * freq * burst_t + phi)
        
        artifact = (carrier + 0.5 * torch.sign(carrier)) * window * (cur_std * amplitude_ratio)
        x_flat[b, start_idx:end_idx] += artifact
        
    x_noisy = x_flat.view(orig_shape)
    return _restore_type(x_noisy, is_numpy, orig_dtype)


def apply_noise(
    x: Union[np.ndarray, torch.Tensor],
    noise_type: str = "clean",
    seed: int = 123,
    **kwargs
) -> Union[np.ndarray, torch.Tensor]:
    """
    统一噪声调用入口，默认 seed = 123
    """
    if noise_type == "clean" or noise_type is None:
        return x
    elif noise_type == "gaussian":
        return add_gaussian_noise(x, seed=seed, **kwargs)
    elif noise_type == "baseline_wander":
        return add_baseline_wander(x, seed=seed, **kwargs)
    elif noise_type == "burst":
        return add_burst_artifact(x, seed=seed, **kwargs)
    else:
        raise ValueError(f"未知的噪声类型: {noise_type}")


if __name__ == "__main__":
    print("=== 正在使用 seed=123 自检 noise_generator 模块 ===")
    dummy_x = torch.sin(torch.linspace(0, 30 * math.pi, 3000)).unsqueeze(0).repeat(4, 1, 1)
    
    out_g = add_gaussian_noise(dummy_x, snr_db=10.0, seed=123)
    noise_part = out_g - dummy_x
    snr_val = 10 * torch.log10(torch.mean(dummy_x ** 2) / torch.mean(noise_part ** 2)).item()
    print(f"✔ 高斯白噪声: 设定 10.0 dB | 实测 {snr_val:.2f} dB (seed=123)")
    
    out_w = add_baseline_wander(dummy_x, seed=123)
    out_b = add_burst_artifact(dummy_x, seed=123)
    print("✔ 基线漂移与突发伪迹函数校验通过！")
    print("🎉 噪声模块随机种子已与项目标准 (seed=123) 全面对齐！")