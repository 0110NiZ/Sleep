"""
scripts/run_e2_evaluation.py
E2 实验：AttnSleep Baseline 模型在各类型噪声干扰下的鲁棒性自动化批量评测。
- 支持 5 折测试集全覆盖
- 支持 6 种工况 (Clean + 3 档高斯白噪声 + 基线漂移 + 突发运动伪迹)
- 严格遵循团队统一随机种子: seed = 123
- 自动检测并适配 CPU / GPU，自动导出 CSV 实验报表
"""

import os
import sys
import glob
import json
import csv
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

# 引入项目根目录与噪声生成器
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "scripts"))
from noise_generator import apply_noise

# 1. 设备与环境配置 (自动适配本地 CPU 或队友 GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== 启动 E2 鲁棒性评测 (执行设备: {device}, 随机种子: seed=123) ===")

# 2. 定义 6 种测试工况
TEST_CONDITIONS = [
    {"name": "Clean (基线)", "type": "clean"},
    {"name": "Gaussian 20dB", "type": "gaussian", "snr_db": 20.0},
    {"name": "Gaussian 10dB", "type": "gaussian", "snr_db": 10.0},
    {"name": "Gaussian 5dB", "type": "gaussian", "snr_db": 5.0},
    {"name": "Baseline Wander", "type": "baseline_wander", "wander_ratio": 1.0},
    {"name": "Burst Artifact", "type": "burst", "amplitude_ratio": 3.5},
]

# 3. 读取划分配置
split_path = PROJECT_ROOT / "Sleep_Baseline_5fold" / "fivefold_subject_splits.json"
if not split_path.exists():
    split_path = PROJECT_ROOT / "configs" / "fivefold_subject_splits.json"

assert split_path.exists(), f"未找到划分文件: {split_path}"
with open(split_path, "r", encoding="utf-8") as f:
    splits_info = json.load(f)

# 解包可能的外壳
if isinstance(splits_info, dict):
    for wrapper in ["folds", "splits", "cv", "cross_validation", "data"]:
        matched_key = next((k for k in splits_info.keys() if str(k).lower().strip() == wrapper), None)
        if matched_key and isinstance(splits_info[matched_key], (list, dict)):
            splits_info = splits_info[matched_key]
            break

# 4. 定位脑电切片数据
data_dir = PROJECT_ROOT / "data" / "processed" / "sleep-edf-20"
if not data_dir.exists():
    data_dir = PROJECT_ROOT / "data" / "processed"

npz_files = glob.glob(str(data_dir / "*.npz"))
assert len(npz_files) >= 39, f"npz 数据数量不足 39 晚，检测到 {len(npz_files)} 个！"

# 引入模型定义
sys.path.append(str(PROJECT_ROOT))
from AttnSleep.model.model import AttnSleep

# 存储各折各工况的评测结果
results_table = []
detailed_metrics = {cond["name"]: {"acc": [], "mf1": [], "kappa": []} for cond in TEST_CONDITIONS}

# 5. 循环执行 5 折评测
for fold_idx in range(5):
    print(f"\n>>> 正在评测 Fold {fold_idx} ...")
    
    # 提取测试受试者名单
    fold_data = splits_info[fold_idx] if isinstance(splits_info, list) else splits_info.get(f"fold{fold_idx}") or splits_info.get(str(fold_idx))
    test_block = fold_data.get("test") if isinstance(fold_data, dict) else fold_data
    raw_items = test_block.get("subjects") or test_block.get("files") or [] if isinstance(test_block, dict) else test_block
    
    target_prefixes = []
    for item in raw_items:
        s = str(item).strip()
        sub_num = int(s[3:5]) if s.upper().startswith("SC4") else int(s.split(".")[0])
        target_prefixes.append(f"SC4{sub_num:02d}")
        
    fold_test_files = sorted([f for f in npz_files if any(os.path.basename(f).startswith(p) for p in target_prefixes)])
    
    # 内存预加载该折测试集 (避免重复读取磁盘，极大提速评测)
    x_list, y_list = [], []
    for fpath in fold_test_files:
        with np.load(fpath) as z:
            cur_x, cur_y = z["x"], z["y"]
            if cur_x.ndim == 3 and cur_x.shape[2] == 1:
                cur_x = np.transpose(cur_x, (0, 2, 1))
            x_list.append(cur_x)
            y_list.append(cur_y)
            
    clean_x = np.concatenate(x_list, axis=0)  # (N, 1, 3000)
    true_y = np.concatenate(y_list, axis=0)
    
    # 载入该折最优 Baseline 权重
    ckpt_path = PROJECT_ROOT / f"Sleep_Baseline_5fold/fold{fold_idx}/best_validation_macro_f1.pth"
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    
    model = AttnSleep().to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # 依次测试 6 种工况
    for cond in TEST_CONDITIONS:
        cond_name = cond["name"]
        noise_type = cond["type"]
        kwargs = {k: v for k, v in cond.items() if k not in ["name", "type"]}
        
        # 在线注入噪声 (严格固定 seed=123)
        if noise_type == "clean":
            input_x = clean_x
        else:
            input_x = apply_noise(clean_x, noise_type=noise_type, seed=123, **kwargs)

        # 批次前向推理
        preds = []
        batch_size = 128
        with torch.no_grad():
            for i in range(0, len(input_x), batch_size):
                bx = torch.tensor(input_x[i:i+batch_size], dtype=torch.float32, device=device)
                out = model(bx)
                pred = torch.argmax(out, dim=1).cpu().numpy()
                preds.extend(pred)
                
        preds = np.array(preds)
        acc = accuracy_score(true_y, preds) * 100
        mf1 = f1_score(true_y, preds, average="macro") * 100
        kappa = cohen_kappa_score(true_y, preds)
        
        detailed_metrics[cond_name]["acc"].append(acc)
        detailed_metrics[cond_name]["mf1"].append(mf1)
        detailed_metrics[cond_name]["kappa"].append(kappa)
        
        results_table.append({
            "Fold": fold_idx,
            "Condition": cond_name,
            "Accuracy": f"{acc:.2f}%",
            "Macro-F1": f"{mf1:.2f}%",
            "Kappa": f"{kappa:.4f}"
        })
        print(f"  [{cond_name:<20}] Acc: {acc:.2f}% | Macro-F1: {mf1:.2f}% | Kappa: {kappa:.4f}")

# 6. 计算五折汇总平均值与衰减量 (Delta)
print("\n" + "=" * 80)
print(f"{'E2 实验五折平均汇总表 (Baseline 鲁棒性)':^80}")
print("=" * 80)
print(f"{'Condition':<25} | {'Acc (%)':<10} | {'MF1 (%)':<10} | {'Kappa':<10} | {'Delta-Acc':<10} | {'Delta-MF1':<10}")
print("-" * 80)

clean_mean_acc = np.mean(detailed_metrics["Clean (基线)"]["acc"])
clean_mean_mf1 = np.mean(detailed_metrics["Clean (基线)"]["mf1"])

summary_rows = []
for cond in TEST_CONDITIONS:
    name = cond["name"]
    m_acc = np.mean(detailed_metrics[name]["acc"])
    m_mf1 = np.mean(detailed_metrics[name]["mf1"])
    m_kappa = np.mean(detailed_metrics[name]["kappa"])
    
    delta_acc = m_acc - clean_mean_acc
    delta_mf1 = m_mf1 - clean_mean_mf1
    
    d_acc_str = f"{delta_acc:+.2f}%" if name != "Clean (基线)" else "--"
    d_mf1_str = f"{delta_mf1:+.2f}%" if name != "Clean (基线)" else "--"
    
    print(f"{name:<25} | {m_acc:<10.2f} | {m_mf1:<10.2f} | {m_kappa:<10.4f} | {d_acc_str:<10} | {d_mf1_str:<10}")
    
    summary_rows.append({
        "Condition": name,
        "Mean_Accuracy": round(m_acc, 2),
        "Mean_Macro_F1": round(m_mf1, 2),
        "Mean_Kappa": round(m_kappa, 4),
        "Delta_Accuracy": round(delta_acc, 2) if name != "Clean (基线)" else 0.0,
        "Delta_Macro_F1": round(delta_mf1, 2) if name != "Clean (基线)" else 0.0
    })

print("=" * 80)

# 7. 导出结构化报表至 reports/ 目录
reports_dir = PROJECT_ROOT / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)
csv_path = reports_dir / "e2_robustness_results.csv"

with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
    writer.writeheader()
    writer.writerows(summary_rows)

print(f"\n✔ 实验汇总结果已成功保存为报表: {csv_path}")