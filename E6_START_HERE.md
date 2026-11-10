# E6：从准备文件到正式结果

## 当前状态

本包基于 E5-dae 分支提交 `3bf547d7fd4862de32d2c318ab9a8468ce6a1ff8`。
新增 E6 训练、评估、结果比较脚本。原 AttnSleep、Noise v2、五折划分均未修改。
正式 E6 指标已归档于 result/e6_noise_augmentation；原始 EEG 和权重仍通过组内存储共享。逐轮验证明细及逐样本预测需补充交接，见 HANDOVER_MAINTENANCE.md。以下训练说明保留供复现参考，不代表需要重新训练。

## 1. 你到底要做什么

E6 直接训练能识别带噪信号的 AttnSleep。每折随机初始化，训练输入随机保留 Clean 或加一种噪声，目标仍是 W/N1/N2/N3/REM 五类标签。它不需要 DAE，也不需要 E5 输出波形。

本次预先固定一个实验配方：50% Clean；另外 50% 在九种噪声条件中等概率抽取。每轮重新抽类型和噪声实例，不叠加不同噪声。该比例是起始设计，不能写成“验证选出的最佳比例”。如小组希望比较其他比例，应在任何 E6 正式测试前预定候选集合，另开输出目录完整训练，并仅凭验证集选择；这会增加训练预算。

每折固定 100 epochs、batch size 128、Adam(lr=0.001, weight_decay=0.001, amsgrad=True)，第10轮结束后 lr=0.0001。无额外归一化，保留微伏。训练类权重只读取本折训练标签，使用 E1 的原公式。每折训练 seed 与 E1 一样为123；增强抽样和训练噪声使用独立 namespace。

每轮用完整验证集的 Clean + 9噪声分别算 Macro-F1，十项等权平均选 best；同分保留更早 epoch。验证与测试严格复用仓库实际 Noise v2 的 seed=123、`filename:epoch_index` 样本 ID 和原文件顺序。两者样本 ID 不同，训练噪声种子另设。正式测试前冻结全部五折。

## 2. 先向组长拿数据

必需：**39个与 E5 使用相同的预处理 Clean NPZ**。总数应为42,308个30秒片段，100 Hz，Fpz-Cz，`x=(N,3000,1)`、`y=(N,)`，五类标签0..4。

可以复制这段给组长：

> 我开始做E6，需要你E5实际使用的39个预处理Clean NPZ（总共42,308片段），请保持原文件名和内容，并提供你训练环境的Python/PyTorch版本。E6从随机初始化训练，不需要DAE权重或去噪波形。为核对共同评估环境，后续可能需要五折E1权重和E2的逐样本预测；它们不用于E6训练。

不要直接把自己重新处理的另一套NPZ替换进来。默认会检查 E5 manifest 的文件哈希。NPZ重新压缩可能改变文件哈希，即使数组相同；此时先验证原始数组逐元素相同，再考虑 `--allow-repacked-data`。该选项仅放宽容器哈希，形状和标签计数相同本身不能证明波形相同。

## 3. 下载与放置

压缩包解压后包含 `Sleep/` 文件夹。把39个NPZ放入：

`Sleep/data/processed/sleep-edf-20/`

或者保留在其他磁盘，运行时加 `--data-dir`。

Windows PowerShell示例，假设你将Sleep文件夹放在F盘：

```powershell
cd F:\Sleep
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

应在组内已有的、可运行 AttnSleep 的环境执行；不要覆盖现有环境。如果没有环境，先安装Python 3.10+并创建专用虚拟环境，优先复用小组的Python/PyTorch版本及依赖。仓库 requirements 文件记录了原环境；requirements-lock.txt 是Ubuntu CUDA环境，不能直接当作Windows通用安装清单。

缺少普通依赖时在已激活环境中安装：

```powershell
python -m pip install numpy==1.26.4 scipy==1.14.1 scikit-learn==1.6.1 pandas==2.2.3
```

PyTorch须是与你GPU驱动兼容的版本，建议对齐小组2.6.0环境。若上面的import或CUDA检查失败，把完整输出给我，再按你的实际环境配置。正式训练建议使用GPU，CPU仅适合检查和小规模试跑。

## 4. 先检查路径和协议

在Sleep根目录运行：

```powershell
python scripts/run_e6_noise_augmentation.py --stage check
```

如果数据在 `F:\SleepData\clean`：

```powershell
python scripts/run_e6_noise_augmentation.py --stage check --data-dir "F:\SleepData\clean"
```

check会核验共享代码/划分/配置，并列出缺失文件，不读取测试NPZ内容。正式train/smoke在每折加载train/validation时才逐文件核验哈希、形状和标签计数。需要看到missing_files为空。若CUDA=False，先解决GPU环境，不要直接开始完整训练。

后续每个命令都要使用相同的 `--data-dir`，或把数据放入默认目录。

## 5. 小规模试跑

```powershell
python scripts/run_e6_noise_augmentation.py --stage smoke --device cuda
```

只用fold0的少量train/validation数据跑1轮，覆盖十种验证条件，不做正式测试。结果单独保存在 `result/local/e6_smoke/`，不会进入正式结果。

成功末尾显示 `SMOKE complete`。重复smoke会跳过已有完成结果；如需重新试跑，使用新的输出目录，例如 `--out result/local/e6_smoke_02`。

## 6. 正式五折训练

```powershell
python -u scripts/run_e6_noise_augmentation.py --stage train --device cuda
```

输出：`result/local/e6_noise_augmentation/`。

每折文件：

- `latest.pth`：当前完成轮次的完整状态，含优化器和PyTorch随机数状态；体积较大，内含best快照用于中断恢复。
- `best.pth`：该折十条件验证平均Macro-F1最好的模型。
- `history.json`：逐轮loss、准确率、验证分数、耗时、增强条件计数。
- `validation/epochNNN.json`：逐轮十种完整验证指标。
- `selection.json`：训练完成后的冻结选模记录。
- `training_data_audit.json`、`class_weights.json`：数据哈希和仅来自train的类权重。

想先做fold0：

```powershell
python -u scripts/run_e6_noise_augmentation.py --stage train --fold 0 --device cuda
```

之后使用不带 `--fold` 的相同命令，会跳过完成折并继续其他折。不要同时运行同一输出目录的两个进程。

程序中断后，重新执行同一命令、保持环境和代码不变。可从上一轮完整完成的位置继续；未完成的那一轮会重新计算。训练中不依赖网络，模型和数据下载完整后，网络断开不影响本地训练。电脑关机或休眠会暂停/终止训练，因此请接电并关闭自动休眠。

## 7. 五折选模全部完成后正式Test

```powershell
python scripts/run_e6_noise_augmentation.py --stage test --device cuda
```

这里 `--device cuda` 用于匹配已记录的训练环境；正式测试按照配置在CPU进行，与E2/E5使用相同batch size和推理设备。它不加载E1或DAE权重。

test会检查五折都完成100轮、best哈希匹配、数据未改变，然后对每折评估Clean+9噪声。首次对比建议在小组同一CPU硬件上运行，以便耗时可比。当前程序会校验训练环境版本，换电脑或版本需先重新核验复现，不要删除校验来强行继续。

应产生：

- 50份预测NPZ，含sample_ids、y_true、y_pred。
- 总共423,080条预测记录。
- `test_per_fold.csv`、`test_summary.csv`、`stage_per_fold.csv`。
- 每折`test_metrics.json`：五类指标、混淆矩阵、Kappa。
- `pooled_details.json`：合并预测后的指标，与五折平均分开报告。
- `overall.json`：每折九噪声均值、五折均值与样本标准差、Clean性能。
- `all_selections_frozen.json`、`manifest.json`：选模与运行追溯信息。

## 8. 生成与E2/E4/E5的比较

```powershell
python scripts/summarize_e6.py
```

生成 `E6_RESULT_REPORT.md` 和 `paired_differences.csv`，包括每条件E6−E2/E4/E5的差值和逐折对比。

E3的逐折结果不在本分支当前结果目录中，比较脚本会明确标注缺失。需要组员提供匹配的50行逐折指标后补充，不能仅用55.98%构造逐折分析。

主要分析：九噪声平均Macro-F1、Clean变化、哪类噪声获益、强高斯是否仍不稳定、N1等各阶段变化、逐折方向是否一致、计算成本。50%配方未证明最优；固定一个训练seed；五折并非五次独立重复实验。不得仅因均值上升就宣称统计显著性。

E6不重建波形，所以RMSE/SNR improvement不适用。耗时以CPU模型前向计时，排除生成噪声、传输和warm-up；仅在相同硬件及口径下与已有结果比较。

## 9. 交付组长

提交源代码、配置、运行说明、环境记录、日志、50行逐折指标、十条件汇总、阶段指标、混淆矩阵、五折权重和50份预测。

建议代码使用独立 `E6-noise-augmentation` 分支，由组员review后合并；本包没有替你推送GitHub或修改远端分支。

`result/local/`、NPZ和PTH被原仓库.gitignore忽略。权重和预测通过组内共享存储交接；把体积小的正式结果复制到 `result/e6_noise_augmentation/` 后再提交，切勿把smoke或合成数据结果当成正式结果。
