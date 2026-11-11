> 2026-11-10 整合更新：下文保留维护时的记录。E6 的500份验证明细已上传并核验；完整共享包的50份预测、5份最佳权重及597项文件哈希均已核验通过，536份对应GitHub文件逐字节一致。关键交接文件已补齐，无需重新索取或重跑。核验记录见 reports/integration_20261110/。本分支已整合E2–E6及维护修改；原始结果与实验参数保持不变。

# 实验交付整理说明（2026-11-10）

## 本次范围与结果来源

仅修复路径、交付入口和说明。不修改噪声参数、划分、架构、优化器、选模规则或归档结果；不重新训练、推理或生成噪声。不合并 main。

- E3 原始提交：`c2547fb`，维护分支：`maintenance/e3-portability`。
- E4–E6 交付原始提交：`a7b1b17`，维护分支：`maintenance/e5-e6-handover`。
- E5 正式结果的历史代码在 `3bf547d` 可取得。
- 归档 manifest 中的 code hash 对应当时的实验代码。维护修改必然改变部分入口文件的 hash，**不得改写旧 manifest 使其假装对应新代码**。审计/恢复已有运行必须使用原始提交；本维护版本不用于继续写入原运行目录。

## 路径与独立共享文件

在仓库根目录设置环境变量；不需要修改 Python 文件：

```bash
export SLEEP_DATA_DIR=/实际路径/processed/sleep-edf-20
export SLEEP_BASELINE_DIR=/实际路径/基线权重根目录
export SLEEP_RESULT_DIR=/实际路径/新的本地输出目录
```

`SLEEP_DATA_DIR` 下应有39个 Clean NPZ。基线目录支持 `fold0`–`fold4` 或 `fold0_seed123`–`fold4_seed123`，每折内有 `config.json` 和 `best_validation_macro_f1.pth`。同一折同时存在两套时明确报错，避免误选权重。E5 保留 `E5_BASELINE_ROOT` 作为可选覆盖。

E5 的 E2 复核可通过 `E2_REFERENCE_CSV` 指向 E3 交付的 `result/e2_v2/predictions_with_ids.csv`；未设置时读取历史提交 `c2547fb` 的同一文件，Git 历史缺失会明确报错。**本次无需运行复核或实验。**

`E5_HANDOVER_DIR` 可设置 E5 整理文件的输出位置，默认在本地结果目录下的 `e5_handover`，不再默认写入归档结果目录。原始 EDF 核验可设置 `SLEEP_RAW_DIR`。

## 统一分析入口

新增 `scripts/summarize_all_experiments.py`：仅读取正式逐折 CSV，包含 E2–E6；E3 必须提供验证选参后的 `selected_per_fold.csv`。所有输入使用明确的 `_pct` 列，不猜测0–1单位。输出目录必须不存在，拒绝覆盖既有文件。

```bash
python scripts/summarize_all_experiments.py   --e3 /E3仓库/result/20261104-050058-E3/comparison/selected_per_fold.csv   --out /新的分析输出目录
```

这是后续可用的汇总命令，本次未对真实结果运行。原 `summarize_e6.py` 保留为历史 E2/E4/E5/E6 比较入口；最终全组汇总使用新增入口。

## E6 请组员补交哪些文件

从**正式实验的原输出目录**打包，保持相对路径，不重新生成：

1. `fold0/validation/epoch001.json` 至 `epoch100.json`，其余四折相同，共500份。每份包含当轮十条件验证指标，用于独立重算选模分数。可提交 GitHub；若丢失请如实说明，不用测试结果或重新运行补造历史记录。
2. `fold0/predictions_*.npz`，每折十份、五折共50份，通过组内存储共享。需含 `sample_ids`、`y_true`、`y_pred`，用于复算指标、核对输入身份及后续配对分析。
3. `fold0/best.pth` 至 `fold4/best.pth`，共5份，通过组内存储共享。文件 SHA-256 应匹配各折 `selection.json` 的 `checkpoint_sha256`。
4. 原始运行命令及完整运行日志（若保留），注明代码提交、运行环境和输出目录。附本次共享文件的 SHA-256 清单，以检查传输完整性。

已有的 `manifest.json`、`all_selections_frozen.json`、各折 `history.json` / `selection.json` / `test_metrics.json` / `class_weights.json` / `training_data_audit.json`、测试及分阶段汇总保留即可。不要重复训练或更换原文件。`latest.pth` 仅需恢复训练时才有用，本轮分析不必传。已有同一份 Clean 数据无需重复共享。

## 分析和写作边界

- E5 的训练总时间含验证，E6 有独立训练耗时；现有硬件和计时范围未统一，不直接排效率名次。若以后确实需要统一计时，另行安排；本次不执行。
- E3 正式汇总按预定验证规则选参，但汇总记录晚于候选 Test 输出产生，不写成“所有候选测试前已保存冻结记录”。
- E3–E5 是去噪加冻结分类器，E6 是重新训练的分类器，结论对应完整策略比较。
- 五折均值与样本标准差作描述性分析，不能将重叠训练集的五折视作五次独立重复实验并直接声称显著性。
- E4/E5 Clean NPZ 已核对 x/y/fs/ch_label 一致，容器文件 hash 不同不单独构成重跑理由；对象类型 header 未在该核对中覆盖。应保存原核验清单及输出，不声称已定位容器差异原因。

## 合并注意事项

两个维护分支未彼此合并，也未合并 main。未来处理 E2 入口冲突时保留 `project_paths` 配置方式；两边共享的 `project_paths.py` 已统一。保留数据、权重及本地输出的忽略规则，保留所有历史结果、manifest 和原始提交。
