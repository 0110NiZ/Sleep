P10 白底马卡龙图表：使用你自己的 E6 测试预测

1. 解压后，把 p10 文件夹放在 F:\sleep\Sleep 内。
   确认 F:\sleep\Sleep\p10\p10.py 存在。
2. CMD 逐条执行：
   cd /d F:\sleep\Sleep
   .venv\Scripts\activate.bat
   python -m pip install matplotlib==3.10.1
   python p10\p10.py
3. 打开结果：
   explorer F:\sleep\Sleep\p10\output

数据来源：
- E6必须读取你电脑 result/local/e6_noise_augmentation/fold0..fold4/
  中50份 predictions_*.npz 的 y_true、y_pred 和 sample_ids。
- E6各阶段F1由真实预测重新生成混淆矩阵后计算，不读取PPT数据，不用硬编码柱高。
- 若相应test_metrics.json存在，逐项核对重算的混淆矩阵、阶段F1和Macro-F1。
- E2/E5优先读你本地项目 result/e2_v2/e2_details.json 和
  result/e5_dae/fold*/test_metrics.json；若没有，使用包内的组内正式测试记录。
  E2/E5包内记录来源提交见 reference_provenance.json；它们不是E6的预测数据。
- 所有方法均不包含Clean：先在每折内等权平均9种噪声的阶段F1，再平均5折。
- Clean预测只用于一致性检查，不混入柱状图均值。
- 此包附带的预生成PNG实际使用你上传的E6_full_results.zip的50份预测重算。
  E6 N1重算后仍为38.43%，与原PPT相同是因为数据本身一致，不是照图填数。

输出：
- p10_stage_f1_white_macaron.png：白色背景、300dpi、柱顶两位小数。
- p10_stage_f1_white_macaron.svg：同款矢量图。
- p10_stage_f1_values.csv：15根柱子的准确数据和五折标准差。
- p10_stage_f1_per_condition.csv：逐折逐条件逐阶段结果，含E6 Clean核对数据。
- p10_data_check.json：实际输入位置、预测文件哈希、汇总口径和核验结果。

若不想显示柱顶数字：
   python p10\p10.py --no-labels

白底#FFFFFF；E2薰衣草紫#C7B7E2、E5薄荷绿#B7DDCA、E6浅粉#F1B8C5。
Windows优先使用Calibri，图中不加入PPT标题与装饰，只替换原页的图表部分。
图片只重绘既有实验指标，不重新训练或测试模型。
