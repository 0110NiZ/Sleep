"""Compare final E6 results with recorded E2/E4/E5, never used for selection."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    result = {(int(r['fold']), r['condition']): r for r in rows}
    assert len(rows) == len(result) == 50, f'{path}: require 50 unique fold/condition rows'
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT / 'result/local/e6_noise_augmentation')
    args = ap.parse_args()
    methods = {
        'E2': ROOT / 'result/e4_wavelet/e2_reference_per_fold.csv',
        'E4': ROOT / 'result/e4_wavelet/test_per_fold.csv',
        'E5': ROOT / 'result/e5_dae/test_per_fold.csv',
        'E6': args.out / 'test_per_fold.csv'}
    tables = {m:load_csv(p) for m,p in methods.items()}
    conditions = [c['name'] for c in json.loads((ROOT / 'configs/noise_conditions_v2.json').read_text())['conditions']]
    assert all(set(t) == set(tables['E6']) for t in tables.values())
    lines = ['# E6 正式测试比较', '',
        '以下为五折均值 ± 样本标准差；单位为百分比，差值单位为百分点。', '',
        '| 方法 | 九噪声平均 Macro-F1 | Clean Macro-F1 |',
        '|---|---:|---:|']
    fold_values = {}
    for method, table in tables.items():
        noisy = [np.mean([float(table[f,c]['macro_f1_pct']) for c in conditions if c != 'Clean']) for f in range(5)]
        clean = [float(table[f,'Clean']['macro_f1_pct']) for f in range(5)]
        fold_values[method] = noisy
        lines.append(f'| {method} | {np.mean(noisy):.2f} ± {np.std(noisy,ddof=1):.2f} | {np.mean(clean):.2f} ± {np.std(clean,ddof=1):.2f} |')
    lines += ['', 'E3 尚未导入逐折结果，不能用总均值代替配对数据。', '',
        '| 条件 | E6 Macro-F1 | ΔE2 | ΔE4 | ΔE5 |', '|---|---:|---:|---:|---:|']
    delta_rows = []
    for c in conditions:
        values = [float(tables['E6'][f,c]['macro_f1_pct']) for f in range(5)]
        deltas = []
        for m in ['E2','E4','E5']:
            diff = [values[f]-float(tables[m][f,c]['macro_f1_pct']) for f in range(5)]
            deltas.append(np.mean(diff))
            for f,d in enumerate(diff):
                delta_rows.append(dict(reference=m, condition=c, fold=f, delta_macro_f1_pp=d))
        lines.append(f'| {c} | {np.mean(values):.2f} ± {np.std(values,ddof=1):.2f} | ' + ' | '.join(f'{d:+.2f}' for d in deltas) + ' |')
    lines += ['', '各折九噪声平均 Macro-F1 的差值：', '']
    for m in ['E2','E4','E5']:
        differences = np.array(fold_values['E6'])-fold_values[m]
        lines.append(f'- E6 − {m}: ' + ', '.join(f'{v:+.2f}' for v in differences) + ' pp')
    lines += ['', '限制：五折训练集重叠，五个差值不应视为五次独立重复实验；不据此直接宣称统计显著性。',
        '单个固定训练 seed；50% Clean 是预先固定的起始配方，未证明为最优。',
        'E6 不重建波形，RMSE 和 SNR improvement 不适用。推理耗时仅在相同硬件和计时口径下比较。', '']
    (args.out / 'E6_RESULT_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    with (args.out / 'paired_differences.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(delta_rows[0])); w.writeheader(); w.writerows(delta_rows)
    print(args.out / 'E6_RESULT_REPORT.md')


if __name__ == '__main__':
    main()
