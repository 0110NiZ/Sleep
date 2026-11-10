"""Independent post-run verification and handoff; never trains or selects a model."""
import csv,hashlib,json,shutil
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,cohen_kappa_score,precision_recall_fscore_support
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from project_paths import ROOT, RESULT_ROOT, DATA_DIR, configured
OUT=RESULT_ROOT/'e5_dae';HAND=configured('E5_HANDOVER_DIR', RESULT_ROOT/'e5_handover')
HAND.mkdir(parents=True, exist_ok=True)
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def readcsv(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def save(p,o):p.write_text(json.dumps(o,indent=2,ensure_ascii=False),encoding='utf-8')
def near(a,b):assert np.isclose(float(a),float(b),rtol=1e-10,atol=1e-10),(a,b)
assert load(OUT/'pipeline_status.json')['stage']=='complete'
splits=load(ROOT/'configs/fivefold_subject_splits.json');sels=load(OUT/'all_selections_frozen.json')
conditions=[c['name'] for c in load(ROOT/'configs/noise_conditions_v2.json')['conditions']]
rows=readcsv(OUT/'test_per_fold.csv');assert len(rows)==50
lookup={(int(r['fold']),r['condition']):r for r in rows};assert len(lookup)==50
summary=readcsv(OUT/'test_summary.csv');pooled=load(OUT/'pooled_details.json')
e2={r['condition']:r for r in readcsv(ROOT/'result/e4_wavelet/e2_e4_comparison.csv')}
allpred={n:[[],[]] for n in conditions};total=0;foldrows=[]
stage_rows={(r['condition'],r['stage']):r for r in readcsv(OUT/'stage_summary.csv')};stage_values={}
for f in range(5):
    d=OUT/f'fold{f}';sel=sels[f];hist=load(d/'history.json')
    best=max(hist,key=lambda r:r['validation_macro_f1']);assert best['epoch']==sel['selected_epoch']
    near(best['validation_macro_f1'],sel['validation_macro_f1'])
    assert len(hist)==sel['epochs_run']
    assert hashlib.sha256((d/'best.pth').read_bytes()).hexdigest()==sel['checkpoint_sha256']
    for h in hist:
        vals=load(d/f"validation/epoch{h['epoch']:03d}.json")
        near(np.mean([v['macro_f1'] for v in vals.values()]),h['validation_macro_f1'])
    ids=[];labels=[]
    for name in splits['folds'][f]['test']['files']:
        with np.load(DATA_DIR/name) as z:
            labels.extend(z['y'].tolist());ids.extend(f'{name}:{i}' for i in range(len(z['y'])))
    metrics=load(d/'test_metrics.json')
    for n in conditions:
        with np.load(d/f'predictions_{n}.npz',allow_pickle=False) as z:
            y=z['y_true'];pr=z['y_pred'];assert z['sample_ids'].tolist()==ids;assert y.tolist()==labels
        m=metrics[n];r=lookup[f,n];total+=len(y)
        cm=confusion_matrix(y,pr,labels=range(5));assert cm.tolist()==m['confusion_matrix']
        near(accuracy_score(y,pr),m['accuracy']);near(f1_score(y,pr,average='macro',labels=range(5),zero_division=0),m['macro_f1'])
        near(cohen_kappa_score(y,pr),m['kappa']);near(m['macro_f1']*100,r['macro_f1_pct']);near(m['accuracy']*100,r['accuracy_pct'])
        pp,rr,ff,ss=precision_recall_fscore_support(y,pr,labels=range(5),zero_division=0)
        for i,stage in enumerate(['W','N1','N2','N3','REM']):
            for key,value in [('precision',pp[i]),('recall',rr[i]),('f1',ff[i])]:
                near(value,m['stage_metrics'][stage][key]);stage_values.setdefault((n,stage,key),[]).append(float(value))
        if n=='Clean':assert m['snr_improvement_db'] is None
        else:near(20*np.log10(m['rmse_noisy_uv']/m['rmse_denoised_uv']),m['snr_improvement_db'])
        assert np.isfinite(m['rmse_denoised_uv']) and m['denoise_and_inference_ms_per_epoch']>0
        allpred[n][0].append(y);allpred[n][1].append(pr)
    foldrows.append(dict(fold=f+1,selected_epoch=sel['selected_epoch'],epochs_run=sel['epochs_run'],
        clean_macro_f1_pct=float(lookup[f,'Clean']['macro_f1_pct']),
        noisy_macro_f1_pct=float(np.mean([float(lookup[f,n]['macro_f1_pct']) for n in conditions[1:]]))))
assert total==423080
for (n,stage,key),vals in stage_values.items():
    near(np.mean(vals),stage_rows[n,stage]['mean_'+key]);near(np.std(vals,ddof=1),stage_rows[n,stage]['sd_'+key])
for r in summary:
    n=r['condition'];vals=[float(lookup[f,n]['macro_f1_pct']) for f in range(5)]
    near(np.mean(vals),r['mean_macro_f1_pct']);near(np.std(vals,ddof=1),r['sd_macro_f1_pct'])
    y=np.concatenate(allpred[n][0]);pr=np.concatenate(allpred[n][1])
    assert confusion_matrix(y,pr,labels=range(5)).tolist()==pooled[n]['confusion_matrix']
    near(f1_score(y,pr,labels=range(5),average='macro',zero_division=0),pooled[n]['macro_f1'])
overall=load(OUT/'overall.json');nv=[r['noisy_macro_f1_pct'] for r in foldrows]
near(np.mean(nv),overall['noisy_macro_f1_mean_pct']);near(np.std(nv,ddof=1),overall['noisy_macro_f1_sd_pp'])
HAND.mkdir(parents=True,exist_ok=True);fig=HAND/'figures';fig.mkdir(exist_ok=True)
for name in ['test_per_fold.csv','test_summary.csv','stage_summary.csv','pooled_details.json','overall.json','all_selections_frozen.json','manifest.json','data_audit.json','e2_reproduction.json']:
    shutil.copy2(OUT/name,HAND/name)
for f in range(5):
    dest=HAND/f'fold{f}';dest.mkdir(exist_ok=True)
    for name in ['history.json','selection.json','test_metrics.json']:shutil.copy2(OUT/f'fold{f}'/name,dest/name)
    shutil.copytree(OUT/f'fold{f}/validation',dest/'validation',dirs_exist_ok=True)
audit=dict(status='passed',prediction_files=50,predictions_checked=total,checks=['sample IDs and ground-truth labels against original test NPZ','50 confusion matrices, accuracy, Macro-F1, kappa, stage precision/recall/F1','pooled confusion matrices and Macro-F1','validation selection across all epochs and checkpoint hashes','condition means and sample SD; nine-noise within-fold averaging','RMSE/SNR algebraic consistency; not a fresh signal reconstruction rerun'],folds=foldrows)
save(HAND/'final_verification.json',audit)
comparison=[]
for r in summary:
    n=r['condition'];comparison.append(dict(condition=n,e2=float(e2[n]['e2_mean']),e4=float(e2[n]['e4_mean']),e5=float(r['mean_macro_f1_pct']),e5_sd=float(r['sd_macro_f1_pct']),delta_e5_e2=float(r['mean_macro_f1_pct'])-float(e2[n]['e2_mean'])))
with (HAND/'e2_e4_e5_comparison.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(comparison[0]));w.writeheader();w.writerows(comparison)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig1,ax=plt.subplots(figsize=(12,5));xx=np.arange(10)
for offset,key,label,color in [(-.25,'e2','E2: Noisy baseline','#8894a0'),(0,'e4','E4: Wavelet','#e0a34e'),(.25,'e5','E5: DAE','#315880')]:
    ax.bar(xx+offset,[r[key] for r in comparison],.25,label=label,color=color)
ax.set_xticks(xx,[r['condition'].replace('_','\n') for r in comparison]);ax.set_ylabel('Five-fold mean Macro-F1 (%)');ax.set_ylim(0,100);ax.legend(ncol=3);fig1.tight_layout();fig1.savefig(fig/'condition_comparison.png',dpi=180);plt.close(fig1)
fig1,ax=plt.subplots(figsize=(8,4));means=[58.14146253053502,55.9807818796,62.930486337362495,np.mean(nv)]
ax.bar(['E2 baseline','E3 band-pass','E4 wavelet','E5 DAE'],means,color=['#8894a0','#a8bdc9','#e0a34e','#315880'])
for i,v in enumerate(means):ax.text(i,v+1,f'{v:.2f}%',ha='center')
ax.set_ylim(0,85);ax.set_ylabel('Nine-noise mean Macro-F1 (%)');fig1.tight_layout();fig1.savefig(fig/'overall_comparison.png',dpi=180);plt.close(fig1)
fig1,axs=plt.subplots(1,2,figsize=(11,4))
for f in range(5):
    hist=load(OUT/f'fold{f}/history.json');x=[h['epoch'] for h in hist]
    axs[0].plot(x,[h['train_mse_scaled'] for h in hist],label=f'Fold {f+1}')
    axs[1].plot(x,[h['validation_macro_f1']*100 for h in hist],label=f'Fold {f+1}')
axs[0].set_ylabel('Training MSE / training RMS squared');axs[1].set_ylabel('Ten-condition validation Macro-F1 (%)')
for ax in axs:ax.set_xlabel('Epoch');ax.legend()
fig1.tight_layout();fig1.savefig(fig/'training_validation_curves.png',dpi=180);plt.close(fig1)
fig1,axs=plt.subplots(2,5,figsize=(17,7))
for ax,n in zip(axs.flat,conditions):
    cm=np.asarray(pooled[n]['confusion_matrix']);norm=cm/cm.sum(axis=1,keepdims=True)
    ax.imshow(norm,vmin=0,vmax=1,cmap='Blues');ax.set_title(n.replace('_',' '));ax.set_xticks(range(5),['W','N1','N2','N3','REM'],rotation=45);ax.set_yticks(range(5),['W','N1','N2','N3','REM'])
    for i in range(5):
        for j in range(5):ax.text(j,i,f'{100*norm[i,j]:.0f}',ha='center',va='center',fontsize=7,color='white' if norm[i,j]>.5 else 'black')
fig1.suptitle('Pooled confusion matrices: row percentages (true rows, predicted columns)');fig1.tight_layout();fig1.savefig(fig/'pooled_confusion_matrices.png',dpi=180);plt.close(fig1)
clean=next(r for r in summary if r['condition']=='Clean')
lines=['# E5 DAE 实验结果与交接','',
f"九噪声 Test Macro-F1：**{np.mean(nv):.2f}% ± {np.std(nv,ddof=1):.2f}%**（五折均值 ± 样本标准差）。相对 E2 提高 **{np.mean(nv)-58.14146253053502:.2f} 个百分点**，相对 E4 提高 **{np.mean(nv)-62.930486337362495:.2f} 个百分点**。",
f"Clean Macro-F1：**{float(clean['mean_macro_f1_pct']):.2f}% ± {float(clean['sd_macro_f1_pct']):.2f}%**，相对 E2 降低 **{75.08227730767501-float(clean['mean_macro_f1_pct']):.2f} 个百分点**。",'',
'## 协议与核验','',
'五折受试者划分、Noise v2、样本 ID、E1 权重保持一致。E2 的 423,080 条预测复现差异为零。训练只使用当前折训练片段及对应 Clean 参考；验证十条件等权平均 Macro-F1 选模。所有五折选模冻结后进行 CPU 测试。没有用测试结果调整配置。',
'独立核验通过：50 个预测文件、423,080 条样本 ID/真实标签、分类指标、混淆矩阵、选模记录和 checkpoint 哈希。NPZ 文件哈希不同于 E4 重新打包文件，不能声称文件逐字节相同；基线预测复现已单独核验。','',
'## 五折结果','', '| 折 | 训练轮数 | 选中轮次 | Clean F1 (%) | 九噪声 F1 (%) |','|---|---:|---:|---:|---:|']
for r in foldrows:lines.append(f"| {r['fold']} | {r['epochs_run']} | {r['selected_epoch']} | {r['clean_macro_f1_pct']:.2f} | {r['noisy_macro_f1_pct']:.2f} |")
lines+=['','## 分条件比较','','| 条件 | E2 F1 (%) | E4 F1 (%) | E5 F1 (%) | E5 − E2 (pp) |','|---|---:|---:|---:|---:|']
for r in comparison:lines.append(f"| {r['condition']} | {r['e2']:.2f} | {r['e4']:.2f} | {r['e5']:.2f} ± {r['e5_sd']:.2f} | {r['delta_e5_e2']:+.2f} |")
lines+=['','## 解释与限制','',
'DAE 的平均优势主要来自中、强漂移和中、强高斯噪声。其高斯噪声表现仍低于 E4，小波与 DAE 的优势条件不同。Clean、低高斯以及低/中 Burst 存在下降，不能概括为所有条件均改善。',
'第1、3折在第7轮提前停止，均选择第1轮；其他折运行30轮。强高斯条件跨折标准差很大（约26.15个百分点），模型学习效果不均匀。这是预先固定的提前停止规则产生的结果，不据测试表现重训或换参。单随机种子、有限训练预算和模拟噪声限制应在报告中说明，不声称统计显著性。',
'RMSE/SNR以预处理Clean EEG为参考，不代表绝对无伪迹。低高斯条件平均SNR improvement为负；即使重建改善，分类也不一定同步改善。SNR improvement按每折所有采样点的误差能量计算，再对五折取平均。',
f"训练加每轮验证总计约 {sum(s['training_and_validation_seconds'] for s in sels)/3600:.2f} 小时。CPU去噪加分类平均约 {np.mean([float(r['mean_denoise_and_inference_ms_per_epoch']) for r in summary]):.3f} ms/epoch，预热后测量，不含加噪、磁盘读取或GPU训练；跨机器不能直接比较耗时。",'',
'## 文件交接','',
'小文件及图表：`result/e5_dae/`。完整结果、五折 `best.pth` / `latest.pth`、50个逐样本预测 NPZ：`result/local/e5_dae/`，通过组内共享存储交接。GitHub不自动上传权重或预测大文件。',
'实现：`scripts/dae_model.py`、`scripts/run_e5_dae.py`；配置：`configs/e5_dae.json`；运行说明：`E5_RUN.md`；代码分支：`E5-dae`，实验实现提交 `887ff31`。','',
'E6可以独立使用同一Clean数据、划分、Noise v2和评价协议，不需要DAE的输出作为训练输入。此次未启动E6。']
(HAND/'E5_RESULT_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(dict(audit='passed',samples=total,overall=overall,folds=foldrows,handoff=str(HAND)),ensure_ascii=False))
