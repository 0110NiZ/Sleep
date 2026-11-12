"""Generate PPT pages 8–9 from archived formal results; no training/inference.
Run in repository root: python scripts/plot_ppt_p8_p9.py
Requires numpy, matplotlib. Refuses to overwrite an existing output directory.
"""
import argparse,csv,hashlib,json,statistics,subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
SOURCES={'E2':'result/e2_v2/e2_per_fold.csv','E3':'result/e3_bandpass/comparison/selected_per_fold.csv','E4':'result/e4_wavelet/test_per_fold.csv','E5':'result/e5_dae/test_per_fold.csv','E6':'result/e6_noise_augmentation/test_per_fold.csv'}
BLUE='#34658B'; GREEN='#348E8E'; BACK='#FBF7EF'
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
 table={(int(r['fold']),r['condition']):r for r in rows}
 assert len(rows)==len(table)==50,(p,'expected 50 unique rows')
 return table
def writecsv(p,rows):
 with p.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,default=ROOT/'reports/ppt_p8_p9');args=parser.parse_args();out=args.out
 if out.exists():raise FileExistsError(f'Refusing to overwrite {out}; use --out NEW_DIRECTORY')
 tables={k:read(ROOT/v) for k,v in SOURCES.items()};conditions=[x['name'] for x in json.loads((ROOT/'configs/noise_conditions_v2.json').read_text())['conditions']];expected={(i,c) for i in range(5) for c in conditions}
 for k,t in tables.items():
  assert set(t)==expected,k
  assert all(0<=float(r['macro_f1_pct'])<=100 for r in t.values()),k
 # Independent reference archive must agree with E2 for all 50 scores.
 ref=read(ROOT/'result/e4_wavelet/e2_reference_per_fold.csv')
 assert all(abs(float(ref[k]['macro_f1_pct'])-float(tables['E2'][k]['macro_f1_pct']))<1e-9 for k in expected)
 overall=[];folds=[]
 for m,t in tables.items():
  c=[float(t[f,'Clean']['macro_f1_pct']) for f in range(5)]
  n=[statistics.mean(float(t[f,k]['macro_f1_pct']) for k in conditions if k!='Clean') for f in range(5)]
  overall.append(dict(method=m,clean_mean=statistics.mean(c),clean_sd=statistics.stdev(c),noisy_mean=statistics.mean(n),noisy_sd=statistics.stdev(n),gap_pp=statistics.mean(c)-statistics.mean(n)))
  folds.extend(dict(method=m,fold=f,clean_macro_f1=c[f],nine_noise_macro_f1=n[f]) for f in range(5))
 comparison=[]
 for m in ['E4','E5']:
  rows=[tables[m][f,'Gaussian_Medium'] for f in range(5)]
  comparison.append(dict(method=m,condition='Gaussian_Medium',rmse_mean_uv=statistics.mean(float(r['rmse_denoised_uv']) for r in rows),macro_f1_mean=statistics.mean(float(r['macro_f1_pct']) for r in rows)))
 out.mkdir(parents=True);writecsv(out/'p8_overall.csv',overall);writecsv(out/'p8_per_fold.csv',folds);writecsv(out/'p9_gaussian_medium.csv',comparison)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':BACK,'axes.facecolor':BACK})
 names=['E2 Baseline','E3 Band-pass','E4 Wavelet','E5 DAE','E6 Augmentation'];c=np.array([r['clean_mean'] for r in overall]);n=np.array([r['noisy_mean'] for r in overall]);y=np.arange(5)
 fig,ax=plt.subplots(figsize=(12,6));ax.hlines(y,n,c,color='#A8B6BC',lw=3)
 ax.scatter(c,y,color=BLUE,s=75,label='Clean',zorder=3);ax.scatter(n,y,color=GREEN,s=75,label='Nine-noise mean',zorder=3)
 for i,(cv,nv) in enumerate(zip(c,n)):
  ax.annotate(f'{cv:.2f}',(cv,i),xytext=(0,14),textcoords='offset points',ha='center',color=BLUE,weight='bold')
  ax.annotate(f'{nv:.2f}',(nv,i),xytext=(0,-23),textcoords='offset points',ha='center',color=GREEN,weight='bold')
 ax.set(yticks=y,yticklabels=names,xlim=(0,80),ylim=(4.65,-.65),xlabel='Five-fold mean Macro-F1 (%)',title='Clean vs. Noisy Performance');ax.set_xticks(np.arange(0,81,20));ax.grid(axis='x',alpha=.2);ax.legend(loc='upper left',frameon=False)
 fig.text(.5,.025,'Smaller gaps alone do not establish better robustness. E3 also lowers clean performance.',ha='center',fontsize=10,color='#607482');fig.tight_layout(rect=(0,.065,1,1))
 for ext in ['png','svg','pdf']:fig.savefig(out/f'p8_clean_noisy_dumbbell.{ext}',dpi=300)
 plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(11,5.4))
 for ax,key,title,label,limit in [(axes[0],'rmse_mean_uv','Reconstruction error','RMSE (µV), lower is better',8),(axes[1],'macro_f1_mean','Sleep-stage classification','Macro-F1 (%), higher is better',80)]:
  vals=[r[key] for r in comparison];bars=ax.bar(['E4 Wavelet','E5 DAE'],vals,color=[BLUE,GREEN],width=.6);ax.set(title=title,ylabel=label,ylim=(0,limit));ax.bar_label(bars,fmt='%.2f',padding=5);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
 rmse_diff=comparison[0]['rmse_mean_uv']-comparison[1]['rmse_mean_uv'];f1_diff=comparison[0]['macro_f1_mean']-comparison[1]['macro_f1_mean']
 fig.suptitle('Gaussian Medium: Reconstruction and Classification');fig.text(.5,.025,f'DAE: {rmse_diff:.2f} µV lower RMSE, but {f1_diff:.2f} pp lower Macro-F1. Differences use unrounded means.',ha='center',fontsize=10,color='#607482');fig.tight_layout(rect=(0,.075,1,.94))
 for ext in ['png','svg','pdf']:fig.savefig(out/f'p9_reconstruction_classification.{ext}',dpi=300)
 plt.close(fig)
 inputs=list(SOURCES.values())+['result/e4_wavelet/e2_reference_per_fold.csv','configs/noise_conditions_v2.json']
 provenance={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'sources_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in inputs},'python':__import__('sys').version,'matplotlib':matplotlib.__version__,'numpy':np.__version__,'e6_vs_e2_noisy_pp':float(n[-1]-n[0]),'e6_vs_e2_clean_pp':float(c[-1]-c[0]),'p9_rmse_difference_uv':rmse_diff,'p9_macro_f1_difference_pp':f1_diff}
 (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
 (out/'README.md').write_text('''# PPT pages 8–9 figures

Generated from formal archived test CSVs, not rounded PPT labels. No training or inference.

- Page 8: clean Macro-F1 and equally weighted nine-noise Macro-F1. Average nine conditions within each fold, then average five folds. CSV SD is sample SD over five fold values. E3 uses validation-selected per-fold filters.
- Page 9: Gaussian_Medium only. Average per-fold reported RMSE and Macro-F1 for E4 and E5. Both methods use corresponding frozen E1 classifiers. The reported RMSE is averaged from archived fold metrics, not recomputed from pooled waveforms.
- `p8_per_fold.csv` supports later paired descriptive analysis. No statistical significance is implied and folds are not independent repetitions.
- PNG: 300 dpi for slides. SVG/PDF: vector outputs. SVG retains text. These are figure files; they do not contain PowerPoint chart workbooks. Source CSV and plotting script support regeneration.
- Differences use unrounded source values; subtraction of displayed two-decimal labels can differ by 0.01 pp.
- `provenance.json` records input hashes and source commit. Original result files remain unchanged.

From repository root in the Sleep environment:

```bash
source /home/nzd/Sleep/.venv/bin/activate
python scripts/plot_ppt_p8_p9.py --out /tmp/sleep-p8-p9-new
```

Use a new output directory. Requirements: numpy and matplotlib. The default output is `reports/ppt_p8_p9`; the script refuses to overwrite an existing directory.
''',encoding='utf-8')
 print(json.dumps({'overall':overall,'p9':comparison,'differences':{k:v for k,v in provenance.items() if '_pp' in k or 'difference' in k}},indent=2))
if __name__=='__main__':main()
