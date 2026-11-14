"""Generate robustness comparison figures from archived formal results; no training/inference.
Run in repository root: python scripts/plot_robustness_comparison.py
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
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
 table={(int(r['fold']),r['condition']):r for r in rows}
 assert len(rows)==len(table)==50,(p,'expected 50 unique rows')
 return table
def writecsv(p,rows):
 with p.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,default=ROOT/'reports/figures/robustness_comparison');args=parser.parse_args();out=args.out
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
 out.mkdir(parents=True);writecsv(out/'clean_noisy_summary.csv',overall);writecsv(out/'clean_noisy_per_fold.csv',folds);writecsv(out/'reconstruction_classification_gaussian_medium.csv',comparison)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none','figure.facecolor':'white','axes.facecolor':'white'})
 c=np.array([r['clean_mean'] for r in overall]);n=np.array([r['noisy_mean'] for r in overall])
 green='#C2E5CF'
 # Match the model colors in the presentation's all-method noisy comparison.
 model_colors={'E2':'#F59790','E6':'#CDE2E8'}
 clean_rows=[dict(method=r['method'],clean_mean=r['clean_mean'],clean_sd=r['clean_sd'],color=model_colors[r['method']]) for r in overall if r['method'] in model_colors]
 writecsv(out/'clean_e2_e6_summary.csv',clean_rows)
 fig,ax=plt.subplots(figsize=(8.4,5.4))
 bars=ax.bar(['E2 Baseline','E6 Augmentation'],[r['clean_mean'] for r in clean_rows],color=[r['color'] for r in clean_rows],width=.48,zorder=3)
 ax.bar_label(bars,labels=[f"{r['clean_mean']:.2f}%" for r in clean_rows],padding=9,fontsize=16,fontweight='bold',color='#394854')
 ax.set_ylim(0,100);ax.set_ylabel('Clean Macro-F1 (%)',fontsize=13)
 ax.set_yticks(np.arange(0,101,20));ax.tick_params(axis='x',labelsize=14,length=0,pad=12)
 ax.tick_params(axis='y',colors='#394854');ax.set_axisbelow(True);ax.grid(axis='y',color='#E5E8EB',linewidth=.8)
 for side in ['top','right']:ax.spines[side].set_visible(False)
 for side in ['left','bottom']:ax.spines[side].set_color('#B7BEC4')
 fig.tight_layout(pad=1.8)
 for ext in ['png','svg','pdf']:fig.savefig(out/f'clean_e2_vs_e6_macro_f1.{ext}',dpi=300,facecolor='white')
 plt.close(fig)
 # Keep new text artifacts free of generated trailing whitespace.
 for name in ['clean_e2_e6_summary.csv','clean_e2_vs_e6_macro_f1.svg']:
  f=out/name;f.write_text('\n'.join(line.rstrip() for line in f.read_text().splitlines())+'\n',encoding='utf-8')
 fig,axes=plt.subplots(1,2,figsize=(9,4.5))
 for ax,key,label,limit in [(axes[0],'rmse_mean_uv','RMSE (µV)',8),(axes[1],'macro_f1_mean','Macro-F1 (%)',80)]:
  vals=[r[key] for r in comparison]
  bars=ax.bar(['E4 Wavelet','E5 DAE'],vals,color=['#EDDDAB',green],width=.55)
  ax.set_ylabel(label);ax.set_ylim(0,limit);ax.bar_label(bars,fmt='%.2f',padding=4)
 fig.suptitle('Gaussian Medium',fontsize=12)
 fig.tight_layout()
 for ext in ['png','svg','pdf']:fig.savefig(out/f'reconstruction_vs_classification_gaussian_medium.{ext}',dpi=300,facecolor='white')
 plt.close(fig)
 rmse_diff=comparison[0]['rmse_mean_uv']-comparison[1]['rmse_mean_uv'];f1_diff=comparison[0]['macro_f1_mean']-comparison[1]['macro_f1_mean']
 inputs=list(SOURCES.values())+['result/e4_wavelet/e2_reference_per_fold.csv','configs/noise_conditions_v2.json']
 provenance={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'plot_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'clean_comparison_colors':model_colors,'sources_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in inputs},'python':__import__('sys').version,'matplotlib':matplotlib.__version__,'numpy':np.__version__,'e6_vs_e2_noisy_pp':float(n[-1]-n[0]),'e6_vs_e2_clean_pp':float(c[-1]-c[0]),'p9_rmse_difference_uv':rmse_diff,'p9_macro_f1_difference_pp':f1_diff}
 (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
 (out/'README.md').write_text('''# Robustness comparison figures

Generated from formal archived test CSVs, not rounded PPT labels. No training or inference.

- Style: white background and two-decimal value labels. Clean comparison model colors: E2 coral #F59790 and E6 pale blue #CDE2E8, matching the presentation all-method comparison. E4/E5 colors: yellow #EDDDAB and green #C2E5CF. Clean comparison uses two vertical bars with a 0–100% axis; reconstruction/classification comparison uses two basic bar charts.
- Clean comparison: E2 and E6 clean Macro-F1 only (75.08% and 74.24%). The E6 minus E2 difference from unrounded means is -0.8431417448038587 percentage points, displayed as -0.84 pp. Clean means no additional synthetic noise. Clean values average the five clean fold scores. Full E2–E6 clean/noisy tables remain available for analysis. For their noisy mean, average nine conditions within each fold, then average five folds. CSV SD is sample SD over five fold values. E3 uses validation-selected per-fold filters.
- Reconstruction/classification comparison: Gaussian_Medium only. Average per-fold reported RMSE and Macro-F1 for E4 and E5. Both methods use corresponding frozen E1 classifiers. The reported RMSE is averaged from archived fold metrics, not recomputed from pooled waveforms.
- `clean_noisy_per_fold.csv` supports later paired descriptive analysis. No statistical significance is implied and folds are not independent repetitions.
- PNG: 300 dpi for slides. SVG/PDF: vector outputs. SVG retains text. These are figure files; they do not contain PowerPoint chart workbooks. Source CSV and plotting script support regeneration.
- Differences use unrounded source values; subtraction of displayed two-decimal labels can differ by 0.01 pp.
- `provenance.json` records input hashes and source commit. Original result files remain unchanged. The obsolete clean_vs_noisy_macro_f1 PNG/SVG/PDF has been replaced by clean_e2_vs_e6_macro_f1; the former plotting block is removed.

## Files and interpretation

| File | Contents |
|---|---|
| `clean_e2_vs_e6_macro_f1.png` | E2/E6 clean Macro-F1 only; zero-based vertical bars |
| `clean_e2_e6_summary.csv` | Full-precision clean means, sample SDs and model colors for the new figure |
| `reconstruction_vs_classification_gaussian_medium.png` | E4/E5 RMSE and Macro-F1 under Gaussian Medium; two bar charts |
| `clean_noisy_summary.csv` | Five-fold means, sample SDs and clean/noisy gaps |
| `clean_noisy_per_fold.csv` | Paired per-fold clean and noisy scores |
| `reconstruction_classification_gaussian_medium.csv` | Values plotted in the E4/E5 comparison |
| `provenance.json` | Formal input hashes, source commit and full-precision differences |

Each chart also has SVG and PDF vector versions. Labels show two decimals; source CSVs retain full precision. In the first figure coral means E2 and pale blue means E6; in the second yellow means E4 and green means E5. RMSE is lower-is-better, Macro-F1 higher-is-better. Both charts show descriptive means without uncertainty bars; sample SDs for the overall comparison are retained in the summary CSV. Gaussian Medium is one illustrative condition, not a conclusion about every condition.

From repository root in the Sleep environment:

```bash
source /home/nzd/Sleep/.venv/bin/activate
python scripts/plot_robustness_comparison.py --out /tmp/sleep-robustness-figures-new
```

Use a new output directory. Requirements: numpy and matplotlib. The default output is `reports/figures/robustness_comparison`; the script refuses to overwrite an existing directory.
''',encoding='utf-8')
 print(json.dumps({'overall':overall,'p9':comparison,'differences':{k:v for k,v in provenance.items() if '_pp' in k or 'difference' in k}},indent=2))
if __name__=='__main__':main()
