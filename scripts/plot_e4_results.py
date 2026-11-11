"""Plot E4 results against the locally reproduced E2 baseline."""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from project_paths import E2_DIR, RESULT_ROOT

def read(p):
 with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def main():
 out=RESULT_ROOT/'e4_wavelet';figs=out/'figures';figs.mkdir(exist_ok=True)
 e4=read(out/'test_per_fold.csv');e2=read(E2_DIR/'e2_per_fold.csv')
 def vals(rows,c):return np.array([float(r['macro_f1_pct']) for r in rows if r['condition']==c])
 fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
 for ax,kind in zip(axes,['Gaussian','Drift','Burst']):
  for rows,label,col in [(e2,'E2 noisy baseline','#64748b'),(e4,'E4 selected wavelet','#007c91')]:
   data=[vals(rows,f'{kind}_{s}') for s in ['Low','Medium','High']]
   ax.errorbar(range(3),[x.mean() for x in data],yerr=[x.std(ddof=1) for x in data],marker='o',capsize=3,label=label,color=col)
  ax.set_title(kind);ax.set_xticks(range(3),['Low','Medium','High']);ax.set_xlabel('Noise severity');ax.grid(alpha=.2);ax.set_ylim(0,100)
 axes[0].set_ylabel('Macro-F1 (%)');axes[2].legend(fontsize=8)
 fig.suptitle('Noise robustness: five-fold mean and sample SD');fig.tight_layout();fig.savefig(figs/'noise_robustness.png',dpi=180);plt.close(fig)
 fig,ax=plt.subplots(figsize=(7,4))
 for i,(rows,label,col) in enumerate([(e2,'E2 baseline','#64748b'),(e4,'E4 selected wavelet','#007c91')]):
  clean=vals(rows,'Clean');noisy=np.array([np.mean([float(r['macro_f1_pct']) for r in rows if int(r['fold'])==f and r['condition']!='Clean']) for f in range(5)])
  means=[clean.mean(),noisy.mean()];sd=[clean.std(ddof=1),noisy.std(ddof=1)]
  bars=ax.bar(np.arange(2)+(i-.5)*.34,means,.34,yerr=sd,capsize=4,label=label,color=col)
  ax.bar_label(bars,labels=[f'{v:.2f}' for v in means],padding=6)
 ax.set_xticks([0,1],['Clean','Nine-noise average']);ax.set_ylim(0,100);ax.set_ylabel('Macro-F1 (%)');ax.legend();ax.set_title('Robustness and clean-input performance');fig.tight_layout();fig.savefig(figs/'clean_vs_noisy.png',dpi=180);plt.close(fig)
 detail=json.loads((out/'pooled_details.json').read_text());fig,axes=plt.subplots(1,2,figsize=(10,4))
 for ax,c in zip(axes,['Clean','Gaussian_High']):
  cm=np.array(detail[c]['confusion_matrix']);norm=cm/cm.sum(axis=1,keepdims=True)*100
  ax.imshow(norm,cmap='Blues',vmin=0,vmax=100)
  for i in range(5):
   for j in range(5):ax.text(j,i,f'{norm[i,j]:.1f}',ha='center',va='center',color='white' if norm[i,j]>55 else 'black',fontsize=8)
  ax.set_xticks(range(5),['W','N1','N2','N3','REM']);ax.set_yticks(range(5),['W','N1','N2','N3','REM']);ax.set_xlabel('Predicted');ax.set_ylabel('True');ax.set_title(c+' (row %)')
 fig.suptitle('E4 pooled test confusion matrices');fig.tight_layout();fig.savefig(figs/'pooled_confusion_matrices.png',dpi=180);plt.close(fig)
 print('Saved figures to',figs)
if __name__=='__main__':main()
