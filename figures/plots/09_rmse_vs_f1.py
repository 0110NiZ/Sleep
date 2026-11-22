"""Compare reconstruction error and classification under medium Gaussian noise."""
import _bootstrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_style import *


def main():
    d=pd.read_csv(DATA/'reconstruction_case.csv').set_index('method').loc[['E4','E5']]
    fig,axes=plt.subplots(1,2,figsize=(13.2,6.2));fig.patch.set_facecolor(PAPER)
    conf=[('rmse_uv','Reconstruction error','RMSE (μV)',8.0,'Lower is better'),
          ('macro_f1','Downstream classification','Macro-F1 (%)',80.0,'Higher is better')]
    for ax,(field,title,ylab,ylim,caption) in zip(axes,conf):
        ax.set_facecolor(PANEL)
        bars=ax.bar([0,1],d[field],width=.56,color=[YELLOW,MINT],edgecolor=INK,linewidth=.7,zorder=3)
        for b,v in zip(bars,d[field]):
            ax.text(b.get_x()+b.get_width()/2,v+ylim*.025,f'{v:.2f}',fontsize=17,ha='center',fontweight='bold')
        ax.set_xticks([0,1],['E4 · Wavelet','E5 · DAE']);ax.set_ylim(0,ylim)
        ax.set_ylabel(ylab);ax.set_title(title,fontsize=17,pad=21)
        ax.text(.96,.95,caption,transform=ax.transAxes,ha='right',va='top',color=SUBTLE,fontsize=11)
        tidy(ax)
    fig.suptitle('Lower RMSE does not always mean higher sleep-stage F1',fontsize=21,fontweight='bold',y=1.04)
    rmse_gain = d.at['E4', 'rmse_uv'] - d.at['E5', 'rmse_uv']
    f1_gain = d.at['E4', 'macro_f1'] - d.at['E5', 'macro_f1']
    fig.text(.5,.015,f'Same medium-Gaussian test segments and frozen classifier within each fold · DAE: −{rmse_gain:.2f} μV RMSE · Wavelet: +{f1_gain:.2f} pp F1',
             ha='center',fontsize=10,color=SUBTLE)
    fig.tight_layout(w_pad=2.4,rect=(0,.05,1,1))
    save(fig,'09_rmse_vs_f1')

if __name__=='__main__':main()
