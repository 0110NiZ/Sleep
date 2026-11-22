"""Compare noisy F1 for the five sleep stages."""
import _bootstrap
import pandas as pd
import numpy as np
from plot_style import *


def main():
    df=pd.read_csv(DATA/'stage_f1.csv')
    x=np.arange(len(df));width=.24
    fig,ax=canvas((13.6,6.4))
    for i,(name,color) in enumerate([('E2',ROSE),('E5',MINT),('E6',PEACH)]):
        bs=ax.bar(x+(i-1)*width,df[name],width=width,color=color,edgecolor=INK,lw=.48,label=name,zorder=3)
        for b,v in zip(bs,df[name]):
            ax.text(b.get_x()+b.get_width()/2,v+1.25,f'{v:.1f}',fontsize=9,ha='center',va='bottom',color=INK)
    ax.set_xticks(x,df.stage);ax.set_ylim(0,100);ax.set_ylabel('Stage-wise F1 (%)')
    ax.set_title('E6 improves all five stages, but N1 remains the weakest',pad=23,fontsize=18)
    tidy(ax);legend(ax,ncol=3,loc='upper left')
    ax.axvspan(.5,1.5,color=YELLOW,alpha=.18,zorder=0)
    n1 = df.set_index('stage').at['N1', 'E6']
    ax.text(1,95,f'N1: E6 = {n1:.2f}%',color=INK,ha='center',fontsize=12,fontweight='bold')
    small_note(ax,'Five-fold mean across nine noisy conditions. F1 alone cannot explain the remaining N1 errors.',y=-.16)
    fig.tight_layout();save(fig,'10_stagewise_f1')

if __name__=='__main__':main()
