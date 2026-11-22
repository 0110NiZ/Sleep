"""Plot the change in Macro-F1 from E2 in each noise condition."""
import _bootstrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
from plot_style import *


def main():
    df=pd.read_csv(DATA/'condition_deltas.csv').set_index('method')
    data=df.to_numpy(dtype=float)
    cmap=LinearSegmentedColormap.from_list('user_dual',[ROSE,PEACH,PANEL,MINT,BLUE],N=256)
    norm=TwoSlopeNorm(vcenter=0,vmin=-12,vmax=50)
    fig,ax=canvas((16.2,5.8))
    im=ax.imshow(data,cmap=cmap,norm=norm,aspect='auto')
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v=data[i,j]
            ax.text(j,i,f'{v:+.1f}',ha='center',va='center',fontsize=12.8,
                    weight='bold' if abs(v)>=20 else 'normal',color=INK)
    ax.set_xticks([])
    ax.set_yticks(np.arange(4),['E3 · Band-pass','E4 · Wavelet','E5 · DAE','E6 · Augmentation'],fontsize=11)
    ax.tick_params(length=0,pad=9)
    for j,sev in enumerate(['Low','Med','High']*3):
        ax.text(j,-.62,sev,ha='center',va='center',fontsize=11,color=SUBTLE)
    for x in [2.5,5.5]:ax.axvline(x,color=INK,linewidth=2.1,alpha=.5)
    for x0,lab in [(1,'GAUSSIAN WHITE NOISE'),(4,'BASELINE DRIFT'),(7,'OSCILLATORY BURST')]:
        ax.text(x0,-1.13,lab,ha='center',va='center',fontsize=13,fontweight='bold',color=INK)
    for i in range(10):ax.axvline(i-.5,color=WHITE,lw=2)
    for i in range(5):ax.axhline(i-.5,color=WHITE,lw=2)
    ax.set_xlim(-.5,8.5);ax.set_ylim(3.52,-1.35)
    ax.set_title('Robustness gains depend on the noise condition',fontsize=20,pad=67)
    for spine in ax.spines.values():spine.set_visible(False)
    cbar=fig.colorbar(im,ax=ax,pad=.02,shrink=.78,aspect=26)
    cbar.set_label('Change in Macro-F1 (percentage points)',fontsize=11,color=INK)
    cbar.outline.set_visible(False)
    fig.text(.52,.045,'Red: worse than E2     ·     Blue/green: better than E2     ·     Same test noise and segments within each fold',
             ha='center',fontsize=10,color=SUBTLE)
    fig.subplots_adjust(left=.12,right=.91,top=.70,bottom=.18)
    save(fig,'08_condition_heatmap')

if __name__=='__main__':main()
