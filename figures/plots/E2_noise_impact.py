"""Plot the E2 baseline response to three noise types and three strengths."""
import _bootstrap
import pandas as pd
import matplotlib.pyplot as plt
from plot_style import *


def main():
    df = pd.read_csv(DATA / 'baseline_noise.csv')
    clean = pd.read_csv(DATA / 'overall_performance.csv').set_index('method').at['E2', 'clean_f1']
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True)
    fig.patch.set_facecolor(PAPER)
    families = [('Gaussian','Gaussian white noise'), ('Drift','Baseline drift (0.5 Hz)'), ('Burst','Oscillatory burst')]
    for ax,(family,title) in zip(axs,families):
        sub = df[df.noise==family].set_index('severity').loc[['Low','Medium','High']]
        ys = sub.macro_f1.to_numpy()
        color = NOISE_COLORS[family]
        ax.set_facecolor(PANEL)
        ax.plot([0,1,2],ys,'-o',color=INK,linewidth=2.3,markersize=6,
                markerfacecolor=color,markeredgecolor=INK, zorder=4)
        ax.axhline(clean,color=ROSE,ls='--',lw=1.4,alpha=.85,zorder=1)
        ax.set_title(title,fontsize=15,pad=19)
        ax.set_xlim(-.25,2.25); ax.set_ylim(0,96)
        ax.set_xticks([0,1,2],labels=[f'{s}\n({u})' for s,u in zip(['Low','Medium','High'],sub.setting)])
        for i,v in enumerate(ys):
            ax.annotate(f'{v:.2f}%',(i,v),xytext=(0,-20 if v>72 else 11),
                        textcoords='offset points',ha='center',fontsize=11,fontweight='bold',color=INK)
        tidy(ax)
    axs[0].set_ylabel('Macro-F1 (%)')
    fig.suptitle('Noise impact on the frozen baseline (E2)', x=.5,y=1.055,fontsize=21,fontweight='bold',color=INK)
    fig.text(.51,-.015,f'Dashed line: E1 clean reference = {clean:.2f}%. Noise types are not calibrated to equal physical severity.',
             ha='center',color=SUBTLE,fontsize=10)
    fig.tight_layout(w_pad=2)
    save(fig,'E2_noise_impact')

if __name__=='__main__':main()
