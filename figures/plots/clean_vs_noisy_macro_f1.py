"""Compare clean and mean noisy Macro-F1 across the five methods."""
import _bootstrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from plot_style import *


def main():
    df = pd.read_csv(DATA / 'overall_performance.csv')
    x = np.arange(len(df))
    w = 0.34
    fig, ax = canvas((13.6, 5.65))
    # Neutral clean bars keep the method-color coding unambiguous.
    CLEAN_GRAY = '#D8DFE3'
    b1 = ax.bar(x-w/2, df.clean_f1, width=w, color=CLEAN_GRAY,
                edgecolor=INK, linewidth=0.55, zorder=3)
    b2 = ax.bar(x+w/2, df.mean_noisy_f1, width=w,
                color=[METHOD_COLORS[m] for m in df.method],
                edgecolor=INK, linewidth=0.55, zorder=3)
    for b, v in zip(b1, df.clean_f1):
        label_bar(ax, b, f'{v:.1f}%', fontsize=11.5, y_offset=1.8)
    for b, v in zip(b2, df.mean_noisy_f1):
        label_bar(ax, b, f'{v:.1f}%', fontsize=11.5, y_offset=1.8)

    # 100% tick now sits INSIDE the plot with headroom, not against the image crop.
    ax.set_ylim(0, 110)
    ax.set_xlim(-0.6, len(df)-0.4)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel('Macro-F1 (%)', fontsize=13)
    ax.set_xticks(x, labels=[f'{m}\n{n}' for m, n in zip(df.method, df.name)])
    ax.tick_params(axis='both', labelsize=11.5, pad=5)
    tidy(ax)
    legend_handles = [
        Patch(facecolor=CLEAN_GRAY, edgecolor=INK, linewidth=.55, label='Clean EEG'),
        Patch(facecolor=PEACH, edgecolor=INK, linewidth=.55, label='Noisy EEG (method-specific colors)'),
    ]
    ax.legend(handles=legend_handles, loc='upper left', frameon=False,
              fontsize=10.4, ncol=2, columnspacing=1.8, handlelength=1.4)
    ax.annotate('E3 also decreases clean-input F1',
                xy=(1-w/2, df.set_index('method').at['E3','clean_f1']), xytext=(1.48, 94),
                arrowprops={'arrowstyle':'->', 'color':ROSE, 'lw':1.6},
                ha='left', va='center', fontsize=11.0, color=INK)
    fig.text(0.5, 0.054,
             'Clean E2 is the E1 baseline · Noisy scores average 9 noise settings · Five-fold means',
             ha='center', va='center', color=SUBTLE, fontsize=9.6)
    # No redundant chart title: the slide already has a title.
    fig.subplots_adjust(left=0.071, right=0.987, top=0.93, bottom=0.205)
    save(fig, 'clean_vs_noisy_macro_f1')


if __name__ == '__main__':
    main()
