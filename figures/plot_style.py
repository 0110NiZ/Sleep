"""Shared visual identity for AttnSleep figures.

Palette matches the five hex codes supplied by the project team.
All chart functions operate on source CSVs, never on hard-coded image pixels.
"""
from __future__ import annotations

from pathlib import Path
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patheffects

ROOT = Path(__file__).resolve().parent
BUILD = Path(os.environ.get('SLEEP_FIGURE_BUILD', ROOT))
DATA = BUILD / 'tables'
OUTPUT = BUILD / 'images'
OUTPUT.mkdir(parents=True, exist_ok=True)

BLUE = '#74A9C5'
MINT = '#C2E5CF'
YELLOW = '#EDDDAB'
PEACH = '#F2B8AE'
ROSE = '#DD7389'
PALETTE = [BLUE, MINT, YELLOW, PEACH, ROSE]

# Neutral text/background colors updated to a clean white figure background.
PAPER = '#FFFFFF'
PANEL = '#FFFFFF'
INK = '#304E60'
SUBTLE = '#697983'
GRID = '#DFE3E1'
WHITE = '#FFFFFF'
METHOD_COLORS = {'E2': ROSE, 'E3': BLUE, 'E4': YELLOW, 'E5': MINT, 'E6': PEACH}
NOISE_COLORS = {'Gaussian': BLUE, 'Drift': MINT, 'Burst': PEACH}

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 15,
    'axes.labelsize': 11,
    'axes.titleweight': 'bold',
    'axes.edgecolor': GRID,
    'axes.labelcolor': INK,
    'text.color': INK,
    'xtick.color': SUBTLE,
    'ytick.color': SUBTLE,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'grid.color': GRID,
    'grid.alpha': 0.72,
    'grid.linewidth': .8,
    'figure.facecolor': PAPER,
    'axes.facecolor': PANEL,
    'savefig.facecolor': PAPER,
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def canvas(figsize=(12.0, 6.75)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PANEL)
    return fig, ax


def tidy(ax, *, y_grid=True, hide_left=False):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(not hide_left)
    ax.spines['bottom'].set_color(GRID)
    if y_grid:
        ax.yaxis.grid(True, zorder=0)
    else:
        ax.grid(False)
    ax.set_axisbelow(True)


def legend(ax, *args, **kwargs):
    return ax.legend(*args, frameon=False, fontsize=10, labelcolor=INK, **kwargs)


def save(fig, name: str):
    """Save a high-resolution PNG in a deterministic location."""
    out = OUTPUT / f'{name}.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches='tight', pad_inches=.18)
    plt.close(fig)
    return out


def small_note(ax, value, x=.01, y=-.14, fontsize=9):
    ax.text(x, y, value, transform=ax.transAxes, color=SUBTLE,
            fontsize=fontsize, va='top', ha='left', clip_on=False)


def label_bar(ax, bar, label, *, fontsize=10, y_offset=1.4):
    x = bar.get_x() + bar.get_width()/2
    y = bar.get_height()
    ax.text(x, y + y_offset, label, ha='center', va='bottom',
            color=INK, fontweight='bold', fontsize=fontsize)


def pct(value, decimals=1):
    return f'{value:.{decimals}f}%'
