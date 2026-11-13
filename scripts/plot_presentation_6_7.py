#!/usr/bin/env python3
"""Render COMP4330 Group 7 presentation figures for slides 6 and 7.

NO training/inference, and NO PowerPoint manipulation.
The plots use the group's archived formal Test Macro-F1 scores (E2–E6).

From the extracted package:
    python plot_presentation_6_7.py --out figures

From the Sleep repository root, with archived 50-row fold files present:
    python scripts/plot_presentation_6_7.py --repo . --out reports/figures/presentation_6_7

Requires numpy and matplotlib. Output: two PNG and two SVG images.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

# Exactly the six colors supplied by the user, in RGB hexadecimal.
PALETTE = {
    'dusty_blue': '#9EAAD1',
    'coral': '#F59790',
    'lavender': '#DACFE5',
    'apricot': '#F5DBB6',
    'ice_blue': '#CDE2E8',
    'pale_blue': '#C8D4E9',
}
BAR_COLORS = [
    PALETTE['coral'],       # E2
    PALETTE['dusty_blue'],  # E3
    PALETTE['apricot'],     # E4
    PALETTE['lavender'],    # E5
    PALETTE['ice_blue'],    # E6
]
METHODS = ('E2', 'E3', 'E4', 'E5', 'E6')
CONDITIONS = (
    'Clean',
    'Gaussian_Low', 'Gaussian_Medium', 'Gaussian_High',
    'Drift_Low', 'Drift_Medium', 'Drift_High',
    'Burst_Low', 'Burst_Medium', 'Burst_High',
)
DISPLAY_NAMES = ['E2\nNoisy baseline', 'E3\nBand-pass', 'E4\nWavelet', 'E5\nDAE', 'E6\nAugmentation']
INK = '#394854'
MUTED = '#62707B'
GRID = '#E6E6E4'
WHITE = '#FFFFFF'


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as fh:
        return list(csv.DictReader(fh))


def load_saved_summary(data_dir: Path):
    """Use the project's checked summary, including original 5-fold SD."""
    conditions_path = data_dir / 'condition_fivefold_means.csv'
    overall_path = data_dir / 'overall_nine_noisy.csv'
    if not conditions_path.exists() or not overall_path.exists():
        raise FileNotFoundError(f'Expected verified summary CSVs in {data_dir}')
    by_cond = {(r['method'], r['condition']): float(r['macro_f1_pct'])
               for r in csv_rows(conditions_path)}
    overall = {r['method']: (float(r['mean_macro_f1_pct']), float(r['sd_fold_means_pp']))
               for r in csv_rows(overall_path)}
    return by_cond, overall


def load_repo_results(repo: Path):
    """Recompute figures from current Sleep/main archived formal test CSVs.

    E3 must use the validation-selected official result, not arbitrary filter results.
    Each method needs 5 folds x (Clean + 9 noise conditions).
    """
    sources = {
        'E2': 'result/e4_wavelet/e2_reference_per_fold.csv',
        'E3': 'result/e3_bandpass/comparison/selected_per_fold.csv',
        'E4': 'result/e4_wavelet/test_per_fold.csv',
        'E5': 'result/e5_dae/test_per_fold.csv',
        'E6': 'result/e6_noise_augmentation/test_per_fold.csv',
    }
    expected = {(fold, cond) for fold in range(5) for cond in CONDITIONS}
    by_cond = {}
    overall = {}
    for method, relative_path in sources.items():
        file_path = repo / relative_path
        if not file_path.is_file():
            raise FileNotFoundError(f'Missing {method}: {file_path} (check the main branch layout)')
        rows = csv_rows(file_path)
        table = {(int(r['fold']), r['condition']): float(r['macro_f1_pct']) for r in rows}
        if len(rows) != 50 or set(table) != expected:
            raise ValueError(f'{method} must contain exactly 50 unique fold-condition rows: {file_path}')
        if any(not math.isfinite(v) or not 0 <= v <= 100 for v in table.values()):
            raise ValueError(f'Invalid Macro-F1 percentages in {file_path}')
        for cond in CONDITIONS:
            by_cond[method, cond] = statistics.mean(table[fold, cond] for fold in range(5))
        noisy_per_fold = [statistics.mean(table[fold, cond] for cond in CONDITIONS[1:])
                          for fold in range(5)]
        overall[method] = (statistics.mean(noisy_per_fold), statistics.stdev(noisy_per_fold))
    return by_cond, overall


def validate_inputs(by_cond, overall):
    if set(by_cond) != {(m, c) for m in METHODS for c in CONDITIONS}:
        raise ValueError('A five-method x ten-condition summary is required')
    if set(overall) != set(METHODS):
        raise ValueError('The overall summary is missing one or more methods')
    for method in METHODS:
        vals = [by_cond[method, cond] for cond in CONDITIONS]
        if any(not math.isfinite(v) or not 0 <= v <= 100 for v in vals):
            raise ValueError(f'Invalid condition summary for {method}')
        mean = statistics.mean(vals[1:])
        reported_mean, reported_sd = overall[method]
        if abs(mean - reported_mean) > 1e-6:
            raise ValueError(f'{method}: nine-noise mean does not match the condition table')
        if reported_sd < 0 or not math.isfinite(reported_sd):
            raise ValueError(f'{method}: invalid SD')
    # Prevent accidental mixing of newer/older incompatible archives.
    checks = {
        ('E2', 'Clean'): 75.08227730767501,
        ('E2', 'Gaussian_High'): 20.098309256235233,
        ('E4', 'Gaussian_High'): 64.54574599998818,
        ('E6', 'Gaussian_High'): 70.07656612541832,
    }
    for key, expected in checks.items():
        if not math.isclose(by_cond[key], expected, abs_tol=1e-4):
            raise ValueError(f'Formal archive mismatch for {key}: found {by_cond[key]}')


def blend(color1: str, color2: str, t: float) -> tuple[float, float, float]:
    """Interpolate two exact palette endpoints; t clamped to [0,1]."""
    t = max(0.0, min(1.0, t))
    def rgb(value):
        return np.asarray([int(value[k:k + 2], 16) / 255 for k in (1, 3, 5)])
    result = rgb(color1) * (1 - t) + rgb(color2) * t
    return tuple(result.tolist())


def heat_color(delta_pp: float):
    # Red = below E2, cool blue = above E2, nearly white = near zero.
    # Pale/mid/deep positive shades are all from the supplied six-color palette.
    if delta_pp < 0:
        return blend(WHITE, PALETTE['coral'], abs(delta_pp) / 12.0)
    if delta_pp <= 12:
        return blend(WHITE, PALETTE['ice_blue'], delta_pp / 12)
    if delta_pp <= 30:
        return blend(PALETTE['ice_blue'], PALETTE['pale_blue'], (delta_pp - 12) / 18)
    return blend(PALETTE['pale_blue'], PALETTE['dusty_blue'], (delta_pp - 30) / 22)


def draw_slide6(overall, output_dir: Path):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13})
    fig, ax = plt.subplots(figsize=(13.2, 5.0), dpi=300)
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)
    fig.subplots_adjust(left=0.077, right=0.985, bottom=0.225, top=0.89)

    xx = np.arange(5)
    means = np.asarray([overall[m][0] for m in METHODS])
    sds = np.asarray([overall[m][1] for m in METHODS])
    bars = ax.bar(xx, means, color=BAR_COLORS, width=0.66, edgecolor='#FFFFFF', linewidth=0.8, zorder=3)
    ax.errorbar(xx, means, yerr=sds, fmt='none', ecolor='#74818B',
                capsize=5, elinewidth=1.55, capthick=1.35, zorder=4)

    for bar, mean, sd in zip(bars, means, sds):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + sd + 1.5,
                f'{mean:.2f}%', ha='center', va='bottom', fontsize=15.5,
                fontweight='bold', color=INK)

    ax.set_xticks(xx, labels=DISPLAY_NAMES)
    ax.tick_params(axis='x', length=0, pad=9, labelsize=12.6, colors=INK)
    ax.set_ylim(0, 94)
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.tick_params(axis='y', length=0, pad=8, labelsize=12.5, colors=INK)
    ax.set_ylabel('Macro-F1 (%)', fontsize=13.5, labelpad=12, color=INK)
    ax.grid(axis='y', color=GRID, linewidth=1, zorder=0)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    for side in ('bottom', 'left'):
        ax.spines[side].set_color('#D6D9D9')
    ax.text(1, 1.025, 'Error bars: five-fold sample SD', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=10.5, color=MUTED)

    fig.savefig(output_dir / 'slide6_overall_morandi.png', dpi=300, facecolor=WHITE)
    fig.savefig(output_dir / 'slide6_overall_morandi.svg', format='svg', facecolor=WHITE)
    plt.close(fig)


def draw_slide7(by_cond, output_dir: Path):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 14})
    row_methods = METHODS[1:]
    rows = ['E3 Band-pass', 'E4 Wavelet', 'E5 DAE', 'E6 Augmentation']
    matrix = np.asarray([[by_cond[m, c] - by_cond['E2', c] for c in CONDITIONS[1:]]
                         for m in row_methods], dtype=float)
    fig, ax = plt.subplots(figsize=(16.0, 5.0), dpi=300)
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)
    fig.subplots_adjust(left=0.16, right=0.99, top=0.73, bottom=0.16)

    for i in range(4):
        for j in range(9):
            value = matrix[i, j]
            ax.add_patch(Rectangle((j - .5, i - .5), 1, 1,
                                   facecolor=heat_color(value), edgecolor=WHITE, linewidth=2.7))
            ax.text(j, i, f'{value:+.1f}', ha='center', va='center',
                    fontsize=14.1, fontweight='semibold', color=INK)

    ax.set_xlim(-.5, 8.5)
    ax.set_ylim(3.5, -.5)
    ax.set_xticks(np.arange(9), labels=['Low', 'Medium', 'High'] * 3)
    ax.xaxis.tick_top()
    ax.tick_params(axis='x', length=0, pad=9, labelsize=12.7, colors=INK)
    ax.set_yticks(np.arange(4), labels=rows)
    ax.tick_params(axis='y', length=0, pad=13, labelsize=13.7, colors=INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for vline in (2.5, 5.5):
        ax.axvline(vline, color='#B1B4B7', linewidth=2.4, zorder=5)

    for pos, name in ((1, 'GAUSSIAN WHITE NOISE'), (4, 'BASELINE DRIFT'), (7, 'OSCILLATORY BURST')):
        ax.text(pos, 1.29, name, transform=ax.get_xaxis_transform(),
                ha='center', va='center', fontsize=12.4, fontweight='bold', color=INK)

    fig.text(.16, .025, 'Negative  −  worse than E2', ha='left', color='#A96B66', fontsize=10.6)
    fig.text(.57, .025, '0  no change', ha='center', color=MUTED, fontsize=10.6)
    fig.text(.95, .025, 'Positive  +  better than E2', ha='right', color='#687CA9', fontsize=10.6)

    fig.savefig(output_dir / 'slide7_noise_conditions_morandi.png', dpi=300, facecolor=WHITE)
    fig.savefig(output_dir / 'slide7_noise_conditions_morandi.svg', format='svg', facecolor=WHITE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repo', type=Path, help='Use the five formal Test CSVs in the current Sleep repository')
    script = Path(__file__).resolve()
    repo_data_dir = script.parent.parent / 'reports' / 'figures' / 'presentation_6_7' / 'data'
    default_data_dir = repo_data_dir if repo_data_dir.is_dir() else script.parent / 'data'
    parser.add_argument('--data-dir', type=Path, default=default_data_dir)
    parser.add_argument('--out', type=Path, default=script.parent / 'figures')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.repo is None:
        by_cond, overall = load_saved_summary(args.data_dir)
        provenance = f'verified offline CSV: {args.data_dir}'
    else:
        by_cond, overall = load_repo_results(args.repo)
        provenance = f'formal test rows from local repo: {args.repo}'
    validate_inputs(by_cond, overall)
    draw_slide6(overall, args.out)
    draw_slide7(by_cond, args.out)
    print('Source:', provenance)
    print('Overall (nine noisy conditions; five folds):')
    for method in METHODS:
        print(f'  {method}: {overall[method][0]:.4f}%  (sample SD {overall[method][1]:.4f} pp)')
    print('Saved:', args.out / 'slide6_overall_morandi.png')
    print('Saved:', args.out / 'slide7_noise_conditions_morandi.png')


if __name__ == '__main__':
    main()
