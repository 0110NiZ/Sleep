"""Plot E2 frozen baseline sensitivity across Noise v2 conditions (white background)."""

import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "noise_conditions_v2.json"
PER_FOLD_CSV = ROOT / "result" / "e2_v2" / "e2_per_fold.csv"
SUMMARY_CSV = ROOT / "result" / "e2_v2" / "e2_fivefold_summary.csv"
OUT_DIR = ROOT / "result" / "local" / "e2_figures"


def load_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def format_level_label(cond: dict) -> str:
    ntype = cond["noise_type"]
    level = cond["name"].split("_")[1]
    if ntype == "gaussian":
        return f"{level}\n({cond['snr_db']:g} dB)"
    if ntype == "baseline_wander":
        return f"{level}\n({cond['amplitude_uv']:g} $\\mu$V)"
    if ntype == "burst":
        return f"{level}\n({cond['amplitude_ratio']:g}$\\sigma$)"
    return level


def main() -> None:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    cond_map = {c["name"]: c for c in cfg["conditions"]}

    per_fold_rows = load_csv(PER_FOLD_CSV)
    summary_rows = {r["condition"]: r for r in load_csv(SUMMARY_CSV)}

    stats = {}
    for cname in cond_map:
        vals = np.array(
            [float(r["macro_f1_pct"]) for r in per_fold_rows if r["condition"] == cname],
            dtype=np.float64,
        )
        assert len(vals) == 5, f"Expected 5 folds for {cname}, got {len(vals)}"
        mean_val = float(vals.mean())
        sd_val = float(vals.std(ddof=1))
        assert np.isclose(mean_val, float(summary_rows[cname]["mean_macro_f1_pct"]), atol=1e-6)
        assert np.isclose(sd_val, float(summary_rows[cname]["sd_macro_f1_pct"]), atol=1e-6)
        stats[cname] = (mean_val, sd_val)

    clean_mean, clean_sd = stats["Clean"]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial"],
        "mathtext.default": "regular",
        "font.size": 10,
        "axes.linewidth": 0.9,
        "axes.edgecolor": "#333333",
    })

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.1), dpi=300, sharey=True)
    fig.patch.set_facecolor("white")

    families = [
        ("Gaussian White Noise", "Gaussian"),
        ("Baseline Drift (0.5 Hz)", "Drift"),
        ("Oscillatory Burst (15–35 Hz)", "Burst"),
    ]
    levels = ["Low", "Medium", "High"]

    line_color = "#1f4e78"
    ref_color = "#595959"

    for idx, (ax, (panel_title, prefix)) in enumerate(zip(axes, families)):
        ax.set_facecolor("white")
        names = [f"{prefix}_{lv}" for lv in levels]
        x = np.arange(len(levels))
        means = np.array([stats[n][0] for n in names])
        sds = np.array([stats[n][1] for n in names])
        xlabels = [format_level_label(cond_map[n]) for n in names]

        ax.axhspan(clean_mean - clean_sd, clean_mean + clean_sd, color="#d9d9d9", alpha=0.28, zorder=1)
        ax.axhline(
            clean_mean,
            color=ref_color,
            linestyle="--",
            linewidth=1.3,
            zorder=2,
            label=f"E1 Clean Baseline ({clean_mean:.2f}% ± {clean_sd:.2f} pp)",
        )

        ax.errorbar(
            x,
            means,
            yerr=sds,
            fmt="-o",
            color=line_color,
            ecolor=line_color,
            elinewidth=1.25,
            capsize=4,
            capthick=1.25,
            markersize=6.2,
            linewidth=2.0,
            zorder=4,
            label="E2 Noisy Baseline (Mean ± sample SD)",
        )

        for xi, (m, s) in enumerate(zip(means, sds)):
            if m >= 62.0:
                y_pos = m - s - 3.2
                va = "top"
            else:
                y_pos = m + s + 3.0
                va = "bottom"
            ax.text(
                xi,
                y_pos,
                f"{m:.2f}%",
                ha="center",
                va=va,
                fontsize=9.2,
                fontweight="bold" if xi == 2 else "normal",
                color="#1a1a1a",
                zorder=5,
            )

        ax.set_title(panel_title, fontsize=11, fontweight="bold", color="#1a1a1a", pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=9.2, color="#222222")
        ax.set_xlim(-0.42, 2.42)
        ax.set_ylim(0, 96)
        ax.set_yticks([0, 20, 40, 60, 80])
        ax.grid(axis="y", linestyle=":", linewidth=0.75, color="#cccccc", zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(colors="#222222")

        if idx == 0:
            ax.set_ylabel("Test Macro-F1 (%)", fontsize=10.5, fontweight="bold", color="#1a1a1a")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=2,
        frameon=False,
        fontsize=9.5,
    )

    fig.subplots_adjust(left=0.065, right=0.985, top=0.81, bottom=0.17, wspace=0.08)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / "e2_baseline_sensitivity.png"
    fig.savefig(out_file, dpi=300, facecolor="white", transparent=False)
    plt.close(fig)
    print(f"[SAVED] {out_file}")


if __name__ == "__main__":
    main()
