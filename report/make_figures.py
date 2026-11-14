"""Figures of the report, from the small result files the Kaggle notebooks brought back into outputs/.

    uv run python report/make_figures.py   # -> report/figures/

Sized for report.md built by pandoc (A4, 2 cm margins, 10 pt): \\textwidth measured at 483.70 pt.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from figure_style import CB_COLORS, measure, new_figure, panel_labels, save_figure, set_style  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).parent / "figures"

# measured from the pandoc build (\typeout of \the\textwidth); figures use 8 pt so that two panels
# (3.2 in each) clear the 27 x 19 em panel floor
measure("thesis", full_pt=483.69687, height_pt=731.23584, base_pt=8)
set_style("thesis")

LABELS = {
    "a0_tta": "A0 + D4 TTA",
    "join_rule": "join pieces",
    "rescue": "join + rescue dropped",
    "join_then_filter": "join, then filter",
    "strict_join": "strict keep rule + join",
    "strict": "strict keep rule",
}


def ablation(ax):
    """OOF PQ change of each post-processing variant against A0, with its paired-bootstrap 95 % interval."""
    d = pd.read_csv(ROOT / "outputs/cnn-baseline-oof-lab/oof_results.csv").dropna(subset=["d_a0_plain"])
    d = d[d["variant"].isin(LABELS)].sort_values("d_a0_plain")
    y = np.arange(len(d))
    colours = [CB_COLORS["vermillion"] if v < 0 else CB_COLORS["blue"] for v in d["d_a0_plain"]]
    ax.errorbar(
        d["d_a0_plain"], y,
        xerr=[d["d_a0_plain"] - d["lo_a0_plain"], d["hi_a0_plain"] - d["d_a0_plain"]],
        fmt="none", ecolor="0.4", capsize=2,
    )
    ax.scatter(d["d_a0_plain"], y, c=colours, zorder=3, s=14)
    ax.axvline(0, color="0.6", lw=0.6)
    ax.set_yticks(y, [LABELS[v] for v in d["variant"]])
    ax.set_xlabel(r"$\Delta$PQ vs A0 (707 OOF frames)")


def contrast(ax):
    """Absorption contrast of annotated pixels: what the network covers, what it leaves out, and the noise."""
    d = pd.read_csv(ROOT / "outputs/cnn-baseline-contrast-lab/contrast_rows.csv")
    bins = np.linspace(-0.05, 0.25, 61)
    series = [
        ("matched filaments", d.loc[d["status"] == "matched", "whole_contrast"], CB_COLORS["blue"]),
        ("near-miss, covered part", d.loc[d["status"] == "near_miss", "covered_contrast"], CB_COLORS["green"]),
        ("near-miss, left-out part", d.loc[d["status"] == "near_miss", "uncovered_contrast"], CB_COLORS["vermillion"]),
    ]
    for label, values, colour in series:
        ax.hist(values.dropna(), bins=bins, density=True, histtype="step", lw=1.2, color=colour, label=label)
    ax.axvline(d["spread"].median(), color="0.3", ls="--", lw=0.8, label=r"background 1$\sigma$")
    ax.set_xlabel("contrast $1 - I/I_{\\mathrm{bg}}$")
    ax.set_ylabel("density")
    ax.legend(frameon=False, loc="upper right")


fig, axes = new_figure(1, 2, span="full", aspect=0.72)
ablation(axes[0])
contrast(axes[1])
panel_labels(axes)
print(save_figure(fig, OUT, "fig1_ablation_contrast"))
